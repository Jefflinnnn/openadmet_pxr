"""
Fine-tuning Training Script for Molecular Property Prediction

This script implements training pipelines for both regression and classification tasks
on molecular property prediction using Graph Neural Networks (GNNs). It supports:

- Transfer learning with pre-trained model loading
- Distributed training across multiple GPUs/nodes
- Automatic Mixed Precision (AMP) training with bfloat16
- Model Exponential Moving Average (EMA) for improved generalization
- Flexible learning rate scheduling and optimization
- Comprehensive logging and checkpoint management

Usage:
    python main.py --mode regression --name <property_name> [additional arguments]
    python main.py --mode classification --name <property_name> [additional arguments]

Author: JunyiAn
Date: 2026-02-28
"""

import argparse
import gc
import os
import resource
import time
from datetime import datetime

import numpy as np
import torch
from torch_geometric.loader import DataLoader

# Logging and utilities
from logger import FileLogger
import utils

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# Model training and optimization utilities from timm (PyTorch Image Models)
from timm.utils import NativeScaler, ModelEmaV2
from timm.scheduler import create_scheduler
from optim_factory import create_optimizer

# Custom dataset and training modules
from suiren_datasets.org_mol2d import PP_smiles_2d
from engine import (
    train_one_epoch, evaluate, compute_stats,
    train_cls_one_epoch, evaluate_cls,
    MMPSampler,
)


def _load_resume_checkpoint(model, resume_path, logger):
    """Load model weights from a checkpoint, returning the full dict for optimizer/scheduler restore."""
    logger.info('Start loading checkpoint')
    ckpt = torch.load(resume_path, map_location=torch.device('cpu'))
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
        ckpt = None
    logger.info('Load checkpoint successfully')
    return ckpt


def _restore_training_state(resume_ckpt, optimizer, lr_scheduler, logger, warm_restart=False):
    """Restore optimizer/scheduler state from checkpoint dict. Returns start_epoch."""
    start_epoch = 0
    if resume_ckpt is not None:
        if warm_restart:
            logger.info('Warm restart: loaded weights only, fresh optimizer/scheduler')
        else:
            if "optimizer" in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
                logger.info('Restored optimizer state from checkpoint')
            if "epoch" in resume_ckpt:
                start_epoch = resume_ckpt["epoch"] + 1
                logger.info(f'Resuming from epoch {start_epoch}')
            if "lr_scheduler" in resume_ckpt:
                lr_scheduler.load_state_dict(resume_ckpt["lr_scheduler"])
        del resume_ckpt
    return start_epoch


def compute_rae(targets, preds, active_threshold=6.0):
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    active_mask = targets >= active_threshold
    inactive_mask = ~active_mask
    abs_err = np.abs(targets - preds)

    active_mae = float(abs_err[active_mask].mean()) if active_mask.sum() > 0 else float('nan')
    inactive_mae = float(abs_err[inactive_mask].mean()) if inactive_mask.sum() > 0 else float('nan')

    parts = []
    if active_mask.sum() > 0:
        parts.append(active_mae)
    if inactive_mask.sum() > 0:
        parts.append(inactive_mae)
    overall_rae = float(np.mean(parts)) if parts else float('nan')

    return overall_rae, active_mae, inactive_mae, int(active_mask.sum()), int(inactive_mask.sum())


def compute_active_rae(targets, preds, active_threshold=6.0, active_weight=2.0):
    """RAE variant that weights active subgroup more heavily for checkpoint selection."""
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    active_mask = targets >= active_threshold
    inactive_mask = ~active_mask
    abs_err = np.abs(targets - preds)

    active_mae = float(abs_err[active_mask].mean()) if active_mask.sum() > 0 else float('nan')
    inactive_mae = float(abs_err[inactive_mask].mean()) if inactive_mask.sum() > 0 else float('nan')

    if active_mask.sum() > 0 and inactive_mask.sum() > 0:
        return (active_weight * active_mae + inactive_mae) / (active_weight + 1.0)
    elif active_mask.sum() > 0:
        return active_mae
    elif inactive_mask.sum() > 0:
        return inactive_mae
    return float('nan')

def get_args_parser():
    """
    Parse command line arguments for model fine-tuning.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser for the training script
    """
    parser = argparse.ArgumentParser(
        'Fine-tuning for molecular property prediction',
        add_help=False
    )
    
    # ========================================================================
    # Model Configuration
    # ========================================================================
    parser.add_argument('--checkpoint-pretrain', type=str, default=None,
                        help='Path to pre-trained model checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to resume training from checkpoint')
    parser.add_argument('--warm-restart', action='store_true',
                        help='Load only model weights from --resume, reset optimizer/scheduler/epoch (fresh cosine restart)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--unfreeze-backbone', action='store_true',
                        help='Train the full model instead of freezing the pretrained backbone')
    
    # ========================================================================
    # Task Configuration
    # ========================================================================
    parser.add_argument('--mode', type=str, default='regression',
                        help='Run mode: regression or classification')
    parser.add_argument('--name', type=str, required=True,
                        help='Property name for dataset and experiment naming')
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'l2', 'balanced_mse'],
                        help='Loss function: l1 (MAE), l2 (MSE), or balanced_mse for regression')
    parser.add_argument('--balanced-mse-bins', type=int, default=20,
                        help='Number of bins for balanced MSE label density estimation')
    parser.add_argument('--balanced-mse-smoothing', type=float, default=1.0,
                        help='Smoothing constant for balanced MSE bin counts')
    parser.add_argument('--main-metric', type=str, default='MAE',
                        help='Primary metric for model selection (MAE, R2, RAE, active_rae, ACC, AUROC, AUPRC)')
    parser.add_argument('--class-num', type=int, default=2,
                        help='Number of classes for classification tasks')
    
    # ========================================================================
    # Data Configuration
    # ========================================================================
    parser.add_argument('--data-mode', type=str, default='smiles_random',
                        choices=['smiles_random', 'smiles_defined'],
                        help='Data loading mode: random split or predefined split')
    parser.add_argument('--tvt', action='store_true',
                        help='Use train/val/test split (default: train/val only)')
    parser.add_argument('--ratio', type=float, default=0.8,
                        help='Train/val split ratio for random splitting')
    parser.add_argument('--compute-stats', action='store_true', dest='compute_stats',
                        help='Compute and display dataset statistics only')
    
    # ========================================================================
    # Training Hyperparameters
    # ========================================================================
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    
    # ========================================================================
    # Optimizer Configuration
    # ========================================================================
    parser.add_argument('--opt', type=str, default='adamw',
                        help='Optimizer type (adamw, sgd, etc.)')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='L2 regularization coefficient')
    parser.add_argument('--opt-eps', type=float, default=1e-8,
                        help='Optimizer epsilon for numerical stability')
    parser.add_argument('--opt-betas', type=float, nargs='+', default=None,
                        help='Adam optimizer betas')
    parser.add_argument('--clip-grad', type=float, default=None,
                        help='Gradient clipping norm threshold')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum coefficient')
    
    # ========================================================================
    # Learning Rate Schedule
    # ========================================================================
    parser.add_argument('--sched', type=str, default='cosine',
                        help='Learning rate scheduler (cosine, step, etc.)')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='Number of warmup epochs')
    parser.add_argument('--warmup-lr', type=float, default=1e-6,
                        help='Warmup initial learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate for scheduler')
    parser.add_argument('--decay-epochs', type=float, default=30,
                        help='Epoch interval to decay learning rate')
    parser.add_argument('--decay-rate', type=float, default=0.1,
                        help='Learning rate decay rate')
    parser.add_argument('--cooldown-epochs', type=int, default=10,
                        help='Cooldown epochs after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10,
                        help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--early-stopping', type=int, default=0,
                        help='Stop training if val metric does not improve for N epochs (0 = disabled)')
    parser.add_argument('--active-weight', type=float, default=1.0,
                        help='Loss weight multiplier for active compounds (pEC50 >= 6). 1.0 = no upweighting')
    parser.add_argument('--tail-weight', type=float, default=1.0,
                        help='Loss weight multiplier for low-activity compounds (pEC50 <= tail-threshold)')
    parser.add_argument('--tail-threshold', type=float, default=3.0,
                        help='pEC50 threshold below which tail-weight applies')
    parser.add_argument('--mmp-pairs', type=str, default=None,
                        help='Path to MMP pairs JSON (from enumerate_mmps.py). Enables delta head auxiliary loss.')
    parser.add_argument('--mmp-weight', type=float, default=0.1,
                        help='Weight for MMP delta loss (default: 0.1)')
    parser.add_argument('--mmp-batch-size', type=int, default=16,
                        help='Number of MMP pairs per training step (default: 16)')
    parser.add_argument('--mmp-mode', type=str, default='main_head',
                        choices=['delta_head', 'main_head'],
                        help='MMP loss mode: delta_head (separate head on emb diff) or main_head (pred_a - pred_b)')
    parser.add_argument('--mmp-start-epoch', type=int, default=0,
                        help='Epoch at which MMP delta loss activates (two-stage training). 0 = from start.')
    parser.add_argument('--mmp-cliff-boost', type=float, default=2.0,
                        help='Loss weight multiplier for activity cliff pairs (high sim, large delta)')
    parser.add_argument('--mmp-max-pairs-per-compound', type=int, default=6,
                        help='Max MMP pairs per compound to prevent scaffold dominance. 0 = no limit.')
    parser.add_argument('--focal-gamma', type=float, default=0.0,
                        help='Focal loss gamma for regression. Weights each sample by |error|^gamma. 0 = standard L1.')
    parser.add_argument('--moe-head', action='store_true',
                        help='Use mixture-of-experts prediction head (separate active/inactive heads with learned gate)')
    parser.add_argument('--heteroscedastic-head', action='store_true',
                        help='Use heteroscedastic head (outputs mu + log_var, trained with Gaussian NLL)')
    parser.add_argument('--log-var-clamp', type=float, nargs=2, default=[-6.0, 6.0],
                        metavar=('MIN', 'MAX'),
                        help='Clamp range for log-variance output (default: -6.0 6.0)')
    parser.add_argument('--confident-error-weight', type=float, default=0.0,
                        help='Boost weight for actives where model is confident but wrong (requires --heteroscedastic-head). 0 = disabled.')
    parser.add_argument('--cew-start-epoch', type=int, default=0,
                        help='Epoch at which confident-error-weight activates (curriculum). 0 = from start.')
    parser.add_argument('--global-feat-train', type=str, default=None,
                        help='Path to .npy file with pre-computed global features for training set')
    parser.add_argument('--global-feat-val', type=str, default=None,
                        help='Path to .npy file with pre-computed global features for validation set')
    parser.add_argument('--contrastive-weight', type=float, default=0.0,
                        help='Weight for contrastive loss on MMP embeddings. 0 = disabled.')
    parser.add_argument('--contrastive-margin', type=float, default=1.0,
                        help='Margin for contrastive loss. Pairs with |delta| > 0.5 std are pushed apart beyond this distance.')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None,
                        help='LR noise schedule (epoch percentages)')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67,
                        help='LR noise amplitude as percentage')
    parser.add_argument('--lr-noise-std', type=float, default=1.0,
                        help='LR noise standard deviation')
    
    # ========================================================================
    # Model EMA (Exponential Moving Average)
    # ========================================================================
    parser.add_argument('--model-ema', action='store_true',
                        help='Enable exponential moving average model')
    parser.add_argument('--model-ema-decay', type=float, default=0.9999,
                        help='EMA decay coefficient')
    parser.add_argument('--model-ema-force-cpu', action='store_true',
                        help='Force EMA model to CPU')
    
    # ========================================================================
    # Regularization
    # ========================================================================
    parser.add_argument('--drop-path', type=float, default=0.0,
                        help='Drop path rate for stochastic depth')
    parser.add_argument('--attn-pool', action='store_true',
                        help='Use attention-weighted pooling instead of mean pooling')

    # ========================================================================
    # Data Loading
    # ========================================================================
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--pin-mem', action='store_true', default=True,
                        help='Pin memory in DataLoader')
    
    # ========================================================================
    # Mixed Precision Training (AMP)
    # ========================================================================
    parser.add_argument('--amp', action='store_true',
                        help='Enable automatic mixed precision (bfloat16) training')
    
    # ========================================================================
    # Distributed Training
    # ========================================================================
    parser.add_argument('--world_size', type=int, default=1,
                        help='Number of distributed processes')
    parser.add_argument('--local-rank', type=int, default=0,
                        help='Local rank in distributed training')
    parser.add_argument('--dist_url', type=str, default='env://',
                        help='URL for distributed training setup')
    
    # ========================================================================
    # Logging
    # ========================================================================
    parser.add_argument('--print-freq', type=int, default=100,
                        help='Print frequency (batches)')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='W&B run name. If omitted, W&B logging is disabled.')
    parser.add_argument('--run-type', type=str, default='cv',
                        choices=['cv', 'final', 'analog-cv'],
                        help='Run type for W&B tagging: "cv" for cross-validation, "final" for production runs, "analog-cv" for analog-mimic CV sweeps')
    parser.add_argument('--save-epochs', type=str, default=None,
                        help='Comma-separated list of epochs at which to save checkpoints (e.g. "19,20,21,22,23")')
    parser.add_argument('--aux-weight', type=float, default=0.1,
                        help='Loss weight for auxiliary multi-task heads (log2fc). 0 = disabled.')
    parser.add_argument('--aux-weights', type=float, nargs='+', default=None,
                        help='Per-head aux weights (overrides --aux-weight). E.g. "0.1 0.05" weights head-0 at 0.1 and head-1 at 0.05.')
    parser.add_argument('--aux-start-epoch', type=int, default=10,
                        help='Epoch at which auxiliary multi-task loss activates.')
    parser.add_argument('--rank-weight', type=float, default=0.0,
                        help='Weight for soft Spearman rank correlation loss. 0 = disabled.')
    parser.add_argument('--rank-start-epoch', type=int, default=10,
                        help='Epoch at which rank loss activates.')
    parser.add_argument('--rank-tau', type=float, default=0.1,
                        help='Temperature for soft rank approximation (smaller = sharper ranks).')

    return parser


def train_regression(args):
    """
    Training pipeline for regression tasks (continuous property prediction).

    This function implements the complete training workflow including:
    - Dataset loading and preprocessing
    - Model initialization with pre-trained weights
    - Optimizer and learning rate scheduler setup
    - Training loop with validation and EMA evaluation
    - Checkpoint saving and best model tracking

    Args:
        args (Namespace): Parsed command line arguments containing all training configurations

    Returns:
        None: Results are saved to checkpoints and logged to files
    """

    # ========================================================================
    # Stage 1: Initialization
    # ========================================================================
    # Generate timestamp for experiment tracking
    exp_time = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # Initialize distributed training environment
    utils.init_distributed_mode(args)
    is_main_process = (args.rank == 0)

    # Create logging directory and logger
    if is_main_process:
        if not os.path.exists('logs/' + args.name):
            os.makedirs('logs/' + args.name)

    _log = FileLogger(
        is_master=is_main_process,
        is_rank0=is_main_process,
        output_dir='logs/' + args.name + '/',
        time_name=exp_time
    )
    _log.info(args)

    _wandb_run = None
    if _WANDB_AVAILABLE and is_main_process and getattr(args, 'wandb_run_name', None):
        wandb_tags = [args.run_type, f"seed-{args.seed}", args.name, f"loss-{args.loss}", "exp1"]
        if getattr(args, 'heteroscedastic_head', False):
            wandb_tags.append("heteroscedastic")
        if args.loss == 'balanced_mse':
            wandb_tags.append(f"bmse-bins{args.balanced_mse_bins}")
        if getattr(args, 'heteroscedastic_head', False) and args.loss == 'balanced_mse':
            wandb_tags.append("hetero+bmse")
        if getattr(args, 'confident_error_weight', 0.0) > 0:
            wandb_tags.append(f"cew-{args.confident_error_weight}")
        if getattr(args, 'moe_head', False):
            wandb_tags.append("moe")
        _wandb_run = wandb.init(
            entity="jefflinnnn-personal",
            project="open-admet-cc-dev-2",
            name=args.wandb_run_name,
            group=f"{args.name}_{args.run_type}",
            config=vars(args),
            tags=wandb_tags,
        )
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ========================================================================
    # Stage 2: Dataset Loading and Preprocessing
    # ========================================================================
    if args.data_mode == 'smiles_random':
        _log.info(f'Random spliting dataset to train dataset and valid dataset with ratio {args.ratio}')
        data_path = 'data/' + args.name
        train_dataset = PP_smiles_2d(data_path, 'train', args.name, args.ratio)
        val_dataset   = PP_smiles_2d(data_path, 'valid', args.name, args.ratio)
    elif args.data_mode == 'smiles_defined':
        if args.tvt:
            _log.info('Using defined train, valid, test dataset')
            data_path = 'data/' + args.name
            train_dataset = PP_smiles_2d(data_path, 'train', args.name, args.ratio)
            val_dataset    = PP_smiles_2d(data_path, 'valid', args.name, args.ratio)
            test_dataset   = PP_smiles_2d(data_path, 'test', args.name, defined=True)
        else:
            _log.info('Using defined train, valid dataset')
            data_path = 'data/' + args.name
            train_dataset = PP_smiles_2d(data_path, 'train', args.name, defined=True)
            val_dataset   = PP_smiles_2d(data_path, 'valid', args.name, defined=True)
    else:
        raise ValueError('Unseen data file.')
    
    # Log dataset warnings and statistics
    if train_dataset.exceed_ele is not None:
        _log.info(train_dataset.exceed_ele)
    if train_dataset.fail_mole is not None and val_dataset.fail_mole is not None:
        _log.info('Fail molecules in Training set: {}, Fail molecules in Valid set size:{}'.format(
            train_dataset.fail_mole, val_dataset.fail_mole))
    _log.info('Training set size: {}, Valid set size:{}'.format(
        len(train_dataset), len(val_dataset)))
    
    # Compute normalization factors (mean, std) for regression target
    norm_factor = [train_dataset.mean(), train_dataset.std()]
    _log.info('Training set mean: {}, std:{}'.format(
        norm_factor[0], norm_factor[1]))

    # Compute per-head norm factors for auxiliary targets if present
    base_ds = train_dataset.base if hasattr(train_dataset, 'base') else train_dataset
    aux_means = base_ds.mean_aux() if hasattr(base_ds, 'mean_aux') else []
    aux_stds  = base_ds.std_aux()  if hasattr(base_ds, 'std_aux')  else []
    if aux_means:
        args._aux_norm_factors = list(zip(aux_means, aux_stds))
        _log.info(f'Aux norm factors: {args._aux_norm_factors}')
    else:
        args._aux_norm_factors = None
    
    # Reset random seeds after dataset loading
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # ========================================================================
    # Stage 3: Model Setup
    # ========================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load global features if provided
    global_feat_train = None
    global_feat_val = None
    global_feat_dim = 0
    if getattr(args, 'global_feat_train', None) is not None:
        global_feat_train = torch.from_numpy(np.load(args.global_feat_train)).float()
        global_feat_dim = global_feat_train.shape[1]
        _log.info(f'Loaded global features (train): {global_feat_train.shape}')
        if getattr(args, 'global_feat_val', None) is not None:
            global_feat_val = torch.from_numpy(np.load(args.global_feat_val)).float()
            _log.info(f'Loaded global features (val): {global_feat_val.shape}')

    # Wrap datasets to inject global features per sample
    if global_feat_train is not None:
        from torch.utils.data import Dataset as TorchDataset
        class GlobalFeatDataset(TorchDataset):
            def __init__(self, base_dataset, feats):
                assert len(feats) >= len(base_dataset), \
                    f"Feature count {len(feats)} < dataset size {len(base_dataset)}"
                self.base = base_dataset
                self.feats = feats
            def __len__(self):
                return len(self.base)
            def __getitem__(self, idx):
                data = self.base[idx]
                data.global_feat = self.feats[idx].unsqueeze(0)
                return data
            def mean(self):
                return self.base.mean()
            def std(self):
                return self.base.std()
        train_dataset = GlobalFeatDataset(train_dataset, global_feat_train)
        if global_feat_val is not None:
            val_dataset = GlobalFeatDataset(val_dataset, global_feat_val)

    # Load model with pre-trained and fine-tune components
    from models.finetune_model import standard_finetune
    model = standard_finetune(
        class_flag=False, class_num=2,
        moe_head=getattr(args, 'moe_head', False),
        global_feat_dim=global_feat_dim,
        heteroscedastic=getattr(args, 'heteroscedastic_head', False),
        attn_pool=getattr(args, 'attn_pool', False),
    )
    model = model.to(device)

    # Load pre-trained model weights if provided
    if args.checkpoint_pretrain is not None:
        _log.info('Start loading pretrain model')
        checkpoint = torch.load(args.checkpoint_pretrain, map_location=torch.device('cpu'))
        model.pretrain_model.load_state_dict(checkpoint)
        _log.info('Load pretrain model successfully')
    
    # Load model weights from checkpoint (resume handled fully after optimizer creation)
    resume_checkpoint = _load_resume_checkpoint(model, args.resume, _log) if args.resume else None

    if args.unfreeze_backbone:
        _log.info('Backbone is UNFROZEN — full model will be fine-tuned')
    else:
        _log.info('Freezing the pretrain model')
        frozen_modules = [model.pretrain_model]
        for module in frozen_modules:
            for _, param in module.named_parameters():
                param.requires_grad = False

    # Initialize Exponential Moving Average (EMA) model for improved generalization
    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else None)

    # Wrap model for distributed training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank])

    # Log model parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.info('Number of trainable params: {}'.format(n_parameters))

    # ========================================================================
    # Stage 4: Optimizer and Learning Rate Scheduler
    # ========================================================================
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    start_epoch = _restore_training_state(resume_checkpoint, optimizer, lr_scheduler, _log,
                                              warm_restart=getattr(args, 'warm_restart', False))

    # Create loss function based on task type
    from loss_functions import BalancedMSELoss, GaussianNLLLoss
    balanced_mse_weighter = None
    if args.loss == 'balanced_mse':
        all_targets = torch.tensor([train_dataset[i].y.item() for i in range(len(train_dataset))])
        all_targets_norm = (all_targets - norm_factor[0]) / norm_factor[1]
        balanced_mse_weighter = BalancedMSELoss(
            targets=all_targets_norm,
            num_bins=args.balanced_mse_bins,
            smoothing=args.balanced_mse_smoothing,
        )
        _log.info(f'Balanced MSE weighting: bins={args.balanced_mse_bins}, smoothing={args.balanced_mse_smoothing}')

    if getattr(args, 'heteroscedastic_head', False):
        criterion = GaussianNLLLoss(
            log_var_min=args.log_var_clamp[0],
            log_var_max=args.log_var_clamp[1],
        )
        _log.info(f'Using GaussianNLL loss (heteroscedastic head), log_var clamp={args.log_var_clamp}')
    elif args.loss == 'balanced_mse':
        criterion = balanced_mse_weighter
    elif args.loss == 'l1':
        criterion = torch.nn.L1Loss(reduction='none')
    elif args.loss == 'l2':
        criterion = torch.nn.MSELoss(reduction='none')
    else:
        raise ValueError(f"Unknown loss type: {args.loss}")
    criterion = criterion.to(device)
    if balanced_mse_weighter is not None and balanced_mse_weighter is not criterion:
        balanced_mse_weighter = balanced_mse_weighter.to(device)

    # ========================================================================
    # Stage 5: Automatic Mixed Precision (AMP) Setup
    # ========================================================================
    # Setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = False  # Disabled by default
    loss_scaler = None
    if args.amp:
        amp_autocast = True
        loss_scaler = NativeScaler()

    # ========================================================================
    # Stage 6: Data Loader Setup
    # ========================================================================
    # ========================================================================
    # Stage 6: Data Loader Setup
    # ========================================================================
    if args.distributed:
        # Use DistributedSampler for multi-GPU training
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset,
            num_replicas=utils.get_world_size(),
            rank=utils.get_rank(),
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler_train,
            num_workers=args.workers,
            pin_memory=args.pin_mem and torch.cuda.is_available(),
            drop_last=True
        )
    else:
        # Single GPU training
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=args.pin_mem and torch.cuda.is_available(),
            drop_last=True
        )
    
    # Validation and test loaders (no shuffling needed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    if args.tvt:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # ========================================================================
    # Stage 7: Compute Dataset Statistics (Optional)
    # ========================================================================
    if args.compute_stats:
        compute_stats(train_loader, max_radius=args.radius, logger=_log, print_freq=args.print_freq)
        return
    
    # ========================================================================
    # Stage 7b: MMP Delta Head Setup (Optional)
    # ========================================================================
    mmp_sampler = None
    if getattr(args, 'mmp_pairs', None) is not None:
        import pandas as _pd
        _train_csv_path = os.path.join('data', args.name, 'raw', f'{args.name}.csv')
        if not os.path.exists(_train_csv_path):
            _train_csv_path = os.path.join('data', args.name, 'raw', f'{args.name}_train.csv')
        _train_smiles = _pd.read_csv(_train_csv_path)['SMILES'].tolist()
        mmp_sampler = MMPSampler(
            pairs_json_path=args.mmp_pairs,
            smiles_list=_train_smiles,
            norm_factor=norm_factor,
            batch_size=args.mmp_batch_size,
            seed=args.seed,
            cliff_boost=getattr(args, 'mmp_cliff_boost', 2.0),
            max_pairs_per_compound=getattr(args, 'mmp_max_pairs_per_compound', 0),
            global_feats=global_feat_train,
        )
        _log.info(f'MMP delta head enabled: {len(mmp_sampler.pairs)} pairs, weight={args.mmp_weight}, '
                  f'mode={args.mmp_mode}, start_epoch={args.mmp_start_epoch}, '
                  f'cliff_boost={args.mmp_cliff_boost}')

    # ========================================================================
    # Stage 8: Training Loop
    # ========================================================================
    # Initialize best metrics tracking
    best_epoch = 0
    best_train_r2, best_train_err = 0, float('inf')
    best_val_r2, best_val_err = 0, float('inf')
    best_val_rae = float('inf')
    best_test_r2, best_test_err = 0, float('inf')
    best_ema_epoch = 0
    best_ema_val_r2, best_ema_val_err = 0, float('inf')
    best_ema_test_r2, best_ema_test_err = 0, float('inf')
    epochs_without_improvement = 0

    _log.info('Start training')
    for epoch in range(start_epoch, args.epochs):
        _log.info(f"Training property: {args.name}")
        epoch_start_time = time.perf_counter()

        # Update learning rate scheduler
        lr_scheduler.step(epoch)

        # Set epoch for distributed sampler (ensures different shuffle each epoch)
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # ====================================================================
        # Training Phase
        # ====================================================================
        train_err, train_r2 = train_one_epoch(
            model=model,
            criterion=criterion,
            norm_factor=norm_factor,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            model_ema=model_ema,
            amp_autocast=amp_autocast,
            loss_scaler=loss_scaler,
            clip_grad=args.clip_grad,
            print_freq=args.print_freq,
            logger=_log,
            active_weight=args.active_weight,
            tail_weight=args.tail_weight,
            tail_threshold=args.tail_threshold,
            mmp_sampler=mmp_sampler,
            mmp_weight=args.mmp_weight if mmp_sampler is not None else 0.0,
            mmp_mode=args.mmp_mode,
            mmp_start_epoch=getattr(args, 'mmp_start_epoch', 0),
            focal_gamma=args.focal_gamma,
            contrastive_weight=args.contrastive_weight if mmp_sampler is not None else 0.0,
            contrastive_margin=args.contrastive_margin,
            heteroscedastic=getattr(args, 'heteroscedastic_head', False),
            balanced_mse_weighter=balanced_mse_weighter,
            confident_error_weight=getattr(args, 'confident_error_weight', 0.0) if epoch >= getattr(args, 'cew_start_epoch', 0) else 0.0,
            aux_norm_factors=getattr(args, '_aux_norm_factors', None),
            aux_weight=getattr(args, 'aux_weight', 0.1),
            aux_weights=getattr(args, 'aux_weights', None),
            aux_start_epoch=getattr(args, 'aux_start_epoch', 10),
            rank_weight=getattr(args, 'rank_weight', 0.0),
            rank_start_epoch=getattr(args, 'rank_start_epoch', 10),
            rank_tau=getattr(args, 'rank_tau', 0.1),
        )

        # ====================================================================
        # Validation Phase
        # ====================================================================
        val_err, val_r2, val_loss, val_targets, val_preds, val_spearman, val_kendall = evaluate(
            model,
            norm_factor,
            val_loader,
            device,
            amp_autocast=amp_autocast,
            print_freq=args.print_freq,
            logger=_log,
            heteroscedastic=getattr(args, 'heteroscedastic_head', False),
        )
        val_rae, val_active_mae, val_inactive_mae, n_active, n_inactive = compute_rae(val_targets, val_preds)
        val_active_rae = compute_active_rae(val_targets, val_preds)

        # ====================================================================
        # Checkpoint Management
        # ====================================================================
        checkpoints_dir = f'checkpoints/{args.name}/{exp_time}'
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        # Determine if this is the best model based on selected metric
        if args.main_metric == 'MAE':
            best_flag = (val_err < best_val_err)
        elif args.main_metric == 'R2':
            best_flag = (val_r2 > best_val_r2)
        elif args.main_metric == 'active_rae':
            best_flag = (val_active_rae < best_val_rae)
        elif args.main_metric == 'RAE':
            best_flag = (val_rae < best_val_rae)
        else:
            best_flag = (val_err < best_val_err)

        # Save best model
        if best_flag:
            best_val_err = val_err
            best_train_err = train_err
            best_train_r2 = train_r2
            best_val_r2 = val_r2
            best_val_rae = val_active_rae if args.main_metric == 'active_rae' else val_rae
            best_epoch = epoch

            # Evaluate on test set if available
            if args.tvt:
                test_err, test_r2, test_loss, _, _, _, _ = evaluate(
                    model,
                    norm_factor,
                    test_loader,
                    device,
                    amp_autocast=amp_autocast,
                    print_freq=args.print_freq,
                    logger=_log,
                    heteroscedastic=getattr(args, 'heteroscedastic_head', False),
                )
                info_str = (f'Best Test -- Epoch: [{epoch}], '
                           f'MAE: {test_err:.5f}, R2: {test_r2:.5f}\n')
                _log.info(info_str)
                best_test_err = test_err
                best_test_r2 = test_r2

            # Save best checkpoint (only on main process in distributed training)
            if is_main_process:
                checkpoint_data = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_err": best_val_err,
                    "best_test_err": best_test_err,
                    "norm_factor": norm_factor
                }
                torch.save(checkpoint_data, f'{checkpoints_dir}/{args.name}_regression.pt')

        # Save latest checkpoint every epoch for crash recovery
        if is_main_process:
            latest_data = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "best_val_err": best_val_err,
                "best_test_err": best_test_err,
                "norm_factor": norm_factor
            }
            torch.save(latest_data, f'{checkpoints_dir}/{args.name}_regression_latest.pt')

            # Save epoch-specific checkpoint if requested
            if hasattr(args, 'save_epochs') and args.save_epochs:
                save_epoch_list = [int(e) for e in args.save_epochs.split(',')]
                if epoch in save_epoch_list:
                    torch.save(latest_data, f'{checkpoints_dir}/{args.name}_regression_epoch{epoch}.pt')
                    _log.info(f'Saved epoch-specific checkpoint: epoch {epoch}')

        # ====================================================================
        # Logging
        # ====================================================================
        train_log = (f'Epoch: [{epoch}], '
                    f'Train MAE: {train_err:.5f}, Train R2: {train_r2:.5f}, '
                    f'Val MAE: {val_err:.5f}, Val R2: {val_r2:.5f}, '
                    f'Spearman: {val_spearman:.5f}, Kendall: {val_kendall:.5f}, '
                    f'Val RAE: {val_rae:.5f} (active={val_active_mae:.5f} [{n_active}], inactive={val_inactive_mae:.5f} [{n_inactive}]), '
                    f'Time: {time.perf_counter() - epoch_start_time:.2f}s')
        _log.info(train_log)

        if _wandb_run is not None:
            _wandb_run.log({
                "epoch": epoch,
                "train/mae": train_err,
                "train/r2": train_r2,
                "val/mae": val_err,
                "val/r2": val_r2,
                "val/spearman": val_spearman,
                "val/kendall": val_kendall,
                "val/rae": val_rae,
                "val/active_rae": val_active_rae,
                "val/active_mae": val_active_mae,
                "val/inactive_mae": val_inactive_mae,
                "best_val/mae": best_val_err,
                "best_val/r2": best_val_r2,
                "best_val/rae": best_val_rae,
                "lr": optimizer.param_groups[0]['lr'],
            })

        best_log = (f'Best: Epoch={best_epoch}, '
                   f'Train MAE: {best_train_err:.5f}, Train R2: {best_train_r2:.5f}, '
                   f'Val MAE: {best_val_err:.5f}, Val R2: {best_val_r2:.5f}, Val RAE: {best_val_rae:.5f}, '
                   f'Test MAE: {best_test_err:.5f}, Test R2: {best_test_r2:.5f}\n')
        _log.info(best_log)

        if best_flag:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if args.early_stopping > 0 and epochs_without_improvement >= args.early_stopping:
            _log.info(f'Early stopping triggered: no improvement for {args.early_stopping} epochs')
            break

        # ====================================================================
        # EMA Model Evaluation (if enabled)
        # ====================================================================
        if model_ema is not None:
            ema_val_err, ema_val_r2, _, _, _, _, _ = evaluate(
                model_ema.module,
                norm_factor,
                val_loader,
                device,
                amp_autocast=amp_autocast,
                print_freq=args.print_freq,
                logger=_log,
                heteroscedastic=getattr(args, 'heteroscedastic_head', False),
            )
            
            # Track best EMA model
            if ema_val_err < best_ema_val_err:
                best_ema_val_err = ema_val_err
                best_ema_val_r2 = ema_val_r2
                best_ema_epoch = epoch

                # Evaluate EMA model on test set
                if args.tvt:
                    test_ema_err, test_ema_r2, _, _, _, _, _ = evaluate(
                        model_ema.module,
                        norm_factor,
                        test_loader,
                        device,
                        amp_autocast=amp_autocast,
                        print_freq=args.print_freq,
                        logger=_log,
                        heteroscedastic=getattr(args, 'heteroscedastic_head', False),
                    )
                    info_str = (f'Best EMA Test -- Epoch: [{epoch}], '
                               f'MAE: {test_ema_err:.5f}, R2: {test_ema_r2:.5f}\n')
                    _log.info(info_str)
                    best_ema_test_err = test_ema_err
                    best_ema_test_r2 = test_ema_r2

                # Save EMA checkpoint
                if is_main_process:
                    checkpoint_data = {
                        "state_dict": model_ema.module.state_dict(),
                        "epoch": epoch,
                        "best_val_ema_err": best_ema_val_err,
                        "best_test_ema_err": best_ema_test_err,
                        "norm_factor": norm_factor
                    }
                    torch.save(checkpoint_data, f'{checkpoints_dir}/{args.name}_regression_ema.pt')
    
            ema_log = (f'EMA -- Epoch: [{epoch}], '
                      f'Val MAE: {ema_val_err:.5f}, Val R2: {ema_val_r2:.5f}, '
                      f'Time: {time.perf_counter() - epoch_start_time:.2f}s')
            _log.info(ema_log)
            
            best_ema_log = (f'Best EMA: Epoch={best_ema_epoch}, '
                           f'Val MAE: {best_ema_val_err:.5f}, Val R2: {best_ema_val_r2:.5f}\n')
            _log.info(best_ema_log)

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        # Log memory usage to detect leaks early (macOS reports bytes, Linux reports KB)
        if is_main_process and epoch % 5 == 0:
            rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_mb = rss_raw / (1024 * 1024) if os.uname().sysname == 'Darwin' else rss_raw / 1024
            _log.info(f'Memory: peak RSS={rss_mb:.0f} MB')


def train_classification(args):
    """
    Training pipeline for classification tasks (categorical property prediction).
    
    This function implements the complete training workflow for classification including:
    - Dataset loading with classification labels
    - Model initialization with appropriate output heads
    - Training with cross-entropy loss
    - Evaluation using accuracy, AUROC, and AUPRC metrics
    - EMA model tracking for improved generalization
    
    Args:
        args (Namespace): Parsed command line arguments containing all training configurations
    
    Returns:
        None: Results are saved to checkpoints and logged to files
    """

    # ========================================================================
    # Stage 1: Initialization
    # ========================================================================
    # Generate timestamp for experiment tracking
    exp_time = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    # Initialize distributed training environment
    utils.init_distributed_mode(args)
    is_main_process = (args.rank == 0)
    
    # Create logging directory and logger
    if is_main_process:
        if not os.path.exists('logs/' + args.name):
            os.makedirs('logs/' + args.name)
    
    _log = FileLogger(
        is_master=is_main_process,
        is_rank0=is_main_process,
        output_dir='logs/' + args.name + '/',
        time_name=exp_time
    )
    _log.info(args)

    _wandb_run = None
    if _WANDB_AVAILABLE and is_main_process and getattr(args, 'wandb_run_name', None):
        wandb_tags = [args.run_type, f"seed-{args.seed}", args.name, "classification"]
        _wandb_run = wandb.init(
            entity="jefflinnnn-personal",
            project="open-admet-cc-dev-2",
            name=args.wandb_run_name,
            group=f"{args.name}_{args.run_type}",
            config=vars(args),
            tags=wandb_tags,
        )

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ========================================================================
    # Stage 2: Dataset Loading and Preprocessing
    # ========================================================================
    if args.data_mode == 'smiles_random':
        _log.info(f'Random spliting dataset to train dataset and valid dataset with ratio {args.ratio}')
        data_path = 'data/' + args.name
        train_dataset = PP_smiles_2d(data_path, 'train', args.name, args.ratio, classification=True)
        val_dataset   = PP_smiles_2d(data_path, 'valid', args.name, args.ratio, classification=True)
    elif args.data_mode == 'smiles_defined':
        if args.tvt:
            _log.info('Using defined train, valid, test dataset')
            data_path = 'data/' + args.name
            train_dataset = PP_smiles_2d(data_path, 'train', args.name, args.ratio, classification=True)
            val_dataset    = PP_smiles_2d(data_path, 'valid', args.name, args.ratio, classification=True)
            test_dataset   = PP_smiles_2d(data_path, 'test', args.name, defined=True, classification=True)
        else:
            _log.info('Using defined train, valid dataset')
            data_path = 'data/' + args.name
            train_dataset = PP_smiles_2d(data_path, 'train', args.name, defined=True, classification=True)
            val_dataset   = PP_smiles_2d(data_path, 'valid', args.name, defined=True, classification=True)
    else:
        raise ValueError('Unseen data file.')
    
    # Log dataset warnings and statistics
    if train_dataset.exceed_ele is not None:
        _log.info(train_dataset.exceed_ele)
    if train_dataset.fail_mole is not None and val_dataset.fail_mole is not None:
        _log.info('Fail molecules in Training set: {}, Fail molecules in Valid set size:{}'.format(
            train_dataset.fail_mole, val_dataset.fail_mole))
    _log.info('Training set size: {}, Valid set size:{}'.format(
        len(train_dataset), len(val_dataset)))
    
    # Get class number from dataset
    class_num = train_dataset.class_num
    _log.info('Number of classes: {}'.format(class_num))
    
    # Reset random seeds after dataset loading
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # ========================================================================
    # Stage 3: Model Setup
    # ========================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model with pre-trained and fine-tune components for classification
    from models.finetune_model import standard_finetune
    model = standard_finetune(class_flag=True, class_num=class_num)
    model = model.to(device)

    # Load pre-trained model weights if provided
    if args.checkpoint_pretrain is not None:
        _log.info('Start loading pretrain model')
        checkpoint = torch.load(args.checkpoint_pretrain, map_location=torch.device('cpu'))
        model.pretrain_model.load_state_dict(checkpoint)
        _log.info('Load pretrain model successfully')
    
    # Load model weights from checkpoint (resume handled fully after optimizer creation)
    resume_checkpoint = _load_resume_checkpoint(model, args.resume, _log) if args.resume else None

    # Freeze pre-trained model parameters to preserve learned representations
    _log.info('Freezing the pretrain model')
    frozen_modules = [model.pretrain_model]
    for module in frozen_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False

    # Move model to device
    model = model.to(device)

    # Initialize Exponential Moving Average (EMA) model for improved generalization
    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else None)

    # Wrap model for distributed training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank])

    # Log model parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.info('Number of trainable params: {}'.format(n_parameters))

    # ========================================================================
    # Stage 4: Optimizer and Learning Rate Scheduler
    # ========================================================================
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    start_epoch = _restore_training_state(resume_checkpoint, optimizer, lr_scheduler, _log,
                                              warm_restart=getattr(args, 'warm_restart', False))

    # Cross-entropy loss for multi-class classification
    criterion = torch.nn.CrossEntropyLoss()
    
    # ========================================================================
    # Stage 5: Automatic Mixed Precision (AMP) Setup
    # ========================================================================
    # Setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = False  # Disabled by default
    loss_scaler = None
    if args.amp:
        amp_autocast = True
        loss_scaler = NativeScaler()
    
    # ========================================================================
    # Stage 6: Data Loader Setup
    # ========================================================================
    # ========================================================================
    # Stage 6: Data Loader Setup
    # ========================================================================
    if args.distributed:
        # Use DistributedSampler for multi-GPU training
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset,
            num_replicas=utils.get_world_size(),
            rank=utils.get_rank(),
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler_train,
            num_workers=args.workers,
            pin_memory=args.pin_mem and torch.cuda.is_available(),
            drop_last=True
        )
    else:
        # Single GPU training
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=args.pin_mem and torch.cuda.is_available(),
            drop_last=True
        )
    
    # Validation and test loaders (no shuffling needed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    if args.tvt:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # ========================================================================
    # Stage 7: Compute Dataset Statistics (Optional)
    # ========================================================================
    if args.compute_stats:
        compute_stats(train_loader, max_radius=args.radius, logger=_log, print_freq=args.print_freq)
        return
    
    # ========================================================================
    # Stage 8: Training Loop
    # ========================================================================
    # Initialize best metrics tracking
    best_epoch = 0
    best_train_acc = 0
    best_train_auprc, best_train_auroc = 0, 0
    best_val_acc = 0
    best_val_auprc, best_val_auroc = 0, 0
    best_test_acc = 0
    best_train_err, best_val_err = float('inf'), float('inf')
    best_ema_epoch = 0
    best_ema_val_acc = 0
    best_ema_val_err = float('inf')
    
    _log.info('Start training')
    for epoch in range(start_epoch, args.epochs):
        _log.info(f"Training property: {args.name}")
        epoch_start_time = time.perf_counter()

        # Update learning rate scheduler
        lr_scheduler.step(epoch)

        # Set epoch for distributed sampler (ensures different shuffle each epoch)
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # ====================================================================
        # Training Phase
        # ====================================================================
        train_loss, train_acc, train_auprc, train_auroc = train_cls_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            model_ema=model_ema,
            amp_autocast=amp_autocast,
            loss_scaler=loss_scaler,
            print_freq=args.print_freq,
            logger=_log
        )
        
        # ====================================================================
        # Validation Phase
        # ====================================================================
        val_loss, val_acc, val_auprc, val_auroc = evaluate_cls(
            model,
            val_loader,
            device,
            criterion,
            amp_autocast=amp_autocast,
            print_freq=args.print_freq,
            logger=_log
        )
        
        # ====================================================================
        # Checkpoint Management
        # ====================================================================
        checkpoints_dir = f'checkpoints/{args.name}/{exp_time}'
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        # Determine if this is the best model based on selected metric
        best_flag = False
        if args.main_metric == 'ACC':
            best_flag = (val_acc > best_val_acc)
        elif args.main_metric == 'AUPRC':
            best_flag = (val_auprc > best_val_auprc)
        elif args.main_metric == 'AUROC':
            best_flag = (val_auroc > best_val_auroc)
        else:
            # Default to accuracy
            best_flag = (val_acc > best_val_acc)

        # Save best model
        if best_flag:
            best_val_err = val_loss
            best_train_err = train_loss
            best_train_acc = train_acc
            best_val_acc = val_acc
            best_epoch = epoch
            best_val_auprc = val_auprc
            best_val_auroc = val_auroc

            # Evaluate on test set if available
            if args.tvt:
                test_loss, test_acc, test_auprc, test_auroc= evaluate_cls(
                    model,
                    test_loader,
                    device,
                    criterion,
                    amp_autocast=amp_autocast,
                    print_freq=args.print_freq,
                    logger=_log
                )
                info_str = (f'Best Test -- Epoch: [{epoch}], '
                           f'Accuracy: {test_acc:.5f}\n')
                _log.info(info_str)
                best_test_acc = test_acc

            # Save best checkpoint (only on main process in distributed training)
            if is_main_process:
                checkpoint_data = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "best_test_acc": best_test_acc
                }
                torch.save(checkpoint_data, f'{checkpoints_dir}/{args.name}_classification.pt')

        # Save latest checkpoint every epoch for crash recovery
        if is_main_process:
            latest_data = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "best_test_acc": best_test_acc
            }
            torch.save(latest_data, f'{checkpoints_dir}/{args.name}_classification_latest.pt')

        # ====================================================================
        # Logging
        # ====================================================================
        train_log = (f'Epoch: [{epoch}], '
                    f'Train Acc: {train_acc:.5f}, Train AUROC: {train_auroc:.5f}, Train AUPRC: {train_auprc:.5f}, '
                    f'Val Acc: {val_acc:.5f}, Val AUROC: {val_auroc:.5f}, Val AUPRC: {val_auprc:.5f}, '
                    f'Time: {time.perf_counter() - epoch_start_time:.2f}s')
        _log.info(train_log)

        if _wandb_run is not None:
            _wandb_run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": float(train_acc),
                "train/auroc": float(train_auroc),
                "train/auprc": float(train_auprc),
                "val/loss": val_loss,
                "val/acc": float(val_acc),
                "val/auroc": float(val_auroc),
                "val/auprc": float(val_auprc),
                "best_val/acc": float(best_val_acc),
                "best_val/auroc": float(best_val_auroc),
                "best_val/auprc": float(best_val_auprc),
                "lr": optimizer.param_groups[0]['lr'],
            })

        best_log = (f'Best: Epoch={best_epoch}, '
                   f'Train Loss: {best_train_err:.5f}, Train Acc: {best_train_acc:.5f}, '
                   f'Val Loss: {best_val_err:.5f}, Val Acc: {best_val_acc:.5f}, '
                   f'Val AUROC: {best_val_auroc:.5f}, Val AUPRC: {best_val_auprc:.5f}, '
                   f'Test Acc: {best_test_acc:.5f}\n')
        _log.info(best_log)

        # ====================================================================
        # EMA Model Evaluation (if enabled)
        # ====================================================================
        if model_ema is not None:
            ema_val_loss, ema_val_acc, _, _ = evaluate_cls(
                model_ema.module,
                val_loader,
                device,
                criterion,
                amp_autocast=amp_autocast,
                print_freq=args.print_freq,
                logger=_log
            )

            # Track best EMA model
            if ema_val_acc > best_ema_val_acc:
                best_ema_val_acc = ema_val_acc
                best_ema_val_err = ema_val_loss
                best_ema_epoch = epoch
                
                # Save EMA checkpoint
                if is_main_process:
                    checkpoint_data = {
                        "state_dict": model_ema.module.state_dict(),
                        "epoch": epoch,
                        "best_val_acc": best_ema_val_acc
                    }
                    torch.save(checkpoint_data, f'{checkpoints_dir}/{args.name}_classification_ema.pt')
    
            ema_log = (f'EMA -- Epoch: [{epoch}], '
                      f'Val Loss: {ema_val_loss:.5f}, Val Acc: {ema_val_acc:.5f}, '
                      f'Time: {time.perf_counter() - epoch_start_time:.2f}s')
            _log.info(ema_log)
            
            best_ema_log = (f'Best EMA: Epoch={best_ema_epoch}, '
                           f'Val Loss: {best_ema_val_err:.5f}, Val Acc: {best_ema_val_acc:.5f}\n')
            _log.info(best_ema_log)

        gc.collect()


if __name__ == "__main__":
    """
    Main entry point for the fine-tuning training script.
    
    This script supports two training modes:
    - regression: For continuous property prediction
    - classification: For categorical property prediction
    
    The mode is selected via the --mode argument when running the script.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        'Fine tuning for various molecular properties',
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    
    # Route to appropriate training function based on mode
    if args.mode == 'regression':
        train_regression(args)
    elif args.mode == 'classification':
        train_classification(args)
    else:
        raise ValueError(f'Unknown training mode: {args.mode}. Must be "regression" or "classification"')
