import json
import numpy as np
import torch
from typing import Iterable, List, Optional

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def _autocast_device_type(device: torch.device) -> str:
    # torch.autocast does not support 'mps'; fall back to 'cpu' for MPS devices.
    return device.type if device.type != "mps" else "cpu"
from timm.utils import ModelEmaV2, dispatch_clip_grad
import time
from torch_cluster import radius_graph
import torch_geometric
from sklearn.metrics import r2_score

from torcheval.metrics.functional import binary_auroc, binary_auprc

ModelEma = ModelEmaV2


class MMPSampler:
    """Samples batches of MMP pairs for delta loss computation."""

    def __init__(self, pairs_json_path: str, smiles_list: List[str],
                 norm_factor: list, batch_size: int = 16, seed: int = 0):
        from suiren_datasets.org_mol2d import from_smiles

        with open(pairs_json_path) as f:
            all_pairs = json.load(f)

        self.norm_mean = norm_factor[0]
        self.norm_std = norm_factor[1]
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        # Build graph data for all unique molecules referenced in pairs
        unique_idx = set()
        for p in all_pairs:
            unique_idx.add(p["idx_a"])
            unique_idx.add(p["idx_b"])

        self.graph_cache = {}
        allowed = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
        for idx in unique_idx:
            if idx >= len(smiles_list):
                continue
            smi = smiles_list[idx]
            try:
                graph_tuple, mol_flag = from_smiles(smi, with_hydrogen=True)
                if not mol_flag:
                    continue
                x, edge_index, edge_attr, edge_index_all = graph_tuple
                if not set(x[:, 0].tolist()).issubset(allowed):
                    continue
                data = Data(
                    x=x.to(torch.long),
                    edge_index=edge_index.to(torch.long),
                    edge_attr=edge_attr.to(torch.long),
                    edge_index_all=edge_index_all.to(torch.long),
                )
                self.graph_cache[idx] = data
            except Exception:
                continue

        # Filter pairs to only those where both molecules have valid graphs
        self.pairs = [p for p in all_pairs
                      if p["idx_a"] in self.graph_cache and p["idx_b"] in self.graph_cache]
        self.pair_indices = np.arange(len(self.pairs))
        self._ptr = len(self.pairs)  # force shuffle on first call

        print(f"MMPSampler: {len(self.pairs)} pairs loaded, "
              f"{len(self.graph_cache)} unique molecules cached")

    def sample(self, device):
        if len(self.pairs) == 0:
            return None

        if self._ptr + self.batch_size > len(self.pairs):
            self.rng.shuffle(self.pair_indices)
            self._ptr = 0

        batch_idx = self.pair_indices[self._ptr:self._ptr + self.batch_size]
        self._ptr += self.batch_size

        selected = [self.pairs[i] for i in batch_idx]
        graphs_a = [self.graph_cache[p["idx_a"]] for p in selected]
        graphs_b = [self.graph_cache[p["idx_b"]] for p in selected]
        deltas = torch.tensor(
            [p["delta_pec50"] / self.norm_std for p in selected],
            dtype=torch.float, device=device
        )

        def make_emb_fn(graphs):
            def fn(model, device):
                loader = DataLoader(graphs, batch_size=len(graphs), shuffle=False)
                batch = next(iter(loader)).to(device)
                return model.forward_embedding(batch)
            return fn

        return {
            "emb_a_fn": make_emb_fn(graphs_a),
            "emb_b_fn": make_emb_fn(graphs_b),
            "delta_norm": deltas,
        }


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        """Initialize the meter with zero values."""
        self.reset()

    def reset(self):
        """Reset all metrics to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the meter with a new value.
        
        Args:
            val: The value to add
            n: The count of items (default: 1)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    norm_factor: list,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    model_ema: Optional[ModelEma] = None,
    amp_autocast: bool = False,
    loss_scaler=None,
    clip_grad=None,
    print_freq: int = 100,
    logger=None,
    active_weight: float = 1.0,
    mmp_sampler=None,
    mmp_weight: float = 0.0,
    focal_gamma: float = 0.0,
):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        criterion: The loss function
        norm_factor: List containing [mean, std] for normalization
        data_loader: DataLoader for training data
        optimizer: The optimizer for updating weights
        device: The device to run on (CPU/GPU)
        epoch: Current epoch number
        model_ema: Optional exponential moving average model
        amp_autocast: Whether to use automatic mixed precision
        loss_scaler: Optional loss scaler for gradient scaling
        clip_grad: Gradient clipping threshold
        print_freq: Frequency of logging (every N steps)
        logger: Logger instance for recording metrics
        
    Returns:
        Tuple of (average_mae, r2_score)
    """
    model.train()
    criterion.train()

    loss_metric = AverageMeter()
    mae_metric = AverageMeter()

    start_time = time.perf_counter()

    task_mean = norm_factor[0]
    task_std = norm_factor[1]

    all_targets = []
    all_preds = []

    for step, data in enumerate(data_loader):
        data = data.to(device)

        # Forward pass with automatic mixed precision
        with torch.autocast(
            device_type=_autocast_device_type(device),
            enabled=amp_autocast,
            dtype=torch.bfloat16,
        ):
            if mmp_sampler is not None and mmp_weight > 0:
                pred, emb = model.forward_with_embedding(data)
            else:
                pred = model(data)
                emb = None
            pred = pred.view(-1)
            target_norm = (data.y - task_mean) / task_std
            per_sample_loss = torch.nn.functional.l1_loss(pred, target_norm, reduction='none')

            weights = torch.ones_like(per_sample_loss)
            if active_weight != 1.0:
                weights = torch.where(data.y >= 6.0, active_weight, 1.0)
            if focal_gamma > 0:
                weights = weights * per_sample_loss.detach().abs().clamp(min=1e-6) ** focal_gamma

            reg_loss = (per_sample_loss * weights).mean()

            # MMP delta loss
            delta_loss = torch.tensor(0.0, device=device)
            if mmp_sampler is not None and mmp_weight > 0 and emb is not None:
                mmp_batch = mmp_sampler.sample(device)
                if mmp_batch is not None:
                    emb_a = mmp_batch["emb_a_fn"](model, device)
                    emb_b = mmp_batch["emb_b_fn"](model, device)
                    delta_pred = model.forward_delta(emb_a, emb_b).view(-1)
                    delta_target = mmp_batch["delta_norm"]
                    delta_loss = torch.nn.functional.l1_loss(delta_pred, delta_target)

            loss = reg_loss + mmp_weight * delta_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            loss.backward()
            if clip_grad is not None:
                dispatch_clip_grad(model.parameters(), value=clip_grad, mode="norm")
            optimizer.step()

        # Update metrics
        loss_metric.update(loss.item(), n=pred.shape[0])
        err = pred.detach() * task_std + task_mean - data.y
        mae_metric.update(torch.mean(torch.abs(err)).item(), n=pred.shape[0])

        all_targets.append(data.y.cpu())
        all_preds.append((pred.detach() * task_std + task_mean).cpu())

        # Update EMA model if provided
        if model_ema is not None:
            model_ema.update(model)

        if torch.cuda.is_available(): torch.cuda.synchronize()

        # Logging
        if step % print_freq == 0 or step == len(data_loader) - 1:
            elapsed_time = time.perf_counter() - start_time
            progress = (step + 1) / len(data_loader)
            time_per_step = 1e3 * elapsed_time / progress / len(data_loader)
            
            info_str = (
                f"Epoch: [{epoch}][{step}/{len(data_loader)}] "
                f"loss: {loss_metric.avg:.5f}, "
                f"MAE: {mae_metric.avg:.5f}, "
                f"time/step={time_per_step:.0f}ms, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
            logger.info(info_str)

    # Compute final metrics
    all_targets = torch.cat(all_targets, dim=0).numpy().flatten()
    all_preds = torch.cat(all_preds, dim=0).to(dtype=torch.float).numpy().flatten()
    r2 = r2_score(all_targets, all_preds)

    return mae_metric.avg, r2


def evaluate(
    model: torch.nn.Module,
    norm_factor: list,
    data_loader: Iterable,
    device: torch.device,
    amp_autocast: bool = False,
    print_freq: int = 100,
    logger=None,
    epoch: int = 0,
    debug_bad_example: bool = False,
    threshold: float = 0.35,
):
    """
    Evaluate the model on validation/test data.
    
    Args:
        model: The neural network model
        norm_factor: List containing [mean, std] for normalization
        data_loader: DataLoader for evaluation data
        device: The device to run on
        amp_autocast: Whether to use automatic mixed precision
        print_freq: Frequency of logging
        logger: Logger instance
        epoch: Current epoch number
        debug_bad_example: Whether to track examples with high errors
        threshold: Error threshold for identifying bad examples
        
    Returns:
        Tuple of (average_mae, r2_score, average_loss)
    """
    model.eval()

    loss_metric = AverageMeter()
    mae_metric = AverageMeter()
    criterion = torch.nn.L1Loss()
    criterion.eval()

    task_mean = norm_factor[0]
    task_std = norm_factor[1]

    all_targets = []
    all_preds = []
    # worse_example = [] if debug_bad_example else None
    # worse_err = [] if debug_bad_example else None

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)

            with torch.autocast(
                device_type=_autocast_device_type(device),
                enabled=amp_autocast,
                dtype=torch.bfloat16,
            ):
                pred = model(data)
                pred = pred.view(-1)

            loss = criterion(pred, (data.y - task_mean) / task_std)
            loss_metric.update(loss.item(), n=pred.shape[0])
            err = pred.detach() * task_std + task_mean - data.y
            mae_metric.update(torch.mean(torch.abs(err)).item(), n=pred.shape[0])

            # # Track examples with prediction errors exceeding threshold
            # if debug_bad_example:
            #     indices = torch.where(torch.abs(err) > threshold)[0]
            #     for i in indices:
            #         worse_example.append(data.smiles[i])
            #         worse_err.append(torch.abs(err)[i].item())

            all_targets.append(data.y.cpu())
            all_preds.append((pred.detach() * task_std + task_mean).cpu())

        # Compute final metrics
        all_targets = torch.cat(all_targets, dim=0).numpy().flatten()
        all_preds = (
            torch.cat(all_preds, dim=0).to(dtype=torch.float).numpy().flatten()
        )
        r2 = r2_score(all_targets, all_preds)

    return mae_metric.avg, r2, loss_metric.avg, all_targets, all_preds


def train_cls_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    model_ema: Optional[ModelEma] = None,
    amp_autocast: bool = False,
    loss_scaler=None,
    clip_grad=None,
    print_freq: int = 100,
    logger=None,
):
    """
    Train a classification model for one epoch.
    
    Args:
        model: The neural network model
        criterion: The loss function
        data_loader: DataLoader for training data
        optimizer: The optimizer for updating weights
        device: The device to run on
        epoch: Current epoch number
        model_ema: Optional exponential moving average model
        amp_autocast: Whether to use automatic mixed precision
        loss_scaler: Optional loss scaler for gradient scaling
        clip_grad: Gradient clipping threshold
        print_freq: Frequency of logging
        logger: Logger instance
        
    Returns:
        Tuple of (average_loss, accuracy, auprc, auroc)
    """
    model.train()
    criterion.train()

    loss_metric = AverageMeter()

    acc_count = 0
    total = 0

    start_time = time.perf_counter()
    all_targets = []
    all_preds = []

    for step, data in enumerate(data_loader):
        data = data.to(device)
        
        with torch.autocast(
            device_type=_autocast_device_type(device),
            enabled=amp_autocast,
            dtype=torch.bfloat16,
        ):
            pred = model(data)
            loss = criterion(pred, data.y)

        # Backward pass and optimization
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            loss.backward()
            if clip_grad is not None:
                dispatch_clip_grad(model.parameters(), value=clip_grad, mode="norm")
            optimizer.step()

        # Update metrics
        loss_metric.update(loss.item(), n=pred.shape[0])
        acc_count += (pred.detach().cpu().argmax(dim=1) == data.y.cpu()).sum()
        total += pred.shape[0]

        all_targets.append(data.y.cpu())
        all_preds.append(torch.softmax(pred, dim=-1).detach().cpu())

        # Update EMA model if provided
        if model_ema is not None:
            model_ema.update(model)

        if torch.cuda.is_available(): torch.cuda.synchronize()

        # Logging
        if step % print_freq == 0 or step == len(data_loader) - 1:
            elapsed_time = time.perf_counter() - start_time
            progress = (step + 1) / len(data_loader)
            time_per_step = 1e3 * elapsed_time / progress / len(data_loader)
            
            info_str = (
                f"Epoch: [{epoch}][{step}/{len(data_loader)}] "
                f"loss: {loss_metric.avg:.5f}, "
                f"Acc: {acc_count / total:.5f}, "
                f"time/step={time_per_step:.0f}ms, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
            logger.info(info_str)

    # Compute final metrics
    all_targets = torch.cat(all_targets, dim=0)
    all_preds = torch.cat(all_preds, dim=0).to(dtype=torch.float)[:, 1]
    auprc = binary_auprc(all_preds, all_targets)
    auroc = binary_auroc(all_preds, all_targets)
    epoch_acc = acc_count / total

    return loss_metric.avg, epoch_acc, auprc, auroc


def evaluate_cls(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    criterion: torch.nn.Module,
    amp_autocast: bool = False,
    print_freq: int = 100,
    logger=None,
):
    """
    Evaluate the classification model.
    
    Args:
        model: The neural network model
        data_loader: DataLoader for evaluation data
        device: The device to run on
        criterion: The loss function
        amp_autocast: Whether to use automatic mixed precision
        print_freq: Frequency of logging
        logger: Logger instance
        
    Returns:
        Tuple of (average_loss, accuracy, auprc, auroc)
    """
    model.eval()

    loss_metric = AverageMeter()
    total = 0
    acc_count = 0

    criterion.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)

            with torch.autocast(
                device_type=_autocast_device_type(device),
                enabled=amp_autocast,
                dtype=torch.bfloat16,
            ):
                pred = model(data)

            loss = criterion(pred, data.y)
            loss_metric.update(loss.item(), n=pred.shape[0])

            total += pred.shape[0]
            acc_count += (
                (pred.detach().cpu().argmax(dim=-1) == data.y.cpu()).sum()
            )

            all_targets.append(data.y.cpu())
            all_preds.append(torch.softmax(pred, dim=-1).detach().cpu())

        # Compute final metrics
        all_targets = torch.cat(all_targets, dim=0)
        all_preds = torch.cat(all_preds, dim=0).to(dtype=torch.float)[:, 1]
        auprc = binary_auprc(all_preds, all_targets)
        auroc = binary_auroc(all_preds, all_targets)
        acc = acc_count / total

    return loss_metric.avg, acc, auprc, auroc


def compute_stats(
    data_loader: Iterable,
    max_radius: float,
    logger,
    print_freq: int = 1000,
):
    """
    Compute statistics (mean nodes, edges, degrees) for the dataset.
    
    Args:
        data_loader: DataLoader for the dataset
        max_radius: Maximum radius for radius graph construction
        logger: Logger instance
        print_freq: Frequency of logging statistics
    """
    log_str = f"\nCalculating statistics with max_radius={max_radius}\n"
    logger.info(log_str)

    avg_node = AverageMeter()
    avg_edge = AverageMeter()
    avg_degree = AverageMeter()

    for step, data in enumerate(data_loader):
        pos = data.pos
        batch = data.batch
        
        # Construct radius graph
        edge_src, edge_dst = radius_graph(
            pos, r=max_radius, batch=batch, max_num_neighbors=1000
        )
        
        batch_size = float(batch.max() + 1)
        num_nodes = pos.shape[0]
        num_edges = edge_src.shape[0]
        num_degree = torch_geometric.utils.degree(edge_src, num_nodes)
        num_degree = torch.sum(num_degree)

        avg_node.update(num_nodes / batch_size, batch_size)
        avg_edge.update(num_edges / batch_size, batch_size)
        avg_degree.update(num_degree / num_nodes, num_nodes)

        if step % print_freq == 0 or step == len(data_loader) - 1:
            log_str = (
                f"[{step}/{len(data_loader)}]\t"
                f"avg node: {avg_node.avg:.2f}, "
                f"avg edge: {avg_edge.avg:.2f}, "
                f"avg degree: {avg_degree.avg:.2f}"
            )
            logger.info(log_str)