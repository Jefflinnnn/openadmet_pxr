import gc
import json
import numpy as np
import torch
from typing import Iterable, List, Optional

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from loss_functions import BalancedMSELoss


def _autocast_device_type(device: torch.device) -> str:
    # torch.autocast does not support 'mps'; fall back to 'cpu' for MPS devices.
    return device.type if device.type != "mps" else "cpu"
from timm.utils import ModelEmaV2, dispatch_clip_grad
import time
from torch_cluster import radius_graph
import torch_geometric
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import r2_score

from torcheval.metrics.functional import binary_auroc, binary_auprc


class RunningR2:
    """Computes R2 incrementally without storing all predictions in memory."""

    def __init__(self):
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.sum_err2 = 0.0
        self.n = 0

    def update(self, targets, preds):
        t = np.asarray(targets)
        p = np.asarray(preds)
        self.sum_y += t.sum()
        self.sum_y2 += (t ** 2).sum()
        self.sum_err2 += ((t - p) ** 2).sum()
        self.n += len(t)

    def compute(self):
        if self.n == 0:
            return 0.0
        mean_y = self.sum_y / self.n
        ss_tot = self.sum_y2 - self.n * mean_y ** 2
        if ss_tot == 0:
            return 0.0
        return 1.0 - self.sum_err2 / ss_tot

ModelEma = ModelEmaV2


class MMPSampler:
    """Samples batches of MMP pairs for delta loss computation.

    Supports per-pair cliff weighting: pairs with high similarity but large
    potency difference receive boosted loss weight.
    """

    def __init__(self, pairs_json_path: str, smiles_list: List[str],
                 norm_factor: list, batch_size: int = 16, seed: int = 0,
                 cliff_sim_threshold: float = 0.25, cliff_delta_threshold: float = 1.5,
                 cliff_boost: float = 2.0, max_pairs_per_compound: int = 0,
                 global_feats=None):
        from suiren_datasets.org_mol2d import from_smiles

        with open(pairs_json_path) as f:
            all_pairs = json.load(f)

        self.norm_mean = norm_factor[0]
        self.norm_std = norm_factor[1]
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.cliff_sim_threshold = cliff_sim_threshold
        self.cliff_delta_threshold = cliff_delta_threshold
        self.cliff_boost = cliff_boost

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
        valid_pairs = [p for p in all_pairs
                       if p["idx_a"] in self.graph_cache and p["idx_b"] in self.graph_cache]

        # Per-compound pair quota: limit pairs per compound to prevent dominance
        if max_pairs_per_compound > 0:
            compound_counts: dict = {}
            filtered_pairs = []
            # Sort by |delta| descending so high-information pairs are kept first
            valid_pairs.sort(key=lambda p: abs(p["delta_pec50"]), reverse=True)
            for p in valid_pairs:
                a, b = p["idx_a"], p["idx_b"]
                count_a = compound_counts.get(a, 0)
                count_b = compound_counts.get(b, 0)
                if count_a >= max_pairs_per_compound and count_b >= max_pairs_per_compound:
                    continue
                filtered_pairs.append(p)
                compound_counts[a] = count_a + 1
                compound_counts[b] = count_b + 1
            self.pairs = filtered_pairs
        else:
            self.pairs = valid_pairs

        # Precompute per-pair weights
        self.pair_weights = np.ones(len(self.pairs), dtype=np.float32)
        n_cliff = 0
        if self.cliff_boost > 0:
            for i, p in enumerate(self.pairs):
                sim = p.get("tanimoto", 0.0)
                abs_delta = abs(p["delta_pec50"])
                w = min(3.0, 1.0 + abs_delta)
                if sim >= self.cliff_sim_threshold and abs_delta >= self.cliff_delta_threshold:
                    w *= self.cliff_boost
                    n_cliff += 1
                self.pair_weights[i] = w

        self.pair_indices = np.arange(len(self.pairs))
        self._ptr = len(self.pairs)  # force shuffle on first call

        self.global_feats = global_feats

        print(f"MMPSampler: {len(self.pairs)} pairs loaded "
              f"({n_cliff} cliff pairs, boost={self.cliff_boost}x), "
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
        graphs_a = [self.graph_cache[p["idx_a"]].clone() for p in selected]
        graphs_b = [self.graph_cache[p["idx_b"]].clone() for p in selected]
        indices_a = [p["idx_a"] for p in selected]
        indices_b = [p["idx_b"] for p in selected]
        deltas = torch.tensor(
            [p["delta_pec50"] / self.norm_std for p in selected],
            dtype=torch.float, device=device
        )
        weights = torch.tensor(
            [self.pair_weights[i] for i in batch_idx],
            dtype=torch.float, device=device
        )

        global_feats = self.global_feats

        def make_emb_fn(graphs, indices):
            def fn(model, device):
                from torch_geometric.data import Batch
                if global_feats is not None:
                    for i, idx in enumerate(indices):
                        if idx < global_feats.shape[0]:
                            graphs[i].global_feat = global_feats[idx].unsqueeze(0)
                batch = Batch.from_data_list(graphs).to(device)
                return model.forward_embedding(batch), batch
            return fn

        return {
            "emb_a_fn": make_emb_fn(graphs_a, indices_a),
            "emb_b_fn": make_emb_fn(graphs_b, indices_b),
            "delta_norm": deltas,
            "weights": weights,
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


def soft_spearman_loss(pred: torch.Tensor, target: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Differentiable Spearman rank correlation loss via pairwise soft-ranking.

    Each element's soft rank = sum of sigmoid((x_i - x_j) / tau) over all j.
    Returns 1 - spearman(pred, target), so minimising this maximises rank correlation.
    O(n^2) in batch size — keep batches <= 256.
    """
    n = pred.shape[0]
    if n < 4:
        return torch.tensor(0.0, device=pred.device)
    # soft ranks: shape [n]
    pred_exp   = pred.unsqueeze(1).expand(n, n)     # [n, n]
    target_exp = target.unsqueeze(1).expand(n, n)
    pred_ranks   = torch.sigmoid((pred_exp   - pred_exp.T)   / tau).sum(dim=1)
    target_ranks = torch.sigmoid((target_exp - target_exp.T) / tau).sum(dim=1)
    # Pearson on soft ranks = soft Spearman
    pred_r   = pred_ranks   - pred_ranks.mean()
    target_r = target_ranks - target_ranks.mean()
    cos = (pred_r * target_r).sum() / (
        pred_r.norm() * target_r.norm() + 1e-8
    )
    return 1.0 - cos


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
    tail_weight: float = 1.0,
    tail_threshold: float = 3.0,
    mmp_sampler=None,
    mmp_weight: float = 0.0,
    mmp_mode: str = "delta_head",
    mmp_start_epoch: int = 0,
    focal_gamma: float = 0.0,
    contrastive_weight: float = 0.0,
    contrastive_margin: float = 1.0,
    heteroscedastic: bool = False,
    balanced_mse_weighter=None,
    confident_error_weight: float = 0.0,
    aux_norm_factors: list = None,
    aux_weight: float = 0.1,
    aux_weights: list = None,
    aux_start_epoch: int = 10,
    rank_weight: float = 0.0,
    rank_start_epoch: int = 10,
    rank_tau: float = 0.1,
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

    running_r2 = RunningR2()

    for step, data in enumerate(data_loader):
        data = data.to(device)

        # Forward pass with automatic mixed precision
        with torch.autocast(
            device_type=_autocast_device_type(device),
            enabled=amp_autocast,
            dtype=torch.bfloat16,
        ):
            _has_y_aux = (aux_norm_factors is not None
                          and aux_weight > 0
                          and epoch >= aux_start_epoch)
            if _has_y_aux:
                try:
                    _y_aux_check = data.y_aux
                    aux_active = _y_aux_check is not None
                except Exception:
                    aux_active = False
            else:
                aux_active = False

            if aux_active:
                pred, aux_preds = model.forward_aux(data)
            else:
                pred = model(data)

            target_norm = (data.y - task_mean) / task_std

            if heteroscedastic:
                per_sample_loss = criterion(pred, target_norm)
                mu = pred[:, 0]
            else:
                pred = pred.view(-1)
                per_sample_loss = criterion(pred, target_norm)
                if per_sample_loss.dim() == 0:
                    per_sample_loss = per_sample_loss.unsqueeze(0)
                mu = pred

            if balanced_mse_weighter is not None:
                weights = balanced_mse_weighter._get_weights(target_norm)
            elif isinstance(criterion, BalancedMSELoss):
                weights = torch.ones_like(per_sample_loss)
            else:
                weights = torch.ones_like(per_sample_loss)
                if active_weight != 1.0:
                    weights = torch.where(data.y >= 6.0, active_weight, 1.0)
                if tail_weight != 1.0:
                    weights = torch.where(data.y <= tail_threshold, tail_weight, weights)
            if hasattr(data, 'sample_weight') and data.sample_weight is not None:
                weights = weights * data.sample_weight.to(weights.device)
            if focal_gamma > 0 and not heteroscedastic:
                weights = weights * per_sample_loss.detach().abs().clamp(min=1e-6) ** focal_gamma

            if heteroscedastic and confident_error_weight > 0:
                log_var = pred[:, 1].detach()
                confidence = (-log_var).exp()
                abs_err = (mu.detach() - target_norm).abs()
                active_mask = data.y >= 6.0
                boost = 1.0 + confident_error_weight * confidence * abs_err * active_mask.float()
                weights = weights * boost

            reg_loss = (per_sample_loss * weights).mean()

            # MMP delta loss + contrastive loss (gated by mmp_start_epoch)
            delta_loss = 0.0
            contrast_loss = 0.0
            mmp_active = (mmp_sampler is not None and mmp_weight > 0
                          and epoch >= mmp_start_epoch)
            if mmp_active:
                mmp_batch = mmp_sampler.sample(device)
                if mmp_batch is not None:
                    emb_a, batch_a = mmp_batch["emb_a_fn"](model, device)
                    emb_b, batch_b = mmp_batch["emb_b_fn"](model, device)
                    delta_target = mmp_batch["delta_norm"]
                    pair_weights = mmp_batch["weights"]

                    if mmp_mode == "main_head":
                        head_a = model._apply_head(emb_a, batch_a)
                        head_b = model._apply_head(emb_b, batch_b)
                        pred_a = head_a[:, 0] if heteroscedastic else head_a.view(-1)
                        pred_b = head_b[:, 0] if heteroscedastic else head_b.view(-1)
                        delta_pred = pred_a - pred_b
                    else:
                        delta_pred = model.forward_delta(emb_a, emb_b).view(-1)

                    per_pair_loss = torch.nn.functional.l1_loss(
                        delta_pred, delta_target, reduction='none')
                    delta_loss = (per_pair_loss * pair_weights).mean()

                    if contrastive_weight > 0:
                        emb_dist = torch.nn.functional.pairwise_distance(emb_a, emb_b)
                        delta_raw_abs = delta_target.abs() * task_std
                        is_cliff = (delta_raw_abs > 0.5).float()
                        pull_margin = 0.5 * contrastive_margin
                        push = is_cliff * torch.nn.functional.relu(contrastive_margin - emb_dist)
                        pull = (1 - is_cliff) * torch.nn.functional.relu(emb_dist - pull_margin)
                        contrast_loss = (push + pull).mean()

                    del emb_a, emb_b, batch_a, batch_b, delta_pred, delta_target, pair_weights, mmp_batch

            # Auxiliary multi-task loss (masked where target is NaN)
            # aux_weights (per-head) takes precedence over the global aux_weight scalar.
            # The final loss contribution is: sum_i(head_weight_i * L1_i), where
            # head_weight_i = aux_weights[i] if provided, else aux_weight.
            aux_loss = 0.0
            if aux_active:
                n_aux = len(aux_norm_factors)
                y_aux = data.y_aux
                if y_aux.dim() == 1:
                    y_aux = y_aux.view(-1, n_aux)
                for i, (aux_pred, (aux_mean, aux_std)) in enumerate(
                    zip(aux_preds, aux_norm_factors)
                ):
                    aux_target = y_aux[:, i]
                    valid_mask = ~torch.isnan(aux_target)
                    if valid_mask.sum() == 0:
                        continue
                    aux_pred_v = aux_pred.view(-1)[valid_mask]
                    aux_target_norm = (aux_target[valid_mask] - aux_mean) / aux_std
                    head_w = (aux_weights[i] if aux_weights is not None and i < len(aux_weights)
                              else aux_weight)
                    aux_loss = aux_loss + head_w * torch.nn.functional.l1_loss(
                        aux_pred_v, aux_target_norm
                    )

            # Soft Spearman rank correlation loss
            rank_loss = 0.0
            if rank_weight > 0 and epoch >= rank_start_epoch:
                rank_loss = soft_spearman_loss(mu, target_norm, tau=rank_tau)

            # When aux_weights is set, each head's contribution is already scaled
            # by its per-head weight inside the loop above; aux_weight is unused.
            loss = (reg_loss
                    + mmp_weight * delta_loss
                    + contrastive_weight * contrast_loss
                    + (1.0 if aux_weights is not None else aux_weight) * aux_loss
                    + rank_weight * rank_loss)

        # Backward pass and optimization
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            loss.backward()
            if clip_grad is not None:
                dispatch_clip_grad(model.parameters(), value=clip_grad, mode="norm")
            optimizer.step()

        # Update metrics (no accumulation — compute incrementally)
        loss_metric.update(loss.item(), n=mu.shape[0])
        preds_np = (mu.detach() * task_std + task_mean).cpu().numpy().flatten()
        targets_np = data.y.cpu().numpy().flatten()
        mae_metric.update(np.mean(np.abs(preds_np - targets_np)), n=mu.shape[0])
        running_r2.update(targets_np, preds_np)

        # Update EMA model if provided
        if model_ema is not None:
            model_ema.update(model)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()

        del data, pred, target_norm, per_sample_loss, weights, reg_loss, loss

        # Flush MPS cache periodically to prevent semaphore leak
        if hasattr(torch, "mps") and torch.backends.mps.is_available() and step % 50 == 0:
            torch.mps.empty_cache()

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

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    return mae_metric.avg, running_r2.compute()


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
    heteroscedastic: bool = False,
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

    all_targets_list = []
    all_preds_list = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)

            with torch.autocast(
                device_type=_autocast_device_type(device),
                enabled=amp_autocast,
                dtype=torch.bfloat16,
            ):
                pred = model(data)
                if heteroscedastic:
                    mu = pred[:, 0]
                else:
                    mu = pred.view(-1)

            loss = criterion(mu, (data.y - task_mean) / task_std)
            loss_metric.update(loss.item(), n=mu.shape[0])
            err = mu.detach() * task_std + task_mean - data.y
            mae_metric.update(torch.mean(torch.abs(err)).item(), n=mu.shape[0])

            all_targets_list.append(data.y.cpu().numpy().flatten())
            all_preds_list.append((mu.detach() * task_std + task_mean).cpu().float().numpy().flatten())

        all_targets = np.concatenate(all_targets_list)
        all_preds = np.concatenate(all_preds_list)
        del all_targets_list, all_preds_list
        r2 = r2_score(all_targets, all_preds)
        spearman = spearmanr(all_targets, all_preds).statistic
        kendall = kendalltau(all_targets, all_preds).statistic

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    return mae_metric.avg, r2, loss_metric.avg, all_targets, all_preds, spearman, kendall


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
    all_targets_list = []
    all_preds_list = []

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

        all_targets_list.append(data.y.cpu().numpy())
        all_preds_list.append(torch.softmax(pred, dim=-1).detach().cpu().numpy())

        # Update EMA model if provided
        if model_ema is not None:
            model_ema.update(model)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()

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
    all_targets = torch.from_numpy(np.concatenate(all_targets_list))
    all_preds = torch.from_numpy(np.concatenate(all_preds_list)).float()[:, 1]
    del all_targets_list, all_preds_list
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
    all_targets_list = []
    all_preds_list = []

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

            all_targets_list.append(data.y.cpu().numpy())
            all_preds_list.append(torch.softmax(pred, dim=-1).detach().cpu().numpy())

        # Compute final metrics
        all_targets = torch.from_numpy(np.concatenate(all_targets_list))
        all_preds = torch.from_numpy(np.concatenate(all_preds_list)).float()[:, 1]
        del all_targets_list, all_preds_list
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