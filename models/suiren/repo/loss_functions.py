import torch
import torch.nn as nn
import numpy as np


class BalancedMSELoss(nn.Module):
    """MSE loss reweighted by inverse label density (Balanced MSE, CVPR 2022).

    Bins the training targets and assigns each sample a weight proportional to
    1 / (frequency of its bin), so rare tails receive higher loss contribution.
    """

    def __init__(self, targets: torch.Tensor, num_bins: int = 20, smoothing: float = 1.0):
        super().__init__()
        targets_np = targets.numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)

        counts, bin_edges = np.histogram(targets_np, bins=num_bins)
        counts = counts.astype(np.float64) + smoothing
        weights = len(targets_np) / (num_bins * counts)

        self.register_buffer("bin_edges", torch.from_numpy(bin_edges).float())
        self.register_buffer("bin_weights", torch.from_numpy(weights).float())
        self.num_bins = num_bins

    def _get_weights(self, targets: torch.Tensor) -> torch.Tensor:
        bin_idx = torch.bucketize(targets, self.bin_edges[1:-1])
        bin_idx = bin_idx.clamp(0, self.num_bins - 1)
        return self.bin_weights[bin_idx]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Returns per-sample weighted MSE (unreduced)."""
        per_sample = (pred - target) ** 2
        weights = self._get_weights(target)
        return per_sample * weights


class GaussianNLLLoss(nn.Module):
    """Gaussian negative log-likelihood for heteroscedastic regression.

    Expects pred to be [B, 2] with columns [mu, log_var].
    Returns per-sample NLL (unreduced) for compatibility with external weighting.
    """

    def __init__(self, log_var_min: float = -6.0, log_var_max: float = 6.0):
        super().__init__()
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu = pred[:, 0]
        log_var = pred[:, 1].clamp(self.log_var_min, self.log_var_max)
        return 0.5 * log_var + 0.5 * (target - mu) ** 2 / log_var.exp()
