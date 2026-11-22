"""Standalone routing, non-target diversity regularization, and DDP deduplication utilities.

This module has no dependency on upstream V-JEPA 2 internals so that ``HeadRouter``
and its supporting functions can be imported, tested, and reused independently.
"""

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def adjust_difficulty_thresholds(epoch: int) -> Tuple[float, float]:
    """Gradually tighten easy/hard confidence thresholds after epoch 10.

    Returns:
        (easy_thresh, hard_low_thresh):
            - ``easy_thresh``: minimum confidence for a correct prediction to be down-weighted.
            - ``hard_low_thresh``: confidence below which even a correct prediction is up-weighted.
    """
    easy_thresh = 0.90
    hard_low_thresh = 0.60
    if epoch >= 10:
        step = (epoch // 10) * 0.01
        easy_thresh = min(0.95, 0.90 + step)
        hard_low_thresh = min(0.65, 0.60 + step)
    return easy_thresh, hard_low_thresh


def compute_difficulty_weights(
    uniform_probs: Tensor,
    labels: Tensor,
    epoch: int,
    easy_weight: float = 0.1,
    hard_weight: float = 2.0,
) -> Tuple[Tensor, Tensor]:
    """Compute per-sample difficulty weights from the uniform probe ensemble.

    Every misclassified sample is treated as hard (``hard_weight``), along with
    low-confidence correct predictions. High-confidence correct predictions are
    down-weighted to ``easy_weight``.

    Note:
        Combining dynamic difficulty weighting with Focal Loss strongly focuses
        updates on misclassified samples and assumes relatively clean labels.

    Args:
        uniform_probs: Uniform ensemble class probabilities of shape ``[B, C]``.
        labels: Ground-truth integer class labels of shape ``[B]``.
        epoch: Zero-based training epoch index.
        easy_weight: Weight assigned to high-confidence correct samples.
        hard_weight: Weight assigned to misclassified or low-confidence samples.

    Returns:
        sample_weights: Float32 tensor of shape ``[B]``.
        hard_mask: Boolean tensor of shape ``[B]`` indicating hard samples.
    """
    confidences, preds = uniform_probs.max(dim=1)
    correct_mask = preds.eq(labels)
    easy_thresh, hard_low_thresh = adjust_difficulty_thresholds(epoch)

    sample_weights = torch.ones_like(confidences, dtype=torch.float32)
    easy_mask = correct_mask & (confidences >= easy_thresh)
    hard_mask = (~correct_mask) | (correct_mask & (confidences < hard_low_thresh))

    sample_weights[easy_mask] = easy_weight
    sample_weights[hard_mask] = hard_weight
    return sample_weights, hard_mask


def _zero_with_grad(reference: Tensor) -> Tensor:
    """Return a zero scalar that still participates in autograd.

    A plain ``new_zeros(())`` detaches from the graph, so calling ``backward()`` on a
    loss that happens to be that zero (for example when every sample is saturated and
    filtered out) raises "element 0 of tensors does not require grad". Multiplying the
    reference tensor by ``0.0`` keeps the value at exactly zero while preserving a
    valid gradient path, so the training loop never has to special-case this branch.
    """
    return reference.sum() * 0.0


def differentiable_diversity_loss(
    head_prob: Tensor,
    peer_probs_detached: Tensor,
    labels: Tensor,
    norm_eps: float = 1e-6,
) -> Tensor:
    """Differentiable non-target cosine-similarity penalty against detached peer probes.

    To avoid conflicting with supervised classification, the ground-truth class ``y``
    is masked out before measuring cosine similarity over the remaining ``C - 1``
    non-target probabilities. Samples whose non-target probability vector has near-zero
    L2 norm (saturated predictions) are excluded to prevent unstable zero-norm gradients.
    For binary classification (``C <= 2``), the probability simplex has only 1 degree
    of freedom so non-target diversity is undefined and this function returns ``0.0``.

    Args:
        head_prob: Class probability tensor ``[B, C]`` for the active probe (with grad).
        peer_probs_detached: Stacked detached probability tensor ``[H-1, B, C]`` from peers.
        labels: Ground-truth integer class labels ``[B]``.
        norm_eps: Minimum L2 norm required for a sample's non-target vector to be included.

    Returns:
        Scalar mean non-target cosine similarity in ``[0, 1]`` (or ``0.0`` if no valid samples).
    """
    num_classes = head_prob.shape[-1]
    if peer_probs_detached.shape[0] == 0 or num_classes <= 2:
        return _zero_with_grad(head_prob)

    non_target_mask = (~F.one_hot(labels.long(), num_classes=num_classes).bool()).to(dtype=head_prob.dtype)
    head_nt = head_prob * non_target_mask
    peer_mean = peer_probs_detached.mean(dim=0) * non_target_mask

    # Filter out saturated zero-norm non-target vectors to keep backward gradients strictly finite
    valid = (head_nt.detach().norm(dim=-1) > norm_eps) & (peer_mean.norm(dim=-1) > norm_eps)
    if not valid.any():
        return _zero_with_grad(head_prob)

    sim = F.cosine_similarity(head_nt[valid], peer_mean[valid], dim=-1)
    return sim.mean()


def deduplicate_by_name(
    names: Sequence[str],
    *items,
    sort_by_name: bool = True,
):
    """Deduplicate gathered distributed outputs by sample identifier.

    ``DistributedSampler(drop_last=False)`` pads datasets whose length is not
    divisible by ``world_size`` by repeating samples from the start. Stripping
    duplicate identifiers prevents validation router train/val leakage and
    duplicate lines in multi-GPU ``predictions.txt``.

    Args:
        names: Sequence of sample identifiers of length ``N``.
        *items: Parallel ``torch.Tensor`` (first dim ``N``) or Python sequences of length ``N``.
        sort_by_name: If True, sort unique samples lexicographically by name so
            ordering is deterministic across different ``world_size`` configurations.

    Returns:
        Tuple ``(unique_names, *deduped_items)`` with the same container types as input.
    """
    first_indices = {}
    for idx, name in enumerate(names):
        if name not in first_indices:
            first_indices[name] = idx

    ordered_names: List[str] = sorted(first_indices.keys()) if sort_by_name else list(first_indices.keys())
    keep_indices = [first_indices[n] for n in ordered_names]

    deduped = []
    for item in items:
        if isinstance(item, torch.Tensor):
            index_tensor = torch.tensor(keep_indices, dtype=torch.long, device=item.device)
            deduped.append(item.index_select(0, index_tensor))
        else:
            deduped.append([item[i] for i in keep_indices])

    return (ordered_names, *deduped)


def stratified_train_val_split(
    labels: Tensor,
    val_ratio: float = 0.3,
    seed: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Deterministically split sample indices into train/val while preserving class ratios."""
    labels_cpu = labels.detach().cpu().long()
    n_samples = labels_cpu.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)

    if n_samples <= 2:
        perm = torch.randperm(n_samples, generator=gen)
        return perm, perm

    train_parts: List[Tensor] = []
    val_parts: List[Tensor] = []
    unique_classes = torch.unique(labels_cpu, sorted=True)

    for cls in unique_classes:
        cls_indices = torch.nonzero(labels_cpu == cls, as_tuple=False).squeeze(1)
        perm = cls_indices[torch.randperm(cls_indices.numel(), generator=gen)]
        n_cls = perm.numel()
        if n_cls == 1:
            train_parts.append(perm)
        else:
            n_val = max(1, min(n_cls - 1, int(round(n_cls * val_ratio))))
            val_parts.append(perm[:n_val])
            train_parts.append(perm[n_val:])

    if not val_parts:
        perm = torch.randperm(n_samples, generator=gen)
        n_val = max(1, int(round(n_samples * val_ratio)))
        return perm[n_val:], perm[:n_val]

    train_idx = torch.cat(train_parts, dim=0)
    val_idx = torch.cat(val_parts, dim=0)
    train_idx = train_idx[torch.randperm(train_idx.numel(), generator=gen)]
    val_idx = val_idx[torch.randperm(val_idx.numel(), generator=gen)]
    return train_idx, val_idx


class HeadRouter(nn.Module):
    """Lightweight probability-level stacking MLP predicting per-video weights over H probes.

    Uses ``LayerNorm`` instead of ``BatchNorm1d`` so both training and inference
    work reliably for any batch size (including ``B == 1``) without running-stat drift.
    Best suited for low-to-moderate class counts (where ``H * C`` remains compact).

    Input:
        all_probs: ``[B, H, C]`` softmax class probabilities from ``H`` probes.
    Output:
        weights: ``[B, H]`` simplex weights summing to 1 across probes.
    """

    def __init__(
        self,
        num_heads: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.input_dim = num_heads * num_classes
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, all_probs: Tensor) -> Tensor:
        batch_size = all_probs.shape[0]
        flat_input = all_probs.reshape(batch_size, -1)  # [B, H * C]
        return F.softmax(self.net(flat_input), dim=1)  # [B, H]


@torch.no_grad()
def evaluate_router(
    router: HeadRouter,
    all_probs: Tensor,
    all_labels: Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Pure forward evaluation of a trained ``HeadRouter`` alongside same-split baselines.

    Args:
        router: Trained ``HeadRouter`` module.
        all_probs: ``[N, H, C]`` deduplicated softmax class probabilities.
        all_labels: ``[N]`` integer class labels.
        device: Target PyTorch device.

    Returns:
        Dictionary of apples-to-apples accuracies (%): ``router_acc``,
        ``best_probe_acc``, and ``uniform_ens_acc``.
    """
    if all_labels.numel() == 0:
        raise ValueError("Cannot evaluate HeadRouter on an empty dataset.")

    eval_probs = all_probs.to(device=device, dtype=torch.float32)
    eval_labels = all_labels.to(device=device, dtype=torch.long)
    n_samples = eval_labels.size(0)

    router.eval()
    weights = router(eval_probs)  # [N, H]
    fused = torch.sum(eval_probs * weights.unsqueeze(2), dim=1)  # [N, C]
    router_acc = 100.0 * fused.max(dim=1).indices.eq(eval_labels).sum().item() / n_samples

    per_probe_preds = eval_probs.max(dim=2).indices  # [N, H]
    per_probe_accs = 100.0 * per_probe_preds.eq(eval_labels.unsqueeze(1)).float().mean(dim=0)
    best_probe_acc = float(per_probe_accs.max().item())

    uniform_probs = eval_probs.mean(dim=1)  # [N, C]
    uniform_ens_acc = 100.0 * uniform_probs.max(dim=1).indices.eq(eval_labels).sum().item() / n_samples

    return {
        "router_acc": router_acc,
        "best_probe_acc": best_probe_acc,
        "uniform_ens_acc": uniform_ens_acc,
    }


def train_and_eval_router(
    all_probs: Tensor,
    all_labels: Tensor,
    num_heads: int,
    num_classes: int,
    device: torch.device,
    val_ratio: float = 0.3,
    steps: int = 100,
    lr: float = 5e-3,
    weight_decay: float = 1e-2,
    seed: int = 0,
) -> Tuple[HeadRouter, Dict[str, float]]:
    """Fit ``HeadRouter`` on a stratified split and evaluate router vs. baselines on ``router_val``.

    Initialization and dropout RNG states are isolated via ``torch.random.fork_rng``
    with a fixed seed so holdout metrics are deterministic across epochs.

    Args:
        all_probs: ``[N, H, C]`` deduplicated softmax class probabilities.
        all_labels: ``[N]`` integer class labels in ``[0, num_classes - 1]``.
        num_heads: Number of attentive probes ``H``.
        num_classes: Number of target classes ``C``.
        device: Target PyTorch device.
        val_ratio: Fraction held out for ``router_val``.
        steps: Number of full-batch AdamW optimization steps with cosine LR decay.
        lr: Initial learning rate.
        weight_decay: AdamW weight decay.
        seed: Random seed controlling the stratified split, router init, and dropout.

    Returns:
        router: Trained ``HeadRouter`` in eval mode.
        metrics: Dictionary of apples-to-apples accuracies (%) measured on ``router_val``:
            ``router_acc``, ``best_probe_acc``, and ``uniform_ens_acc``.
    """
    n_samples = all_labels.shape[0]
    if n_samples == 0:
        raise ValueError("Cannot fit HeadRouter on an empty validation set.")

    labels_long = all_labels.long()
    if int(labels_long.min().item()) < 0 or int(labels_long.max().item()) >= num_classes:
        raise ValueError(
            f"Router labels must be in [0, {num_classes - 1}], "
            f"got range [{int(labels_long.min().item())}, {int(labels_long.max().item())}]."
        )

    train_idx, val_idx = stratified_train_val_split(labels_long, val_ratio=val_ratio, seed=seed)

    train_probs = all_probs[train_idx].to(device=device, dtype=torch.float32)
    train_labels = labels_long[train_idx].to(device=device)
    val_probs = all_probs[val_idx]
    val_labels = labels_long[val_idx]

    fork_devices = [device.index if device.index is not None else 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)

        router = HeadRouter(num_heads, num_classes).to(device)
        optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps))

        router.train()
        for _ in range(steps):
            optimizer.zero_grad()
            weights = router(train_probs)  # [N_train, H]
            fused_probs = torch.sum(train_probs * weights.unsqueeze(2), dim=1)  # [N_train, C]
            log_probs = torch.log(fused_probs.clamp(min=1e-8))
            loss = F.nll_loss(log_probs, train_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

    metrics = evaluate_router(router, val_probs, val_labels, device)
    return router, metrics
