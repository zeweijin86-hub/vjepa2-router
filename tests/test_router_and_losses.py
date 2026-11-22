"""Unit tests for standalone routing, non-target diversity regularization, and FocalLoss.

Run without upstream V-JEPA 2 dependencies:
    python -m unittest discover -s tests -v
"""

import unittest

import torch

from evals.video_classification_frozen.losses import FocalLoss
from evals.video_classification_frozen.router import (
    HeadRouter,
    compute_difficulty_weights,
    deduplicate_by_name,
    differentiable_diversity_loss,
    evaluate_router,
    stratified_train_val_split,
    train_and_eval_router,
)


class TestFocalLoss(unittest.TestCase):
    def test_fp16_extreme_logits_are_finite(self):
        criterion = FocalLoss(gamma=2.0, reduction="none")
        logits = torch.tensor([[20.0, -20.0], [-15.0, 15.0]], dtype=torch.float16, requires_grad=True)
        labels = torch.tensor([0, 0], dtype=torch.long)

        loss = criterion(logits, labels)
        self.assertEqual(loss.shape, (2,))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss).all())

        loss.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_invalid_reduction_raises(self):
        with self.assertRaises(ValueError):
            FocalLoss(gamma=2.0, reduction="invalid")

    def test_class_weights_scale_loss(self):
        logits = torch.tensor([[0.2, 0.8], [0.2, 0.8]], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.long)
        unweighted = FocalLoss(gamma=2.0, reduction="none")(logits, labels)
        weighted = FocalLoss(gamma=2.0, weight=torch.tensor([3.0, 0.5]), reduction="none")(logits, labels)
        self.assertTrue(torch.allclose(weighted[0], unweighted[0] * 3.0))
        self.assertTrue(torch.allclose(weighted[1], unweighted[1] * 0.5))


class TestDifficultyAndDiversity(unittest.TestCase):
    def test_misclassified_low_confidence_is_always_hard(self):
        probs = torch.tensor(
            [
                [0.48, 0.52],  # wrong (label=0), low confidence -> hard (2.0)
                [0.52, 0.48],  # right (label=0), low confidence -> hard (2.0)
                [0.96, 0.04],  # right (label=0), high confidence -> easy (0.1)
                [0.75, 0.25],  # right (label=0), mid confidence -> neutral (1.0)
            ]
        )
        labels = torch.tensor([0, 0, 0, 0])
        weights, hard_mask = compute_difficulty_weights(probs, labels, epoch=30)

        self.assertAlmostEqual(weights[0].item(), 2.0)
        self.assertAlmostEqual(weights[1].item(), 2.0)
        self.assertAlmostEqual(weights[2].item(), 0.1)
        self.assertAlmostEqual(weights[3].item(), 1.0)
        self.assertEqual(hard_mask.tolist(), [True, True, False, False])

    def test_diversity_loss_is_noop_for_binary_classification(self):
        head = torch.tensor([[0.8, 0.2], [0.3, 0.7]], requires_grad=True)
        peers = torch.tensor([[[0.7, 0.3], [0.2, 0.8]]])
        labels = torch.tensor([0, 1])
        loss = differentiable_diversity_loss(head, peers, labels)
        self.assertEqual(loss.item(), 0.0)

    def test_diversity_loss_masks_ground_truth_class_in_multiclass(self):
        head = torch.tensor([[0.70, 0.15, 0.10, 0.05], [0.10, 0.60, 0.20, 0.10]], requires_grad=True)
        peers = torch.tensor([[[0.75, 0.10, 0.10, 0.05], [0.15, 0.55, 0.15, 0.15]]])
        labels = torch.tensor([0, 1])

        loss = differentiable_diversity_loss(head, peers, labels)
        self.assertTrue(0.0 <= loss.item() <= 1.0 + 1e-6)
        loss.backward()
        self.assertIsNotNone(head.grad)
        self.assertTrue(torch.isfinite(head.grad).all())
        # Gradient wrt the ground-truth class probability must be strictly zero
        self.assertAlmostEqual(head.grad[0, 0].item(), 0.0, places=6)
        self.assertAlmostEqual(head.grad[1, 1].item(), 0.0, places=6)

    def test_diversity_loss_saturated_zero_nontarget_is_finite(self):
        # Regression test: saturated prediction where all non-target probs underflow to 0.0
        head = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
        peers = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        labels = torch.tensor([0, 1])

        loss = differentiable_diversity_loss(head, peers, labels)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(head.grad is None or torch.isfinite(head.grad).all())

    def test_diversity_loss_empty_peers_is_zero(self):
        head = torch.tensor([[0.7, 0.2, 0.1]], requires_grad=True)
        empty_peers = torch.empty((0, 1, 3))
        labels = torch.tensor([0])
        loss = differentiable_diversity_loss(head, empty_peers, labels)
        self.assertEqual(loss.item(), 0.0)


class TestRouterAndDeduplication(unittest.TestCase):
    def test_router_single_sample_train_and_eval(self):
        router = HeadRouter(num_heads=4, num_classes=2)
        single = torch.randn(1, 4, 2).softmax(dim=-1)

        router.train()
        w_train = router(single)
        self.assertEqual(w_train.shape, (1, 4))
        self.assertTrue(torch.allclose(w_train.sum(dim=-1), torch.ones(1), atol=1e-5))

        router.eval()
        w_eval = router(single)
        self.assertEqual(w_eval.shape, (1, 4))
        self.assertTrue(torch.allclose(w_eval.sum(dim=-1), torch.ones(1), atol=1e-5))

    def test_stratified_split_preserves_class_balance(self):
        labels = torch.cat([torch.zeros(80, dtype=torch.long), torch.ones(20, dtype=torch.long)])
        train_idx, val_idx = stratified_train_val_split(labels, val_ratio=0.3, seed=0)

        self.assertEqual(train_idx.numel() + val_idx.numel(), 100)
        self.assertEqual(len(set(train_idx.tolist()) & set(val_idx.tolist())), 0)
        val_ones = int((labels[val_idx] == 1).sum().item())
        self.assertEqual(val_ones, 6)  # 30% of 20

    def test_train_and_eval_router_deterministic_and_reports_baselines(self):
        n = 60
        labels = torch.cat([torch.zeros(n // 2, dtype=torch.long), torch.ones(n // 2, dtype=torch.long)])
        probs = torch.zeros(n, 2, 2)
        probs[: n // 2, 0] = torch.tensor([0.95, 0.05])
        probs[: n // 2, 1] = torch.tensor([0.45, 0.55])
        probs[n // 2 :, 0] = torch.tensor([0.55, 0.45])
        probs[n // 2 :, 1] = torch.tensor([0.05, 0.95])

        router_a, metrics_a = train_and_eval_router(
            probs, labels, num_heads=2, num_classes=2, device=torch.device("cpu"), steps=80, seed=0
        )
        router_b, metrics_b = train_and_eval_router(
            probs, labels, num_heads=2, num_classes=2, device=torch.device("cpu"), steps=80, seed=0
        )

        self.assertGreaterEqual(metrics_a["router_acc"], 95.0)
        self.assertAlmostEqual(metrics_a["best_probe_acc"], 50.0, places=4)
        self.assertEqual(metrics_a, metrics_b)
        for p_a, p_b in zip(router_a.parameters(), router_b.parameters()):
            self.assertTrue(torch.allclose(p_a, p_b))

        # Pure forward evaluation on full set via evaluate_router
        full_metrics = evaluate_router(router_a, probs, labels, torch.device("cpu"))
        self.assertGreaterEqual(full_metrics["router_acc"], 95.0)

    def test_train_and_eval_router_rejects_unlabeled(self):
        probs = torch.full((4, 2, 2), 0.5)
        labels = torch.tensor([-1, -1, -1, -1])
        with self.assertRaises(ValueError):
            train_and_eval_router(probs, labels, num_heads=2, num_classes=2, device=torch.device("cpu"))

    def test_deduplicate_by_name_strips_ddp_padding_deterministically(self):
        names = ["vid_c.mp4", "vid_a.mp4", "vid_b.mp4", "vid_c.mp4"]
        probs = torch.tensor([[1.0], [2.0], [3.0], [1.0]])
        labels = torch.tensor([1, 0, 1, 1])

        u_names, u_probs, u_labels = deduplicate_by_name(names, probs, labels)
        self.assertEqual(u_names, ["vid_a.mp4", "vid_b.mp4", "vid_c.mp4"])
        self.assertEqual(u_probs.squeeze(-1).tolist(), [2.0, 3.0, 1.0])
        self.assertEqual(u_labels.tolist(), [0, 1, 1])


if __name__ == "__main__":
    unittest.main()
