# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import logging
import math
import pprint
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from .losses import FocalLoss
from .router import (
    HeadRouter,
    compute_difficulty_weights,
    deduplicate_by_name,
    differentiable_diversity_loss,
    evaluate_router,
    train_and_eval_router,
)

from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.data_manager import init_data
from src.models.attentive_pooler import AttentiveClassifier
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter, CSVLogger

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "[%(levelname)s][%(asctime)s][%(name)s][%(funcName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
for handler in logger.handlers:
    handler.setFormatter(formatter)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def _atomic_torch_save(obj, target_path):
    """Write checkpoint to a temporary file and atomically replace target_path."""
    tmp_path = f"{target_path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, target_path)


def _get_dataset_names(dataset):
    """Retrieve sample names from a dataset, unwrapping MonitoredDataset if present."""
    return getattr(dataset, "names", getattr(getattr(dataset, "dataset", None), "names", None))


def _resolve_saved_router(device, num_heads, num_classes, folder, embedded_router_state, expected_epoch, rank):
    """Load an epoch-synchronized HeadRouter from best_probes.pt (preferred) or router.pt."""
    if embedded_router_state is not None:
        router = HeadRouter(num_heads, num_classes).to(device)
        router.load_state_dict(embedded_router_state)
        router.eval()
        if rank == 0:
            logger.info(f"Loaded joint-atomic HeadRouter embedded in probe checkpoint (epoch={expected_epoch}).")
        return router

    router_path = os.path.join(folder, "router.pt")
    if not os.path.exists(router_path):
        return None

    try:
        payload = torch.load(router_path, map_location=device, weights_only=True)
        if isinstance(payload, dict) and "state_dict" in payload:
            saved_epoch = payload.get("epoch")
            if expected_epoch is not None and saved_epoch is not None and saved_epoch != expected_epoch:
                if rank == 0:
                    logger.warning(
                        f"Router epoch mismatch ({router_path}: epoch={saved_epoch} vs probes: epoch={expected_epoch}); "
                        "ignoring stale router.pt."
                    )
                return None
            state_dict = payload["state_dict"]
        else:
            state_dict = payload

        router = HeadRouter(num_heads, num_classes).to(device)
        router.load_state_dict(state_dict)
        router.eval()
        if rank == 0:
            logger.info(f"Loaded HeadRouter checkpoint from {router_path}")
        return router
    except Exception as e:
        if rank == 0:
            logger.warning(f"Failed to load HeadRouter from {router_path} ({e}).")
        return None


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- VAL ONLY
    val_only = args_eval.get("val_only", False)
    if val_only:
        logger.info("VAL ONLY")

    # -- INFERENCE ONLY
    infer_only = args_eval.get("infer_only", False)
    if infer_only:
        logger.info("INFER ONLY")

    # -- EXPERIMENT
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", args_eval.get("max_workers", 12))
    log_interval = args_eval.get("log_interval", 2)

    # -- PRETRAIN
    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment")

    # -- CLASSIFIER
    args_classifier = args_exp.get("classifier")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)

    # -- DATA
    args_data = args_exp.get("data")
    dataset_type = args_data.get("dataset_type", "VideoDataset")
    num_classes = args_data.get("num_classes")
    train_data_path = [args_data.get("dataset_train")]
    val_data_path = [args_data.get("dataset_val")]
    resolution = args_data.get("resolution", 224)
    num_segments = args_data.get("num_segments", 1)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 4)
    duration = args_data.get("clip_duration", None)
    num_views_per_segment = args_data.get("num_views_per_segment", 1)
    normalization = args_data.get("normalization", None)
    color_jitter = args_data.get("color_jitter", True)

    # -- OPTIMIZATION
    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    focal_gamma = float(args_opt.get("focal_gamma", 2.0))
    use_difficulty_weighting = bool(args_opt.get("use_difficulty_weighting", True))
    diversity_weight = float(args_opt.get("diversity_weight", 0.0))
    opt_kwargs = [
        dict(
            ref_wd=kwargs.get("weight_decay"),
            final_wd=kwargs.get("final_weight_decay"),
            start_lr=kwargs.get("start_lr"),
            ref_lr=kwargs.get("lr"),
            final_lr=kwargs.get("final_lr"),
            warmup=kwargs.get("warmup"),
        )
        for kwargs in args_opt.get("multihead_kwargs")
    ]
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    if rank == 0 and diversity_weight > 0.0 and num_classes <= 2:
        logger.warning(
            "diversity_weight > 0 with num_classes <= 2: non-target probability diversity is "
            "undefined on a 1-DoF binary simplex and will be skipped."
        )

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, "video_classification_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    best_probes_path = os.path.join(folder, "best_probes.pt")
    router_save_path = os.path.join(folder, "router.pt")

    # -- make csv_logger (training runs only; avoid polluting logs during val_only/infer_only)
    if rank == 0 and not (infer_only or val_only):
        csv_logger = CSVLogger(
            log_file,
            ("%d", "epoch"),
            ("%.5f", "train_acc"),
            ("%.5f", "val_best_probe_acc"),
            ("%.5f", "holdout_best_probe_acc"),
            ("%.5f", "holdout_uniform_ens_acc"),
            ("%.5f", "router_holdout_acc"),
            ("%.5f", "best_router_holdout_acc"),
        )

    # -- init models
    encoder = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )
    # -- init classifier probes
    classifiers = [
        AttentiveClassifier(
            embed_dim=encoder.embed_dim,
            num_heads=num_heads,
            depth=num_probe_blocks,
            num_classes=num_classes,
            use_activation_checkpointing=True,
        ).to(device)
        for _ in opt_kwargs
    ]
    classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]

    val_loader, _ = make_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_segments=num_segments,
        eval_duration=duration,
        num_views_per_segment=num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=num_workers,
        normalization=normalization,
        color_jitter=False,
    )

    # INFERENCE ONLY: skip train_loader/optimizer setup and enforce epoch-synced checkpoint loading
    if infer_only:
        if os.path.exists(best_probes_path):
            ckpt_to_load = best_probes_path
            allow_router = True
        elif os.path.exists(latest_path):
            ckpt_to_load = latest_path
            allow_router = False
            if rank == 0:
                logger.warning(
                    f"best_probes.pt not found at {best_probes_path}; falling back to {latest_path} "
                    "with uniform probe averaging (router disabled to avoid epoch mismatch)."
                )
        else:
            raise FileNotFoundError(
                f"Inference requires a trained probe checkpoint, but neither {best_probes_path} "
                f"nor {latest_path} exists."
            )

        classifiers, _, _, ckpt_epoch, best_router_acc_tracker, embedded_router = load_checkpoint(
            device=device,
            r_path=ckpt_to_load,
            classifiers=classifiers,
            opt=None,
            scaler=None,
            val_only=True,
        )
        saved_router = (
            _resolve_saved_router(
                device, len(classifiers), num_classes, folder, embedded_router, ckpt_epoch, rank
            )
            if allow_router
            else None
        )
        logger.info(f"Running inference only (checkpoint epoch={ckpt_epoch}, best router acc={best_router_acc_tracker:.2f}%)...")
        infer_one_epoch(
            device=device,
            encoder=encoder,
            classifiers=classifiers,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            folder=folder,
            num_classes=num_classes,
            saved_router=saved_router,
        )
        return

    # VAL ONLY: pure forward evaluation using saved router (no refitting/leakage on held-out test sets)
    if val_only:
        if os.path.exists(best_probes_path):
            ckpt_to_load = best_probes_path
            allow_router = True
        elif os.path.exists(latest_path):
            ckpt_to_load = latest_path
            allow_router = False
        else:
            raise FileNotFoundError(
                f"Validation requires a trained probe checkpoint, but neither {best_probes_path} "
                f"nor {latest_path} exists."
            )
        classifiers, _, _, ckpt_epoch, best_router_acc_tracker, embedded_router = load_checkpoint(
            device=device,
            r_path=ckpt_to_load,
            classifiers=classifiers,
            opt=None,
            scaler=None,
            val_only=True,
        )
        saved_router = (
            _resolve_saved_router(
                device, len(classifiers), num_classes, folder, embedded_router, ckpt_epoch, rank
            )
            if allow_router
            else None
        )
        val_acc, eval_metrics, _, _, _ = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            classifiers=classifiers,
            scaler=None,
            optimizer=None,
            scheduler=None,
            wd_scheduler=None,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            log_interval=log_interval,
            rank=rank,
            epoch=0,
            best_router_acc=best_router_acc_tracker,
            focal_gamma=focal_gamma,
            use_difficulty_weighting=False,
            diversity_weight=0.0,
            saved_router=saved_router,
        )
        if rank == 0:
            mode_tag = "SAVED ROUTER (FULL DEDUP EVAL)" if saved_router is not None else "REFIT ROUTER (30% HOLDOUT)"
            logger.info(
                "[VAL ONLY | %s] batch_best_probe: %.3f%% | same_split_best_probe: %.3f%% | "
                "same_split_uniform_ens: %.3f%% | router: %.3f%%"
                % (
                    mode_tag,
                    val_acc,
                    eval_metrics["best_probe_acc"],
                    eval_metrics["uniform_ens_acc"],
                    eval_metrics["router_acc"],
                )
            )
        return

    train_loader, train_sampler = make_dataloader(
        dataset_type=dataset_type,
        root_path=train_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        eval_duration=duration,
        num_segments=num_segments,
        num_views_per_segment=1,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        num_workers=num_workers,
        normalization=normalization,
        color_jitter=color_jitter,
    )
    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")

    # -- optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=(use_bfloat16 and device.type == "cuda"),
    )

    # -- load training checkpoint
    start_epoch = 0
    best_router_acc_tracker = 0.0

    if resume_checkpoint and os.path.exists(latest_path):
        classifiers, optimizer, scaler, start_epoch, best_router_acc_tracker, _ = load_checkpoint(
            device=device,
            r_path=latest_path,
            classifiers=classifiers,
            opt=optimizer,
            scaler=scaler,
            val_only=False,
        )
        logger.info(f"Resuming with Best Router Holdout Acc: {best_router_acc_tracker:.2f}%")

        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

    def save_checkpoint(epoch, target_path=latest_path, include_opt=True, router_state=None):
        if rank != 0:
            return
        save_dict = {
            "classifiers": [c.state_dict() for c in classifiers],
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
            "best_router_acc": best_router_acc_tracker,
        }
        if include_opt:
            save_dict["opt"] = [o.state_dict() for o in optimizer]
            save_dict["scaler"] = None if scaler is None else [s.state_dict() for s in scaler]
        if router_state is not None:
            save_dict["router"] = router_state
        _atomic_torch_save(save_dict, target_path)

    # TRAIN LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        train_sampler.set_epoch(epoch)
        train_acc, _, _, _, _ = run_one_epoch(
            device=device,
            training=True,
            encoder=encoder,
            classifiers=classifiers,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=train_loader,
            use_bfloat16=use_bfloat16,
            log_interval=log_interval,
            rank=rank,
            epoch=epoch,
            best_router_acc=best_router_acc_tracker,
            focal_gamma=focal_gamma,
            use_difficulty_weighting=use_difficulty_weighting,
            diversity_weight=diversity_weight,
        )

        logger.info("Epoch %d Eval" % (epoch + 1))

        val_acc, holdout_metrics, new_best_router_acc, router_improved, router_state = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            classifiers=classifiers,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            log_interval=log_interval,
            rank=rank,
            epoch=epoch,
            best_router_acc=best_router_acc_tracker,
            focal_gamma=focal_gamma,
            use_difficulty_weighting=False,
            diversity_weight=0.0,
        )
        best_router_acc_tracker = new_best_router_acc

        logger.info(
            "[%5d] train: %.3f%% | val_best_probe: %.3f%% | holdout(probe: %.3f%%, ens: %.3f%%, router: %.3f%%, best: %.3f%%)"
            % (
                epoch + 1,
                train_acc,
                val_acc,
                holdout_metrics["best_probe_acc"],
                holdout_metrics["uniform_ens_acc"],
                holdout_metrics["router_acc"],
                best_router_acc_tracker,
            )
        )
        if rank == 0:
            csv_logger.log(
                epoch + 1,
                train_acc,
                val_acc,
                holdout_metrics["best_probe_acc"],
                holdout_metrics["uniform_ens_acc"],
                holdout_metrics["router_acc"],
                best_router_acc_tracker,
            )

        save_checkpoint(epoch + 1, target_path=latest_path, include_opt=True)
        if router_improved and rank == 0 and router_state is not None:
            # Joint atomic snapshot: embed router weights inside lightweight best_probes.pt
            save_checkpoint(epoch + 1, target_path=best_probes_path, include_opt=False, router_state=router_state)
            # Also export standalone router.pt with epoch metadata for external reuse
            _atomic_torch_save({"state_dict": router_state, "epoch": epoch + 1}, router_save_path)
            logger.info(
                f"[Saved Synchronized Pair] best_probes.pt (with embedded router) + router.pt "
                f"(epoch={epoch + 1}, Holdout Acc={best_router_acc_tracker:.2f}%)"
            )


# =========================================================================== #
# --- Custom Modification: Distributed Inference with HeadRouter ---
# =========================================================================== #
@torch.no_grad()
def infer_one_epoch(
    device,
    encoder,
    classifiers,
    data_loader,
    use_bfloat16,
    folder,
    num_classes,
    saved_router=None,
):
    """Run distributed inference over val_loader using saved_router if provided, else uniform mean."""
    encoder.eval()
    for c in classifiers:
        c.eval()

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    router = saved_router
    num_heads = len(classifiers)

    if rank == 0:
        logger.info(f"Inference config: num_heads={num_heads}, num_classes={num_classes}, router={'enabled' if router is not None else 'uniform_avg'}")

    amp_enabled = bool(use_bfloat16 and device.type == "cuda")
    output_path = os.path.join(folder, "predictions.txt")
    local_names = []
    local_results = []
    total_batches = len(data_loader)

    for itr, data in enumerate(data_loader):
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            clips = [[dij.to(device, non_blocking=True) for dij in di] for di in data[0]]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            names = data[3]

            outputs = encoder(clips, clip_indices)
            outputs = [[c(o) for o in outputs] for c in classifiers]

        # Average FP32 class probabilities across views for each probe: [B, H, C]
        processed_heads = [
            sum([F.softmax(o.float(), dim=1) for o in coutputs]) / len(coutputs)
            for coutputs in outputs
        ]
        batch_probs = torch.stack(processed_heads, dim=1)

        if router is not None:
            weights = router(batch_probs)  # [B, H]
            fused_probs = torch.sum(batch_probs * weights.unsqueeze(2), dim=1)  # [B, C]
        else:
            fused_probs = batch_probs.mean(dim=1)  # [B, C]

        final_predictions = fused_probs.max(dim=1).indices
        local_names.extend(names)
        local_results.extend([r.item() for r in final_predictions])

        if rank == 0 and itr % 10 == 0:
            logger.info(f"[Rank {rank}] Inference progress: {itr}/{total_batches}")

    if world_size > 1:
        gathered_names = [None for _ in range(world_size)] if rank == 0 else None
        gathered_results = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_names, gathered_names, dst=0)
        dist.gather_object(local_results, gathered_results, dst=0)
        if rank == 0:
            all_names = [n for sub in gathered_names for n in sub]
            all_results = [r for sub in gathered_results for r in sub]
    else:
        all_names = local_names
        all_results = local_results

    error_msg = None
    if rank == 0:
        # Strip duplicate samples introduced by DistributedSampler padding and sort deterministically
        unique_names, unique_results = deduplicate_by_name(all_names, all_results, sort_by_name=True)

        # Verify no corrupted/unreadable videos were silently replaced and dropped
        dataset_names = _get_dataset_names(data_loader.dataset)
        if dataset_names is not None:
            expected = set(dataset_names)
            missing = sorted(expected - set(unique_names))
            if missing:
                preview = ", ".join(missing[:5])
                error_msg = (
                    f"Inference coverage mismatch: expected {len(expected)} unique videos from manifest, "
                    f"got {len(unique_names)} ({len(missing)} unreadable/missing, e.g. [{preview}])."
                )

        if error_msg is None:
            tmp_output = f"{output_path}.tmp.{os.getpid()}"
            with open(tmp_output, "w") as f:
                for name, label in zip(unique_names, unique_results):
                    f.write(f"{name} {label}\n")
            os.replace(tmp_output, output_path)
            logger.info(f"Inference complete ({len(unique_names)} unique samples). Saved predictions to {output_path}")

    # Broadcast error across all ranks BEFORE barrier to avoid NCCL deadlock on failure
    if world_size > 1:
        err_box = [error_msg]
        dist.broadcast_object_list(err_box, src=0)
        error_msg = err_box[0]

    if error_msg is not None:
        raise RuntimeError(error_msg)

    if dist.is_initialized():
        dist.barrier()

    return 0


def run_one_epoch(
    device,
    training,
    encoder,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    log_interval,
    rank,
    epoch,
    best_router_acc=0.0,
    focal_gamma=2.0,
    use_difficulty_weighting=True,
    diversity_weight=0.0,
    saved_router=None,
):
    for c in classifiers:
        c.train(mode=training)

    amp_enabled = bool(use_bfloat16 and device.type == "cuda")
    div_weight = 0.0 if epoch < 10 else (diversity_weight if epoch < 20 else 2.0 * diversity_weight)
    use_diversity = training and div_weight > 0.0 and len(classifiers) > 1

    criterion = FocalLoss(gamma=focal_gamma, weight=None, reduction="none")
    top1_meters = [AverageMeter() for _ in classifiers]
    uniform_ensemble_meter = AverageMeter()

    val_probs_buffer = []
    val_labels_buffer = []
    val_names_buffer = []

    for itr, data in enumerate(data_loader):
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            clips = [[dij.to(device, non_blocking=True) for dij in di] for di in data[0]]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1].to(device)
            names = data[3] if len(data) > 3 else [f"r{rank}_sample_{itr}_{i}" for i in range(len(labels))]
            batch_size = len(labels)

            with torch.no_grad():
                outputs = encoder(clips, clip_indices)
                if not training:
                    outputs = [[c(o) for o in outputs] for c in classifiers]
            if training:
                outputs = [[c(o) for o in outputs] for c in classifiers]

        # Per-sample FP32 Focal Loss per probe and clip view
        losses = [[criterion(o, labels) for o in coutputs] for coutputs in outputs]

        # Differentiable per-probe mean probabilities [B, C] only when non-target diversity is active
        num_classes_cur = outputs[0][0].shape[-1]
        active_diversity = use_diversity and num_classes_cur > 2
        if active_diversity:
            head_probs_grad = [
                sum([F.softmax(o.float(), dim=1) for o in coutputs]) / len(coutputs)
                for coutputs in outputs
            ]
        else:
            head_probs_grad = None

        with torch.no_grad():
            if head_probs_grad is not None:
                ensemble_outputs = [hp.detach() for hp in head_probs_grad]
            else:
                ensemble_outputs = [
                    sum([F.softmax(o.float(), dim=1) for o in coutputs]) / len(coutputs)
                    for coutputs in outputs
                ]

            if not training:
                batch_stack = torch.stack(ensemble_outputs, dim=1).cpu().float()  # [B, H, C]
                val_probs_buffer.append(batch_stack)
                val_labels_buffer.append(labels.cpu())
                val_names_buffer.extend(list(names))

            # Sample-weighted per-probe and uniform-ensemble accuracy (handles uneven tail batches)
            correct_counts = [ens_out.max(dim=1).indices.eq(labels).sum() for ens_out in ensemble_outputs]
            top1_accs = [100.0 * count / batch_size for count in correct_counts]
            top1_accs_synced = [float(AllReduce.apply(t1a.clone().detach())) for t1a in top1_accs]
            for t1m, t1a in zip(top1_meters, top1_accs_synced):
                t1m.update(t1a, n=batch_size)

            uniform_probs = sum(ensemble_outputs) / len(ensemble_outputs)  # [B, C]
            uniform_acc = 100.0 * uniform_probs.max(dim=1).indices.eq(labels).sum() / batch_size
            uniform_ensemble_meter.update(float(AllReduce.apply(uniform_acc.clone().detach())), n=batch_size)

            # Optional training-only dynamic difficulty sample weights
            sample_weights = None
            hard_mask = None
            if training:
                if use_difficulty_weighting:
                    sample_weights, hard_mask = compute_difficulty_weights(uniform_probs, labels, epoch)
                    sample_weights = sample_weights.to(device=device)
                else:
                    sample_weights = torch.ones(batch_size, device=device, dtype=torch.float32)

        # --- Backpropagation ---
        if training:
            detached_heads = ensemble_outputs if active_diversity else None

            probe_losses = []
            for cls_idx in range(len(classifiers)):
                cls_losses = losses[cls_idx]
                weighted_seg_losses = [(seg_loss * sample_weights).mean() for seg_loss in cls_losses]
                total_probe_loss = sum(weighted_seg_losses)

                if active_diversity:
                    peers = torch.stack(
                        [detached_heads[j] for j in range(len(classifiers)) if j != cls_idx], dim=0
                    )  # [H-1, B, C]
                    div_term = differentiable_diversity_loss(head_probs_grad[cls_idx], peers, labels) * div_weight
                    total_probe_loss = total_probe_loss + div_term

                probe_losses.append(total_probe_loss)

            if amp_enabled and scaler is not None and scaler[0] is not None:
                for s, loss_i, o in zip(scaler, probe_losses, optimizer):
                    s.scale(loss_i).backward()
                    s.step(o)
                    s.update()
            else:
                for loss_i, o in zip(probe_losses, optimizer):
                    loss_i.backward()
                    o.step()
            for o in optimizer:
                o.zero_grad()

        # --- Logging ---
        _agg_top1 = np.array([t1m.avg for t1m in top1_meters])
        mem_mb = torch.cuda.max_memory_allocated() / 1024.0**2 if device.type == "cuda" else 0.0
        if itr % 10 == 0 and rank == 0:
            if training and hard_mask is not None:
                hard_ratio = (hard_mask.sum().item() / batch_size) * 100
                logger.info(
                    "[%5d] hard_ratio: %.1f%% | uniform_ens: %.3f%% | best_probe: %.3f%% [mem: %.2e]"
                    % (itr, hard_ratio, uniform_ensemble_meter.avg, _agg_top1.max(), mem_mb)
                )
            else:
                logger.info(
                    "[%5d] uniform_ens: %.3f%% | best_probe: %.3f%% [mem: %.2e]"
                    % (itr, uniform_ensemble_meter.avg, _agg_top1.max(), mem_mb)
                )

        with torch.no_grad():
            if training:
                avg_loss = torch.stack([li.detach() for li in probe_losses]).mean()
            else:
                avg_loss = torch.stack([loss_j.detach().mean() for loss_i in losses for loss_j in loss_i]).mean()
            if dist.is_initialized():
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss / dist.get_world_size()

        if training and itr % log_interval == 0 and rank == 0:
            lr = optimizer[0].param_groups[0]["lr"]
            time_str = datetime.now().strftime("%Y%m%d %H:%M")
            acc_mean = float(np.mean(top1_accs_synced))
            logger.info(
                "[Time: %s] [Epoch: %d] [Step: %5d] [Loss: %.4f] [LR: %.2e] [Mean Acc: %.1f%%]"
                % (time_str, epoch + 1, itr, avg_loss.item(), lr, acc_mean)
            )

    # ======================================================================= #
    # --- Custom Modification: Hold-out Router Fitting & Pure Forward Eval ----
    # ======================================================================= #
    router_improved = False
    router_state = None
    holdout_metrics = {"router_acc": 0.0, "best_probe_acc": 0.0, "uniform_ens_acc": 0.0}

    if not training and len(val_probs_buffer) > 0:
        local_probs = torch.cat(val_probs_buffer, dim=0)
        local_labels = torch.cat(val_labels_buffer, dim=0)

        if dist.is_initialized():
            world_size = dist.get_world_size()
            all_probs_list = [None for _ in range(world_size)] if rank == 0 else None
            all_labels_list = [None for _ in range(world_size)] if rank == 0 else None
            all_names_list = [None for _ in range(world_size)] if rank == 0 else None
            dist.gather_object(local_probs, all_probs_list, dst=0)
            dist.gather_object(local_labels, all_labels_list, dst=0)
            dist.gather_object(val_names_buffer, all_names_list, dst=0)

            if rank == 0:
                global_probs = torch.cat(all_probs_list, dim=0)
                global_labels = torch.cat(all_labels_list, dim=0)
                global_names = [n for sub in all_names_list for n in sub]
        else:
            global_probs = local_probs
            global_labels = local_labels
            global_names = val_names_buffer

        if rank == 0:
            # Strip DistributedSampler padding duplicates and sort deterministically
            _, global_probs, global_labels = deduplicate_by_name(
                global_names, global_probs, global_labels, sort_by_name=True
            )
            if saved_router is not None:
                # Pure forward evaluation with pre-trained router (zero refitting/leakage)
                holdout_metrics = evaluate_router(saved_router, global_probs, global_labels, device)
            else:
                num_heads = global_probs.shape[1]
                num_classes = global_probs.shape[2]
                router_model, holdout_metrics = train_and_eval_router(
                    global_probs, global_labels, num_heads, num_classes, device, seed=_GLOBAL_SEED
                )
                epoch_router_acc = holdout_metrics["router_acc"]

                if epoch_router_acc > best_router_acc:
                    best_router_acc = epoch_router_acc
                    router_improved = True
                    router_state = {k: v.detach().cpu() for k, v in router_model.state_dict().items()}
                    logger.info(
                        f"[New Best Router] Holdout Acc: {epoch_router_acc:.2f}% "
                        f"(vs. same-split best_probe={holdout_metrics['best_probe_acc']:.2f}%, "
                        f"uniform_ens={holdout_metrics['uniform_ens_acc']:.2f}%)"
                    )
                else:
                    logger.info(
                        f"[Skip Router Save] Holdout Acc ({epoch_router_acc:.2f}%) <= Best ({best_router_acc:.2f}%)"
                    )

        if dist.is_initialized():
            flag_tensor = torch.tensor(
                [
                    holdout_metrics["router_acc"],
                    holdout_metrics["best_probe_acc"],
                    holdout_metrics["uniform_ens_acc"],
                    best_router_acc,
                    1.0 if router_improved else 0.0,
                ],
                device=device,
            )
            dist.broadcast(flag_tensor, src=0)
            holdout_metrics = {
                "router_acc": float(flag_tensor[0].item()),
                "best_probe_acc": float(flag_tensor[1].item()),
                "uniform_ens_acc": float(flag_tensor[2].item()),
            }
            best_router_acc = float(flag_tensor[3].item())
            router_improved = bool(flag_tensor[4].item() > 0.5)
            dist.barrier()

    return (
        float(np.array([t1m.avg for t1m in top1_meters]).max()),
        holdout_metrics,
        best_router_acc,
        router_improved,
        router_state,
    )


def load_checkpoint(device, r_path, classifiers, opt, scaler, val_only=False):
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    logger.info(f"read-path: {r_path}")

    pretrained_dict = checkpoint["classifiers"]
    msg = [c.load_state_dict(pd) for c, pd in zip(classifiers, pretrained_dict)]
    epoch = checkpoint.get("epoch", 0)
    best_router_acc = checkpoint.get("best_router_acc", 0.0)
    embedded_router = checkpoint.get("router", None)

    if val_only:
        logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {msg}")
        return classifiers, opt, scaler, epoch, best_router_acc, embedded_router

    logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {msg}")
    [o.load_state_dict(pd) for o, pd in zip(opt, checkpoint["opt"])]

    if scaler is not None and checkpoint.get("scaler") is not None:
        [s.load_state_dict(pd) for s, pd in zip(scaler, checkpoint["scaler"])]

    logger.info(f"loaded optimizers from epoch {epoch}")
    return classifiers, opt, scaler, epoch, best_router_acc, embedded_router


def load_pretrained(encoder, pretrained, checkpoint_key="target_encoder"):
    logger.info(f"Loading pretrained model from {pretrained}")
    checkpoint = robust_checkpoint_loader(pretrained, map_location="cpu")
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint["encoder"]

    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f"key '{k}' could not be found in loaded state dict")
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f"{pretrained_dict[k].shape} | {v.shape}")
            logger.info(f"key '{k}' is of different shape in model and loaded state dict")
            exit(1)
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained model with msg: {msg}")
    logger.info(f"loaded pretrained encoder from epoch: {checkpoint['epoch']}\n path: {pretrained}")
    del checkpoint
    return encoder


DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type="VideoDataset",
    img_size=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None,
    normalization=None,
    color_jitter=False,
):
    if normalization is None:
        normalization = DEFAULT_NORMALIZATION

    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=True,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.3, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=img_size,
        normalize=normalization,
        color_jitter=color_jitter,
    )

    data_loader, data_sampler = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        drop_last=False,
        subset_file=subset_file,
    )
    return data_loader, data_sampler


def init_opt(classifiers, iterations_per_epoch, opt_kwargs, num_epochs, use_bfloat16=False):
    optimizers, schedulers, wd_schedulers, scalers = [], [], [], []
    for c, kwargs in zip(classifiers, opt_kwargs):
        param_groups = [
            {
                "params": (p for n, p in c.named_parameters()),
                "mc_warmup_steps": int(kwargs.get("warmup") * iterations_per_epoch),
                "mc_start_lr": kwargs.get("start_lr"),
                "mc_ref_lr": kwargs.get("ref_lr"),
                "mc_final_lr": kwargs.get("final_lr"),
                "mc_ref_wd": kwargs.get("ref_wd"),
                "mc_final_wd": kwargs.get("final_wd"),
            }
        ]
        logger.info("Using AdamW")
        optimizers += [torch.optim.AdamW(param_groups)]
        schedulers += [WarmupCosineLRSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        wd_schedulers += [CosineWDSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        scalers += [torch.amp.GradScaler("cuda") if use_bfloat16 else None]
    return optimizers, scalers, schedulers, wd_schedulers


class WarmupCosineLRSchedule(object):

    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_lr = group.get("mc_ref_lr")
            final_lr = group.get("mc_final_lr")
            start_lr = group.get("mc_start_lr")
            warmup_steps = group.get("mc_warmup_steps")
            T_max = self.T_max - warmup_steps
            if self._step < warmup_steps:
                progress = float(self._step) / float(max(1, warmup_steps))
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                progress = float(self._step - warmup_steps) / float(max(1, T_max))
                new_lr = max(
                    final_lr,
                    final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
                )
            group["lr"] = new_lr


class CosineWDSchedule(object):

    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max

        for group in self.optimizer.param_groups:
            ref_wd = group.get("mc_ref_wd")
            final_wd = group.get("mc_final_wd")
            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if final_wd <= ref_wd:
                new_wd = max(final_wd, new_wd)
            else:
                new_wd = min(final_wd, new_wd)
            group["weight_decay"] = new_wd
