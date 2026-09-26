from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cross_layer.batch_builder import (
    CrossLayerBatchBuilder,
    make_plan,
)
from model.preact_resnet import preact_resnet18


CIFAR100_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR100_STD = (0.2023, 0.1994, 0.2010)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split(
    targets,
    *,
    val_per_class: int,
    seed: int,
):
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)

    train_idx = []
    val_idx = []

    for cls in range(100):
        idx = np.flatnonzero(targets == cls)
        idx = rng.permutation(idx)

        val_idx.extend(idx[:val_per_class].tolist())
        train_idx.extend(idx[val_per_class:].tolist())

    return train_idx, val_idx


def build_loaders(
    data_root: str,
    *,
    source_batch: int,
    workers: int,
    seed: int,
):
    raw_train = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=transforms.PILToTensor(),
    )

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    eval_train = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=False,
        transform=eval_transform,
    )

    test_set = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=False,
        transform=eval_transform,
    )

    train_idx, val_idx = stratified_split(
        raw_train.targets,
        val_per_class=50,
        seed=seed,
    )

    train_set = Subset(raw_train, train_idx)
    val_set = Subset(eval_train, val_idx)

    train_gen = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=source_batch,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        pin_memory=True,
        generator=train_gen,
        persistent_workers=(workers > 0),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=256,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )

    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )

    return train_loader, val_loader, test_loader


def soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total = 0
    correct1 = 0
    loss_sum = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction="sum")

        loss_sum += float(loss.item())
        correct1 += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.numel()

    return {
        "loss": loss_sum / total,
        "acc1": 100.0 * correct1 / total,
    }


def set_cosine_lr(
    optimizer,
    *,
    base_lr: float,
    step: int,
    total_steps: int,
):
    progress = step / max(total_steps, 1)
    lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))

    for group in optimizer.param_groups:
        group["lr"] = lr

    return lr


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    if a.numel() < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1].item())


def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-layer augmentation pilot trainer"
    )

    p.add_argument(
        "--group",
        choices=["g2", "g3", "g4", "g5"],
        required=True,
    )

    p.add_argument(
        "--data-root",
        default="/root/autodl-tmp/CIFAR_100",
    )
    p.add_argument(
        "--output-dir",
        default="cross_layer/results",
    )

    p.add_argument("--seed", type=int, default=20170922)
    p.add_argument("--workers", type=int, default=4)

    p.add_argument("--source-batch", type=int, default=32)
    p.add_argument("--repeats", type=int, default=4)

    p.add_argument("--max-updates", type=int, default=10000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)

    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)

    p.add_argument("--ra-n", type=int, default=3)
    p.add_argument("--fixed-m", type=int, default=5)
    p.add_argument("--role-b-m", type=int, default=9)
    p.add_argument("--m-min", type=int, default=3)
    p.add_argument("--m-max", type=int, default=9)

    p.add_argument("--amp", action="store_true")
    p.add_argument(
        "--test-at-end",
        action="store_true",
        help="For final experiments only. Pilot should select on validation.",
    )

    return p.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    if args.source_batch * args.repeats != 128:
        raise ValueError(
            "Pilot protocol expects source_batch * repeats == 128."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader, val_loader, test_loader = build_loaders(
        args.data_root,
        source_batch=args.source_batch,
        workers=args.workers,
        seed=args.seed,
    )

    model = preact_resnet18(num_classes=100).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(args.amp and device.type == "cuda"),
    )

    builder = CrossLayerBatchBuilder(
        num_classes=100,
        ra_n=args.ra_n,
        role_b_m=args.role_b_m,
        fixed_m=args.fixed_m,
        base_seed=args.seed,
    )

    run_name = f"{args.group}_seed{args.seed}_u{args.max_updates}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    log_file = (run_dir / "metrics.jsonl").open(
        "w", encoding="utf-8", buffering=1
    )

    best_val = -1.0
    best_step = -1
    best_path = run_dir / "best.pt"

    running_loss = 0.0
    running_count = 0

    corr_mix_sum = 0.0
    corr_cut_sum = 0.0
    corr_count = 0

    m_mix_hist = Counter()
    m_cut_hist = Counter()

    start_time = time.time()

    step = 0

    print("=" * 72)
    print(f"Group          : {args.group}")
    print(f"Device         : {device}")
    print(f"Source batch   : {args.source_batch}")
    print(f"Effective batch: {args.source_batch * args.repeats}")
    print(f"Max updates    : {args.max_updates}")
    print(f"RA             : N={args.ra_n}")
    print(f"Fixed M        : {args.fixed_m}")
    print(f"Dynamic M      : [{args.m_min}, {args.m_max}]")
    print(f"AMP            : {args.amp}")
    print("=" * 72)

    while step < args.max_updates:
        for raw_images, labels in train_loader:
            if step >= args.max_updates:
                break

            lr = set_cosine_lr(
                optimizer,
                base_lr=args.lr,
                step=step,
                total_steps=args.max_updates,
            )

            raw_images = raw_images.cpu()
            labels = labels.cpu()

            if args.group == "g2":
                images, soft_targets, metadata = (
                    builder.build_simple_stack(
                        raw_images,
                        labels,
                        step=step,
                        repeats=args.repeats,
                        switch_prob=0.5,
                    )
                )

                if metadata["self_pair_count"] != 0:
                    raise RuntimeError("G2 self pairing detected.")

            else:
                plan = make_plan(
                    batch_size=raw_images.size(0),
                    height=raw_images.size(-2),
                    width=raw_images.size(-1),
                    seed=args.seed + step * 1_000_003,
                    m_min=args.m_min,
                    m_max=args.m_max,
                )

                if args.group == "g3":
                    mode = "shuffle"
                elif args.group == "g4":
                    mode = "inverse"
                elif args.group == "g5":
                    mode = "positive"
                else:
                    raise RuntimeError(args.group)

                images, soft_targets, metadata = builder.build_roles(
                    raw_images,
                    labels,
                    plan,
                    mode=mode,
                    step=step,
                )

                base = torch.arange(raw_images.size(0))

                if torch.any(metadata["mix_partner"] == base):
                    raise RuntimeError("Mixup self pairing detected.")
                if torch.any(metadata["cut_partner"] == base):
                    raise RuntimeError("CutMix self pairing detected.")

                cm = corr(
                    metadata["mix_degree_mix"],
                    metadata["m_mix"],
                )
                cc = corr(
                    metadata["mix_degree_cut"],
                    metadata["m_cut"],
                )

                if math.isfinite(cm):
                    corr_mix_sum += cm
                if math.isfinite(cc):
                    corr_cut_sum += cc
                corr_count += 1

                m_mix_hist.update(metadata["m_mix"].tolist())
                m_cut_hist.update(metadata["m_cut"].tolist())

            images = images.to(device, non_blocking=True)
            soft_targets = soft_targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(args.amp and device.type == "cuda"),
            ):
                logits = model(images)
                loss = soft_cross_entropy(logits, soft_targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            running_count += 1

            step += 1

            if step % args.log_every == 0:
                mean_loss = running_loss / max(running_count, 1)

                msg = (
                    f"step={step:6d}/{args.max_updates} "
                    f"lr={lr:.6f} "
                    f"loss={mean_loss:.4f}"
                )

                if args.group != "g2" and corr_count > 0:
                    msg += (
                        f" corr_mix={corr_mix_sum/corr_count:+.3f}"
                        f" corr_cut={corr_cut_sum/corr_count:+.3f}"
                    )

                print(msg)

                log_record = {
                    "type": "train",
                    "step": step,
                    "lr": lr,
                    "mean_train_loss": mean_loss,
                }

                if args.group != "g2" and corr_count > 0:
                    log_record.update({
                        "mean_corr_mix": corr_mix_sum / corr_count,
                        "mean_corr_cut": corr_cut_sum / corr_count,
                    })

                log_file.write(json.dumps(log_record) + "\n")

                running_loss = 0.0
                running_count = 0

            if step % args.eval_every == 0 or step == args.max_updates:
                val = evaluate(model, val_loader, device)

                print(
                    f"[VAL] step={step:6d} "
                    f"loss={val['loss']:.4f} "
                    f"acc1={val['acc1']:.2f}%"
                )

                log_file.write(
                    json.dumps({
                        "type": "val",
                        "step": step,
                        **val,
                    }) + "\n"
                )

                if val["acc1"] > best_val:
                    best_val = val["acc1"]
                    best_step = step

                    torch.save({
                        "model": model.state_dict(),
                        "step": step,
                        "best_val_acc1": best_val,
                        "args": vars(args),
                    }, best_path)

    elapsed = time.time() - start_time

    summary = {
        "group": args.group,
        "seed": args.seed,
        "max_updates": args.max_updates,
        "best_val_acc1": best_val,
        "best_val_step": best_step,
        "elapsed_seconds": elapsed,
        "mean_corr_mix": (
            corr_mix_sum / corr_count
            if corr_count > 0 else None
        ),
        "mean_corr_cut": (
            corr_cut_sum / corr_count
            if corr_count > 0 else None
        ),
        "m_mix_hist": dict(sorted(m_mix_hist.items())),
        "m_cut_hist": dict(sorted(m_cut_hist.items())),
    }

    if args.test_at_end:
        checkpoint = torch.load(
            best_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])

        test = evaluate(model, test_loader, device)
        summary["test_acc1_from_best_val"] = test["acc1"]
        summary["test_loss_from_best_val"] = test["loss"]

        print(
            f"[TEST] best-val checkpoint "
            f"step={best_step} acc1={test['acc1']:.2f}%"
        )

    with (run_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log_file.close()

    print("\n" + "=" * 72)
    print(
        f"DONE {args.group}: "
        f"best val={best_val:.2f}% @ step {best_step}, "
        f"time={elapsed/60:.1f} min"
    )

    if summary["mean_corr_mix"] is not None:
        print(
            f"mean corr(s,M): "
            f"mix={summary['mean_corr_mix']:+.3f}, "
            f"cut={summary['mean_corr_cut']:+.3f}"
        )
    print(f"saved: {run_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
