from __future__ import annotations

# ============================================================
# Standard library imports
# 标准库导入
# ============================================================
import argparse
import json
import math
import random
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

# ============================================================
# Third-party imports
# 第三方库导入
# ============================================================
import torch
from torch.utils.data import DataLoader

# ============================================================
# Make project root importable
# 让项目根目录可导入
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Local imports
# 本地导入
# ============================================================
from src.data.stage1_fewshot_dataset import Stage1FewshotDataset, Stage1FewshotCollator
from src.adapters.stage1_fewshot_adapter import Stage1FewshotAdapter


# ============================================================
# Small utilities
# 小工具函数
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json_safe(path: str | Path) -> Any | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def now_sec() -> float:
    return time.time()


def is_finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x.detach()).all().item())


def tensor_num_tokens(batch: dict[str, Any]) -> int:
    # forward_old internally appends EOS target, so semantic_len + batch_size is a better CE-token estimate.
    return int(batch["semantic_ids_len"].sum().item()) + int(batch["semantic_ids"].shape[0])


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def make_autocast_context(device: torch.device, enabled: bool):
    if enabled and str(device).startswith("cuda"):
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def make_grad_scaler(device: torch.device, enabled: bool):
    return torch.cuda.amp.GradScaler(enabled=bool(enabled and str(device).startswith("cuda")))


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    total_steps: int,
    warmup_steps: int,
):
    scheduler_name = str(scheduler_name).lower()
    if scheduler_name == "none" or scheduler_name == "constant":
        return None
    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported lr_scheduler={scheduler_name!r}. Use constant/none/cosine.")

    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ============================================================
# Forward / epoch helpers
# forward 与 epoch 辅助函数
# ============================================================
def forward_stage1_loss(
    adapter: Stage1FewshotAdapter,
    batch: dict[str, Any],
    use_amp: bool,
    train_mode: bool,
) -> tuple[torch.Tensor, float, int]:
    batch = adapter.move_batch_to_device(batch)
    if train_mode:
        adapter.model.train()
    else:
        adapter.model.eval()

    with make_autocast_context(adapter.device, enabled=use_amp):
        loss, acc = adapter.model.forward_old(
            batch["phoneme_ids"],
            batch["phoneme_ids_len"],
            batch["semantic_ids"],
            batch["semantic_ids_len"],
            batch["bert_feature"],
        )
    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"forward_old loss must be Tensor, got {type(loss)}")
    return loss, float(acc), tensor_num_tokens(batch)


def summarize_epoch(
    sums: dict[str, float],
    prefix: str,
) -> dict[str, float]:
    steps = max(int(sums.get("steps", 0)), 1)
    tokens = max(float(sums.get("tokens", 0.0)), 1.0)
    return {
        f"{prefix}_loss": float(sums.get("loss_sum", 0.0) / steps),
        f"{prefix}_loss_per_token": float(sums.get("loss_sum", 0.0) / tokens),
        f"{prefix}_top3_acc": float(sums.get("acc_sum", 0.0) / steps),
        f"{prefix}_tokens": float(sums.get("tokens", 0.0)),
        f"{prefix}_steps": float(sums.get("steps", 0.0)),
    }


def run_train_epoch(
    adapter: Stage1FewshotAdapter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    scheduler,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
) -> tuple[dict[str, float], int]:
    sums = {"loss_sum": 0.0, "acc_sum": 0.0, "tokens": 0.0, "steps": 0.0}
    optimizer.zero_grad(set_to_none=True)

    grad_accum_steps = max(int(args.grad_accum_steps), 1)
    max_train_steps = None if args.max_train_steps_per_epoch is None else int(args.max_train_steps_per_epoch)

    for step_idx, batch in enumerate(loader, start=1):
        if max_train_steps is not None and step_idx > max_train_steps:
            break

        loss, acc, tokens = forward_stage1_loss(
            adapter=adapter,
            batch=batch,
            use_amp=bool(args.use_amp),
            train_mode=True,
        )
        if not is_finite_tensor(loss):
            raise FloatingPointError(f"Non-finite train loss at epoch={epoch}, step={step_idx}: {loss}")

        raw_loss_value = float(loss.detach().float().cpu().item())
        loss_for_backward = loss / float(grad_accum_steps)

        scaler.scale(loss_for_backward).backward()

        should_step = (step_idx % grad_accum_steps == 0) or (step_idx == len(loader))
        if should_step:
            if args.grad_clip is not None and float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(adapter.iter_trainable_parameters()), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            global_step += 1

        sums["loss_sum"] += raw_loss_value
        sums["acc_sum"] += float(acc)
        sums["tokens"] += float(tokens)
        sums["steps"] += 1.0

        if args.log_every_steps > 0 and step_idx % int(args.log_every_steps) == 0:
            print(
                f"[train][epoch={epoch} step={step_idx}] "
                f"loss={raw_loss_value:.4f} "
                f"loss/token={raw_loss_value / max(tokens, 1):.6f} "
                f"top3={float(acc):.4f} "
                f"lr={get_lr(optimizer):.8g}"
            )

    return summarize_epoch(sums, prefix="train"), global_step


@torch.no_grad()
def run_val_epoch(
    adapter: Stage1FewshotAdapter,
    loader: DataLoader,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    sums = {"loss_sum": 0.0, "acc_sum": 0.0, "tokens": 0.0, "steps": 0.0}
    max_val_steps = None if args.max_val_steps_per_epoch is None else int(args.max_val_steps_per_epoch)

    for step_idx, batch in enumerate(loader, start=1):
        if max_val_steps is not None and step_idx > max_val_steps:
            break
        loss, acc, tokens = forward_stage1_loss(
            adapter=adapter,
            batch=batch,
            use_amp=bool(args.use_amp),
            train_mode=False,
        )
        if not is_finite_tensor(loss):
            raise FloatingPointError(f"Non-finite val loss at epoch={epoch}, step={step_idx}: {loss}")
        raw_loss_value = float(loss.detach().float().cpu().item())
        sums["loss_sum"] += raw_loss_value
        sums["acc_sum"] += float(acc)
        sums["tokens"] += float(tokens)
        sums["steps"] += 1.0

    return summarize_epoch(sums, prefix="val")


# ============================================================
# Main
# 主流程
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Stage1 v6.7.0 Original-style Few-shot."
    )
    parser.add_argument("--base_stage1_ckpt", type=str, default=None, help="Optional base stage1 ckpt override.")
    parser.add_argument("--train_manifest", type=str, required=True, help="Stage1 train manifest JSONL.")
    parser.add_argument("--val_manifest", type=str, default=None, help="Stage1 val manifest JSONL.")
    parser.add_argument("--root_dir", type=str, default=None, help="Dataset root. Defaults to train manifest parent.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu / cuda")
    parser.add_argument("--use_half", action="store_true", help="Convert stage1 model to half on CUDA.")
    parser.add_argument("--use_amp", action="store_true", help="Use CUDA AMP autocast + GradScaler.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--trainable_scope", type=str, default="head_and_last_n")
    parser.add_argument("--last_n_layers", type=int, default=4)
    parser.add_argument("--lr_scheduler", type=str, default="constant", choices=["constant", "none", "cosine"])
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_train_items", type=int, default=None)
    parser.add_argument("--max_val_items", type=int, default=None)
    parser.add_argument("--max_train_steps_per_epoch", type=int, default=None)
    parser.add_argument("--max_val_steps_per_epoch", type=int, default=None)
    parser.add_argument("--log_every_steps", type=int, default=20)
    parser.add_argument("--save_every_epoch", action="store_true", help="Also save epoch_N.ckpt after each epoch.")
    parser.add_argument("--dry_run", action="store_true", help="Run dataset/model/forward check and exit without optimizer step.")
    parser.add_argument("--load_strict", action="store_true", help="Use strict=True for stage1 ckpt load_state_dict.")
    args = parser.parse_args()

    start_time = now_sec()
    set_seed(int(args.seed))

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "stage1_fewshot_train_report.json"
    best_ckpt_path = output_dir / "best_model.ckpt"
    last_ckpt_path = output_dir / "last_model.ckpt"

    train_manifest = Path(args.train_manifest).resolve()
    val_manifest = Path(args.val_manifest).resolve() if args.val_manifest else None
    root_dir = Path(args.root_dir).resolve() if args.root_dir else train_manifest.parent

    report: dict[str, Any] = {
        "task": "Stage1 v6.7.0 Original-style Few-shot training",
        "status": "UNKNOWN",
        "args": vars(args),
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "train_manifest": str(train_manifest),
            "val_manifest": None if val_manifest is None else str(val_manifest),
            "root_dir": str(root_dir),
            "output_dir": str(output_dir),
            "best_ckpt": str(best_ckpt_path),
            "last_ckpt": str(last_ckpt_path),
            "history": str(history_path),
            "summary": str(summary_path),
        },
    }

    try:
        print("============================================================")
        print("Stage1 v6.7.0 few-shot training")
        print("============================================================")
        print("train_manifest :", train_manifest)
        print("val_manifest   :", val_manifest)
        print("root_dir       :", root_dir)
        print("output_dir     :", output_dir)
        print("device         :", args.device)
        print("use_half       :", args.use_half)
        print("use_amp        :", args.use_amp)
        print("scope          :", args.trainable_scope)
        print("last_n_layers  :", args.last_n_layers)
        print("dry_run        :", args.dry_run)
        print("============================================================")

        print("[1/5] Loading datasets...")
        train_ds = Stage1FewshotDataset(
            manifest_path=train_manifest,
            root_dir=root_dir,
            max_items=args.max_train_items,
            validate_on_load=False,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            collate_fn=Stage1FewshotCollator(),
            drop_last=False,
        )

        val_ds = None
        val_loader = None
        if val_manifest is not None and val_manifest.exists():
            val_ds = Stage1FewshotDataset(
                manifest_path=val_manifest,
                root_dir=root_dir,
                max_items=args.max_val_items,
                validate_on_load=False,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(args.num_workers),
                collate_fn=Stage1FewshotCollator(),
                drop_last=False,
            )

        print("train items:", len(train_ds))
        print("val items  :", 0 if val_ds is None else len(val_ds))
        report["dataset"] = {
            "num_train": len(train_ds),
            "num_val": 0 if val_ds is None else len(val_ds),
            "batch_size": int(args.batch_size),
        }

        print("[2/5] Loading Stage1FewshotAdapter...")
        adapter = Stage1FewshotAdapter(
            stage1_ckpt_path=args.base_stage1_ckpt,
            device=args.device,
            use_half=bool(args.use_half),
            trainable_scope=args.trainable_scope,
            last_n_layers=int(args.last_n_layers),
            load_strict=bool(args.load_strict),
        )
        param_report = adapter.get_param_report()
        report["param_report"] = param_report
        print("total_params    :", param_report["total_params"])
        print("trainable_params:", param_report["trainable_params"])
        print("frozen_params   :", param_report["frozen_params"])
        print("trainable_ratio :", param_report["trainable_ratio"])

        if int(param_report["trainable_params"]) <= 0 and not args.dry_run:
            raise RuntimeError(
                f"trainable_scope={args.trainable_scope!r} exposes 0 trainable params. "
                "Use --dry_run for frozen_debug checks, or choose a trainable scope."
            )

        print("[3/5] Running initial forward check...")
        first_batch = next(iter(train_loader))
        check_loss, check_acc, check_tokens = forward_stage1_loss(
            adapter=adapter,
            batch=first_batch,
            use_amp=bool(args.use_amp),
            train_mode=True,
        )
        check_loss_value = float(check_loss.detach().float().cpu().item())
        report["initial_forward"] = {
            "loss_value": check_loss_value,
            "loss_per_token": check_loss_value / max(int(check_tokens), 1),
            "top3_acc": float(check_acc),
            "tokens": int(check_tokens),
            "loss_requires_grad": bool(check_loss.requires_grad),
            "item_ids": first_batch.get("item_ids", []),
        }
        print("initial loss/token:", report["initial_forward"]["loss_per_token"])
        print("initial top3      :", report["initial_forward"]["top3_acc"])

        if args.dry_run:
            report["status"] = "DRY_RUN_OK"
            report["elapsed_sec"] = round(now_sec() - start_time, 4)
            write_json(report_path, report)
            write_json(summary_path, report)
            print("[DRY_RUN] Wrote report:", report_path)
            return

        print("[4/5] Preparing optimizer...")
        trainable_params = list(adapter.iter_trainable_parameters())
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        steps_per_epoch = math.ceil(len(train_loader) / max(int(args.grad_accum_steps), 1))
        if args.max_train_steps_per_epoch is not None:
            steps_per_epoch = min(steps_per_epoch, int(args.max_train_steps_per_epoch))
        total_optim_steps = max(steps_per_epoch * int(args.epochs), 1)
        scheduler = make_scheduler(
            optimizer,
            scheduler_name=args.lr_scheduler,
            total_steps=total_optim_steps,
            warmup_steps=int(args.warmup_steps),
        )
        scaler = make_grad_scaler(adapter.device, enabled=bool(args.use_amp))

        history: list[dict[str, Any]] = []
        best_metric = float("inf")
        best_epoch = -1
        global_step = 0

        print("[5/5] Training loop...")
        for epoch in range(1, int(args.epochs) + 1):
            epoch_start = now_sec()
            train_metrics, global_step = run_train_epoch(
                adapter=adapter,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                args=args,
                epoch=epoch,
                global_step=global_step,
            )

            if val_loader is not None:
                val_metrics = run_val_epoch(adapter=adapter, loader=val_loader, args=args, epoch=epoch)
                current_metric = float(val_metrics["val_loss_per_token"])
            else:
                val_metrics = {}
                current_metric = float(train_metrics["train_loss_per_token"])

            improved = current_metric < best_metric
            if improved:
                best_metric = current_metric
                best_epoch = epoch
                adapter.save_stage1_checkpoint(
                    best_ckpt_path,
                    extra_meta={
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_metric_name": "val_loss_per_token" if val_loader is not None else "train_loss_per_token",
                        "best_metric": best_metric,
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                        "args": vars(args),
                    },
                )

            adapter.save_stage1_checkpoint(
                last_ckpt_path,
                extra_meta={
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
            )
            if args.save_every_epoch:
                adapter.save_stage1_checkpoint(
                    output_dir / f"epoch_{epoch:04d}.ckpt",
                    extra_meta={
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                        "args": vars(args),
                    },
                )

            row: dict[str, Any] = {
                "epoch": epoch,
                "global_step": global_step,
                "lr": get_lr(optimizer),
                "epoch_elapsed_sec": round(now_sec() - epoch_start, 4),
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "improved": bool(improved),
            }
            row.update(train_metrics)
            row.update(val_metrics)
            history.append(row)
            write_json(history_path, history)

            print(
                f"[epoch {epoch}/{args.epochs}] "
                f"train_loss/token={train_metrics['train_loss_per_token']:.6f} "
                f"train_top3={train_metrics['train_top3_acc']:.4f} "
                + (
                    f"val_loss/token={val_metrics.get('val_loss_per_token', float('nan')):.6f} "
                    f"val_top3={val_metrics.get('val_top3_acc', float('nan')):.4f} "
                    if val_metrics else ""
                )
                + f"best_epoch={best_epoch} improved={improved}"
            )

        summary = {
            "status": "OK",
            "version": "v6.7.0",
            "training_mode": "original_style_stage1_fewshot_teacher_forcing",
            "output_dir": str(output_dir),
            "best_model": str(best_ckpt_path),
            "last_model": str(last_ckpt_path),
            "best_epoch": best_epoch,
            "best_metric_name": "val_loss_per_token" if val_loader is not None else "train_loss_per_token",
            "best_metric": best_metric,
            "last_epoch": int(args.epochs),
            "global_step": global_step,
            "param_report": param_report,
            "dataset": report["dataset"],
            "history_tail": history[-3:],
            "elapsed_sec": round(now_sec() - start_time, 4),
        }
        report["status"] = "OK"
        report["summary"] = summary
        report["elapsed_sec"] = round(now_sec() - start_time, 4)

        write_json(summary_path, summary)
        write_json(report_path, report)

        print("============================================================")
        print("Stage1 few-shot training summary")
        print("============================================================")
        print("status     : OK")
        print("best_epoch :", best_epoch)
        print("best_metric:", best_metric)
        print("best_model :", best_ckpt_path)
        print("last_model :", last_ckpt_path)
        print("history    :", history_path)
        print("summary    :", summary_path)
        print("report     :", report_path)
        print("============================================================")

    except Exception as exc:
        report["status"] = "FAILED"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        report["elapsed_sec"] = round(now_sec() - start_time, 4)
        write_json(report_path, report)
        print("[FAILED]", exc)
        print("report_path:", report_path)
        raise


if __name__ == "__main__":
    main()
