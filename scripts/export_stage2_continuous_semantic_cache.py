from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.gsv_semantic_codec import GSVSemanticCodecV2  # noqa: E402


def safe_torch_load(path: str | Path) -> Any:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_1d_long(x: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be torch.Tensor, got {type(x)}")

    y = x.detach().cpu().long()

    if y.ndim == 2 and y.shape[0] == 1:
        y = y.squeeze(0)

    if y.ndim != 1:
        y = y.reshape(-1)

    return y.contiguous().long()


def normalize_optional_length(x: Any, fallback: int) -> int:
    if x is None:
        return int(fallback)

    try:
        if isinstance(x, int):
            value = int(x)
        elif torch.is_tensor(x):
            value = int(x.detach().cpu().view(-1)[0].item())
        else:
            value = int(x)
    except Exception:
        return int(fallback)

    if value <= 0:
        return int(fallback)

    return int(min(value, fallback))


def tensor_stats(x: torch.Tensor) -> dict[str, Any]:
    x = x.detach().cpu()
    xf = x.float()

    if x.numel() <= 0:
        return {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "numel": int(x.numel()),
        }

    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "numel": int(x.numel()),
        "mean": float(xf.mean().item()),
        "std": float(xf.std().item()) if x.numel() > 1 else 0.0,
        "min": float(xf.min().item()),
        "max": float(xf.max().item()),
        "norm_mean": (
            float(torch.linalg.norm(xf, dim=-1).mean().item())
            if x.ndim >= 2
            else None
        ),
    }


def decode_one_sequence(
    codec: GSVSemanticCodecV2,
    tokens: torch.Tensor,
    *,
    sovits_checkpoint_path: str | None,
    continuous_dtype: str,
) -> torch.Tensor:
    if continuous_dtype == "float16":
        dtype = torch.float16
    elif continuous_dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported continuous_dtype: {continuous_dtype}")

    out = codec.decode_codes_to_continuous(
        tokens,
        sovits_checkpoint_path=sovits_checkpoint_path,
        return_cpu=True,
        dtype=dtype,
    )

    # out.continuous: [1, T, 768]
    continuous = out.continuous[0].contiguous()

    if continuous.shape[0] != tokens.numel():
        raise RuntimeError(
            f"Decoded continuous length mismatch: "
            f"tokens={tokens.numel()}, continuous={tuple(continuous.shape)}"
        )

    if continuous.shape[-1] != 768:
        raise RuntimeError(
            f"Expected continuous dim 768, got shape={tuple(continuous.shape)}"
        )

    return continuous


def process_one_file(
    *,
    src_path: Path,
    dst_path: Path,
    codec: GSVSemanticCodecV2,
    sovits_checkpoint_path: str | None,
    continuous_dtype: str,
    require_predicted: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if dst_path.exists() and not overwrite:
        return {
            "src_path": str(src_path),
            "dst_path": str(dst_path),
            "status": "skipped_exists",
        }

    sample = safe_torch_load(src_path)
    if not isinstance(sample, dict):
        raise TypeError(f"Expected sample dict from {src_path}, got {type(sample)}")

    if "semantic_tokens" not in sample:
        raise KeyError(f"{src_path} missing semantic_tokens")

    oracle_tokens = ensure_1d_long(sample["semantic_tokens"], "semantic_tokens")
    oracle_len = normalize_optional_length(
        sample.get("semantic_lengths", None),
        fallback=int(oracle_tokens.numel()),
    )
    oracle_tokens = oracle_tokens[:oracle_len]

    oracle_cont = decode_one_sequence(
        codec,
        oracle_tokens,
        sovits_checkpoint_path=sovits_checkpoint_path,
        continuous_dtype=continuous_dtype,
    )

    sample["oracle_semantic_continuous"] = oracle_cont
    sample["oracle_semantic_continuous_lengths"] = torch.LongTensor([oracle_cont.shape[0]])

    pred_cont = None
    pred_len = 0

    if "stage1_pred_semantic_tokens" in sample:
        pred_tokens = ensure_1d_long(
            sample["stage1_pred_semantic_tokens"],
            "stage1_pred_semantic_tokens",
        )
        pred_len = normalize_optional_length(
            sample.get("stage1_pred_semantic_lengths", None),
            fallback=int(pred_tokens.numel()),
        )
        pred_tokens = pred_tokens[:pred_len]

        pred_cont = decode_one_sequence(
            codec,
            pred_tokens,
            sovits_checkpoint_path=sovits_checkpoint_path,
            continuous_dtype=continuous_dtype,
        )

        sample["stage1_pred_semantic_continuous"] = pred_cont
        sample["stage1_pred_semantic_continuous_lengths"] = torch.LongTensor(
            [pred_cont.shape[0]]
        )

    elif require_predicted:
        raise KeyError(
            f"{src_path} missing stage1_pred_semantic_tokens, "
            "but --require_predicted was set."
        )

    # Metadata
    meta = dict(sample.get("continuous_semantic_meta", {}))
    meta.update(
        {
            "codec": "GSVSemanticCodecV2",
            "continuous_dtype": continuous_dtype,
            "sovits_checkpoint_path": sovits_checkpoint_path,
            "oracle_semantic_continuous_shape": list(oracle_cont.shape),
            "stage1_pred_semantic_continuous_shape": (
                list(pred_cont.shape) if pred_cont is not None else None
            ),
        }
    )
    sample["continuous_semantic_meta"] = meta

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sample, dst_path)

    result = {
        "src_path": str(src_path),
        "dst_path": str(dst_path),
        "status": "ok",
        "oracle_len": int(oracle_len),
        "oracle_continuous_shape": list(oracle_cont.shape),
        "oracle_stats": tensor_stats(oracle_cont),
        "pred_len": int(pred_len),
        "pred_continuous_shape": list(pred_cont.shape) if pred_cont is not None else None,
        "pred_stats": tensor_stats(pred_cont) if pred_cont is not None else None,
    }

    if pred_len > 0 and oracle_len > 0:
        result["pred_over_oracle_len_ratio"] = float(pred_len) / float(oracle_len)
    else:
        result["pred_over_oracle_len_ratio"] = None

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GPT-SoVITS quantizer.decode continuous semantic cache for Stage2 .pt samples."
    )

    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--sovits_checkpoint_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_half", action="store_true")

    parser.add_argument(
        "--continuous_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
    )

    parser.add_argument("--require_predicted", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_files", type=int, default=None)

    parser.add_argument("--report_json", type=str, default=None)
    parser.add_argument("--copy_non_pt_files", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"input_root not found: {input_root}")

    pt_files = sorted(input_root.glob("*.pt"))
    if args.max_files is not None:
        pt_files = pt_files[: int(args.max_files)]

    if not pt_files:
        raise FileNotFoundError(f"No .pt files found under: {input_root}")

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = requested_device

    codec = GSVSemanticCodecV2(
        version="v2",
        device=device,
        use_half=bool(args.use_half),
    )
    codec.ensure_loaded(sovits_checkpoint_path=args.sovits_checkpoint_path)

    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    print("====================================================")
    print("[v6.4.1] Export continuous semantic cache")
    print(f"input_root       : {input_root}")
    print(f"output_root      : {output_root}")
    print(f"num_pt_files     : {len(pt_files)}")
    print(f"device           : {device}")
    print(f"continuous_dtype : {args.continuous_dtype}")
    print("====================================================")

    for idx, src_path in enumerate(pt_files, start=1):
        rel = src_path.relative_to(input_root)
        dst_path = output_root / rel

        try:
            result = process_one_file(
                src_path=src_path,
                dst_path=dst_path,
                codec=codec,
                sovits_checkpoint_path=args.sovits_checkpoint_path,
                continuous_dtype=args.continuous_dtype,
                require_predicted=bool(args.require_predicted),
                overwrite=bool(args.overwrite),
            )
        except Exception as e:
            result = {
                "src_path": str(src_path),
                "dst_path": str(dst_path),
                "status": "error",
                "error": repr(e),
            }

        results.append(result)

        if idx == 1 or idx % 50 == 0 or result["status"] != "ok":
            print(
                f"[{idx}/{len(pt_files)}] "
                f"{src_path.name} status={result.get('status')} "
                f"oracle_shape={result.get('oracle_continuous_shape')} "
                f"pred_shape={result.get('pred_continuous_shape')} "
                f"err={result.get('error')}"
            )

    if args.copy_non_pt_files:
        for src_path in input_root.iterdir():
            if src_path.is_file() and src_path.suffix.lower() != ".pt":
                dst_path = output_root / src_path.name
                if not dst_path.exists() or args.overwrite:
                    shutil.copy2(src_path, dst_path)

    ok_count = sum(1 for r in results if r.get("status") == "ok")
    skipped_count = sum(1 for r in results if r.get("status") == "skipped_exists")
    error_count = sum(1 for r in results if r.get("status") == "error")

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_files": len(pt_files),
        "ok_count": int(ok_count),
        "skipped_count": int(skipped_count),
        "error_count": int(error_count),
        "continuous_dtype": args.continuous_dtype,
        "device": str(device),
        "sovits_checkpoint_path": args.sovits_checkpoint_path,
        "results": results,
    }

    report_json = (
        Path(args.report_json).resolve()
        if args.report_json is not None
        else output_root / "continuous_semantic_export_report.json"
    )
    save_json(summary, report_json)

    print("====================================================")
    print("[DONE] export continuous semantic cache")
    print(f"ok_count      : {ok_count}")
    print(f"skipped_count : {skipped_count}")
    print(f"error_count   : {error_count}")
    print(f"report_json   : {report_json}")
    print("====================================================")

    if error_count > 0:
        raise RuntimeError(f"Export finished with {error_count} errors. See {report_json}")


if __name__ == "__main__":
    main()