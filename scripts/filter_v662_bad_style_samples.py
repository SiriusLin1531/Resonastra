from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import torch


# ============================================================
# CLI
# ============================================================
def str2bool(x: str | bool) -> bool:
    if isinstance(x, bool):
        return x

    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"Cannot parse bool from: {x}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and filter Stage2 v6.6.2 style-cache samples. "
            "检查并隔离缺失 F0 / speaker embedding 的异常样本。"
        )
    )

    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--rejected_dir", type=str, required=True)
    parser.add_argument("--recursive", type=str2bool, default=False)
    parser.add_argument("--mode", type=str, default="move", choices=["move", "copy", "delete", "report_only"])

    parser.add_argument("--require_target_f0", type=str2bool, default=True)
    parser.add_argument("--require_target_voiced_mask", type=str2bool, default=True)
    parser.add_argument("--reject_all_unvoiced_target_f0", type=str2bool, default=True)
    parser.add_argument("--require_prompt_f0", type=str2bool, default=False)
    parser.add_argument("--check_speaker_fields", type=str2bool, default=True)
    parser.add_argument("--speaker_dim", type=int, default=192)

    parser.add_argument("--report_json", type=str, default=None)
    parser.add_argument("--report_csv", type=str, default=None)

    return parser.parse_args()


# ============================================================
# File helpers
# ============================================================
def list_pt_files(root: Path, recursive: bool) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")

    pattern = "**/*.pt" if recursive else "*.pt"
    return sorted(p.resolve() for p in root.glob(pattern) if p.is_file())


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["status", "source_path", "rejected_path", "root", "relative_path", "sample_id", "issues"],
        )
        writer.writeheader()

        for r in rows:
            writer.writerow(
                {
                    "status": r.get("status", ""),
                    "source_path": r.get("source_path", ""),
                    "rejected_path": r.get("rejected_path", ""),
                    "root": r.get("root", ""),
                    "relative_path": r.get("relative_path", ""),
                    "sample_id": r.get("sample_id", ""),
                    "issues": " | ".join(r.get("issues", [])),
                }
            )


# ============================================================
# Validation helpers
# ============================================================
def add_issue(issues: list[str], message: str) -> None:
    issues.append(message)


def is_finite_tensor(x: torch.Tensor) -> bool:
    if not x.dtype.is_floating_point:
        return True
    return bool(torch.isfinite(x.detach().float()).all().item())


def check_target_acoustic(sample: dict[str, Any], issues: list[str]) -> int | None:
    x = sample.get("target_acoustic")

    if not torch.is_tensor(x):
        add_issue(issues, "missing_or_invalid: target_acoustic")
        return None

    if x.ndim != 2:
        add_issue(issues, f"invalid_shape: target_acoustic shape={tuple(x.shape)}, expected [T,80]")
        return None

    if x.shape[0] <= 0:
        add_issue(issues, f"invalid_length: target_acoustic T={x.shape[0]}")
        return None

    if not is_finite_tensor(x):
        add_issue(issues, "non_finite: target_acoustic")

    return int(x.shape[0])


def check_1d_float(
    sample: dict[str, Any],
    key: str,
    *,
    required: bool,
    expected_len: int | None,
    reject_all_unvoiced_f0: bool,
    issues: list[str],
) -> None:
    value = sample.get(key)

    if value is None:
        if required:
            add_issue(issues, f"missing: {key}")
        return

    if not torch.is_tensor(value):
        add_issue(issues, f"not_tensor: {key}, type={type(value)}")
        return

    x = value.detach().cpu()

    if x.ndim != 1:
        add_issue(issues, f"non_1d: {key}, shape={tuple(x.shape)}")
        x = x.reshape(-1)

    if x.numel() <= 0:
        add_issue(issues, f"empty: {key}")
        return

    if not x.dtype.is_floating_point:
        add_issue(issues, f"not_float: {key}, dtype={x.dtype}")
        return

    x = x.float()

    if not torch.isfinite(x).all():
        add_issue(issues, f"non_finite: {key}")

    if expected_len is not None and int(x.numel()) != int(expected_len):
        add_issue(issues, f"length_mismatch: {key} len={int(x.numel())}, target_acoustic_len={int(expected_len)}")

    if key == "target_f0" and reject_all_unvoiced_f0:
        positive_ratio = float((x > 1.0).float().mean().item())
        if positive_ratio <= 0.0:
            add_issue(issues, f"all_unvoiced_or_zero: {key}, positive_ratio={positive_ratio:.6f}")


def check_1d_bool(
    sample: dict[str, Any],
    key: str,
    *,
    required: bool,
    expected_len: int | None,
    reject_all_false: bool,
    issues: list[str],
) -> None:
    value = sample.get(key)

    if value is None:
        if required:
            add_issue(issues, f"missing: {key}")
        return

    if not torch.is_tensor(value):
        add_issue(issues, f"not_tensor: {key}, type={type(value)}")
        return

    x = value.detach().cpu()

    if x.ndim != 1:
        add_issue(issues, f"non_1d: {key}, shape={tuple(x.shape)}")
        x = x.reshape(-1)

    if x.numel() <= 0:
        add_issue(issues, f"empty: {key}")
        return

    if expected_len is not None and int(x.numel()) != int(expected_len):
        add_issue(issues, f"length_mismatch: {key} len={int(x.numel())}, target_acoustic_len={int(expected_len)}")

    true_ratio = float(x.bool().float().mean().item())

    if reject_all_false and true_ratio <= 0.0:
        add_issue(issues, f"voiced_mask_all_false: {key}, true_ratio={true_ratio:.6f}")


def check_speaker_embedding(
    sample: dict[str, Any],
    key: str,
    *,
    required: bool,
    expected_dim: int,
    issues: list[str],
) -> None:
    value = sample.get(key)

    if value is None:
        if required:
            add_issue(issues, f"missing: {key}")
        return

    if not torch.is_tensor(value):
        add_issue(issues, f"not_tensor: {key}, type={type(value)}")
        return

    x = value.detach().cpu().float().reshape(-1)

    if x.numel() != expected_dim:
        add_issue(issues, f"speaker_dim_mismatch: {key} dim={x.numel()}, expected={expected_dim}")

    if x.numel() <= 0:
        add_issue(issues, f"empty: {key}")
        return

    if not torch.isfinite(x).all():
        add_issue(issues, f"non_finite: {key}")

    norm = float(x.norm().item())
    if norm < 0.5 or norm > 1.5:
        add_issue(issues, f"speaker_norm_suspicious: {key} norm={norm:.6f}")


def validate_sample(
    path: Path,
    *,
    require_target_f0: bool,
    require_target_voiced_mask: bool,
    reject_all_unvoiced_target_f0: bool,
    require_prompt_f0: bool,
    check_speaker_fields: bool,
    speaker_dim: int,
) -> dict[str, Any]:
    issues: list[str] = []

    try:
        sample = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as e:
        return {"sample_id": "", "issues": [f"load_error: {type(e).__name__}: {e}"]}

    if not isinstance(sample, dict):
        return {"sample_id": "", "issues": [f"not_dict: {type(sample)}"]}

    sample_id = str(sample.get("sample_id", ""))
    target_len = check_target_acoustic(sample, issues)

    check_1d_float(
        sample,
        "target_f0",
        required=require_target_f0,
        expected_len=target_len,
        reject_all_unvoiced_f0=reject_all_unvoiced_target_f0,
        issues=issues,
    )

    check_1d_bool(
        sample,
        "target_f0_voiced_mask",
        required=require_target_voiced_mask,
        expected_len=target_len,
        reject_all_false=reject_all_unvoiced_target_f0,
        issues=issues,
    )

    if require_prompt_f0:
        prompt_len = None
        prompt_acoustic = sample.get("prompt_acoustic")
        if torch.is_tensor(prompt_acoustic) and prompt_acoustic.ndim == 2:
            prompt_len = int(prompt_acoustic.shape[0])

        check_1d_float(
            sample,
            "prompt_f0",
            required=True,
            expected_len=prompt_len,
            reject_all_unvoiced_f0=False,
            issues=issues,
        )

        check_1d_bool(
            sample,
            "prompt_f0_voiced_mask",
            required=True,
            expected_len=prompt_len,
            reject_all_false=False,
            issues=issues,
        )

    if check_speaker_fields:
        check_speaker_embedding(
            sample,
            "target_speaker_embedding",
            required=True,
            expected_dim=speaker_dim,
            issues=issues,
        )
        check_speaker_embedding(
            sample,
            "prompt_speaker_embedding",
            required=True,
            expected_dim=speaker_dim,
            issues=issues,
        )

    return {"sample_id": sample_id, "issues": issues}


# ============================================================
# Filter
# ============================================================
def build_rejected_path(*, rejected_dir: Path, root: Path, source_path: Path) -> Path:
    rel = source_path.relative_to(root)
    return rejected_dir / root.name / rel


def handle_bad_sample(*, source_path: Path, rejected_path: Path, mode: str) -> None:
    rejected_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "report_only":
        return
    if mode == "copy":
        shutil.copy2(str(source_path), str(rejected_path))
        return
    if mode == "move":
        if rejected_path.exists():
            rejected_path.unlink()
        shutil.move(str(source_path), str(rejected_path))
        return
    if mode == "delete":
        source_path.unlink()
        return

    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    args = parse_args()

    roots = [Path(x).expanduser().resolve() for x in args.roots]
    rejected_dir = Path(args.rejected_dir).expanduser().resolve()
    rejected_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total = 0
    bad = 0

    for root in roots:
        files = list_pt_files(root, recursive=bool(args.recursive))
        print(f"[SCAN] root={root} num_pt={len(files)}")

        for idx, path in enumerate(files, start=1):
            total += 1

            result = validate_sample(
                path,
                require_target_f0=bool(args.require_target_f0),
                require_target_voiced_mask=bool(args.require_target_voiced_mask),
                reject_all_unvoiced_target_f0=bool(args.reject_all_unvoiced_target_f0),
                require_prompt_f0=bool(args.require_prompt_f0),
                check_speaker_fields=bool(args.check_speaker_fields),
                speaker_dim=int(args.speaker_dim),
            )

            issues = result["issues"]
            rejected_path = ""

            if issues:
                bad += 1
                dst = build_rejected_path(rejected_dir=rejected_dir, root=root, source_path=path)
                rejected_path = str(dst)

                print(f"[BAD] {idx}/{len(files)} {path}")
                for issue in issues:
                    print(f"      - {issue}")
                print(f"      -> {args.mode}: {dst}")

                handle_bad_sample(source_path=path, rejected_path=dst, mode=str(args.mode))

            rows.append(
                {
                    "status": "BAD" if issues else "OK",
                    "source_path": str(path),
                    "rejected_path": rejected_path,
                    "root": str(root),
                    "relative_path": str(path.relative_to(root)),
                    "sample_id": result.get("sample_id", ""),
                    "issues": issues,
                }
            )

    bad_rows = [r for r in rows if r["status"] == "BAD"]
    ok_rows = [r for r in rows if r["status"] == "OK"]

    report = {
        "status": "OK" if not bad_rows else "BAD",
        "mode": str(args.mode),
        "num_total": int(total),
        "num_ok": int(len(ok_rows)),
        "num_bad": int(len(bad_rows)),
        "roots": [str(x) for x in roots],
        "rejected_dir": str(rejected_dir),
        "config": {
            "recursive": bool(args.recursive),
            "require_target_f0": bool(args.require_target_f0),
            "require_target_voiced_mask": bool(args.require_target_voiced_mask),
            "reject_all_unvoiced_target_f0": bool(args.reject_all_unvoiced_target_f0),
            "require_prompt_f0": bool(args.require_prompt_f0),
            "check_speaker_fields": bool(args.check_speaker_fields),
            "speaker_dim": int(args.speaker_dim),
        },
        "bad_samples": bad_rows,
    }

    report_json = args.report_json or str(rejected_dir / "filter_report.json")
    report_csv = args.report_csv or str(rejected_dir / "bad_samples.csv")

    save_json(report, report_json)
    save_csv(bad_rows, report_csv)

    print("====================================================")
    print("Stage2 v6.6.2 style sample filter finished")
    print("====================================================")
    print(f"status       : {report['status']}")
    print(f"mode         : {args.mode}")
    print(f"num_total    : {total}")
    print(f"num_ok       : {len(ok_rows)}")
    print(f"num_bad      : {len(bad_rows)}")
    print(f"rejected_dir : {rejected_dir}")
    print(f"report_json  : {report_json}")
    print(f"report_csv   : {report_csv}")
    print("====================================================")


if __name__ == "__main__":
    main()
