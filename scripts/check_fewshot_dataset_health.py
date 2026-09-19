from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data_factory.io.audio_io import get_audio_duration_sec
except Exception:
    get_audio_duration_sec = None

STAGE2_PT_REQUIRED_KEYS = [
    "semantic_tokens",
    "bert_feature",
    "prompt_tokens",
    "target_acoustic",
    "phoneme_ids",
    "phoneme_lens",
    "raw_text",
    "norm_text",
    "language",
    "sample_id",
    "prompt_wav_path",
    "target_wav_path",
]

V66_STYLE_KEYS = [
    "prompt_acoustic",
    "prompt_acoustic_lengths",
    "target_energy",
    "prompt_energy",
]

CONTINUOUS_KEYS = [
    "oracle_semantic_continuous",
    "oracle_semantic_continuous_lengths",
]


def str2bool(x: str | bool) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {x}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Few-shot dataset health checker with v6.6 style-cache field support."
    )

    parser.add_argument("--manifest_jsonl", type=str, default=None)
    parser.add_argument("--stage2_manifest_jsonl", type=str, default=None)
    parser.add_argument("--stage2_pt_root", type=str, default=None)
    parser.add_argument("--report_path", type=str, required=True)

    parser.add_argument("--print_samples", type=str2bool, default=True)
    parser.add_argument("--max_print_items", type=int, default=20)
    parser.add_argument("--exit_nonzero_on_fail", type=str2bool, default=False)

    parser.add_argument("--min_samples_pass", type=int, default=20)
    parser.add_argument("--min_samples_warn", type=int, default=10)
    parser.add_argument("--min_total_duration_sec_pass", type=float, default=60.0)
    parser.add_argument("--min_total_duration_sec_warn", type=float, default=30.0)
    parser.add_argument("--min_clip_sec_fail", type=float, default=0.8)
    parser.add_argument("--min_clip_sec_warn", type=float, default=2.0)
    parser.add_argument("--max_clip_sec_warn", type=float, default=12.0)
    parser.add_argument("--max_clip_sec_fail", type=float, default=30.0)
    parser.add_argument("--min_prompt_sec", type=float, default=3.0)
    parser.add_argument("--max_prompt_sec", type=float, default=10.0)
    parser.add_argument("--min_text_chars_warn", type=int, default=2)
    parser.add_argument("--max_text_chars_warn", type=int, default=120)
    parser.add_argument("--allow_multi_speaker", type=str2bool, default=False)
    parser.add_argument("--allow_multi_language", type=str2bool, default=False)
    parser.add_argument("--check_audio_duration", type=str2bool, default=True)

    parser.add_argument("--require_continuous_semantic", type=str2bool, default=False)
    parser.add_argument("--require_v66_style_fields", type=str2bool, default=False)
    parser.add_argument("--require_reference_acoustic_style", type=str2bool, default=False)
    parser.add_argument("--require_energy", type=str2bool, default=False)
    parser.add_argument("--require_f0", type=str2bool, default=False)
    parser.add_argument("--require_speaker_embedding", type=str2bool, default=False)
    parser.add_argument("--expected_acoustic_dim", type=int, default=80)
    parser.add_argument("--expected_continuous_semantic_dim", type=int, default=768)

    return parser.parse_args()


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path).resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {p}:{line_no}, got {type(obj)}")
            rows.append(obj)
    return rows


def as_text(x: Any) -> str:
    return "" if x is None else str(x).strip()


def as_float(x: Any, default: float = math.nan) -> float:
    try:
        if x is None:
            return default
        if torch.is_tensor(x):
            if x.numel() == 0:
                return default
            return float(x.detach().float().view(-1)[0].item())
        return float(x)
    except Exception:
        return default


def as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if torch.is_tensor(x):
            if x.numel() == 0:
                return default
            return int(x.detach().long().view(-1)[0].item())
        return int(x)
    except Exception:
        return default


def finite_num(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def file_exists(path: str) -> bool:
    return bool(path) and Path(path).exists() and Path(path).is_file()


def norm_path(x: Any, base_dir: str | Path | None = None) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    p = Path(s)
    if not p.is_absolute() and base_dir is not None:
        p = Path(base_dir).resolve() / p
    return str(p.resolve())


def duration_sec(path: str) -> float:
    if not path or get_audio_duration_sec is None:
        return math.nan
    p = Path(path)
    if not p.exists() or not p.is_file():
        return math.nan
    try:
        return float(get_audio_duration_sec(p))
    except Exception:
        return math.nan


def add_issue(issues: list[dict[str, Any]], level: str, sample_id: Any, issue_type: str, message: str) -> None:
    issues.append(
        {
            "level": level,
            "sample_id": None if sample_id is None else str(sample_id),
            "issue_type": issue_type,
            "message": message,
        }
    )


def worst_status(issues: list[dict[str, Any]]) -> str:
    if any(x["level"] == "FAIL" for x in issues):
        return "FAIL"
    if any(x["level"] == "WARN" for x in issues):
        return "WARN"
    return "PASS"


def stats(xs: list[float | int]) -> dict[str, Any]:
    vals = [float(x) for x in xs if finite_num(float(x))]
    if not vals:
        return {"count": 0}
    return {
        "count": len(vals),
        "mean": float(mean(vals)),
        "median": float(median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def tensor_finite(x: torch.Tensor) -> bool:
    try:
        return bool(torch.isfinite(x.detach().float()).all().item())
    except Exception:
        return False


def tensor_2d_info(x: Any, expected_dim: int | None = None) -> tuple[bool, int, int, str]:
    if not torch.is_tensor(x):
        return False, 0, 0, "not_tensor"
    y = x.detach().cpu()
    if y.ndim == 3 and y.shape[0] == 1:
        y = y.squeeze(0)
    if y.ndim != 2:
        return False, 0, 0, f"bad_ndim_{y.ndim}"
    if y.shape[0] <= 0 or y.shape[1] <= 0:
        return False, int(y.shape[0]), int(y.shape[1]), "empty"
    if expected_dim is not None and int(y.shape[1]) != int(expected_dim):
        return False, int(y.shape[0]), int(y.shape[1]), f"bad_dim_{y.shape[1]}"
    if not tensor_finite(y):
        return False, int(y.shape[0]), int(y.shape[1]), "nan_inf"
    return True, int(y.shape[0]), int(y.shape[1]), "ok"


def tensor_1d_info(x: Any, expected_len: int | None = None, allow_bool: bool = False) -> tuple[bool, int, str]:
    if not torch.is_tensor(x):
        return False, 0, "not_tensor"
    y = x.detach().cpu()
    if y.ndim == 2 and y.shape[0] == 1:
        y = y.squeeze(0)
    y = y.reshape(-1)
    if y.numel() <= 0:
        return False, 0, "empty"
    if expected_len is not None and int(y.numel()) != int(expected_len):
        return False, int(y.numel()), f"bad_len_{y.numel()}_expected_{expected_len}"
    if not allow_bool and not tensor_finite(y.float()):
        return False, int(y.numel()), "nan_inf"
    return True, int(y.numel()), "ok"


def check_manifest(rows: list[dict[str, Any]], args: argparse.Namespace, *, kind: str) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    durations: list[float] = []
    text_lens: list[int] = []
    languages: Counter[str] = Counter()
    speakers: Counter[str] = Counter()

    for idx, row in enumerate(rows):
        sample_id = as_text(row.get("sample_id", row.get("id", idx)))
        text = as_text(row.get("raw_text", row.get("text", row.get("norm_text", ""))))
        language = as_text(row.get("language", ""))
        speaker = as_text(row.get("speaker_name", row.get("speaker", "")))
        if language:
            languages[language] += 1
        if speaker:
            speakers[speaker] += 1
        text_lens.append(len(text))
        if len(text) == 0:
            add_issue(issues, "FAIL", sample_id, "empty_text", "text is empty")
        elif len(text) < args.min_text_chars_warn:
            add_issue(issues, "WARN", sample_id, "text_too_short", f"len={len(text)}")
        elif len(text) > args.max_text_chars_warn:
            add_issue(issues, "WARN", sample_id, "text_too_long", f"len={len(text)}")

        wav_keys = ["wav_path"] if kind == "manifest" else ["prompt_wav_path", "target_wav_path"]
        for key in wav_keys:
            p = norm_path(row.get(key, row.get("target_wav_path", "")))
            if not p:
                add_issue(issues, "FAIL", sample_id, f"missing_{key}", "empty path")
            elif not file_exists(p):
                add_issue(issues, "FAIL", sample_id, f"missing_file_{key}", p)
            elif args.check_audio_duration:
                d = duration_sec(p)
                if finite_num(d):
                    durations.append(d)
                    if key == "prompt_wav_path" and (d < args.min_prompt_sec or d > args.max_prompt_sec):
                        add_issue(issues, "WARN", sample_id, "prompt_duration_out_of_range", f"{d:.3f}")
                    if key != "prompt_wav_path":
                        if d < args.min_clip_sec_fail or d > args.max_clip_sec_fail:
                            add_issue(issues, "FAIL", sample_id, "clip_duration_fail", f"{d:.3f}")
                        elif d < args.min_clip_sec_warn or d > args.max_clip_sec_warn:
                            add_issue(issues, "WARN", sample_id, "clip_duration_warn", f"{d:.3f}")

    if len(rows) < args.min_samples_warn:
        add_issue(issues, "FAIL", None, "too_few_samples", f"num_samples={len(rows)}")
    elif len(rows) < args.min_samples_pass:
        add_issue(issues, "WARN", None, "few_samples", f"num_samples={len(rows)}")

    if not args.allow_multi_language and len(languages) > 1:
        add_issue(issues, "WARN", None, "multi_language", str(dict(languages)))
    if not args.allow_multi_speaker and len(speakers) > 1:
        add_issue(issues, "WARN", None, "multi_speaker", str(dict(speakers)))

    return {
        "kind": kind,
        "final_status": worst_status(issues),
        "num_samples": len(rows),
        "duration_stats": stats(durations),
        "text_len_stats": stats(text_lens),
        "languages": dict(languages),
        "speakers": dict(speakers),
        "num_failures": sum(1 for x in issues if x["level"] == "FAIL"),
        "num_warnings": sum(1 for x in issues if x["level"] == "WARN"),
        "issues": issues,
    }


def check_pt_root(root: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    pt_root = Path(root).resolve()
    if not pt_root.exists() or not pt_root.is_dir():
        raise FileNotFoundError(f"stage2_pt_root not found: {pt_root}")
    files = sorted(pt_root.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files under: {pt_root}")

    issues: list[dict[str, Any]] = []
    sem_lens: list[int] = []
    ac_lens: list[int] = []
    prompt_ac_lens: list[int] = []
    target_energy_lens: list[int] = []
    prompt_energy_lens: list[int] = []
    languages: Counter[str] = Counter()

    continuous_ok = 0
    prompt_acoustic_ok = 0
    target_energy_ok = 0
    prompt_energy_ok = 0
    f0_ok = 0
    speaker_embedding_ok = 0

    require_v66 = args.require_v66_style_fields or args.require_reference_acoustic_style
    require_energy = require_v66 or args.require_energy

    for path in files:
        sample_id = path.stem
        try:
            sample = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception as e:
            add_issue(issues, "FAIL", sample_id, "pt_load_failed", str(e))
            continue
        if not isinstance(sample, dict):
            add_issue(issues, "FAIL", sample_id, "pt_not_dict", str(type(sample)))
            continue

        sample_id = as_text(sample.get("sample_id", sample_id))
        lang = as_text(sample.get("language", ""))
        if lang:
            languages[lang] += 1

        for key in STAGE2_PT_REQUIRED_KEYS:
            if key not in sample:
                add_issue(issues, "FAIL", sample_id, "missing_required_key", key)

        if torch.is_tensor(sample.get("semantic_tokens")):
            sem_len = int(sample["semantic_tokens"].reshape(-1).numel())
            sem_lens.append(sem_len)
        else:
            sem_len = 0
            add_issue(issues, "FAIL", sample_id, "bad_semantic_tokens", "semantic_tokens is not tensor")

        ok_ac, ac_len, ac_dim, msg = tensor_2d_info(sample.get("target_acoustic"), args.expected_acoustic_dim)
        if ok_ac:
            ac_lens.append(ac_len)
        else:
            add_issue(issues, "FAIL", sample_id, "bad_target_acoustic", msg)
            ac_len = None

        if args.require_continuous_semantic:
            for key in CONTINUOUS_KEYS:
                if key not in sample:
                    add_issue(issues, "FAIL", sample_id, "missing_continuous_semantic", key)
        if "oracle_semantic_continuous" in sample:
            ok_cont, cont_len, cont_dim, msg = tensor_2d_info(
                sample.get("oracle_semantic_continuous"), args.expected_continuous_semantic_dim
            )
            if ok_cont:
                continuous_ok += 1
                if sem_len and cont_len != sem_len:
                    add_issue(issues, "FAIL", sample_id, "continuous_len_mismatch", f"{cont_len} vs {sem_len}")
            elif args.require_continuous_semantic:
                add_issue(issues, "FAIL", sample_id, "bad_oracle_semantic_continuous", msg)

        if require_v66:
            for key in V66_STYLE_KEYS:
                if key not in sample:
                    add_issue(issues, "FAIL", sample_id, "missing_v66_style_key", key)

        ok_pa, pa_len, pa_dim, msg = tensor_2d_info(sample.get("prompt_acoustic"), args.expected_acoustic_dim)
        if ok_pa:
            prompt_acoustic_ok += 1
            prompt_ac_lens.append(pa_len)
            pa_len_field = as_int(sample.get("prompt_acoustic_lengths"), pa_len)
            if pa_len_field != pa_len:
                add_issue(issues, "FAIL", sample_id, "prompt_acoustic_length_mismatch", f"{pa_len_field} vs {pa_len}")
        elif require_v66:
            add_issue(issues, "FAIL", sample_id, "bad_prompt_acoustic", msg)

        ok_te, te_len, msg = tensor_1d_info(sample.get("target_energy"), ac_len)
        if ok_te:
            target_energy_ok += 1
            target_energy_lens.append(te_len)
        elif require_energy:
            add_issue(issues, "FAIL", sample_id, "bad_target_energy", msg)

        ok_pe, pe_len, msg = tensor_1d_info(sample.get("prompt_energy"), pa_len if ok_pa else None)
        if ok_pe:
            prompt_energy_ok += 1
            prompt_energy_lens.append(pe_len)
        elif require_energy:
            add_issue(issues, "FAIL", sample_id, "bad_prompt_energy", msg)

        if args.require_f0:
            for key in ["target_f0", "target_f0_voiced_mask", "prompt_f0", "prompt_f0_voiced_mask"]:
                if key not in sample:
                    add_issue(issues, "FAIL", sample_id, "missing_f0_key", key)
        if "target_f0" in sample:
            ok_f0, _, msg = tensor_1d_info(sample.get("target_f0"), ac_len)
            if ok_f0:
                f0_ok += 1
            elif args.require_f0:
                add_issue(issues, "FAIL", sample_id, "bad_target_f0", msg)
        if "target_f0_voiced_mask" in sample:
            ok_mask, _, msg = tensor_1d_info(sample.get("target_f0_voiced_mask"), ac_len, allow_bool=True)
            if not ok_mask and args.require_f0:
                add_issue(issues, "FAIL", sample_id, "bad_target_f0_voiced_mask", msg)

        if args.require_speaker_embedding:
            for key in ["target_speaker_embedding", "prompt_speaker_embedding"]:
                if key not in sample:
                    add_issue(issues, "FAIL", sample_id, "missing_speaker_embedding", key)
        if torch.is_tensor(sample.get("target_speaker_embedding")):
            emb = sample["target_speaker_embedding"].detach().cpu().float().reshape(-1)
            if emb.numel() > 0 and torch.isfinite(emb).all():
                speaker_embedding_ok += 1

        if sem_len > 0 and ac_len:
            ratio = float(ac_len) / max(float(sem_len), 1.0)
            if ratio < 2.0 or ratio > 5.0:
                add_issue(issues, "WARN", sample_id, "acoustic_semantic_ratio_unusual", f"ratio={ratio:.3f}")

    if len(files) < args.min_samples_warn:
        add_issue(issues, "FAIL", None, "too_few_samples", f"num_samples={len(files)}")
    elif len(files) < args.min_samples_pass:
        add_issue(issues, "WARN", None, "few_samples", f"num_samples={len(files)}")
    if not args.allow_multi_language and len(languages) > 1:
        add_issue(issues, "WARN", None, "multi_language", str(dict(languages)))

    return {
        "kind": "stage2_pt_root",
        "final_status": worst_status(issues),
        "stage2_pt_root": str(pt_root),
        "num_samples": len(files),
        "semantic_len_stats": stats(sem_lens),
        "acoustic_len_stats": stats(ac_lens),
        "languages": dict(languages),
        "v641_continuous_semantic": {
            "required": bool(args.require_continuous_semantic),
            "ok_count": continuous_ok,
            "coverage": float(continuous_ok / max(len(files), 1)),
        },
        "v66_style_fields": {
            "required": bool(require_v66),
            "prompt_acoustic_ok_count": prompt_acoustic_ok,
            "target_energy_ok_count": target_energy_ok,
            "prompt_energy_ok_count": prompt_energy_ok,
            "prompt_acoustic_coverage": float(prompt_acoustic_ok / max(len(files), 1)),
            "target_energy_coverage": float(target_energy_ok / max(len(files), 1)),
            "prompt_energy_coverage": float(prompt_energy_ok / max(len(files), 1)),
            "prompt_acoustic_len_stats": stats(prompt_ac_lens),
            "target_energy_len_stats": stats(target_energy_lens),
            "prompt_energy_len_stats": stats(prompt_energy_lens),
            "f0_ok_count": f0_ok,
            "speaker_embedding_ok_count": speaker_embedding_ok,
        },
        "num_failures": sum(1 for x in issues if x["level"] == "FAIL"),
        "num_warnings": sum(1 for x in issues if x["level"] == "WARN"),
        "issues": issues,
    }


def main() -> None:
    args = parse_args()
    provided = [bool(args.manifest_jsonl), bool(args.stage2_manifest_jsonl), bool(args.stage2_pt_root)]
    if sum(1 for x in provided if x) != 1:
        raise ValueError("Provide exactly one of --manifest_jsonl, --stage2_manifest_jsonl, --stage2_pt_root")

    if args.manifest_jsonl:
        rows = read_jsonl(args.manifest_jsonl)
        report = check_manifest(rows, args, kind="manifest")
        input_path = str(Path(args.manifest_jsonl).resolve())
    elif args.stage2_manifest_jsonl:
        rows = read_jsonl(args.stage2_manifest_jsonl)
        report = check_manifest(rows, args, kind="stage2_manifest")
        input_path = str(Path(args.stage2_manifest_jsonl).resolve())
    else:
        report = check_pt_root(args.stage2_pt_root, args)
        input_path = str(Path(args.stage2_pt_root).resolve())

    payload = {
        "tool": "check_fewshot_dataset_health.py",
        "version": "v6.6.0-B",
        "input_path": input_path,
        "final_status": report["final_status"],
        "config": vars(args),
        "report": report,
    }
    save_json(payload, args.report_path)

    print("==== Few-shot dataset health check ====")
    print(f"input_path   : {input_path}")
    print(f"report_path  : {Path(args.report_path).resolve()}")
    print(f"final_status : {report['final_status']}")
    print(f"num_samples  : {report.get('num_samples')}")
    print(f"failures     : {report.get('num_failures')}")
    print(f"warnings     : {report.get('num_warnings')}")

    if report.get("kind") == "stage2_pt_root":
        cont = report.get("v641_continuous_semantic", {})
        v66 = report.get("v66_style_fields", {})
        print("----------------------------------------")
        print("continuous semantic:")
        print(f"  required  : {cont.get('required')}")
        print(f"  ok_count  : {cont.get('ok_count')}")
        print(f"  coverage  : {cont.get('coverage')}")
        print("v6.6 style fields:")
        print(f"  required                 : {v66.get('required')}")
        print(f"  prompt_acoustic_ok_count : {v66.get('prompt_acoustic_ok_count')}")
        print(f"  target_energy_ok_count   : {v66.get('target_energy_ok_count')}")
        print(f"  prompt_energy_ok_count   : {v66.get('prompt_energy_ok_count')}")
        print(f"  prompt_acoustic_coverage : {v66.get('prompt_acoustic_coverage')}")
        print(f"  target_energy_coverage   : {v66.get('target_energy_coverage')}")
        print(f"  prompt_energy_coverage   : {v66.get('prompt_energy_coverage')}")

    if args.print_samples:
        preview = report.get("issues", [])[: args.max_print_items]
        if preview:
            print("----------------------------------------")
            print("issue preview:")
            for x in preview:
                print(f"[{x['level']}] sample={x['sample_id']} type={x['issue_type']} msg={x['message']}")

    print("========================================")

    if report["final_status"] == "FAIL" and args.exit_nonzero_on_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
