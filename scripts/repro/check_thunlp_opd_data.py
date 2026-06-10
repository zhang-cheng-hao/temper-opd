#!/usr/bin/env python3
"""Validate THUNLP OPD parquet inputs before launching long runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {"prompt", "reward_model", "data_source", "ability", "extra_info"}


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def prompt_is_valid(prompt: Any) -> bool:
    prompt_json = to_jsonable(prompt)
    if not isinstance(prompt_json, list) or not prompt_json:
        return False
    return all(
        isinstance(item, dict)
        and isinstance(item.get("role"), str)
        and isinstance(item.get("content"), str)
        and bool(item.get("content"))
        for item in prompt_json
    )


def reward_value(reward_model: Any, key: str) -> Any:
    reward_json = to_jsonable(reward_model)
    if not isinstance(reward_json, dict):
        return None
    return reward_json.get(key)


def prompt_token_lengths(
    df: pd.DataFrame,
    tokenizer_path: str | None,
    max_prompt_length: int | None,
    sample_rows: int,
    trust_remote_code: bool,
) -> dict[str, Any] | None:
    if not tokenizer_path:
        return None

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
    rows = df.head(sample_rows) if sample_rows > 0 else df
    lengths = []
    overlong = 0

    for prompt in rows["prompt"].tolist():
        chat = to_jsonable(prompt)
        text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        lengths.append(len(ids))
        if max_prompt_length is not None and len(ids) > max_prompt_length:
            overlong += 1

    return {
        "tokenizer": tokenizer_path,
        "checked_rows": int(len(rows)),
        "max_prompt_length_limit": max_prompt_length,
        "min": min(lengths) if lengths else None,
        "max": max(lengths) if lengths else None,
        "mean": sum(lengths) / len(lengths) if lengths else None,
        "overlong_count": overlong,
    }


def inspect_parquet(
    path: Path,
    sample_rows: int,
    tokenizer_path: str | None,
    max_prompt_length: int | None,
    token_length_sample_rows: int,
    trust_remote_code: bool,
    fail_on_overlong: bool,
) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "ok": False, "error": "missing file"}
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"path": str(path), "exists": True, "ok": False, "error": repr(exc)}

    columns = list(df.columns)
    missing = sorted(REQUIRED_COLUMNS - set(columns))
    sample = []
    if len(df) and sample_rows:
        sample = [to_jsonable(row) for row in df.head(sample_rows).to_dict("records")]

    prompt_invalid_count = 0
    ground_truth_missing_count = 0
    reward_style_missing_count = 0
    data_source_missing_count = 0
    if not missing:
        prompt_invalid_count = int((~df["prompt"].map(prompt_is_valid)).sum())
        ground_truth_missing_count = int(
            df["reward_model"].map(lambda item: not bool(reward_value(item, "ground_truth"))).sum()
        )
        reward_style_missing_count = int(
            df["reward_model"].map(lambda item: not bool(reward_value(item, "style"))).sum()
        )
        data_source_missing_count = int(df["data_source"].isna().sum() + (df["data_source"].astype(str) == "").sum())

    token_lengths = None
    token_length_error = None
    if tokenizer_path and not missing:
        try:
            token_lengths = prompt_token_lengths(
                df,
                tokenizer_path,
                max_prompt_length,
                token_length_sample_rows,
                trust_remote_code,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            token_length_error = repr(exc)

    ok = (
        not missing
        and prompt_invalid_count == 0
        and ground_truth_missing_count == 0
        and reward_style_missing_count == 0
        and data_source_missing_count == 0
        and not (fail_on_overlong and token_lengths and token_lengths["overlong_count"])
        and token_length_error is None
    )

    return {
        "path": str(path),
        "exists": True,
        "ok": ok,
        "rows": int(len(df)),
        "columns": columns,
        "missing_required_columns": missing,
        "prompt_invalid_count": prompt_invalid_count,
        "ground_truth_missing_count": ground_truth_missing_count,
        "reward_style_missing_count": reward_style_missing_count,
        "data_source_missing_count": data_source_missing_count,
        "prompt_token_lengths": token_lengths,
        "prompt_token_length_error": token_length_error,
        "sample": sample,
    }


def parse_val_files(raw: str | None, test_data_dir: Path) -> list[Path]:
    if not raw:
        return [
            test_data_dir / "AIME25" / "test.parquet",
            test_data_dir / "AMC23" / "test.parquet",
            test_data_dir / "AIME24" / "test.parquet",
        ]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, str):
            return [Path(parsed)]
        return [Path(item) for item in parsed]
    except json.JSONDecodeError:
        return [Path(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--test-data-dir", required=True)
    parser.add_argument(
        "--val-files",
        default=None,
        help="Optional JSON list/string or comma-separated parquet paths. Defaults to AIME25/AMC23/AIME24.",
    )
    parser.add_argument("--sample-rows", type=int, default=1)
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer/model path for prompt length checks.")
    parser.add_argument("--max-prompt-length", type=int, default=None)
    parser.add_argument(
        "--token-length-sample-rows",
        type=int,
        default=0,
        help="Rows to check for token lengths. 0 means all rows.",
    )
    parser.add_argument(
        "--fail-on-overlong",
        action="store_true",
        help="Fail when sampled prompt lengths exceed --max-prompt-length. Default only reports counts.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()

    train_file = Path(args.train_file)
    test_data_dir = Path(args.test_data_dir)
    val_files = parse_val_files(args.val_files, test_data_dir)
    inspect_kwargs = {
        "sample_rows": args.sample_rows,
        "tokenizer_path": args.tokenizer,
        "max_prompt_length": args.max_prompt_length,
        "token_length_sample_rows": args.token_length_sample_rows,
        "trust_remote_code": args.trust_remote_code,
        "fail_on_overlong": args.fail_on_overlong,
    }

    train = inspect_parquet(train_file, **inspect_kwargs)
    vals = [inspect_parquet(path, **inspect_kwargs) for path in val_files]
    warnings = []
    if not train["ok"]:
        warnings.append(f"train not ok: {train_file}")
    for val in vals:
        if not val["ok"]:
            warnings.append(f"val not ok: {val['path']}")

    payload = {
        "train": train,
        "validation": vals,
        "warnings": warnings,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if warnings:
        print("\nWARN: THUNLP OPD data preflight found issues:")
        for warning in warnings:
            print(f"  - {warning}")

    if warnings and args.fail_on_warning:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
