#!/usr/bin/env python3
"""Check the Python/CUDA stack expected by THUNLP OPD."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
from typing import Any


PACKAGE_NAMES = [
    "torch",
    "vllm",
    "ray",
    "transformers",
    "accelerate",
    "datasets",
    "peft",
    "pyarrow",
    "pandas",
    "numpy",
    "tensordict",
    "torchdata",
    "flashinfer-python",
    "sglang",
    "liger-kernel",
    "math-verify",
    "latex2sympy2-extended",
    "sympy",
    "hydra-core",
    "omegaconf",
]

IMPORT_NAMES = [
    "torch",
    "ray",
    "vllm",
    "flash_attn",
    "flashinfer",
    "sglang",
    "pandas",
    "pyarrow.parquet",
    "transformers",
    "hydra",
    "omegaconf",
    "tensordict",
    "torchdata",
    "latex2sympy2_extended",
    "math_verify",
    "verl",
    "verl.trainer.main_ppo",
    "verl.trainer.ppo.ray_trainer",
    "verl.utils.reward_score.ttrl_math",
]


def package_versions() -> dict[str, str | None]:
    versions = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def import_checks() -> dict[str, str]:
    checks = {}
    for module in IMPORT_NAMES:
        try:
            importlib.import_module(module)
            checks[module] = "ok"
        except Exception as exc:  # pragma: no cover - diagnostic path
            checks[module] = repr(exc)
    return checks


def torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"import_error": repr(exc)}
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-gpus", type=int, default=0)
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args()

    versions = package_versions()
    imports = import_checks()
    torch = torch_info()
    warnings = []

    missing_packages = [name for name, version in versions.items() if version is None]
    failed_imports = {name: err for name, err in imports.items() if err != "ok"}
    if missing_packages:
        warnings.append(f"missing packages: {missing_packages}")
    if failed_imports:
        warnings.append(f"failed imports: {failed_imports}")
    if args.min_gpus and torch.get("gpu_count", 0) < args.min_gpus:
        warnings.append(f"gpu_count {torch.get('gpu_count', 0)} < required {args.min_gpus}")

    payload = {
        "packages": versions,
        "imports": imports,
        "torch": torch,
        "warnings": warnings,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if warnings:
        print("\nWARN: THUNLP environment preflight found issues:")
        for warning in warnings:
            print(f"  - {warning}")
    if warnings and args.fail_on_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
