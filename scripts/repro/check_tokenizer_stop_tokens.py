#!/usr/bin/env python3
"""Check tokenizer stop-token compatibility before OPD reproduction.

This is a lightweight preflight for OPD runs. It does not prove that two models
are behaviorally compatible, but it catches common tokenizer/EOS mismatches that
can make top-k RKL supervision push the student away from stopping.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

from transformers import AutoTokenizer


DEFAULT_CANDIDATES = [
    "<|endoftext|>",
    "<|im_end|>",
    "<|end|>",
    "<|eot_id|>",
    "</s>",
]


@dataclass
class TokenizerSummary:
    path: str
    name_or_path: str
    vocab_size: int
    eos_token: str | None
    eos_token_id: int | None
    pad_token: str | None
    pad_token_id: int | None
    bos_token: str | None
    bos_token_id: int | None
    has_chat_template: bool
    special_tokens_map: dict[str, Any]
    stop_candidates: dict[str, list[int] | None]


def encode_candidate(tokenizer: Any, text: str) -> list[int] | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids if ids else None


def summarize(path: str, candidates: list[str], trust_remote_code: bool) -> TokenizerSummary:
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    return TokenizerSummary(
        path=path,
        name_or_path=getattr(tokenizer, "name_or_path", ""),
        vocab_size=len(tokenizer),
        eos_token=tokenizer.eos_token,
        eos_token_id=tokenizer.eos_token_id,
        pad_token=tokenizer.pad_token,
        pad_token_id=tokenizer.pad_token_id,
        bos_token=tokenizer.bos_token,
        bos_token_id=tokenizer.bos_token_id,
        has_chat_template=bool(getattr(tokenizer, "chat_template", None)),
        special_tokens_map=dict(tokenizer.special_tokens_map),
        stop_candidates={candidate: encode_candidate(tokenizer, candidate) for candidate in candidates},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True, help="Student model/tokenizer path")
    parser.add_argument("--teacher", required=True, help="Teacher model/tokenizer path")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Extra literal stop-token candidate to encode. Can be repeated.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit non-zero if EOS ID/token/chat-template presence differs.",
    )
    args = parser.parse_args()

    candidates = DEFAULT_CANDIDATES + args.candidate
    student = summarize(args.student, candidates, args.trust_remote_code)
    teacher = summarize(args.teacher, candidates, args.trust_remote_code)

    checks = {
        "same_vocab_size": student.vocab_size == teacher.vocab_size,
        "same_eos_token": student.eos_token == teacher.eos_token,
        "same_eos_token_id": student.eos_token_id == teacher.eos_token_id,
        "same_pad_token_id": student.pad_token_id == teacher.pad_token_id,
        "same_chat_template_presence": student.has_chat_template == teacher.has_chat_template,
        "same_candidate_encodings": student.stop_candidates == teacher.stop_candidates,
    }
    warnings = [
        name for name, passed in checks.items() if not passed
    ]

    payload = {
        "student": asdict(student),
        "teacher": asdict(teacher),
        "checks": checks,
        "warnings": warnings,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if warnings:
        print("\nWARN: tokenizer stop-token compatibility issues detected:")
        for warning in warnings:
            print(f"  - {warning}")
        print(
            "\nIf this pairing is intentional, document the EOS/stop-token remapping "
            "or loss mask before running long OPD jobs."
        )

    if args.fail_on_mismatch and warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
