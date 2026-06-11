#!/usr/bin/env python3
"""Run the THUNLP OPD README-style vLLM eval with explicit paths."""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import multiprocessing
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
except ImportError:
    destroy_distributed_environment = None
    destroy_model_parallel = None


DEFAULT_TASKS = ("AIME24", "AIME25", "AMC23")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--data-dir", default="baselines/thunlp-opd/scripts/val/data")
    parser.add_argument("--output-root", default="runs/thunlp_eval_ckpt_compare_20260610/readme_eval_outputs")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=31744)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def load_samples(path: Path) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    samples = []
    for i in range(len(df)):
        if "problem" in df.columns:
            prompt = str(df.at[i, "problem"]).strip()
            answer = str(df.at[i, "answer"]).strip()
        else:
            prompt = df.at[i, "prompt"][0]["content"].strip()
            answer = df.at[i, "reward_model"]["ground_truth"].strip()
        samples.append({"example_id": i, "prompt": prompt, "answer": answer})
    return samples


def split_rollout_ids(rollout_ids: list[int], num_workers: int) -> list[list[int]]:
    chunks = [[] for _ in range(num_workers)]
    for idx, rollout_id in enumerate(rollout_ids):
        chunks[idx % num_workers].append(rollout_id)
    return chunks


def format_prompt(tokenizer, prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def worker_process(args_tuple):
    (
        model,
        samples,
        rollout_ids,
        gpu_id,
        enable_thinking,
        temperature,
        top_p,
        max_tokens,
        gpu_memory_utilization,
    ) = args_tuple

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    results = []
    llm = None
    try:
        llm = LLM(
            model=model,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
        )
        tokenizer = llm.get_tokenizer()
        stop_token_ids = []
        for token in ("<|im_end|>", "<|endoftext|>"):
            try:
                ids = tokenizer.encode(token, add_special_tokens=False)
                if ids:
                    stop_token_ids.append(ids[0])
            except Exception:
                pass
        stop_token_ids = sorted(set(stop_token_ids))

        formatted_prompts = [format_prompt(tokenizer, s["prompt"], enable_thinking) for s in samples]
        for rollout_id in rollout_ids:
            sampling = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop_token_ids=stop_token_ids or None,
            )
            outputs = llm.generate(formatted_prompts, sampling, use_tqdm=False)
            for sample, out in zip(samples, outputs):
                results.append(
                    {
                        "example_id": sample["example_id"],
                        "prompt": sample["prompt"],
                        "answer": sample["answer"],
                        "seed": rollout_id,
                        "response": out.outputs[0].text,
                    }
                )
    finally:
        if llm is not None:
            del llm
        if destroy_model_parallel is not None:
            try:
                destroy_model_parallel()
            except Exception:
                pass
        if destroy_distributed_environment is not None:
            try:
                destroy_distributed_environment()
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()
    return results


def grade_outputs(output_dir: Path, tokenizer_path: str) -> list[dict[str, Any]]:
    eval_dir = Path("baselines/thunlp-opd/scripts/val/eval").resolve()
    sys.path.insert(0, str(eval_dir))
    from utils import grade_answer_verl

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    summaries = []
    for jsonl_path in sorted(output_dir.glob("*.jsonl")):
        grouped: dict[int, dict[str, Any]] = {}
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                item = grouped.setdefault(int(row["example_id"]), {"answer": row["answer"], "responses": []})
                item["responses"].append(row["response"])

        per_question = []
        all_scores = []
        lengths = []
        format_errors = 0
        for example_id in sorted(grouped):
            item = grouped[example_id]
            scores = []
            for response in item["responses"]:
                score = bool(grade_answer_verl(response, item["answer"]))
                scores.append(score)
                all_scores.append(score)
                lengths.append(len(tokenizer.encode(response)))
                if "boxed" not in response:
                    format_errors += 1
            per_question.append(sum(scores) / max(len(scores), 1))

        task_name = jsonl_path.name.split("_", 1)[0].upper()
        n_questions = len(per_question)
        n_rollouts = len(all_scores)
        summaries.append(
            {
                "task": task_name,
                "questions": n_questions,
                "rollouts": n_rollouts,
                "mean_score": sum(per_question) / n_questions if n_questions else 0.0,
                "best_score": sum(1 for x in per_question if x > 0) / n_questions if n_questions else 0.0,
                "solve_none": sum(1 for x in per_question if x == 0),
                "solve_all": sum(1 for x in per_question if x == 1),
                "avg_output_length": sum(lengths) / len(lengths) if lengths else 0.0,
                "format_error_rollouts": format_errors,
            }
        )
    with (output_dir / "grading_results.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    return summaries


def main() -> None:
    args = parse_args()
    gpu_ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must not be empty")

    output_dir = Path(args.output_root) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        samples = load_samples(Path(args.data_dir) / task / "test.parquet")
        out_path = output_dir / f"{task.lower()}_t{args.temperature}_p{args.top_p}_n{args.n}-MNT{args.max_tokens}.jsonl"
        if out_path.exists() and not args.replace:
            print(f"skip_existing={out_path}", flush=True)
            continue

        rollout_chunks = split_rollout_ids(list(range(args.n)), len(gpu_ids))
        args_list = [
            (
                args.model,
                samples,
                rollout_chunks[i],
                gpu_ids[i],
                args.enable_thinking,
                args.temperature,
                args.top_p,
                args.max_tokens,
                args.gpu_memory_utilization,
            )
            for i in range(len(gpu_ids))
        ]
        ctx = multiprocessing.get_context("spawn")
        all_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=ctx) as ex:
            futures = [ex.submit(worker_process, item) for item in args_list]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=task):
                all_results.extend(fut.result())

        expected = len(samples) * args.n
        if len(all_results) != expected:
            raise RuntimeError(f"{task}: expected {expected} generations, got {len(all_results)}")
        with out_path.open("w", encoding="utf-8") as f:
            for row in sorted(all_results, key=lambda x: (x["example_id"], x["seed"])):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"saved={out_path} rows={len(all_results)}", flush=True)

    summaries = grade_outputs(output_dir, args.model)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
