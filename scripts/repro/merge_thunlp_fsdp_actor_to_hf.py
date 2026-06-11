#!/usr/bin/env python3
"""Merge a THUNLP/verl FSDP actor checkpoint into a Hugging Face model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardedStateDictConfig, StateDictType
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from verl.utils.device import get_device_id, get_nccl_backend
from verl.utils.fsdp_utils import (
    get_fsdp_full_state_dict,
    get_fsdp_state_ctx,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.workers.engine.fsdp.utils import create_device_mesh, get_sharding_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-dir", required=True, help="Path to global_step_N/actor.")
    parser.add_argument("--base-model", required=True, help="Original HF model path.")
    parser.add_argument("--output-dir", required=True, help="Destination HF model directory.")
    parser.add_argument("--torch-dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--save-dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def main() -> None:
    args = parse_args()
    actor_dir = Path(args.actor_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    base_model = Path(args.base_model).resolve()

    dist.init_process_group(backend=get_nccl_backend())
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device_mesh = create_device_mesh(world_size=world_size, fsdp_size=-1)
    model_config = AutoConfig.from_pretrained(base_model, trust_remote_code=args.trust_remote_code)
    model_dtype = dtype_from_name(args.torch_dtype)

    init_context = get_init_weight_context_manager(
        use_meta_tensor=not getattr(model_config, "tie_word_embeddings", False),
        mesh=device_mesh,
    )
    with init_context():
        module = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=model_dtype,
            config=model_config,
            trust_remote_code=args.trust_remote_code,
        )
    module.to(model_dtype)

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    wrap_policy = get_fsdp_wrap_policy(module, {"min_num_params": 0}, is_lora=False)
    fsdp_model = FSDP(
        module,
        param_init_fn=init_fn,
        auto_wrap_policy=wrap_policy,
        device_id=get_device_id(),
        sharding_strategy=get_sharding_strategy(device_mesh),
        mixed_precision=mixed_precision,
        sync_module_states=True,
        device_mesh=device_mesh,
        forward_prefetch=True,
        use_orig_params=False,
    )

    shard_path = actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
    if not shard_path.exists():
        raise FileNotFoundError(f"Missing shard for rank {rank}: {shard_path}")

    state_cfg = ShardedStateDictConfig(offload_to_cpu=True)
    with get_fsdp_state_ctx(fsdp_model, StateDictType.SHARDED_STATE_DICT, state_cfg, None):
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        fsdp_model.load_state_dict(shard)
        del shard
    dist.barrier()

    state_dict = get_fsdp_full_state_dict(fsdp_model, offload_to_cpu=True, rank0_only=True)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_dtype = dtype_from_name(args.save_dtype)
        with init_empty_weights():
            save_model = AutoModelForCausalLM.from_config(model_config, torch_dtype=save_dtype)
        save_model.to_empty(device="cpu")
        save_model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=True)

        tokenizer_source = actor_dir / "huggingface"
        if not tokenizer_source.exists():
            tokenizer_source = base_model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=args.trust_remote_code)
        tokenizer.save_pretrained(output_dir)

        try:
            generation_config = GenerationConfig.from_pretrained(tokenizer_source)
        except Exception:
            generation_config = GenerationConfig.from_pretrained(base_model)
        generation_config.save_pretrained(output_dir)
        print(f"merged_hf_model={output_dir}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
