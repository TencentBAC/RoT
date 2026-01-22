#!/usr/bin/env python3
"""
将 HuggingFace safetensors 格式转换回 DeepSpeed checkpoint 格式（.pt）

用法:
    python convert_from_safetensors.py \
        --input /path/to/safetensors_dir \
        --output /path/to/output.pt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import torch
from safetensors.torch import load_file
from tqdm import tqdm


def load_safetensors(
    input_dir: Path, 
    model_name: str = "model",
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """从 safetensors 文件加载 state_dict"""
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Loading safetensors from: {input_dir}")
        print(f"{'='*80}")
    
    state_dict = {}
    
    # 检查是否有 index.json（多文件分片）
    index_path = input_dir / f"{model_name}.safetensors.index.json"
    
    if index_path.exists():
        # 多文件分片格式
        if verbose:
            print(f"Found index file: {index_path}")
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        weight_map = index["weight_map"]
        shard_files = sorted(set(weight_map.values()))
        
        if verbose:
            print(f"Loading {len(shard_files)} shard(s)...")
        
        for shard_file in tqdm(shard_files, desc="Loading shards", disable=not verbose):
            shard_path = input_dir / shard_file
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard file not found: {shard_path}")
            
            shard_state = load_file(str(shard_path))
            state_dict.update(shard_state)
            
            if verbose:
                shard_size = shard_path.stat().st_size / (1024**3)
                print(f"  Loaded {shard_file} ({shard_size:.2f} GB, {len(shard_state)} tensors)")
    
    else:
        # 单文件格式
        single_file = input_dir / f"{model_name}.safetensors"
        
        if not single_file.exists():
            raise FileNotFoundError(
                f"Neither index file nor single safetensors file found.\n"
                f"Expected: {index_path} or {single_file}"
            )
        
        if verbose:
            print(f"Loading single safetensors file: {single_file}")
        
        state_dict = load_file(str(single_file))
        
        if verbose:
            file_size = single_file.stat().st_size / (1024**3)
            print(f"  Loaded {file_size:.2f} GB, {len(state_dict)} tensors")
    
    # 统计信息
    total_params = sum(v.numel() for v in state_dict.values())
    total_size = sum(v.numel() * v.element_size() for v in state_dict.values())
    
    if verbose:
        print(f"\nTotal parameters loaded: {total_params:,}")
        print(f"Total size: {total_size / (1024**3):.2f} GB")
        print(f"Total keys: {len(state_dict)}")
    
    return state_dict


def restore_state_dict_keys(
    state_dict: Dict[str, torch.Tensor], 
    prefix: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """恢复 state_dict 的 key 前缀（如果需要）"""
    
    if prefix is None:
        if verbose:
            print("\nNo prefix restoration (using keys as-is)")
        return state_dict
    
    restored_state = {}
    
    for key, value in state_dict.items():
        restored_key = f"{prefix}{key}"
        restored_state[restored_key] = value
    
    if verbose:
        print(f"\nRestored keys with prefix: '{prefix}'")
        print(f"  Example: {list(state_dict.keys())[0]} -> {list(restored_state.keys())[0]}")
    
    return restored_state


def restore_weight_tying(
    state_dict: Dict[str, torch.Tensor],
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """
    恢复 weight tying（可选）
    
    某些模型会共享 embedding 和 lm_head 的权重。在转换为 safetensors 时，
    这些权重被克隆为独立副本。此函数可以恢复 weight tying 以节省内存。
    
    注意：这是可选的，对于推理来说，不恢复 weight tying 也可以正常工作。
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Checking for weight tying opportunities")
        print(f"{'='*80}")
    
    # 常见的 weight tying 模式
    tying_patterns = [
        # (source_key_pattern, target_key_pattern)
        ("language_model.embed_tokens.weight", "llm_lm_head.weight"),
        ("base_model.model.model.language_model.embed_tokens.weight", "base_model.model.lm_head.weight"),
    ]
    
    tied_count = 0
    for source_pattern, target_pattern in tying_patterns:
        source_keys = [k for k in state_dict.keys() if source_pattern in k]
        target_keys = [k for k in state_dict.keys() if target_pattern in k]
        
        for source_key in source_keys:
            for target_key in target_keys:
                if source_key in state_dict and target_key in state_dict:
                    source_tensor = state_dict[source_key]
                    target_tensor = state_dict[target_key]
                    
                    # 检查形状是否匹配
                    if source_tensor.shape == target_tensor.shape:
                        # 检查值是否相同
                        if torch.allclose(source_tensor, target_tensor, rtol=1e-5, atol=1e-8):
                            if verbose:
                                print(f"  Tying: {target_key} -> {source_key}")
                            # 让 target 指向 source（通过直接赋值实现 weight tying）
                            state_dict[target_key] = state_dict[source_key]
                            tied_count += 1
    
    if verbose:
        if tied_count > 0:
            print(f"\n✓ Restored {tied_count} weight tying relationships")
        else:
            print(f"\n✓ No weight tying opportunities found")
    
    return state_dict


def save_deepspeed_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    output_path: Path,
    format: str = "module",
    verbose: bool = True
):
    """保存为 DeepSpeed checkpoint 格式"""
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Saving DeepSpeed checkpoint to: {output_path}")
        print(f"{'='*80}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 根据格式包装 state_dict
    if format == "module":
        checkpoint_state = {"module": state_dict}
    elif format == "model":
        checkpoint_state = {"model": state_dict}
    elif format == "state_dict":
        checkpoint_state = {"state_dict": state_dict}
    elif format == "direct":
        checkpoint_state = state_dict
    else:
        raise ValueError(f"Unknown format: {format}")
    
    # 保存
    torch.save(checkpoint_state, str(output_path))
    
    if verbose:
        file_size = output_path.stat().st_size / (1024**3)
        print(f"✓ Saved checkpoint ({file_size:.2f} GB)")
        print(f"  Format: {format}")
        print(f"  Path: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace safetensors to DeepSpeed checkpoint format")
    parser.add_argument(
        "--input", 
        type=str,
        required=True,
        help="Path to safetensors directory"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Output path for DeepSpeed checkpoint (.pt)"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="RoT",
        help="Model name prefix (should match the conversion)"
    )
    parser.add_argument(
        "--prefix", 
        type=str, 
        default=None,
        help="Key prefix to restore (e.g., 'module.', 'model.')"
    )
    parser.add_argument(
        "--format", 
        type=str, 
        default="module",
        choices=["module", "model", "state_dict", "direct"],
        help="Checkpoint format (how to wrap state_dict)"
    )
    parser.add_argument(
        "--restore_weight_tying",
        action="store_true",
        help="Restore weight tying (e.g., share embedding and lm_head weights)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        default=True,
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    # 验证输入目录
    input_dir = Path(args.input)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    output_path = Path(args.output)
    
    print(f"\n{'='*80}")
    print(f"SafeTensors to DeepSpeed Conversion")
    print(f"{'='*80}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_path}")
    print(f"Format: {args.format}")
    if args.prefix:
        print(f"Key prefix: {args.prefix}")
    print(f"{'='*80}")
    
    # 1. 加载 safetensors
    state_dict = load_safetensors(input_dir, model_name=args.model_name, verbose=args.verbose)
    
    # 2. 恢复 weight tying（如果需要）
    if args.restore_weight_tying:
        state_dict = restore_weight_tying(state_dict, verbose=args.verbose)
    
    # 3. 恢复 key 前缀（如果需要）
    if args.prefix:
        state_dict = restore_state_dict_keys(state_dict, prefix=args.prefix, verbose=args.verbose)
    
    # 4. 保存为 DeepSpeed checkpoint
    save_deepspeed_checkpoint(
        state_dict,
        output_path,
        format=args.format,
        verbose=args.verbose
    )
    
    print(f"\n{'='*80}")
    print(f"✓ Conversion complete!")
    print(f"{'='*80}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
