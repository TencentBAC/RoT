#!/usr/bin/env python3
"""
将 DeepSpeed checkpoint 转换为 HuggingFace safetensors 格式

用法:
    python convert_to_safetensors.py \
        --input /path/to/mp_rank_00_model_states.pt \
        --output /path/to/output_dir \
        --max_shard_size 5GB
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import torch
from safetensors.torch import save_file
from tqdm import tqdm


def convert_size_to_bytes(size_str: str) -> int:
    """将大小字符串转换为字节数 (如 '5GB' -> 5368709120)"""
    size_str = size_str.upper().strip()
    
    if size_str.endswith('KB'):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    elif size_str.endswith('TB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024 * 1024)
    else:
        return int(size_str)


def get_tensor_size(tensor: torch.Tensor) -> int:
    """获取 tensor 的字节大小"""
    return tensor.numel() * tensor.element_size()


def load_deepspeed_checkpoint(checkpoint_path: str, verbose: bool = True) -> Dict[str, torch.Tensor]:
    """加载 DeepSpeed checkpoint 并提取 state_dict"""
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Loading DeepSpeed checkpoint: {checkpoint_path}")
        print(f"{'='*80}")
    
    checkpoint_state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 提取 state_dict（DeepSpeed 保存的格式可能有多种嵌套）
    state_dict = None
    for key in ["module", "model_state_dict", "model", "state_dict"]:
        if key in checkpoint_state:
            state_dict = checkpoint_state[key]
            if verbose:
                print(f"Found state_dict under key: '{key}'")
            break
    
    if state_dict is None and isinstance(checkpoint_state, dict):
        state_dict = checkpoint_state
        if verbose:
            print(f"Using checkpoint_state directly as state_dict")
    
    if state_dict is None:
        raise ValueError("Could not find model state dict in checkpoint")
    
    # 计算总大小
    total_size = sum(get_tensor_size(v) for v in state_dict.values())
    total_params = sum(v.numel() for v in state_dict.values())
    
    if verbose:
        print(f"Total parameters: {total_params:,}")
        print(f"Total size: {total_size / (1024**3):.2f} GB")
        print(f"Total keys: {len(state_dict)}")
    
    return state_dict


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor], verbose: bool = True) -> Dict[str, torch.Tensor]:
    """清理 state_dict 的 key，移除训练时的前缀"""
    
    clean_state = {}
    
    for key, value in state_dict.items():
        clean_key = key
        
        # 移除训练时的前缀（DeepSpeed 或 DDP 包装产生的）
        for prefix in ["module.", "model.", "vision_encoder.mllm_model."]:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
                break
        
        clean_state[clean_key] = value
    
    if verbose:
        print(f"\nCleaned keys: {len(state_dict)} -> {len(clean_state)}")
        if len(state_dict) != len(clean_state):
            print("Warning: Some keys were dropped during cleaning")
    
    return clean_state


def deduplicate_shared_tensors_global(
    state_dict: Dict[str, torch.Tensor],
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """
    全局处理共享内存的 tensor（在分片之前）
    
    检测所有共享内存的 tensor，并为除第一个之外的所有 tensor 创建独立副本。
    这样可以避免 safetensors 的共享内存错误。
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Checking for shared tensors (weight tying)")
        print(f"{'='*80}")
    
    seen_data_ptrs = {}
    deduplicated = {}
    cloned_keys = []
    shared_groups = {}  # data_ptr -> list of keys
    
    # 第一遍：收集所有共享内存的信息
    for key, tensor in state_dict.items():
        data_ptr = tensor.data_ptr()
        if data_ptr not in shared_groups:
            shared_groups[data_ptr] = []
        shared_groups[data_ptr].append(key)
    
    # 找出所有共享内存的组（有多个 key 指向同一内存）
    shared_memory_groups = {ptr: keys for ptr, keys in shared_groups.items() if len(keys) > 1}
    
    if shared_memory_groups and verbose:
        print(f"Found {len(shared_memory_groups)} groups of tensors sharing memory:")
        for i, (data_ptr, keys) in enumerate(shared_memory_groups.items(), 1):
            print(f"\n  Group {i}: {len(keys)} tensors share memory")
            for key in keys:
                tensor_size = get_tensor_size(state_dict[key])
                print(f"    - {key} ({tensor_size / (1024**2):.2f} MB)")
    
    # 第二遍：处理共享内存
    for key, tensor in state_dict.items():
        data_ptr = tensor.data_ptr()
        
        if data_ptr in seen_data_ptrs:
            # 这个 tensor 与之前的 tensor 共享内存 - 创建独立副本
            deduplicated[key] = tensor.clone()
            cloned_keys.append(key)
        else:
            # 第一次见到这个内存地址 - 直接使用
            seen_data_ptrs[data_ptr] = key
            deduplicated[key] = tensor
    
    if verbose:
        if cloned_keys:
            print(f"\n✓ Cloned {len(cloned_keys)} tensors to resolve shared memory")
            print(f"  Original tensors: {len(state_dict)}")
            print(f"  After deduplication: {len(deduplicated)}")
        else:
            print(f"\n✓ No shared tensors found")
    
    return deduplicated


def shard_state_dict(
    state_dict: Dict[str, torch.Tensor], 
    max_shard_size: int,
    verbose: bool = True
) -> List[Dict[str, torch.Tensor]]:
    """将 state_dict 分片"""
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Sharding state_dict (max shard size: {max_shard_size / (1024**3):.2f} GB)")
        print(f"{'='*80}")
    
    # 按照参数大小排序（大的参数优先，避免最后一个shard太小）
    sorted_items = sorted(
        state_dict.items(), 
        key=lambda x: get_tensor_size(x[1]), 
        reverse=True
    )
    
    shards = []
    current_shard = {}
    current_size = 0
    
    for key, tensor in tqdm(sorted_items, desc="Sharding", disable=not verbose):
        tensor_size = get_tensor_size(tensor)
        
        # 如果单个 tensor 超过 max_shard_size，单独放在一个 shard 中
        if tensor_size > max_shard_size:
            if current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0
            
            shards.append({key: tensor})
            if verbose:
                print(f"  Large tensor '{key}' ({tensor_size / (1024**3):.2f} GB) in separate shard")
            continue
        
        # 如果加入当前 tensor 会超过 max_shard_size，创建新的 shard
        if current_size + tensor_size > max_shard_size and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        
        current_shard[key] = tensor
        current_size += tensor_size
    
    # 添加最后一个 shard
    if current_shard:
        shards.append(current_shard)
    
    if verbose:
        print(f"\nCreated {len(shards)} shards:")
        for i, shard in enumerate(shards):
            shard_size = sum(get_tensor_size(v) for v in shard.values())
            print(f"  Shard {i+1}: {len(shard)} tensors, {shard_size / (1024**3):.2f} GB")
    
    return shards


def save_sharded_safetensors(
    shards: List[Dict[str, torch.Tensor]], 
    output_dir: Path,
    model_name: str = "model",
    verbose: bool = True
):
    """保存分片的 safetensors 文件"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Saving safetensors to: {output_dir}")
        print(f"{'='*80}")
    
    # 保存每个 shard
    shard_files = []
    weight_map = {}
    
    for i, shard in enumerate(tqdm(shards, desc="Saving shards", disable=not verbose)):
        if len(shards) == 1:
            shard_filename = f"{model_name}.safetensors"
        else:
            shard_filename = f"{model_name}-{i+1:05d}-of-{len(shards):05d}.safetensors"
        
        shard_path = output_dir / shard_filename
        
        # 保存 safetensors（共享内存问题已在全局处理）
        save_file(shard, str(shard_path))
        
        shard_files.append(shard_filename)
        
        # 记录每个参数在哪个文件中
        for key in shard.keys():
            weight_map[key] = shard_filename
        
        if verbose:
            shard_size = shard_path.stat().st_size / (1024**3)
            print(f"  Saved {shard_filename} ({shard_size:.2f} GB)")
    
    # 保存 index.json（HuggingFace 格式）
    if len(shards) > 1:
        index = {
            "metadata": {
                "total_size": sum(
                    sum(get_tensor_size(v) for v in shard.values()) 
                    for shard in shards
                )
            },
            "weight_map": weight_map
        }
        
        index_path = output_dir / f"{model_name}.safetensors.index.json"
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        if verbose:
            print(f"\n  Saved index: {index_path}")
    
    return shard_files


def save_metadata(
    output_dir: Path,
    original_checkpoint: str,
    total_params: int,
    total_size: int,
    num_shards: int
):
    """保存转换的元数据"""
    
    metadata = {
        "source": "DeepSpeed checkpoint",
        "original_checkpoint": str(original_checkpoint),
        "conversion_format": "safetensors",
        "total_parameters": total_params,
        "total_size_bytes": total_size,
        "num_shards": num_shards,
    }
    
    metadata_path = output_dir / "conversion_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert DeepSpeed checkpoint to HuggingFace safetensors format")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Path to DeepSpeed checkpoint (mp_rank_00_model_states.pt)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Output directory for safetensors files"
    )
    parser.add_argument(
        "--max_shard_size", 
        type=str, 
        default="5GB",
        help="Maximum size per shard (e.g., '5GB', '2GB', '500MB')"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="RoT",
        help="Model name prefix for output files"
    )
    parser.add_argument(
        "--no_clean_keys", 
        action="store_true",
        help="Do not clean state_dict keys (keep original prefixes)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        default=True,
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    # 验证输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")
    
    output_dir = Path(args.output)
    max_shard_size = convert_size_to_bytes(args.max_shard_size)
    
    print(f"\n{'='*80}")
    print(f"DeepSpeed to SafeTensors Conversion")
    print(f"{'='*80}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Max shard size: {args.max_shard_size}")
    print(f"{'='*80}")
    
    # 1. 加载 checkpoint
    state_dict = load_deepspeed_checkpoint(str(input_path), verbose=args.verbose)
    
    # 2. 清理 keys（可选）
    if not args.no_clean_keys:
        state_dict = clean_state_dict_keys(state_dict, verbose=args.verbose)
    
    # 3. 处理共享内存的 tensor（必须在分片之前）
    state_dict = deduplicate_shared_tensors_global(state_dict, verbose=args.verbose)
    
    # 4. 分片
    shards = shard_state_dict(state_dict, max_shard_size, verbose=args.verbose)
    
    # 5. 保存
    shard_files = save_sharded_safetensors(
        shards, 
        output_dir,
        model_name=args.model_name,
        verbose=args.verbose
    )
    
    # 6. 保存元数据
    total_size = sum(get_tensor_size(v) for v in state_dict.values())
    total_params = sum(v.numel() for v in state_dict.values())
    
    save_metadata(
        output_dir,
        original_checkpoint=str(input_path),
        total_params=total_params,
        total_size=total_size,
        num_shards=len(shards)
    )
    
    print(f"\n{'='*80}")
    print(f"✓ Conversion complete!")
    print(f"{'='*80}")
    print(f"Saved {len(shards)} shard(s) to: {output_dir}")
    print(f"Total parameters: {total_params:,}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
