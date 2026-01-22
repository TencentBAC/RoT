"""
统一评估脚本

支持两种模式：
1. 评估模式：计算准确率和压缩统计
2. 生成模式：只生成结果，保存为 JSONL 文件
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from tqdm import tqdm
import torch
import re

sys.path.append(str(Path(__file__).parent.parent))
from models.cot_compressor import CoTCompressor
from models.cot_compressor_v2 import CoTCompressorV2


def load_model(
    checkpoint_path: str,
    config: Dict[str, Any],
    model_type: str = "v2",
    verbose: bool = True,
    stage1_checkpoint: Optional[str] = None
) -> torch.nn.Module:
    """Load model weights from the Stage 2 DeepSpeed checkpoint layout."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"\nUsing device: {device}")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage 2 checkpoint path not found: {checkpoint_path}")

    if model_type != "v2":
        raise ValueError("This loader currently supports only model_type='v2'.")

    enable_lora = config["training"].get("enable_lora", True)
    full_finetuning = config["training"].get("full_finetuning", False)
    lora_r = config["training"].get("lora_r", 16)
    lora_alpha = config["training"].get("lora_alpha", 32)
    lora_dropout = config["training"].get("lora_dropout", 0.05)
    lora_target_modules = config["training"].get("lora_target_modules", None)

    if stage1_checkpoint is None:
        stage1_checkpoint = config.get("logging", {}).get("stage1_checkpoint", None)
    if stage1_checkpoint is None:
        raise ValueError(
            "Stage 1 checkpoint path must be provided for Stage 2 evaluation.\n"
            "Special token embeddings (special_tokens.bin) are saved in Stage 1 training directory.\n"
            "Please provide --stage1_checkpoint argument or set it in config file."
        )

    stage1_path = Path(stage1_checkpoint)
    if not stage1_path.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint path not found: {stage1_path}")

    if verbose:
        print("\n" + "=" * 80)
        print("Loading Stage 2 checkpoint with Stage 1 projection head")
        print("=" * 80)
        print(f"Stage 2 checkpoint dir: {checkpoint_path}")
        print(f"Stage 1 checkpoint dir: {stage1_path}")
        if full_finetuning:
            print(f"Training mode: Full Fine-tuning (all parameters)")
        elif enable_lora:
            print(f"Training mode: LoRA (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})")
        else:
            print(f"Training mode: lm_head-only (legacy)")

    # 🔧 修复：stage2 checkpoint 需要设置 stage2_mode=True 以正确初始化模型
    model = CoTCompressorV2(
        ocr_model_path=config["ocr_model"]["model_path"],
        llm_model_path=config["llm_model"].get("model_path", "/apdcephfs_zwfy/share_1355410/hunyuan/wyattyfwang/ckpt/Qwen3-VL-4B-Thinking"),
        image_size=config["rendering"]["image_size"],
        font_size=config["rendering"]["font_size"],
        device=device,
        freeze_vision=True,
        use_projection_head=True,
        projection_hidden_dim=config["training"].get("projection_hidden_dim", 2048),
        enable_lora=enable_lora,
        full_finetuning=full_finetuning,  # 新增：全参数微调支持
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        use_custom_llm=config["llm_model"].get("use_custom_llm", False),
        loss_type=config["training"].get("loss_type", "mse_only"),
        stage2_mode=True,  # 🔧 修复：评估 stage2 checkpoint 时设置为 True
        freeze_projection_head=True,  # projection_head 已在 stage1 训练好
        include_vision_loss=False,  # 评估时不需要 vision loss
        include_img_end_loss=False,  # 评估时不需要 img_end loss
    )

    # ====================================================================
    # 完全对应 train.py 的保存逻辑来加载检查点
    # ====================================================================
    
    # [Step 1/3] 从 Stage 1 checkpoint 根目录加载 projection_head.bin
    # 对应 train.py (1055): torch.save(projection_head.state_dict(), output_dir / "projection_head.bin")
    if verbose:
        print(f"\n[Step 1/3] Loading projection_head from Stage 1 checkpoint")
    
    projection_file = stage1_path / "projection_head.bin"
    if not projection_file.exists():
        raise FileNotFoundError(
            f"projection_head.bin not found in Stage 1 checkpoint: {projection_file}\n"
            f"Stage 1 training should save projection_head.bin in the root directory."
        )
    
    try:
        if verbose:
            print(f"  File: {projection_file}")
        projection_state = torch.load(projection_file, map_location=device)
        model.projection_head.load_state_dict(projection_state)
        if verbose:
            param_count = sum(p.numel() for p in model.projection_head.parameters())
            size_mb = projection_file.stat().st_size / (1024**2)
            print(f"  ✓ Loaded projection_head ({param_count:,} parameters, {size_mb:.2f} MB)")
    except Exception as e:
        raise RuntimeError(f"Failed to load projection_head from {projection_file}: {e}")
    
    # [Step 2/3] 从 Stage 1 checkpoint 根目录加载 special_tokens.bin
    # 对应 train.py (1090): torch.save(special_tokens_state, output_dir / "special_tokens.bin")
    if verbose:
        print(f"\n[Step 2/3] Loading special token embeddings from Stage 1 checkpoint")
    
    special_tokens_file = stage1_path / "special_tokens.bin"
    if not special_tokens_file.exists():
        raise FileNotFoundError(
            f"special_tokens.bin not found in Stage 1 checkpoint: {special_tokens_file}\n"
            f"Stage 1 training should save special_tokens.bin in the root directory.\n"
            f"Without special tokens, evaluation will produce incorrect results!"
        )
    
    try:
        if verbose:
            print(f"  File: {special_tokens_file}")
        special_tokens_state = torch.load(special_tokens_file, map_location=device)
        
        # 获取 embedding table
        embed_table = model.language_model.get_input_embeddings()
        
        # 加载 <img_begin> embedding（到 embedding table 中）
        # 对应 train.py (1072-1073): embed_table.weight[img_begin_token_id]
        if 'img_begin_emb' in special_tokens_state and hasattr(model, 'img_begin_token_id'):
            embed_table.weight.data[model.img_begin_token_id].copy_(special_tokens_state['img_begin_emb'])
            if verbose:
                norm = embed_table.weight.data[model.img_begin_token_id].norm().item()
                print(f"    ✓ Loaded <img_begin> embedding (norm={norm:.4f})")
        else:
            raise ValueError("<img_begin> embedding not found in special_tokens.bin")
        
        # 加载 <img_end> embedding（到 embedding table 中）
        # 对应 train.py (1079-1080): embed_table.weight[img_end_token_id]
        if 'img_end_emb' in special_tokens_state and hasattr(model, 'img_end_token_id'):
            embed_table.weight.data[model.img_end_token_id].copy_(special_tokens_state['img_end_emb'])
            if verbose:
                norm = embed_table.weight.data[model.img_end_token_id].norm().item()
                print(f"    ✓ Loaded <img_end> embedding (norm={norm:.4f})")
        else:
            raise ValueError("<img_end> embedding not found in special_tokens.bin")
        
        # 验证 token IDs 是否匹配
        if 'img_begin_token_id' in special_tokens_state:
            if special_tokens_state['img_begin_token_id'] != model.img_begin_token_id:
                if verbose:
                    print(f"    ⚠️  Warning: img_begin_token_id mismatch!")
                    print(f"       Checkpoint: {special_tokens_state['img_begin_token_id']}")
                    print(f"       Current model: {model.img_begin_token_id}")
        
        if 'img_end_token_id' in special_tokens_state:
            if special_tokens_state['img_end_token_id'] != model.img_end_token_id:
                if verbose:
                    print(f"    ⚠️  Warning: img_end_token_id mismatch!")
                    print(f"       Checkpoint: {special_tokens_state['img_end_token_id']}")
                    print(f"       Current model: {model.img_end_token_id}")
        
        if verbose:
            size_mb = special_tokens_file.stat().st_size / (1024**2)
            print(f"  ✓ Special token embeddings loaded ({size_mb:.4f} MB)")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load special tokens from {special_tokens_file}: {e}")
    
    # [Step 3/3] 从 Stage 2 checkpoint 的 DeepSpeed 目录加载模型权重
    # 对应 train.py (1136): self.model_engine.save_checkpoint(output_dir)
    # DeepSpeed 保存格式: {output_dir}/global_step_XXX/mp_rank_00_model_states.pt
    if verbose:
        print(f"\n[Step 3/3] Loading model weights from Stage 2 DeepSpeed checkpoint")
    
    # 查找最新的 global_step_* 目录
    ds_dirs = [d for d in checkpoint_path.glob("global_step*") if d.is_dir()]
    if not ds_dirs:
        raise ValueError(f"No global_step_* directories found in {checkpoint_path}")
    
    # 优先使用 latest 文件指定的目录
    latest_file = checkpoint_path / "latest"
    if latest_file.exists():
        try:
            tag = latest_file.read_text().strip()
            latest_step_dir = checkpoint_path / tag
            if not latest_step_dir.exists():
                raise FileNotFoundError(f"Latest file points to non-existent directory: {latest_step_dir}")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Failed to read latest file, using newest global_step: {e}")
            latest_step_dir = max(ds_dirs, key=lambda x: int(x.name.replace("global_step", "")))
    else:
        # 使用最新的 global_step 目录
        try:
            latest_step_dir = max(ds_dirs, key=lambda x: int(x.name.replace("global_step", "")))
        except ValueError:
            latest_step_dir = ds_dirs[-1]
    
    model_states_file = latest_step_dir / "mp_rank_00_model_states.pt"
    if not model_states_file.exists():
        raise FileNotFoundError(f"Model states file not found: {model_states_file}")

    if verbose:
        print(f"  Directory: {latest_step_dir}")
        print(f"  File: {model_states_file}")

    # 加载 DeepSpeed checkpoint
    # 对应 train.py 中 DeepSpeed 保存的格式：mp_rank_00_model_states.pt
    try:
        checkpoint_state = torch.load(model_states_file, map_location=device, weights_only=False)
        
        # 提取 state_dict（DeepSpeed 保存的格式可能有多种嵌套）
        state_dict = None
        for key in ["module", "model_state_dict", "model", "state_dict"]:
            if key in checkpoint_state:
                state_dict = checkpoint_state[key]
                break
        if state_dict is None and isinstance(checkpoint_state, dict):
            state_dict = checkpoint_state
        if state_dict is None:
            raise ValueError("Could not find model state dict in checkpoint")

        if verbose:
            print(f"  Total parameters in checkpoint: {len(state_dict)}")

        # 根据训练模式选择加载策略
        if not (hasattr(model, 'vision_encoder') and hasattr(model.vision_encoder, 'mllm_model')):
            raise ValueError("Expected model.vision_encoder.mllm_model to be available")
        
        mllm_model = model.vision_encoder.mllm_model
        
        # 清理 checkpoint 的 key 前缀
        clean_state: Dict[str, torch.Tensor] = {}
        
        for key, value in state_dict.items():
            clean_key = key
            
            # 移除训练时的前缀（DeepSpeed 或 DDP 包装产生的）
            for prefix in ["module.", "model.", "vision_encoder.mllm_model."]:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    break
            
            # 跳过不属于 mllm_model 的组件
            if any(component in key for component in [
                "text_renderer",      # 文本渲染器（不需要）
                "vision_encoder.ocr", # OCR 模型（已在初始化时加载）
                "projection_head",    # 投影头（已在 Step 1 加载）
                "vision_encoder.clip",    # CLIP 模型（不需要）
                "vision_encoder.sam",     # SAM 模型（不需要）
                "vision_encoder.sam_model",
                "vision_encoder.clip_model",
            ]):
                continue
            
            clean_state[clean_key] = value
        
        if verbose:
            print(f"  Cleaned {len(clean_state)} parameters")
        
        # 🔧 修复：统一checkpoint keys的格式
        # 问题：训练时保存的checkpoint中，keys的层级不一致
        # 发现三种格式：
        #   格式1: base_model.model.model.xxx (正确格式)
        #   格式2: base_model.model.xxx (缺少中间的 model.)
        #   格式3: language_model.xxx 或 llm_lm_head.xxx (完全缺少 base_model.model.model.)
        # 需要将格式2和格式3统一为格式1
        normalized_state: Dict[str, torch.Tensor] = {}
        keys_normalized = 0
        
        for key, value in clean_state.items():
            normalized_key = key
            
            # 情况1: 完全没有 base_model.model. 前缀
            if not key.startswith("base_model.model."):
                # 检查是否需要添加完整前缀
                needs_full_prefix = any(key.startswith(pattern) for pattern in [
                    "language_model.",
                    "visual.",
                    "llm_lm_head",
                ])
                
                if needs_full_prefix:
                    normalized_key = f"base_model.model.model.{key}"
                    keys_normalized += 1
            
            # 情况2: 有 base_model.model. 但缺少中间的 model.
            elif key.startswith("base_model.model.") and not key.startswith("base_model.model.model."):
                suffix = key[len("base_model.model."):]
                
                needs_model_layer = any(suffix.startswith(pattern) for pattern in [
                    "language_model.",
                    "visual.",
                    "llm_lm_head",
                ])
                
                if needs_model_layer:
                    normalized_key = f"base_model.model.model.{suffix}"
                    keys_normalized += 1
            
            normalized_state[normalized_key] = value
        
        clean_state = normalized_state
        
        if verbose and keys_normalized > 0:
            print(f"  🔧 Normalized {keys_normalized} keys (added 'base_model.model.model.' prefix)")
            print(f"  Total keys after normalization: {len(clean_state)}")
        
        # 策略1：全参数微调模式 - 直接加载到 language_model
        if full_finetuning:
            from peft import PeftModel
            
            if isinstance(mllm_model, PeftModel):
                raise ValueError(
                    "Model is initialized as PeftModel (LoRA enabled), but checkpoint was trained with full_finetuning.\n"
                    "Please set enable_lora=False and full_finetuning=True in config file."
                )
            
            if verbose:
                print(f"  Loading full fine-tuning weights")
                # 统计参数数量
                param_count = len(clean_state)
                print(f"    - Total parameters: {param_count}")
            
            # 直接加载到 mllm_model
            missing_keys, unexpected_keys = mllm_model.load_state_dict(clean_state, strict=False)
            
            if verbose:
                print(f"  ✓ Loaded checkpoint into language model")
                
                # 显示重要的 missing keys
                if missing_keys:
                    important_missing = [k for k in missing_keys]
                    if important_missing:
                        print(f"    ⚠️  Missing keys: {len(important_missing)}")
                        for k in important_missing[:10]:
                            print(f"       - {k}")
                        if len(important_missing) > 10:
                            print(f"       ... and {len(important_missing) - 10} more")
                
                # 显示 unexpected keys
                if unexpected_keys:
                    print(f"    ⚠️  Unexpected keys: {len(unexpected_keys)}")
                    for k in unexpected_keys[:10]:
                        print(f"       - {k}")
                    if len(unexpected_keys) > 10:
                        print(f"       ... and {len(unexpected_keys) - 10} more")
        
        # 策略2：LoRA 微调模式 - 加载到 PeftModel
        else:
            from peft import PeftModel
            
            if not isinstance(mllm_model, PeftModel):
                raise ValueError(
                    "Model is not a PeftModel, but Stage 2 training uses LoRA.\n"
                    "Ensure stage2_mode=True and enable_lora=True are set during model initialization."
                )
            
            if verbose:
                print(f"  Model is PeftModel (LoRA enabled)")
                # 统计 base model 和 LoRA 参数数量
                base_params = sum(1 for k in clean_state.keys() if "lora" not in k.lower())
                lora_params = sum(1 for k in clean_state.keys() if "lora" in k.lower())
                print(f"    - Base model parameters: {base_params}")
                print(f"    - LoRA parameters: {lora_params}")
            
            # 加载到 PeftModel
            # PeftModel 内部结构: base_model.model.xxx (base weights) + base_model.model.xxx.lora_xxx (LoRA weights)
            missing_keys, unexpected_keys = mllm_model.load_state_dict(clean_state, strict=False)
            
            if verbose:
                print(f"  ✓ Loaded checkpoint into PeftModel")
                
                # 显示重要的 missing keys（排除预期的 base_model 前缀）
                if missing_keys:
                    important_missing = [k for k in missing_keys if not k.startswith("base_model.model.")]
                    if important_missing:
                        print(f"    ⚠️  Missing keys: {len(important_missing)}")
                        for k in important_missing[:10]:
                            print(f"       - {k}")
                        if len(important_missing) > 10:
                            print(f"       ... and {len(important_missing) - 10} more")
                
                # 显示 unexpected keys
                if unexpected_keys:
                    print(f"    ⚠️  Unexpected keys: {len(unexpected_keys)}")
                    for k in unexpected_keys[:10]:
                        print(f"       - {k}")
                    if len(unexpected_keys) > 10:
                        print(f"       ... and {len(unexpected_keys) - 10} more")
            
            # 验证 LoRA 权重是否成功加载
            lora_loaded = False
            for name, param in mllm_model.named_parameters():
                if "lora" in name.lower():
                    lora_loaded = True
                    break
            
            if not lora_loaded:
                raise ValueError(
                    "Failed to load LoRA weights from checkpoint!\n"
                    "No LoRA parameters found in the loaded model."
                )
            
            if verbose:
                print(f"  ✓ LoRA weights successfully loaded and verified")
            
    except Exception as e:
        if verbose:
            print(f"\n❌ Error loading Stage 2 checkpoint: {e}")
            import traceback
            traceback.print_exc()
        raise

    if verbose:
        print("\n" + "=" * 80)
        print("Checkpoint Loading Complete")
        print("=" * 80)
        print("All model components loaded successfully:")
        print("  ✓ projection_head (from Stage 1)")
        print("  ✓ special_tokens (from Stage 1)")
        if full_finetuning:
            print("  ✓ mllm_model weights (from Stage 2, full fine-tuning)")
        else:
            print("  ✓ mllm_model weights (from Stage 2, including LoRA)")
        print("=" * 80)
    
    # 将模型移到目标设备并设置为评估模式
    model.to(device)
    model.eval()
    
    if verbose:
        print("\n✓ Model ready for evaluation")
    
    return model


def extract_answer(text: str) -> str:
    """
    从生成的文本中提取答案

    Args:
        text: 生成的文本

    Returns:
        提取的答案
    """
    patterns = [
        r"####\s*(\d+\.?\d*)",              # GSM8K 格式
        r"答案是[：:]\s*(\d+\.?\d*)",        # 中文格式
        r"Therefore,?\s+.*?(\d+\.?\d*)",    # Therefore, ... 42
        r"Answer[：:]\s*(\d+\.?\d*)",        # Answer: 42
        r"\$(\d+\.?\d*)",                   # $42
        r"(\d+\.?\d*)\s*$",                 # 最后的数字
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

    return text.strip()


def compare_answers(pred: str, gt: str) -> bool:
    """
    比较两个答案是否一致，支持数字比较（如 8.0 == 8）

    Args:
        pred: 预测答案
        gt: 真实答案

    Returns:
        是否匹配
    """
    pred = pred.strip()
    gt = gt.strip()
    
    # 尝试转换为浮点数进行比较
    try:
        pred_num = float(pred)
        gt_num = float(gt)
        # 使用一个小阈值处理浮点数精度问题
        return abs(pred_num - gt_num) < 1e-9
    except ValueError:
        # 如果无法转换为数字，则进行严格字符串比较
        return pred == gt



def load_dataset_from_file(data_file: str) -> List[Dict[str, Any]]:
    """
    从 JSONL 文件加载数据集

    Args:
        data_file: 数据文件路径

    Returns:
        数据集列表
    """
    dataset = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset


def generate_results(
    model: torch.nn.Module,
    dataset: List[Dict[str, Any]],
    max_samples: Optional[int] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    num_vision_tokens: Optional[int] = None,
    stop_threshold: float = 0.02,
    compute_accuracy: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    生成结果（可选计算准确率）

    Args:
        model: CoT 压缩器模型
        dataset: 数据集
        max_samples: 最多评估多少样本（None = 全部）
        max_new_tokens: 最大生成 token 数
        temperature: 生成温度
        num_vision_tokens: vision token 数量上限（用于自适应停止）
        stop_threshold: 自适应停止阈值
        compute_accuracy: 是否计算准确率
        verbose: 是否显示进度条

    Returns:
        评估结果字典
    """
    model.eval()

    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    
    if verbose:
        print(f"\nProcessing {num_samples} samples...")
        iterator = tqdm(range(num_samples), desc="Generating")
    else:
        iterator = range(num_samples)

    correct = 0
    total = 0
    compression_stats = []
    results = []

    with torch.no_grad():
        for idx in iterator:
            sample = dataset[idx]
            question = sample["question"]
            cot = sample["cot"]
            ground_truth = sample["answer"]

            try:
                # 生成答案（使用自适应停止）
                # cot_text 仅用于 verbose 模式下的对比（可选）
                generated = model.generate(
                    question_text=question,
                    cot_text=cot if verbose else None,  # verbose 模式下传入 cot 用于对比
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    max_vision_tokens=num_vision_tokens if num_vision_tokens else 1024,
                    stop_threshold=stop_threshold,
                    verbose=False  # 避免每个样本都输出详细信息
                )

                # 如果需要计算准确率
                is_correct = None
                predicted_answer = None
                compression_ratio = None
                
                if compute_accuracy:
                    # 提取答案
                    predicted_answer = extract_answer(generated)
                    gt_extracted = extract_answer(ground_truth)
                    
                    # 判断正确性
                    is_correct = compare_answers(predicted_answer, gt_extracted)
                    
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    # 压缩统计
                    if cot and len(str(cot).strip()) > 0:
                        try:
                            stats = model.compute_compression_stats(cot)
                            compression_stats.append(stats)
                            compression_ratio = stats["compression_ratio"]
                        except Exception as e:
                            if verbose:
                                print(f"Warning: Failed to compute compression stats: {e}")
                            compression_ratio = None
                    else:
                        compression_ratio = None

                # 保存结果
                result_entry = {
                    "id": idx,
                    "question": question,
                    "cot": cot,
                    "ground_truth": ground_truth,
                    "prediction": generated,
                }
                
                if compute_accuracy:
                    result_entry.update({
                        "predicted_answer": predicted_answer,
                        "is_correct": is_correct,
                        "compression_ratio": compression_ratio,
                    })
                
                results.append(result_entry)

            except Exception as e:
                if verbose:
                    print(f"\nWarning: Failed to process sample {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                results.append({
                    "id": idx,
                    "question": question,
                    "cot": cot,
                    "ground_truth": ground_truth,
                    "prediction": f"ERROR: {str(e)}",
                })
                continue

    # 准备返回结果
    output = {
        "results": results,
        "total_samples": len(results),
    }
    
    if compute_accuracy:
        accuracy = correct / total if total > 0 else 0.0
        avg_compression_ratio = sum(s["compression_ratio"] for s in compression_stats) / len(compression_stats) if compression_stats else 0.0
        avg_saved_percentage = sum(s["saved_percentage"] for s in compression_stats) / len(compression_stats) if compression_stats else 0.0
        
        output.update({
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "compression": {
                "avg_compression_ratio": avg_compression_ratio,
                "avg_saved_percentage": avg_saved_percentage,
            }
        })

    return output


def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation Script for CoT Compressor")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--stage1_checkpoint", type=str, default=None, 
                        help="Stage 1 checkpoint path (required for Stage 2 evaluation)")
    parser.add_argument("--data_file", type=str, default=None, help="Data file path (overrides dataset/split)")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset name")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Data split")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--num_vision_tokens", type=int, default=None, 
                        help="Max vision tokens limit (default: 1024, uses adaptive stop)")
    parser.add_argument("--stop_threshold", type=float, default=0.02,
                        help="Stop threshold for adaptive generation (default: 0.02)")
    parser.add_argument("--model_type", type=str, default="v2", choices=["v1", "v2"], help="Model type")
    parser.add_argument("--mode", type=str, default="evaluate", choices=["evaluate", "generate"], 
                        help="Mode: 'evaluate' (compute accuracy) or 'generate' (only generate results)")
    parser.add_argument("--output_file", type=str, default=None, help="Output results file")
    parser.add_argument("--output_format", type=str, default="json", choices=["json", "jsonl"], 
                        help="Output format")

    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print(f"CoT Compressor - {args.mode.capitalize()} Mode")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    if args.stage1_checkpoint:
        print(f"Stage 1 checkpoint: {args.stage1_checkpoint}")
    if args.data_file:
        print(f"Data file: {args.data_file}")
    else:
        print(f"Dataset: {args.dataset}")
        print(f"Split: {args.split}")
    print("=" * 80)

    # 加载模型
    model = load_model(
        args.checkpoint, 
        config, 
        model_type=args.model_type, 
        verbose=True,
        stage1_checkpoint=args.stage1_checkpoint
    )

    # 加载数据
    if args.data_file:
        data_file = args.data_file
    else:
        data_file = Path(config["data"]["processed_dir"]) / f"{args.dataset}_{args.split}_processed.jsonl"
        if not Path(data_file).exists():
            # 尝试不带 _processed 后缀
            data_file = Path(config["data"]["processed_dir"]) / f"{args.dataset}_{args.split}.jsonl"

    if not Path(data_file).exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    print(f"\nLoading data from: {data_file}")
    dataset = load_dataset_from_file(str(data_file))
    print(f"Loaded {len(dataset)} samples")

    # 生成/评估
    compute_accuracy = (args.mode == "evaluate")

    print("!" * 80)
    print("args.max_new_tokens:", args.max_new_tokens)
    print("args.temperature:", args.temperature)
    print("args.num_vision_tokens:", args.num_vision_tokens)
    print("args.stop_threshold:", args.stop_threshold)
    print("!" * 80)
    
    eval_results = generate_results(
        model=model,
        dataset=dataset,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_vision_tokens=args.num_vision_tokens,
        stop_threshold=args.stop_threshold,
        compute_accuracy=compute_accuracy,
        verbose=True
    )

    # 打印结果
    if compute_accuracy:
        print("\n" + "=" * 80)
        print("Evaluation Results")
        print("=" * 80)
        print(f"Accuracy: {eval_results['accuracy']*100:.2f}%")
        print(f"Correct: {eval_results['correct']}/{eval_results['total']}")
        print(f"\nCompression:")
        print(f"  Avg compression ratio: {eval_results['compression']['avg_compression_ratio']:.2f}:1")
        print(f"  Avg tokens saved: {eval_results['compression']['avg_saved_percentage']:.1f}%")
        print("=" * 80)
        
        # 打印前几个样本
        print("\nFirst 3 examples:")
        for i, example in enumerate(eval_results['results'][:3]):
            print(f"\n--- Example {i+1} ---")
            print(f"Question: {example['question'][:100]}...")
            print(f"Ground truth: {example['ground_truth']}")
            print(f"Predicted: {example.get('predicted_answer', 'N/A')}")
            print(f"Correct: {'✓' if example.get('is_correct') else '✗'}")
    else:
        print(f"\n✓ Generated {eval_results['total_samples']} results")
        
        # 打印前几个样本
        print("\nFirst 3 examples:")
        for i, example in enumerate(eval_results['results'][:3]):
            print(f"\n--- Example {i+1} ---")
            print(f"Question: {example['question'][:100]}...")
            print(f"Ground truth: {example['ground_truth']}")
            print(f"Prediction: {example['prediction'][:200]}...")

    # 保存结果
    if args.output_file is None:
        checkpoint_name = Path(args.checkpoint).stem
        if args.mode == "evaluate":
            output_file = f"outputs/results/{args.dataset}_{args.split}_{checkpoint_name}_eval.{args.output_format}"
        else:
            output_file = f"outputs/results/{args.dataset}_{args.split}_{checkpoint_name}_results.{args.output_format}"
    else:
        output_file = args.output_file

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存为指定格式
    if args.output_format == "jsonl":
        # JSONL 格式：每行一个 JSON 对象
        with open(output_path, "w", encoding="utf-8") as f:
            for result in eval_results['results']:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    else:
        # JSON 格式：完整的结构化对象
        # 简化输出（只保留前10个样本的详细结果）
        save_data = {
            "total_samples": eval_results['total_samples'],
            "sample_results": eval_results['results'][:10],
        }
        if compute_accuracy:
            save_data.update({
                "accuracy": eval_results['accuracy'],
                "correct": eval_results['correct'],
                "total": eval_results['total'],
                "compression": eval_results['compression'],
            })
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved results to {output_path}")


if __name__ == "__main__":
    main()
