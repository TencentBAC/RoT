"""
训练脚本

训练 CoT 压缩器模型
"""

import os
import sys
import json
import yaml
import argparse
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import PreTrainedModel
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import shutil

# DeepSpeed 支持
try:
    import deepspeed

    DEEPSPEED_AVAILABLE = True

    # 抑制 DeepSpeed 的 INFO 级别日志（避免 MPI 检测等警告）
    import logging

    # 设置 deepspeed 相关 logger 的级别
    deepspeed_logger = logging.getLogger("deepspeed")
    deepspeed_logger.setLevel(logging.WARNING)

    # 也抑制 deepspeed.comm 的日志（这是 MPI 检测日志的来源）
    comm_logger = logging.getLogger("deepspeed.comm")
    comm_logger.setLevel(logging.ERROR)  # 设置为 ERROR 以完全隐藏 INFO 消息

    # 抑制 root logger 中可能的路由
    logging.getLogger("deepspeed.utils").setLevel(logging.WARNING)
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("Warning: DeepSpeed not available. Install with: pip install deepspeed")

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from models.cot_compressor import CoTCompressor
from models.cot_compressor_v2 import CoTCompressorV2


class CoTDataset(Dataset):
    """CoT 数据集"""

    def __init__(self, data_file: str, tokenizer=None, silent: bool = False):
        """
        Args:
            data_file: 预处理后的数据文件路径
            tokenizer: tokenizer对象，用于获取eos_token_id和应用chat template
            silent: 是否静默模式（多卡训练时避免重复输出）
        """
        self.data = []
        self.tokenizer = tokenizer
        
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

        if not silent:
            print(f"✓ Loaded {len(self.data)} samples from {data_file}")
            if tokenizer is not None:
                eos_token = tokenizer.eos_token if hasattr(tokenizer, 'eos_token') else None
                print(f"  Using Qwen3-VL chat template with EOS token: {eos_token}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        question = item["question"]
        cot = item["cot"]
        answer = item["answer"]
        
        # 应用Qwen3-VL的chat template格式
        # 格式：<|im_start|>system\n系统提示<|im_end|>\n<|im_start|>user\n问题<|im_end|>\n<|im_start|>assistant\n回答<|im_end|>
        # "You are a helpful assistant. The final output format is as follows: Answer: <answer>. <|im_end|>\n"
        
        # 构建格式化的question（包含system和user部分）
        '''
        formatted_question = (
            "<|im_start|>system\n"
            "You are a helpful assistant. The final output format is as follows: Answer: <answer>. <|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        '''
        formatted_question = question
        # 构建格式化的CoT（思维链推理过程）
        formatted_cot = cot
        # 构建格式化的answer，在末尾添加<|im_end|>
        # <|im_end|>在Qwen3-VL中会被tokenizer转换为eos_token_id
        # formatted_answer = "### " + answer + "<|im_end|>"
        formatted_answer = "### " + answer + " "
        return {
            "question": formatted_question,
            "cot": formatted_cot,
            "answer": formatted_answer,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    批量数据整理
    由于每个样本的 CoT 长度不同，我们不做 padding，直接返回列表
    """
    return {
        "questions": [item["question"] for item in batch],
        "cots": [item["cot"] for item in batch],
        "answers": [item["answer"] for item in batch],
    }


class Trainer:
    """训练器"""

    def __init__(self, config: Dict[str, Any], deepspeed_config_path: Optional[str] = None):
        self.config = config
        self.deepspeed_config_path = deepspeed_config_path

        # 初始化分布式训练
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.use_deepspeed = (
            DEEPSPEED_AVAILABLE
            and self.deepspeed_config_path is not None
            and os.path.exists(self.deepspeed_config_path)
            and self.local_rank >= 0
        )

        if self.use_deepspeed:
            # DeepSpeed 会自动初始化分布式环境（通过 deepspeed.initialize）
            # DeepSpeed launcher 通常会自动设置 MASTER_ADDR 和 MASTER_PORT
            # 如果没有设置，我们才设置默认值
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = "localhost"
            
            # 自动选择可用端口（仅当 MASTER_PORT 未设置时）
            # 注意：DeepSpeed launcher 应该会自动设置 MASTER_PORT
            # 如果没有设置，我们使用基于 PID 和时间的端口选择
            if "MASTER_PORT" not in os.environ:
                import socket
                import os as os_module
                
                # 所有进程需要使用相同的端口，所以使用一个确定性的方法
                # 基于进程组的 PID（主进程的 PID）和时间戳来选择端口
                base_port = 29500
                # 使用主进程的 PID（如果可用）或者当前进程 PID
                # 对于 DeepSpeed，所有子进程的父进程 PID 相同
                try:
                    # 尝试获取父进程 PID（在 DeepSpeed 中，所有进程共享同一个父进程）
                    parent_pid = os_module.getppid()
                    pid = parent_pid
                except:
                    pid = os_module.getpid()
                
                # 使用 PID 和时间戳生成一个唯一的起始端口
                import time
                time_hash = int(time.time()) % 1000
                port_offset = (pid + time_hash) % 400  # 限制在 29500-29899 范围内
                start_port = base_port + port_offset
                
                # 尝试从起始端口开始查找可用端口（最多尝试 50 个）
                selected_port = None
                for i in range(50):
                    port = start_port + i
                    if port > 29999:  # 限制最大端口
                        port = base_port + (port - 29999 - 1)
                    
                    try:
                        # 尝试绑定端口来检查是否可用（只检查，不占用）
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                            s.bind(("localhost", port))
                            selected_port = port
                            if self.local_rank == 0:
                                print(f"✓ Auto-selected MASTER_PORT={port} (from PID {pid})")
                            break
                    except (OSError, socket.error):
                        # 端口被占用，尝试下一个
                        continue
                
                # 如果找不到可用端口，使用基于 PID 的确定性端口（即使可能被占用）
                if selected_port is None:
                    selected_port = base_port + (pid % 400)
                    if self.local_rank == 0:
                        print(f"⚠️  Warning: Using deterministic port {selected_port} based on PID")
                
                # 设置环境变量（所有进程都需要）
                os.environ["MASTER_PORT"] = str(selected_port)

            # 设置标志告诉 DeepSpeed 我们使用了 DeepSpeed launcher
            # DeepSpeed 通过检查这些环境变量来判断是否使用了 launcher
            os.environ["DEEPSPEED_LAUNCHER"] = "1"

            # 检查并设置 NCCL 超时时间（如果未设置，设置一个较长的默认值）
            # DeepSpeed ZeRO Stage 2 的检查点保存需要较长时间，需要足够长的超时
            if "NCCL_TIMEOUT" not in os.environ:
                # 设置默认超时为 30 分钟（1800 秒）
                os.environ["NCCL_TIMEOUT"] = "1800"
                if self.local_rank == 0:
                    print("  Note: NCCL_TIMEOUT not set, using default 1800 seconds")
            else:
                nccl_timeout = int(os.environ.get("NCCL_TIMEOUT", "1800"))
                if nccl_timeout < 1800 and self.local_rank == 0:
                    print(f"  Warning: NCCL_TIMEOUT={nccl_timeout} may be too short for DeepSpeed checkpoint saving")
                    print(f"  Recommended: export NCCL_TIMEOUT=1800 or higher")

            # 但我们需要先设置设备（在 deepspeed.initialize 之前）
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.is_main_process = self.local_rank == 0

            # 重要：不要在 DeepSpeed 模式下手动初始化分布式环境
            # DeepSpeed 会在 deepspeed.initialize() 时自动初始化
        elif self.local_rank >= 0:
            # 使用标准的 PyTorch DDP
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.is_main_process = self.local_rank == 0
        else:
            # 单卡训练
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.is_main_process = True

        # 创建输出目录
        self.output_dir = Path(config["logging"]["checkpoint_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(config["logging"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 TensorBoard（只在主进程）
        if self.is_main_process:
            try:
                self.tensorboard_dir = self.log_dir / "tensorboard"
                self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
                print(f"✓ TensorBoard logging to: {self.tensorboard_dir}", flush=True)
                print(f"  Log directory: {self.log_dir.absolute()}", flush=True)
            except Exception as e:
                print(f"⚠️  Warning: Failed to initialize TensorBoard: {e}", flush=True)
                print(f"  Continuing without TensorBoard logging...", flush=True)
                self.writer = None
        else:
            self.writer = None

        # 初始化模型
        if self.is_main_process:
            print("\n" + "=" * 80)
            print("Initializing Model")
            print("=" * 80)

        # 对于 DeepSpeed，先初始化模型到 CPU，然后让 DeepSpeed 自动分配到 GPU
        # 对于非 DeepSpeed，直接放到指定设备
        model_device = "cpu" if self.use_deepspeed else self.device

        self.model = CoTCompressorV2(
            ocr_model_path=config["ocr_model"]["model_path"],
            llm_model_path=config["llm_model"]["model_path"],
            image_size=config["rendering"]["image_size"],
            font_size=config["rendering"]["font_size"],
            device=model_device,  # DeepSpeed 模式下先用 CPU，DeepSpeed 会自动分配到 GPU
            freeze_vision=config["training"].get("freeze_vision", True),
            use_projection_head=config["training"].get("use_projection_head", True),
            projection_hidden_dim=config["training"].get("projection_hidden_dim", 2048),
            enable_lora=config["training"].get("enable_lora", True),
            lora_r=config["training"].get("lora_r", 16),
            lora_alpha=config["training"].get("lora_alpha", 32),
            lora_dropout=config["training"].get("lora_dropout", 0.05),
            lora_target_modules=config["training"].get("lora_target_modules", None),
            # 损失权重
            use_uncertainty_weighting=config["training"].get("use_uncertainty_weighting", True),
            vision_loss_weight=config["training"].get("loss_weights", {}).get("vision_loss_weight", 1.0),
            lm_loss_weight=config["training"].get("loss_weights", {}).get("lm_loss_weight", 1.0),
            use_custom_llm=config["llm_model"].get("use_custom_llm", False),
            loss_type=config["training"].get("loss_type", "stable_similarity"),  # 传递损失类型
            # 第二阶段训练参数
            stage2_mode=config["training"].get("stage2_mode", False),
            train_lm_head_only=config["training"].get("train_lm_head_only", False),
            freeze_projection_head=config["training"].get("freeze_projection_head", False),
            include_img_end_loss=config["training"].get("include_img_end_loss", False),
            include_vision_loss=config["training"].get("include_vision_loss", True),
            full_finetuning=config["training"].get("full_finetuning", False),  # 新增：全参数微调
        )

        # 非 DeepSpeed 模式下才手动移动到设备
        if not self.use_deepspeed:
            self.model = self.model.to(self.device)

        # Loss权重
        self.vision_loss_weight = config["training"]["loss_weights"].get("vision_loss_weight", 1.0)
        self.lm_loss_weight = config["training"]["loss_weights"].get("lm_loss_weight", 1.0)

        if self.is_main_process:
            print(f"  Model config:")
            print(f"    - Use projection head: {config['training'].get('use_projection_head', True)}")
            print(f"    - Projection hidden dim: {config['training'].get('projection_hidden_dim', 2048)}")
            print(f"    - Freeze vision: {config['training'].get('freeze_vision', True)}")
            print(f"    - Enable LoRA: {config['training'].get('enable_lora', True)}")
            if config["training"].get("enable_lora", True):
                print(f"    - LoRA r: {config['training'].get('lora_r', 16)}")
                print(f"    - LoRA alpha: {config['training'].get('lora_alpha', 32)}")
                print(f"    - LoRA dropout: {config['training'].get('lora_dropout', 0.05)}")
            print(f"    - Vision loss weight: {self.vision_loss_weight}")
            print(f"    - LM loss weight: {self.lm_loss_weight}")

        # 获取tokenizer（需要在加载数据之前）
        self.tokenizer = self.model.tokenizer
        
        # 加载数据
        if self.is_main_process:
            print("\n" + "=" * 80, flush=True)
            print("Loading Data", flush=True)
            print("=" * 80, flush=True)
        train_file = Path(config["data"]["processed_dir"]) / f"{config['data']['dataset_name']}_train_processed.jsonl"
        # 非主进程使用静默模式，传入tokenizer以应用chat template
        self.train_dataset = CoTDataset(train_file, tokenizer=self.tokenizer, silent=not self.is_main_process)

        # 计算训练步数（需要先创建临时 DataLoader 或使用数据集长度）
        # 对于 DeepSpeed，我们需要先创建一个临时 DataLoader 来计算步数
        if self.use_deepspeed:
            # DeepSpeed 模式下稍后创建 DataLoader（需要 DistributedSampler）
            # 创建一个临时 DataLoader 来计算训练步数
            temp_loader = DataLoader(
                self.train_dataset,
                batch_size=config["training"]["batch_size"],
                shuffle=False,  # 临时，不需要采样
                collate_fn=collate_fn,
                num_workers=0,  # 临时，不需要多进程
            )
            num_training_steps = len(temp_loader) * config["training"]["num_epochs"]
            self.train_loader = None  # 稍后在 DeepSpeed 初始化后创建
        else:
            # 标准训练模式
            if self.local_rank >= 0:
                # 多卡 DDP 模式，使用 DistributedSampler
                from torch.utils.data.distributed import DistributedSampler

                train_sampler = DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.world_size,
                    rank=self.local_rank,
                    shuffle=True,
                )
                self.train_sampler = train_sampler  # 保存引用，用于在每个 epoch 调用 set_epoch
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=config["training"]["batch_size"],
                    sampler=train_sampler,
                    collate_fn=collate_fn,
                    num_workers=config["misc"]["num_workers"],
                    pin_memory=config["training"].get("dataloader_pin_memory", True),
                )
            else:
                # 单卡模式
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=config["training"]["batch_size"],
                    shuffle=True,
                    collate_fn=collate_fn,
                    num_workers=config["misc"]["num_workers"],
                    pin_memory=config["training"].get("dataloader_pin_memory", True),
                )
            num_training_steps = len(self.train_loader) * config["training"]["num_epochs"]

        # 初始化优化器和学习率调度器（DeepSpeed 会自动管理）
        if self.use_deepspeed:
            # 初始化 DeepSpeed engine
            if self.is_main_process:
                print(f"\n" + "=" * 80)
                print("Initializing DeepSpeed")
                print("=" * 80)
                print(f"  DeepSpeed config: {self.deepspeed_config_path}")
                print(f"  Local rank: {self.local_rank}")
                print(f"  World size: {self.world_size}")

            # 读取并更新 DeepSpeed 配置，设置实际的训练参数
            import json

            with open(self.deepspeed_config_path, "r") as f:
                ds_config = json.load(f)
            
            # 检查是否使用了 ZeRO Stage 3 + CPU Offload
            zero_config = ds_config.get("zero_optimization", {})
            zero_stage = zero_config.get("stage", 0)
            use_cpu_offload = (
                zero_config.get("offload_optimizer", {}).get("device", None) == "cpu"
                or zero_config.get("offload_param", {}).get("device", None) == "cpu"
            )
            
            # 如果使用 ZeRO Stage 3 + CPU Offload，不传递自定义 optimizer
            # 让 DeepSpeed 使用配置文件中定义的优化器
            use_custom_optimizer = not (zero_stage == 3 and use_cpu_offload)
            
            if self.is_main_process:
                print(f"  ZeRO Stage: {zero_stage}")
                print(f"  CPU Offload: {use_cpu_offload}")
                print(f"  Use custom optimizer: {use_custom_optimizer}")
            
            # 只有在不使用 ZeRO Stage 3 + CPU Offload 时才创建自定义优化器
            optimizer = None
            lr_scheduler = None
            
            if use_custom_optimizer:
                # 只优化可训练参数（对于LoRA或stage2训练很重要）
                model_parameters = [p for p in self.model.parameters() if p.requires_grad]
                if len(model_parameters) == 0:
                    raise ValueError(
                        "No trainable parameters found! Check your training configuration. "
                        "For stage2 training, ensure enable_lora=True or train_lm_head_only=True."
                    )
                optimizer = AdamW(
                    model_parameters,
                    lr=config["training"]["learning_rate"],
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0.01,
                )
                lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=config["training"]["warmup_steps"],
                    num_training_steps=num_training_steps,
                )
                if self.is_main_process:
                    print(f"  Created custom AdamW optimizer")
            else:
                # 验证可训练参数存在（即使不创建优化器）
                model_parameters = [p for p in self.model.parameters() if p.requires_grad]
                if len(model_parameters) == 0:
                    raise ValueError(
                        "No trainable parameters found! Check your training configuration. "
                        "For stage2 training, ensure enable_lora=True, full_finetuning=True, or train_lm_head_only=True."
                    )
                if self.is_main_process:
                    print(f"  Will use DeepSpeed optimizer from config (required for ZeRO Stage 3 + CPU Offload)")
                    print(f"  Trainable parameters: {len(model_parameters)}")

            # 设置实际的训练参数（而不是 "auto"）
            batch_size = config["training"]["batch_size"]
            gradient_accumulation_steps = config["training"].get("gradient_accumulation_steps", 1)
            learning_rate = config["training"]["learning_rate"]
            warmup_steps = config["training"].get("warmup_steps", 100)
            max_grad_norm = config["training"].get("max_grad_norm", 1.0)

            # 计算全局 batch size: per_gpu_batch_size * num_gpus * gradient_accumulation_steps
            train_batch_size = batch_size * self.world_size * gradient_accumulation_steps

            ds_config["train_batch_size"] = train_batch_size
            ds_config["train_micro_batch_size_per_gpu"] = batch_size
            ds_config["gradient_accumulation_steps"] = gradient_accumulation_steps
            ds_config["gradient_clipping"] = max_grad_norm

            # 设置优化器学习率
            if "optimizer" in ds_config and "params" in ds_config["optimizer"]:
                ds_config["optimizer"]["params"]["lr"] = learning_rate

            # 设置学习率调度器的预热步数
            if "scheduler" in ds_config and "params" in ds_config["scheduler"]:
                ds_config["scheduler"]["params"]["warmup_num_steps"] = warmup_steps
                ds_config["scheduler"]["params"]["warmup_min_lr"] = 0
                ds_config["scheduler"]["params"]["warmup_max_lr"] = learning_rate

            if self.is_main_process:
                print(f"  Train batch size: {train_batch_size}")
                print(f"  Micro batch size per GPU: {batch_size}")
                print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
                print(f"  Learning rate: {learning_rate}")

            # DeepSpeed 初始化：传入更新后的配置
            # DeepSpeed 会自动将模型分配到正确的 GPU
            # 临时抑制 INFO 级别的日志（包括 MPI 检测警告）
            import logging

            old_levels = {}
            for logger_name in ["deepspeed", "deepspeed.comm", "deepspeed.utils"]:
                logger = logging.getLogger(logger_name)
                old_levels[logger_name] = logger.level
                logger.setLevel(logging.ERROR)  # 临时设置为 ERROR 以完全抑制 INFO

            try:
                # 根据是否使用自定义优化器来决定传递参数
                init_kwargs = {
                    "model": self.model,
                    "config": ds_config,  # 传入配置字典而不是文件路径
                }
                
                # 只有在使用自定义优化器时才传递 optimizer 和 lr_scheduler
                if use_custom_optimizer:
                    init_kwargs["optimizer"] = optimizer
                    init_kwargs["lr_scheduler"] = lr_scheduler
                
                self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(**init_kwargs)
            finally:
                # 恢复日志级别
                for logger_name, old_level in old_levels.items():
                    logging.getLogger(logger_name).setLevel(old_level)

            # 重新创建 DataLoader（DeepSpeed 需要分布式采样）
            from torch.utils.data.distributed import DistributedSampler

            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True,
            )
            self.train_sampler = train_sampler  # 保存引用，用于在每个 epoch 调用 set_epoch

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=config["training"]["batch_size"],
                sampler=train_sampler,
                collate_fn=collate_fn,
                num_workers=config["misc"]["num_workers"],
                pin_memory=config["training"].get("dataloader_pin_memory", True),
            )

            # DeepSpeed 会自动设置模型到正确的设备
            # 获取实际模型（DeepSpeed 会包装模型）
            self.model = self.model_engine.module  # 获取实际模型

            # 确保模型的 device 属性正确更新
            if hasattr(self.model, "device"):
                # 从模型参数中获取实际设备
                model_device = next(self.model.parameters()).device
                self.model.device = str(model_device)

            # 确保所有子模块都在正确的设备上（包括参数和缓冲区）
            # DeepSpeed 可能只移动了部分模块，需要递归确保所有子模块都在正确设备
            model_device = next(self.model.parameters()).device

            def move_module_to_device(module, device):
                """递归地将模块及其所有子模块移动到指定设备"""
                try:
                    module.to(device)
                except Exception as e:
                    if self.is_main_process:
                        print(f"Warning: Could not move module {type(module).__name__} to {device}: {e}")

                # 确保所有参数都在正确设备上
                for name, param in module.named_parameters(recurse=False):
                    if param.device != device:
                        try:
                            param.data = param.data.to(device)
                        except Exception as e:
                            if self.is_main_process:
                                print(f"Warning: Could not move parameter {name} to {device}: {e}")

                # 确保所有注册缓冲区都在正确设备上（这对于 rotary_emb 等模块很重要）
                for name, buffer in module.named_buffers(recurse=False):
                    if buffer.device != device:
                        try:
                            buffer.data = buffer.data.to(device)
                        except Exception as e:
                            if self.is_main_process:
                                print(f"Warning: Could not move buffer {name} to {device}: {e}")

                # 递归处理所有子模块
                for child_name, child_module in module.named_children():
                    move_module_to_device(child_module, device)

            # 移动 vision_encoder 及其所有子模块
            if hasattr(self.model, "vision_encoder"):
                move_module_to_device(self.model.vision_encoder, model_device)
                # 特别处理 vision_encoder 的子模块
                if hasattr(self.model.vision_encoder, "mllm_model"):
                    move_module_to_device(self.model.vision_encoder.mllm_model, model_device)

            # 移动 language_model 及其所有子模块（包括 rotary_emb 等）
            # 这是最重要的，因为 rotary_emb 的缓冲区必须在正确设备上
            if hasattr(self.model, "language_model"):
                move_module_to_device(self.model.language_model, model_device)
                # 特别检查 rotary_emb 模块（Qwen3-VL 可能在多个地方有 rotary_emb）
                try:
                    # 遍历所有子模块，找到所有包含 rotary 或 rope 的模块
                    for name, child in self.model.language_model.named_modules():
                        if "rotary" in name.lower() or "rope" in name.lower():
                            move_module_to_device(child, model_device)
                            if self.is_main_process:
                                print(f"  Found and moved {name} to {model_device}")
                except Exception as e:
                    if self.is_main_process:
                        print(f"Warning: Could not find/move rotary_emb modules: {e}")

            # 移动 projection_head
            if hasattr(self.model, "projection_head") and self.model.projection_head is not None:
                move_module_to_device(self.model.projection_head, model_device)

            if self.is_main_process:
                print("✓ DeepSpeed initialized successfully")
                print(f"  Model device: {next(self.model.parameters()).device}")
                if hasattr(self.model, "vision_encoder"):
                    try:
                        vision_device = next(self.model.vision_encoder.parameters()).device
                        print(f"  Vision encoder device: {vision_device}")
                    except:
                        pass
                if hasattr(self.model, "language_model"):
                    try:
                        lm_device = next(self.model.language_model.parameters()).device
                        print(f"  Language model device: {lm_device}")
                        # 检查 rotary_emb 的设备
                        if hasattr(self.model.language_model, "model") and hasattr(
                            self.model.language_model.model, "embed_tokens"
                        ):
                            if hasattr(self.model.language_model.model.embed_tokens, "rotary_emb"):
                                try:
                                    rotary_device = next(
                                        self.model.language_model.model.embed_tokens.rotary_emb.parameters()
                                    ).device
                                    print(f"  Rotary embedding device: {rotary_device}")
                                except:
                                    pass
                    except:
                        pass
        else:
            # 标准 PyTorch 训练
            # 只优化可训练参数
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if len(trainable_params) == 0:
                raise ValueError(
                    "No trainable parameters found! Check your training configuration. "
                    "For stage2 training, ensure enable_lora=True or train_lm_head_only=True."
                )
            self.optimizer = AdamW(
                trainable_params,
                lr=config["training"]["learning_rate"],
            )

            num_training_steps = len(self.train_loader) * config["training"]["num_epochs"]
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=config["training"]["warmup_steps"],
                num_training_steps=num_training_steps,
            )

            # 如果使用多卡 DDP（非 DeepSpeed）
            if self.local_rank >= 0:
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False,
                )

            self.model_engine = None

        # 训练状态
        self.global_step = 0
        self.best_loss = float("inf")
        
        # 保存 sampler 引用（用于在每个 epoch 调用 set_epoch）
        self.train_sampler = None

        # 第二阶段训练：加载第一阶段的检查点
        if config["training"].get("stage2_mode", False):
            stage1_checkpoint = config["logging"].get("stage1_checkpoint", None)
            if stage1_checkpoint and os.path.exists(stage1_checkpoint):
                if self.is_main_process:
                    print(f"\n" + "="*80)
                    print("Loading Stage 1 Checkpoint")
                    print("="*80)
                    print(f"  Checkpoint path: {stage1_checkpoint}")
                
                # 只加载 projection_head 的权重（第一阶段训练的模块）
                self._load_stage1_checkpoint(stage1_checkpoint)
                
                if self.is_main_process:
                    print("✓ Stage 1 checkpoint loaded successfully")
                    print("="*80)

        if self.is_main_process:
            print(f"\n✓ Training setup complete", flush=True)
            print(f"  Device: {self.device}", flush=True)
            print(f"  Local rank: {self.local_rank}", flush=True)
            print(f"  World size: {self.world_size}", flush=True)
            print(f"  Use DeepSpeed: {self.use_deepspeed}", flush=True)
            print(f"  Training samples: {len(self.train_dataset)}", flush=True)
            print(f"  Batch size: {config['training']['batch_size']}", flush=True)
            print(f"  Total steps: {num_training_steps}", flush=True)
            
            # 验证日志目录和文件
            print(f"\n📁 Logging Configuration:", flush=True)
            print(f"  Log directory: {self.log_dir.absolute()}", flush=True)
            print(f"  TensorBoard dir: {self.tensorboard_dir.absolute() if hasattr(self, 'tensorboard_dir') and self.tensorboard_dir else 'Not initialized'}", flush=True)
            
            # 检查 JSONL 日志文件
            jsonl_file = self.log_dir / "training_log.jsonl"
            if jsonl_file.exists():
                print(f"  JSONL log file: {jsonl_file.absolute()} (exists)", flush=True)
            else:
                print(f"  JSONL log file: {jsonl_file.absolute()} (will be created)", flush=True)
            
            # 检查 TensorBoard 文件
            if hasattr(self, 'tensorboard_dir') and self.tensorboard_dir.exists():
                tb_files = list(self.tensorboard_dir.glob("events.out.tfevents.*"))
                if tb_files:
                    print(f"  TensorBoard files: {len(tb_files)} file(s) found", flush=True)
                else:
                    print(f"  TensorBoard files: Will be created during training", flush=True)
            print("", flush=True)

    def train_epoch(self, epoch: int, skip_batches: int = 0) -> Dict[str, float]:
        """训练一个 epoch

        Args:
            epoch: 当前 epoch 索引
            skip_batches: 跳过的 batch 数（用于断点续训）

        Returns:
            包含平均损失的字典 {'total_loss', 'vision_loss', 'lm_loss'}
        """
        if self.model_engine is not None:
            self.model_engine.train()
        else:
            self.model.train()
        epoch_losses = []
        epoch_vision_losses = []
        epoch_lm_losses = []

        # 只在主进程显示进度条
        if self.is_main_process:
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        else:
            progress_bar = self.train_loader

        # 记录跳过的 batch
        skipped_count = 0

        for batch_idx, batch in enumerate(progress_bar):
            # 跳过已经训练过的 batch
            if skipped_count < skip_batches:
                skipped_count += 1
                continue

            # 批量处理
            questions = batch["questions"]
            cots = batch["cots"]
            answers = batch["answers"]

            # 前向传播（批量）
            if self.model_engine is not None:
                # DeepSpeed 模式下
                outputs = self.model_engine(
                    question_texts=questions, cot_texts=cots, answer_texts=answers, return_loss=True
                )
            else:
                outputs = self.model(question_texts=questions, cot_texts=cots, answer_texts=answers, return_loss=True)

            loss = outputs["loss"]
            vision_loss = outputs.get("vision_loss", None)
            lm_loss = outputs.get("lm_loss", None)

            if loss is not None:
                # DeepSpeed 自动处理梯度累积和反向传播
                if self.model_engine is not None:
                    self.model_engine.backward(loss)
                    self.model_engine.step()
                else:
                    # 标准 PyTorch 训练
                    loss = loss / self.config["training"]["gradient_accumulation_steps"]
                    loss.backward()

                # 只保留loss的数值，不保留tensor引用
                loss_value = loss.item()
                if self.model_engine is None:
                    loss_value = loss_value * self.config["training"]["gradient_accumulation_steps"]
                epoch_losses.append(loss_value)

                # 记录分项loss
                if vision_loss is not None:
                    epoch_vision_losses.append(vision_loss.item())
                if lm_loss is not None:
                    epoch_lm_losses.append(lm_loss.item())

            # 清理outputs字典，释放显存
            del outputs, loss, vision_loss, lm_loss
            # 定期清空loss历史，只保留最近100个
            if len(epoch_losses) > 100:
                epoch_losses = epoch_losses[-100:]
                epoch_vision_losses = epoch_vision_losses[-100:]
                epoch_lm_losses = epoch_lm_losses[-100:]

            # 梯度累积（仅非 DeepSpeed 模式，DeepSpeed 会自动处理）
            if self.model_engine is None:
                if (batch_idx + 1) % self.config["training"]["gradient_accumulation_steps"] == 0:
                    # 检查梯度是否有效
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["max_grad_norm"])

                    # 更新参数
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True更彻底地清理梯度

                    self.global_step += 1

                    # 计算当前损失（用于记录）
                    current_loss = epoch_losses[-1] if epoch_losses else 0.0
                    current_vision_loss = epoch_vision_losses[-1] if epoch_vision_losses else 0.0
                    current_lm_loss = epoch_lm_losses[-1] if epoch_lm_losses else 0.0

                    # 每个 step 都记录到 TensorBoard（仅主进程）
                    if self.is_main_process and self.writer is not None:
                        try:
                            self.writer.add_scalar("Step/total_loss", current_loss, self.global_step)
                            self.writer.add_scalar("Step/vision_loss", current_vision_loss, self.global_step)
                            self.writer.add_scalar("Step/lm_loss", current_lm_loss, self.global_step)
                            self.writer.add_scalar("Step/learning_rate", self.scheduler.get_last_lr()[0], self.global_step)
                            # 定期刷新 TensorBoard（每10个step）
                            if self.global_step % 10 == 0:
                                self.writer.flush()
                        except Exception as e:
                            if self.global_step % 100 == 0:  # 每100步只打印一次警告，避免刷屏
                                print(f"⚠️  Warning: Failed to write to TensorBoard: {e}", flush=True)
                    
                    # 定期打印日志和记录文件（仅主进程）- 单卡训练模式
                    if self.is_main_process and self.global_step % self.config["logging"]["log_interval"] == 0:
                        avg_loss = np.mean(epoch_losses[-10:]) if epoch_losses else 0.0
                        avg_vision_loss = np.mean(epoch_vision_losses[-10:]) if epoch_vision_losses else 0.0
                        avg_lm_loss = np.mean(epoch_lm_losses[-10:]) if epoch_lm_losses else 0.0

                        if isinstance(progress_bar, tqdm):
                            current_lr = self.scheduler.get_last_lr()[0]
                            progress_bar.set_postfix(
                                {
                                    "loss": f"{avg_loss:.4f}",
                                    "v_loss": f"{avg_vision_loss:.4f}",
                                    "lm_loss": f"{avg_lm_loss:.4f}",
                                    "lr": f"{current_lr:.2e}",
                                }
                            )

                        # 记录到 JSONL 文件
                        current_lr = self.scheduler.get_last_lr()[0]
                        self._log_metrics(
                            {
                                "loss": avg_loss,
                                "vision_loss": avg_vision_loss,
                                "lm_loss": avg_lm_loss,
                                "lr": current_lr,
                            }
                        )

                        # 额外记录平均损失到 TensorBoard（用于平滑曲线）
                        if self.writer is not None:
                            try:
                                self.writer.add_scalar("Average/total_loss", avg_loss, self.global_step)
                                self.writer.add_scalar("Average/vision_loss", avg_vision_loss, self.global_step)
                                self.writer.add_scalar("Average/lm_loss", avg_lm_loss, self.global_step)
                                self.writer.flush()  # 立即刷新
                            except Exception as e:
                                print(f"⚠️  Warning: Failed to write average metrics to TensorBoard: {e}", flush=True)
                    
                    # 定期清理显存
                    if self.global_step % 10 == 0:
                        torch.cuda.empty_cache()
                    
                    # 保存检查点（按 step 间隔）- 单卡训练模式
                    should_save = self.global_step % self.config["logging"]["save_interval"] == 0
                    if should_save:
                        if self.is_main_process:
                            print(f"\n[Step {self.global_step}] Saving checkpoint...", flush=True)
                        self.save_checkpoint(f"checkpoint_step_{self.global_step}")
            else:
                # DeepSpeed 模式：每次 backward 和 step 都会更新（DeepSpeed 自动处理梯度累积）
                self.global_step += 1

                # 计算当前损失（用于记录）
                current_loss = epoch_losses[-1] if epoch_losses else 0.0
                current_vision_loss = epoch_vision_losses[-1] if epoch_vision_losses else 0.0
                current_lm_loss = epoch_lm_losses[-1] if epoch_lm_losses else 0.0

                # 每个 step 都记录到 TensorBoard（仅主进程）
                if self.is_main_process and self.writer is not None:
                    current_lr = (
                        self.scheduler.get_last_lr()[0]
                        if hasattr(self.scheduler, "get_last_lr")
                        else self.scheduler.get_lr()[0]
                    )
                    self.writer.add_scalar("Step/total_loss", current_loss, self.global_step)
                    self.writer.add_scalar("Step/vision_loss", current_vision_loss, self.global_step)
                    self.writer.add_scalar("Step/lm_loss", current_lm_loss, self.global_step)
                    self.writer.add_scalar("Step/learning_rate", current_lr, self.global_step)

                # 定期清理显存
                if self.global_step % 10 == 0:
                    torch.cuda.empty_cache()

                # 定期打印日志和记录文件（仅主进程）
                if self.is_main_process and self.global_step % self.config["logging"]["log_interval"] == 0:
                    avg_loss = np.mean(epoch_losses[-10:]) if epoch_losses else 0.0
                    avg_vision_loss = np.mean(epoch_vision_losses[-10:]) if epoch_vision_losses else 0.0
                    avg_lm_loss = np.mean(epoch_lm_losses[-10:]) if epoch_lm_losses else 0.0

                    if isinstance(progress_bar, tqdm):
                        current_lr = (
                            self.scheduler.get_last_lr()[0]
                            if hasattr(self.scheduler, "get_last_lr")
                            else self.scheduler.get_lr()[0]
                        )
                        progress_bar.set_postfix(
                            {
                                "loss": f"{avg_loss:.4f}",
                                "v_loss": f"{avg_vision_loss:.4f}",
                                "lm_loss": f"{avg_lm_loss:.4f}",
                                "lr": f"{current_lr:.2e}",
                            }
                        )

                    # 记录到文件
                    current_lr = (
                        self.scheduler.get_last_lr()[0]
                        if hasattr(self.scheduler, "get_last_lr")
                        else self.scheduler.get_lr()[0]
                    )
                    self._log_metrics(
                        {
                            "loss": avg_loss,
                            "vision_loss": avg_vision_loss,
                            "lm_loss": avg_lm_loss,
                            "lr": current_lr,
                        }
                    )

                    # 额外记录平均损失到 TensorBoard（用于平滑曲线）
                    if self.writer is not None:
                        self.writer.add_scalar("Average/total_loss", avg_loss, self.global_step)
                        self.writer.add_scalar("Average/vision_loss", avg_vision_loss, self.global_step)
                        self.writer.add_scalar("Average/lm_loss", avg_lm_loss, self.global_step)

                # 保存检查点（按 step 间隔）
                # DeepSpeed 模式下，所有进程都需要参与保存，但只在指定间隔时触发
                should_save = self.global_step % self.config["logging"]["save_interval"] == 0
                if should_save:
                    if self.is_main_process:
                        print(f"\n[Step {self.global_step}] Saving checkpoint...")
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}")

        # 计算 epoch 平均损失
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else float("inf")
        avg_epoch_vision_loss = np.mean(epoch_vision_losses) if epoch_vision_losses else 0.0
        avg_epoch_lm_loss = np.mean(epoch_lm_losses) if epoch_lm_losses else 0.0

        return {
            "total_loss": avg_epoch_loss,
            "vision_loss": avg_epoch_vision_loss,
            "lm_loss": avg_epoch_lm_loss,
        }

    def train(self):
        """完整训练流程"""
        if self.is_main_process:
            print("\n" + "=" * 80)
            print("Starting Training")
            print("=" * 80)

        # 计算断点续训的起始位置
        start_epoch = 0
        skip_batches_first_epoch = 0
        
        if self.global_step > 0:
            # 计算已经处理过的 micro-batches 总数
            if self.use_deepspeed:
                # 在本脚本中，DeepSpeed 模式下的 global_step 实际上是 micro-batches 计数
                batches_processed = self.global_step
            else:
                # 非 DeepSpeed 模式，global_step 是优化器步数
                batches_processed = self.global_step * self.config["training"].get("gradient_accumulation_steps", 1)
            
            # 计算每个 epoch 的 batch 数
            batches_per_epoch = len(self.train_loader)
            
            start_epoch = batches_processed // batches_per_epoch
            skip_batches_first_epoch = batches_processed % batches_per_epoch
            
            if self.is_main_process:
                print(f"\nResume training info:")
                print(f"  Global step: {self.global_step}")
                print(f"  Batches processed: {batches_processed}")
                print(f"  Start epoch: {start_epoch + 1}")
                print(f"  Skip batches in first epoch: {skip_batches_first_epoch}")

        for epoch in range(start_epoch, self.config["training"]["num_epochs"]):
            # 重要：在每个 epoch 开始时调用 set_epoch
            # 这对于 DistributedSampler 至关重要，确保每个 epoch 的数据分布不同
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            
            if self.is_main_process:
                print(f"\n{'='*80}")
                print(f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
                print(f"{'='*80}")

            # 计算当前 epoch 需要跳过的 batch 数
            current_skip = skip_batches_first_epoch if epoch == start_epoch else 0
            if current_skip > 0 and self.is_main_process:
                print(f"  Skipping first {current_skip} batches...")
                
            epoch_metrics = self.train_epoch(epoch, skip_batches=current_skip)

            epoch_total_loss = epoch_metrics["total_loss"]
            epoch_vision_loss = epoch_metrics["vision_loss"]
            epoch_lm_loss = epoch_metrics["lm_loss"]

            # 重要：在 DeepSpeed 模式下，先同步所有进程，确保所有进程都完成 epoch 训练
            if self.use_deepspeed:
                torch.distributed.barrier()
                # 清理未完成的 CUDA 操作，避免阻塞后续的 NCCL 通信
                torch.cuda.synchronize()

            if self.is_main_process:
                print(f"\nEpoch {epoch+1} completed:")
                print(f"  Total Loss: {epoch_total_loss:.4f}")
                print(f"  Vision Loss: {epoch_vision_loss:.6f}")
                print(f"  LM Loss: {epoch_lm_loss:.4f}")

                # 记录 epoch 级别的损失到 TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar("Epoch/total_loss", epoch_total_loss, epoch + 1)
                    self.writer.add_scalar("Epoch/vision_loss", epoch_vision_loss, epoch + 1)
                    self.writer.add_scalar("Epoch/lm_loss", epoch_lm_loss, epoch + 1)

            # 重要：在 DeepSpeed 模式下，所有进程都需要参与保存检查点
            # 因为 ZeRO Stage 2 需要聚合分布在各个进程上的参数分片
            # 保存前再次同步，确保所有进程都准备好
            if self.use_deepspeed:
                torch.distributed.barrier()
            
            # 保存 epoch 检查点（DeepSpeed 模式下所有进程都会调用，但只有主进程打印日志）
            self.save_checkpoint(f"checkpoint_epoch_{epoch+1}")

            # 更新最佳损失并保存最佳模型
            if self.use_deepspeed:
                # 在 DeepSpeed 模式下，使用主进程的损失值来判断，但所有进程都需要参与保存
                # 使用 broadcast 让所有进程知道是否需要保存最佳模型
                if self.is_main_process:
                    # 主进程判断是否需要更新最佳模型
                    should_save_best = epoch_total_loss < self.best_loss
                    save_flag = torch.tensor(1.0 if should_save_best else 0.0, device=self.device)
                else:
                    save_flag = torch.tensor(0.0, device=self.device)
                
                # 广播保存标志，让所有进程知道是否需要保存
                torch.distributed.broadcast(save_flag, src=0)
                
                if save_flag.item() > 0.5:
                    # 🔧 修复：所有进程都需要更新 best_loss，确保同步
                    # 主进程先更新，然后广播给所有进程
                    if self.is_main_process:
                        self.best_loss = epoch_total_loss
                        best_loss_tensor = torch.tensor(self.best_loss, device=self.device)
                    else:
                        best_loss_tensor = torch.tensor(0.0, device=self.device)
                    
                    # 广播 best_loss 到所有进程
                    torch.distributed.broadcast(best_loss_tensor, src=0)
                    
                    # 非主进程也更新 best_loss
                    if not self.is_main_process:
                        self.best_loss = best_loss_tensor.item()
                    
                    # 所有进程都需要参与保存（DeepSpeed ZeRO Stage 2 要求）
                    torch.distributed.barrier()
                    self.save_checkpoint("best_model")
                    if self.is_main_process:
                        print(f"✓ Saved best model (loss: {epoch_total_loss:.4f})")
            else:
                # 非 DeepSpeed 模式：只在主进程保存
                if epoch_total_loss < self.best_loss:
                    self.best_loss = epoch_total_loss
                    if self.is_main_process:
                        self.save_checkpoint("best_model")
                        print(f"✓ Saved best model (loss: {self.best_loss:.4f})")

            # 保存后同步所有进程，确保保存完成后再进入下一个 epoch
            if self.use_deepspeed:
                torch.distributed.barrier()

        # 关闭 TensorBoard writer（仅主进程）
        if self.is_main_process:
            if self.writer is not None:
                self.writer.close()
                print(f"\n✓ TensorBoard writer closed", flush=True)

            print("\n" + "=" * 80, flush=True)
            print("Training Completed!", flush=True)
            print("=" * 80, flush=True)
            print(f"Best loss: {self.best_loss:.4f}", flush=True)
            
            # 验证日志文件
            print(f"\n📊 Logging Summary:", flush=True)
            if hasattr(self, 'tensorboard_dir') and self.tensorboard_dir:
                print(f"  TensorBoard logs: {self.tensorboard_dir.absolute()}", flush=True)
                tb_files = list(self.tensorboard_dir.glob("events.out.tfevents.*"))
                print(f"  TensorBoard files: {len(tb_files)} file(s)", flush=True)
                if tb_files:
                    print(f"  To view: tensorboard --logdir={self.tensorboard_dir}", flush=True)
            
            jsonl_file = self.log_dir / "training_log.jsonl"
            if jsonl_file.exists():
                file_size = jsonl_file.stat().st_size
                try:
                    line_count = sum(1 for _ in open(jsonl_file, encoding='utf-8'))
                except:
                    line_count = 0
                print(f"  JSONL log file: {jsonl_file.absolute()}", flush=True)
                print(f"  JSONL size: {file_size:,} bytes, {line_count} lines", flush=True)
            else:
                print(f"  ⚠️  JSONL log file not found: {jsonl_file.absolute()}", flush=True)
            
            print("=" * 80 + "\n", flush=True)

        # DeepSpeed 需要同步所有进程
        if self.use_deepspeed:
            torch.distributed.barrier()

    def save_checkpoint(self, name: str):
        """保存检查点为 HuggingFace 格式"""
        # 检查是否是 stage1 训练模式
        model_to_check = self.model.module if hasattr(self.model, "module") else self.model
        if self.model_engine is not None:
            model_to_check = self.model_engine.module
        is_stage1 = not getattr(model_to_check, 'stage2_mode', False)
        
        # DeepSpeed 检查点保存
        if self.model_engine is not None:
            # 重要：DeepSpeed 分布式训练中，所有进程都需要参与保存检查点
            # （特别是使用 ZeRO 时，每个进程保存不同的分片）
            # 所有进程都需要调用 model_engine.save_checkpoint()

            output_dir = self.output_dir / name

            # 主进程创建目录并打印日志
            if self.is_main_process:
                output_dir.mkdir(parents=True, exist_ok=True)
                if is_stage1:
                    print(f"\nSaving Stage 1 checkpoint (projection_head only): {output_dir}")
                else:
                    print(f"\nSaving DeepSpeed checkpoint: {output_dir}")
                    print("  Note: DeepSpeed checkpoint saving may take a while...")
                    print("  All processes are participating in checkpoint saving...")
                import time

                start_time = time.time()
            else:
                # 非主进程：确保目录存在（避免竞态条件）
                output_dir.mkdir(parents=True, exist_ok=True)
                import time

                start_time = time.time()

            # 重要：在保存检查点之前，确保所有进程都已完成当前的计算并同步
            # 这有助于避免在 DeepSpeed save_checkpoint 内部的参数聚合时出现同步问题
            if self.use_deepspeed:
                # 清理可能未完成的 CUDA 操作，避免阻塞后续的 NCCL 通信
                torch.cuda.synchronize()
                # 确保所有进程同步（在保存前，确保目录已创建且所有进程都准备好）
                torch.distributed.barrier()

            # Stage 1: 保存 projection_head + special_tokens + optimizer/scheduler（用于断点续训）
            if is_stage1:
                try:
                    # 只在主进程保存 projection_head
                    model_to_save = self.model_engine.module
                    if hasattr(model_to_save, 'projection_head') and model_to_save.projection_head is not None:
                        if self.is_main_process:
                            print("  Saving projection_head.bin (Stage 1 training)...")
                            step_start = time.time()
                        torch.save(model_to_save.projection_head.state_dict(), output_dir / "projection_head.bin")
                        if self.is_main_process:
                            elapsed = time.time() - step_start
                            size_mb = (output_dir / "projection_head.bin").stat().st_size / (1024**2)
                            print(f"  ✓ Saved projection_head.bin ({size_mb:.2f} MB, took {elapsed:.2f} seconds)")
                    
                    # 🔧 重要修复：保存 special token embeddings（DeepSpeed 模式）
                    # 只在主进程保存（避免多进程同时写入冲突）
                    if self.is_main_process:
                        try:
                            special_tokens_state = {}
                            
                            # 获取 embedding table
                            embed_table = model_to_save.language_model.get_input_embeddings()
                            
                            # 保存 <img_begin> embedding（从 embedding table 中提取）
                            if hasattr(model_to_save, 'img_begin_token_id'):
                                img_begin_emb = embed_table.weight[model_to_save.img_begin_token_id].data.cpu()
                                special_tokens_state['img_begin_emb'] = img_begin_emb
                                norm = img_begin_emb.norm().item()
                                print(f"  ✓ Saved <img_begin> embedding (norm={norm:.4f})")
                            
                            # 保存 <img_end> embedding（从 embedding table 中提取）
                            if hasattr(model_to_save, 'img_end_token_id'):
                                img_end_emb = embed_table.weight[model_to_save.img_end_token_id].data.cpu()
                                special_tokens_state['img_end_emb'] = img_end_emb
                                norm = img_end_emb.norm().item()
                                print(f"  ✓ Saved <img_end> embedding (norm={norm:.4f})")
                            
                            # 保存 token IDs（用于验证）
                            if hasattr(model_to_save, 'img_begin_token_id') and hasattr(model_to_save, 'img_end_token_id'):
                                special_tokens_state['img_begin_token_id'] = model_to_save.img_begin_token_id
                                special_tokens_state['img_end_token_id'] = model_to_save.img_end_token_id
                            
                            if special_tokens_state:
                                torch.save(special_tokens_state, output_dir / "special_tokens.bin")
                                size_mb = (output_dir / "special_tokens.bin").stat().st_size / (1024**2)
                                print(f"  ✓ Saved special token embeddings ({size_mb:.4f} MB)")
                        except Exception as e:
                            print(f"  ⚠️  Warning: Failed to save special token embeddings: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # 💾 保存 optimizer 和 scheduler 状态（用于断点续训）
                    # DeepSpeed 模式：保存完整的 DeepSpeed checkpoint（包含 optimizer 和 scheduler）
                    if self.is_main_process:
                        print("  Saving optimizer and scheduler state for resume training...")
                    
                    try:
                        # 保存训练状态到 client_state
                        client_state = {
                            "global_step": self.global_step,
                            "best_loss": self.best_loss,
                        }
                        
                        # 调用 DeepSpeed save_checkpoint 来保存 optimizer 和 scheduler
                        # 这会在 output_dir 下创建 global_step_XXX 目录
                        self.model_engine.save_checkpoint(str(output_dir), tag=None, client_state=client_state)
                        
                        if self.is_main_process:
                            print(f"  ✓ Saved optimizer and scheduler state (DeepSpeed checkpoint)")
                    except Exception as e:
                        if self.is_main_process:
                            print(f"  ⚠️  Warning: Failed to save optimizer/scheduler state: {e}")
                            print(f"     Resume training will use fresh optimizer/scheduler")
                    
                    # 同步所有进程
                    if self.use_deepspeed:
                        torch.cuda.synchronize()
                        torch.distributed.barrier()
                    
                    if self.is_main_process:
                        elapsed = time.time() - start_time
                        print(f"✓ Saved Stage 1 checkpoint to {output_dir} (took {elapsed:.2f} seconds)")
                        print(f"  Components: projection_head + special tokens + optimizer/scheduler")
                except Exception as e:
                    if self.is_main_process:
                        print(f"❌ Failed to save Stage 1 checkpoint: {e}")
                        import traceback
                        traceback.print_exc()
                    raise
            else:
                # Stage 2: 保存完整的 DeepSpeed checkpoint
                # DeepSpeed 保存检查点（所有进程都会调用，DeepSpeed 内部会处理同步）
                try:
                    if self.is_main_process:
                        print(f"  Process {self.local_rank}: Calling model_engine.save_checkpoint()...")
                        print(f"  This may take a while due to parameter aggregation (ZeRO Stage 2)...")
                        print(f"  Note: If this times out, try increasing NCCL_TIMEOUT environment variable")

                    # 重要：在使用 ZeRO 时，save_checkpoint 会触发参数聚合（ALLREDUCE）
                    # 这需要所有 GPU 同步，确保在调用前所有进程都已准备好
                    # 保存训练状态到 client_state，以便恢复训练
                    client_state = {
                        "global_step": self.global_step,
                        "best_loss": self.best_loss,
                    }
                    
                    # 🔧 重要修复：保存 training_state.json 到根目录（用于断点续训）
                    # 即使 DeepSpeed 保存失败，也尝试保存这个文件，以便知道进度
                    if self.is_main_process:
                        try:
                            training_state = {
                                "global_step": self.global_step,
                                "best_loss": self.best_loss,
                                "stage": "stage2",
                                "use_lora": self.config["training"].get("enable_lora", True),
                                "full_finetuning": self.config["training"].get("full_finetuning", False)
                            }
                            # 保存到 checkpoint 根目录
                            with open(output_dir / "training_state.json", "w") as f:
                                json.dump(training_state, f, indent=2)
                            print(f"  ✓ Saved training_state.json (step={self.global_step}, best_loss={self.best_loss:.4f})")
                        except Exception as e:
                            print(f"  ⚠️  Warning: Failed to save training_state.json: {e}")

                    # 调用 DeepSpeed save_checkpoint
                    # 注意：在使用 ZeRO Stage 2/3 时，这会触发参数聚合，需要较长时间
                    # 如果网络文件系统较慢，可能会增加等待时间
                    # 如果出现 NCCL 超时，建议：
                    #   1. 增加 NCCL_TIMEOUT 环境变量（例如：export NCCL_TIMEOUT=1800）
                    #   2. 检查网络文件系统性能（如果 checkpoint 保存在网络文件系统上）
                    self.model_engine.save_checkpoint(str(output_dir), tag=None, client_state=client_state)

                    # 确保所有进程完成保存后再继续（DeepSpeed save_checkpoint 内部可能已经同步，但为了安全再次同步）
                    if self.use_deepspeed:
                        # 在 barrier 之前再次同步 CUDA，确保所有保存操作完成
                        torch.cuda.synchronize()
                        torch.distributed.barrier()

                    if self.is_main_process:
                        print(f"  Process {self.local_rank}: save_checkpoint() completed successfully")

                    # 在 DeepSpeed 检查点保存完成后，也保存 projection_head.bin
                    # 这是为了与其他检查点格式保持一致，便于后续加载
                    model_to_save = self.model.module if hasattr(self.model, "module") else self.model
                    if hasattr(model_to_save, 'projection_head') and model_to_save.projection_head is not None:
                        if self.is_main_process:
                            print("  Saving projection_head.bin for compatibility...")
                            step_start = time.time()
                        torch.save(model_to_save.projection_head.state_dict(), output_dir / "projection_head.bin")
                        if self.is_main_process:
                            elapsed = time.time() - step_start
                            size_mb = (output_dir / "projection_head.bin").stat().st_size / (1024**2)
                            print(f"  ✓ Saved projection_head.bin ({size_mb:.2f} MB, took {elapsed:.2f} seconds)")

                    # 🔧 重要修复：保存 training_state.json 到根目录（用于断点续训）
                    # 虽然 client_state 被传递给 DeepSpeed，但不保证能正确读取
                    # 因此主进程额外保存 training_state.json
                    if self.is_main_process:
                        try:
                            training_state = {
                                "global_step": self.global_step,
                                "best_loss": self.best_loss,
                                "stage": "stage2"
                            }
                            with open(output_dir / "training_state.json", "w") as f:
                                json.dump(training_state, f, indent=2)
                            print(f"  ✓ Saved training_state.json (step={self.global_step}, best_loss={self.best_loss:.4f})")
                        except Exception as e:
                            print(f"  ⚠️  Warning: Failed to save training_state.json: {e}")
                    
                    # 注意：文件系统统计操作移到 barrier 之后，且使用非阻塞方式
                    # 避免文件系统操作阻塞导致后续同步超时
                    if self.is_main_process:
                        elapsed = time.time() - start_time
                        print(f"✓ Saved DeepSpeed checkpoint to {output_dir} (took {elapsed:.2f} seconds)")

                        # 简化文件系统统计，避免递归遍历大量文件导致阻塞
                        # 只在快速操作失败时才跳过，不阻塞主进程
                        try:
                            # 只检查直接子目录和文件，不递归（避免网络文件系统阻塞）
                            all_items = list(output_dir.iterdir())
                            print(f"  Checkpoint contains {len(all_items)} items")
                            # 只显示前5个项目，避免遍历过多
                            for item in all_items[:5]:
                                try:
                                    if item.is_dir():
                                        print(f"    - {item.name}/ (directory)")
                                    else:
                                        size_mb = item.stat().st_size / (1024**2)
                                        print(f"    - {item.name} ({size_mb:.2f} MB)")
                                except:
                                    pass  # 忽略单个文件统计失败
                            if len(all_items) > 5:
                                print(f"    ... and {len(all_items) - 5} more items")
                        except Exception as e:
                            # 文件系统统计失败不影响保存成功
                            print(f"  Note: Could not check checkpoint details (non-critical): {e}")
                except Exception as e:
                    # 在异常情况下，尝试同步所有进程，但不要因为同步失败而掩盖原始错误
                    if self.use_deepspeed:
                        try:
                            # 尝试同步，但不要阻塞太久
                            torch.cuda.synchronize()
                            # 尝试 barrier，但如果失败也不影响原始错误的报告
                            torch.distributed.barrier()
                        except Exception as sync_error:
                            # 同步失败不影响原始错误的报告
                            if self.is_main_process:
                                print(f"  Warning: Barrier failed during error handling: {sync_error}")
                    
                    if self.is_main_process:
                        print(f"❌ Failed to save DeepSpeed checkpoint: {e}")
                        print(f"  Process {self.local_rank} encountered error during save_checkpoint()")
                        import traceback

                        traceback.print_exc()
                    else:
                        # 非主进程也打印错误信息（虽然可能不会显示，但有助于调试）
                        print(f"[Process {self.local_rank}] Error in save_checkpoint: {e}", flush=True)
                    
                    # 重新抛出异常，让调用者知道保存失败
                    raise

            return

        # 非 DeepSpeed 模式：只在主进程保存
        if not self.is_main_process:
            return

        # 使用 HF 格式保存（分片）
        output_dir = self.output_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 获取实际模型（如果是 DDP，需要 .module）
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        # Stage 1: 保存 projection_head + special token embeddings + optimizer/scheduler
        if is_stage1:
            if self.is_main_process:
                print(f"\nSaving Stage 1 checkpoint: {output_dir}")
                import time
                checkpoint_start_time = time.time()

            # 保存 projection_head
            if model_to_save.projection_head is not None:
                torch.save(model_to_save.projection_head.state_dict(), output_dir / "projection_head.bin")
                if self.is_main_process:
                    elapsed = time.time() - checkpoint_start_time
                    size_mb = (output_dir / "projection_head.bin").stat().st_size / (1024**2)
                    print(f"  ✓ Saved projection_head ({size_mb:.2f} MB, took {elapsed:.2f} seconds)")
            else:
                if self.is_main_process:
                    print("⚠️  Warning: No projection head to save")
            
            # 🔧 重要修复：保存 special token embeddings
            # 第一阶段训练了三个组件：
            # 1. projection_head（已保存）
            # 2. <img_begin> embedding（在 embedding table 中）
            # 3. <img_end> embedding（在 embedding table 中）
            try:
                special_tokens_state = {}
                
                # 获取 embedding table
                embed_table = model_to_save.language_model.get_input_embeddings()
                
                # 保存 <img_begin> embedding（从 embedding table 中提取）
                if hasattr(model_to_save, 'img_begin_token_id'):
                    img_begin_emb = embed_table.weight[model_to_save.img_begin_token_id].data.cpu()
                    special_tokens_state['img_begin_emb'] = img_begin_emb
                    if self.is_main_process:
                        norm = img_begin_emb.norm().item()
                        print(f"  ✓ Saved <img_begin> embedding (norm={norm:.4f})")
                
                # 保存 <img_end> embedding（从 embedding table 中提取）
                if hasattr(model_to_save, 'img_end_token_id'):
                    img_end_emb = embed_table.weight[model_to_save.img_end_token_id].data.cpu()
                    special_tokens_state['img_end_emb'] = img_end_emb
                    if self.is_main_process:
                        norm = img_end_emb.norm().item()
                        print(f"  ✓ Saved <img_end> embedding (norm={norm:.4f})")
                
                # 保存 token IDs（用于验证）
                if hasattr(model_to_save, 'img_begin_token_id') and hasattr(model_to_save, 'img_end_token_id'):
                    special_tokens_state['img_begin_token_id'] = model_to_save.img_begin_token_id
                    special_tokens_state['img_end_token_id'] = model_to_save.img_end_token_id
                
                if special_tokens_state:
                    torch.save(special_tokens_state, output_dir / "special_tokens.bin")
                    if self.is_main_process:
                        size_mb = (output_dir / "special_tokens.bin").stat().st_size / (1024**2)
                        print(f"  ✓ Saved special token embeddings ({size_mb:.2f} MB)")
            except Exception as e:
                if self.is_main_process:
                    print(f"  ⚠️  Warning: Failed to save special token embeddings: {e}")
                    import traceback
                    traceback.print_exc()

            # 💾 保存 optimizer 和 scheduler 状态（用于断点续训）
            if self.optimizer is not None and self.scheduler is not None:
                try:
                    optimizer_state = {
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                    }
                    torch.save(optimizer_state, output_dir / "optimizer.bin")
                    if self.is_main_process:
                        size_mb = (output_dir / "optimizer.bin").stat().st_size / (1024**2)
                        print(f"  ✓ Saved optimizer and scheduler ({size_mb:.2f} MB)")
                except Exception as e:
                    if self.is_main_process:
                        print(f"  ⚠️  Warning: Failed to save optimizer/scheduler: {e}")

            # 保存训练信息
            training_state = {
                "global_step": self.global_step,
                "best_loss": self.best_loss,
                "stage": "stage1",
            }
            with open(output_dir / "training_state.json", "w") as f:
                json.dump(training_state, f, indent=2)

            # 保存配置文件
            with open(output_dir / "config.yaml", "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)

            if self.is_main_process:
                elapsed_total = time.time() - checkpoint_start_time
                print(f"✓ Saved Stage 1 checkpoint to {output_dir} (took {elapsed_total:.2f} seconds)")
                print(f"  Components: projection_head + special tokens + optimizer/scheduler")
            
            return

        # Stage 2: 保存完整模型（LoRA + projection head + 其他组件）
        if self.is_main_process:
            print(f"\nSaving checkpoint in HF format: {output_dir}")

        # 检查是否使用了 LoRA
        use_lora = self.config["training"].get("enable_lora", True)

        if use_lora:
            # 使用了 LoRA 训练，保存 LoRA 权重和投影层权重
            if self.is_main_process:
                print("  Detected LoRA training, saving LoRA weights and projection head...")
                import time

                checkpoint_start_time = time.time()

            # 1. 保存 LoRA 权重
            try:
                from peft import PeftModel

                if hasattr(model_to_save, "language_model") and isinstance(model_to_save.language_model, PeftModel):
                    if self.is_main_process:
                        print("    Step 1/2: Saving LoRA weights...")
                    model_to_save.language_model.save_pretrained(output_dir)
                    if self.is_main_process:
                        elapsed = time.time() - checkpoint_start_time
                        print(f"    ✓ Saved LoRA weights to {output_dir} (took {elapsed:.2f} seconds)")
                        # 检查 LoRA 文件大小
                        try:
                            lora_size = sum(f.stat().st_size for f in output_dir.glob("adapter*.bin"))
                            size_mb = lora_size / (1024**2)
                            print(f"      LoRA files size: {size_mb:.2f} MB")
                        except:
                            pass
                else:
                    if self.is_main_process:
                        print("⚠️  Warning: LoRA enabled but language_model is not PeftModel")
            except Exception as e:
                if self.is_main_process:
                    print(f"⚠️  Failed to save LoRA weights: {e}")
                    print("  Continuing with projection head only...")
                    import traceback

                    traceback.print_exc()

            # 2. 保存 projection_head
            if model_to_save.projection_head is not None:
                if self.is_main_process:
                    print("    Step 2/2: Saving projection head...")
                    step_start = time.time()
                torch.save(model_to_save.projection_head.state_dict(), output_dir / "projection_head.bin")
                if self.is_main_process:
                    elapsed = time.time() - step_start
                    size_mb = (output_dir / "projection_head.bin").stat().st_size / (1024**2)
                    print(f"    ✓ Saved projection_head ({size_mb:.2f} MB, took {elapsed:.2f} seconds)")

        else:
            # 没有使用 LoRA 训练，只保存投影层权重
            if self.is_main_process:
                print("  Detected non-LoRA training, saving projection head only...")
                import time

                checkpoint_start_time = time.time()

            # 只保存 projection_head
            if model_to_save.projection_head is not None:
                torch.save(model_to_save.projection_head.state_dict(), output_dir / "projection_head.bin")
                if self.is_main_process:
                    elapsed = time.time() - checkpoint_start_time
                    size_mb = (output_dir / "projection_head.bin").stat().st_size / (1024**2)
                    print(f"  ✓ Saved projection_head ({size_mb:.2f} MB, took {elapsed:.2f} seconds)")
            else:
                if self.is_main_process:
                    print("⚠️  Warning: No projection head to save")

        # 3. 保存其他训练信息
        training_state = {
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "use_lora": use_lora,  # 记录是否使用了 LoRA
        }
        with open(output_dir / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)

        # 4. 保存配置文件
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        # 5. 保存优化器和调度器状态（可选，用于恢复训练）
        if self.optimizer is not None and self.scheduler is not None:
            optimizer_state = {
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            }
            torch.save(optimizer_state, output_dir / "optimizer.bin")

        if self.is_main_process:
            print(f"✓ Saved checkpoint in HF format to {output_dir}")

        # 清理旧检查点（保留最近 N 个）
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """清理旧检查点（保留最近的 N 个目录）"""
        keep_last_n = self.config["logging"]["keep_last_n_checkpoints"]

        # 获取所有 checkpoint 目录
        checkpoints = sorted(
            [d for d in self.output_dir.glob("checkpoint_step_*") if d.is_dir()], key=lambda x: x.stat().st_mtime
        )

        # 删除旧的目录
        for old_checkpoint_dir in checkpoints[:-keep_last_n]:
            shutil.rmtree(old_checkpoint_dir)
            if self.is_main_process:
                print(f"  Removed old checkpoint: {old_checkpoint_dir.name}")

    def _log_metrics(self, metrics: Dict[str, float]):
        """记录指标到 JSONL 文件"""
        if not self.is_main_process:
            return  # 只在主进程记录
        
        try:
            log_file = self.log_dir / "training_log.jsonl"
            
            # 确保目录存在
            log_file.parent.mkdir(parents=True, exist_ok=True)

            log_entry = {"step": self.global_step, **metrics}

            # 使用追加模式写入，并立即刷新
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                f.flush()  # 立即刷新到磁盘
            
            # 验证文件是否成功写入
            if log_file.exists() and log_file.stat().st_size > 0:
                pass  # 文件正常
            else:
                print(f"⚠️  Warning: Log file {log_file} may not be written correctly", flush=True)
        except Exception as e:
            print(f"⚠️  Warning: Failed to write to log file: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    def _verify_stage1_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        验证第一阶段检查点的完整性
        
        Returns:
            True if checkpoint is valid, False otherwise
        """
        if not checkpoint_path.exists():
            return False
        
        required_files = {
            "projection_head.bin": "Projection head weights",
            "special_tokens.bin": "Special token embeddings",  # 🔧 新增
        }
        
        optional_files = {
            "adapter_config.json": "LoRA configuration",
            "adapter_model.bin": "LoRA weights",
            "training_state.json": "Training state",
        }
        
        if self.is_main_process:
            print(f"\n  Validating Stage 1 Checkpoint Structure:")
        
        all_valid = True
        for filename, description in required_files.items():
            file_path = checkpoint_path / filename
            if file_path.exists():
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                if self.is_main_process:
                    print(f"    ✓ {description}: {filename} ({file_size:.2f} MB)")
            else:
                if self.is_main_process:
                    print(f"    ❌ {description}: {filename} (MISSING - REQUIRED)")
                all_valid = False
        
        for filename, description in optional_files.items():
            file_path = checkpoint_path / filename
            if file_path.exists():
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                if self.is_main_process:
                    print(f"    ℹ️  {description}: {filename} ({file_size:.2f} MB)")
        
        return all_valid

    def _load_stage1_checkpoint(self, checkpoint_path: str):
        """
        加载第一阶段的检查点
        
        第一阶段训练的组件：
        - projection_head: 必须加载
        - LoRA weights (如果使用): 可选加载
        
        注意：在 DeepSpeed 模式下，需要通过 model_engine.module 访问模型
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            if self.is_main_process:
                print(f"⚠️  Stage 1 checkpoint not found: {checkpoint_path}")
            return
        
        if self.is_main_process:
            print(f"\n{'='*60}")
            print("Loading Stage 1 Model Components")
            print(f"{'='*60}")
        
        # 验证检查点完整性
        is_valid = self._verify_stage1_checkpoint(checkpoint_path)
        if not is_valid:
            if self.is_main_process:
                print(f"\n❌ Stage 1 checkpoint validation failed!")
                print(f"   Please check that the checkpoint is complete.")
            # 仍然尝试加载，但会在后面报错
        else:
            if self.is_main_process:
                print(f"\n✓ Stage 1 checkpoint validation passed")
        
        # 获取实际模型（处理 DeepSpeed 包装）
        if self.use_deepspeed and self.model_engine is not None:
            model = self.model_engine.module
        else:
            model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 1. 加载 projection_head（必须）
        projection_file = checkpoint_path / "projection_head.bin"
        if projection_file.exists():
            if model.projection_head is not None:
                try:
                    state_dict = torch.load(projection_file, map_location='cpu')
                    
                    # 使用 strict=False 以获取 missing 和 unexpected keys
                    incompatible_keys = model.projection_head.load_state_dict(state_dict, strict=False)
                    
                    # 确保加载后 projection_head 在正确的设备上
                    if self.use_deepspeed:
                        # DeepSpeed 会自动处理设备分配
                        pass
                    else:
                        model.projection_head = model.projection_head.to(self.device)
                    
                    if self.is_main_process:
                        param_count = sum(p.numel() for p in model.projection_head.parameters())
                        print(f"  ✓ Loaded projection_head ({param_count:,} parameters)")
                        
                        # 检查是否有 missing 或 unexpected keys
                        if incompatible_keys.missing_keys:
                            print(f"    ⚠️  Warning: {len(incompatible_keys.missing_keys)} missing keys in projection_head:")
                            for key in incompatible_keys.missing_keys[:5]:  # 只显示前5个
                                print(f"       - {key}")
                            if len(incompatible_keys.missing_keys) > 5:
                                print(f"       ... and {len(incompatible_keys.missing_keys) - 5} more")
                        
                        if incompatible_keys.unexpected_keys:
                            print(f"    ⚠️  Warning: {len(incompatible_keys.unexpected_keys)} unexpected keys in projection_head checkpoint:")
                            for key in incompatible_keys.unexpected_keys[:5]:  # 只显示前5个
                                print(f"       - {key}")
                            if len(incompatible_keys.unexpected_keys) > 5:
                                print(f"       ... and {len(incompatible_keys.unexpected_keys) - 5} more")
                        
                        # 只有当所有权重都完美匹配时才显示成功消息
                        if not incompatible_keys.missing_keys and not incompatible_keys.unexpected_keys:
                            print(f"    ✓ All projection_head weights matched perfectly")
                        
                        # 验证参数是否被正确冻结
                        trainable = sum(p.numel() for p in model.projection_head.parameters() if p.requires_grad)
                        if trainable > 0:
                            print(f"    ⚠️  Warning: projection_head has {trainable:,} trainable parameters (should be 0)")
                        else:
                            print(f"    ✓ projection_head is correctly frozen")
                except Exception as e:
                    if self.is_main_process:
                        print(f"  ❌ Failed to load projection_head: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                if self.is_main_process:
                    print(f"  ⚠️  Model has no projection_head, skipping")
        else:
            if self.is_main_process:
                print(f"  ❌ projection_head.bin not found in {checkpoint_path}")
                print(f"     Stage 2 training requires projection_head from stage 1!")
        
        # 🔧 重要修复：加载 special token embeddings
        # 第一阶段训练了 <img_begin> 和 <img_end> embeddings，必须加载它们
        special_tokens_file = checkpoint_path / "special_tokens.bin"
        if special_tokens_file.exists():
            try:
                special_tokens_state = torch.load(special_tokens_file, map_location='cpu')
                
                if self.is_main_process:
                    print(f"\n  Loading special token embeddings...")
                
                # 获取 embedding table
                embed_table = model.language_model.get_input_embeddings()
                
                # 统计成功和失败的加载
                loaded_embeddings = []
                missing_embeddings = []
                unexpected_keys = []
                
                # 检查 checkpoint 中的所有 keys
                expected_keys = {'img_begin_emb', 'img_end_emb', 'img_begin_token_id', 'img_end_token_id'}
                for key in special_tokens_state.keys():
                    if key not in expected_keys:
                        unexpected_keys.append(key)
                
                # 加载 <img_begin> embedding（到 embedding table 中）
                if 'img_begin_emb' in special_tokens_state and hasattr(model, 'img_begin_token_id'):
                    embed_table.weight.data[model.img_begin_token_id].copy_(special_tokens_state['img_begin_emb'])
                    if self.is_main_process:
                        norm = embed_table.weight.data[model.img_begin_token_id].norm().item()
                        print(f"    ✓ Loaded <img_begin> embedding (norm={norm:.4f})")
                        loaded_embeddings.append('img_begin_emb')
                elif 'img_begin_emb' not in special_tokens_state:
                    missing_embeddings.append('img_begin_emb')
                    if self.is_main_process:
                        print(f"    ⚠️  Warning: 'img_begin_emb' not found in checkpoint")
                elif not hasattr(model, 'img_begin_token_id'):
                    if self.is_main_process:
                        print(f"    ⚠️  Warning: Model has no 'img_begin_token_id' attribute")
                
                # 加载 <img_end> embedding（到 embedding table 中）
                if 'img_end_emb' in special_tokens_state and hasattr(model, 'img_end_token_id'):
                    embed_table.weight.data[model.img_end_token_id].copy_(special_tokens_state['img_end_emb'])
                    if self.is_main_process:
                        norm = embed_table.weight.data[model.img_end_token_id].norm().item()
                        print(f"    ✓ Loaded <img_end> embedding (norm={norm:.4f})")
                        loaded_embeddings.append('img_end_emb')
                elif 'img_end_emb' not in special_tokens_state:
                    missing_embeddings.append('img_end_emb')
                    if self.is_main_process:
                        print(f"    ⚠️  Warning: 'img_end_emb' not found in checkpoint")
                elif not hasattr(model, 'img_end_token_id'):
                    if self.is_main_process:
                        print(f"    ⚠️  Warning: Model has no 'img_end_token_id' attribute")
                
                # 验证 token IDs 是否匹配
                if 'img_begin_token_id' in special_tokens_state and hasattr(model, 'img_begin_token_id'):
                    if special_tokens_state['img_begin_token_id'] != model.img_begin_token_id:
                        if self.is_main_process:
                            print(f"    ⚠️  Warning: img_begin_token_id mismatch!")
                            print(f"       Checkpoint: {special_tokens_state['img_begin_token_id']}")
                            print(f"       Current model: {model.img_begin_token_id}")
                
                if 'img_end_token_id' in special_tokens_state and hasattr(model, 'img_end_token_id'):
                    if special_tokens_state['img_end_token_id'] != model.img_end_token_id:
                        if self.is_main_process:
                            print(f"    ⚠️  Warning: img_end_token_id mismatch!")
                            print(f"       Checkpoint: {special_tokens_state['img_end_token_id']}")
                            print(f"       Current model: {model.img_end_token_id}")
                
                # 汇总报告
                if self.is_main_process:
                    if missing_embeddings:
                        print(f"    ⚠️  Warning: {len(missing_embeddings)} embeddings missing in checkpoint: {missing_embeddings}")
                    
                    if unexpected_keys:
                        print(f"    ⚠️  Warning: {len(unexpected_keys)} unexpected keys in checkpoint: {unexpected_keys}")
                    
                    # 只有当所有预期的 embeddings 都加载成功且没有 unexpected keys 时才显示成功消息
                    if len(loaded_embeddings) == 2 and not missing_embeddings and not unexpected_keys:
                        print(f"  ✓ All special token embeddings loaded successfully")
                    elif loaded_embeddings:
                        print(f"  ⚠️  Partial success: loaded {len(loaded_embeddings)}/2 embeddings")
                    
            except Exception as e:
                if self.is_main_process:
                    print(f"  ❌ Failed to load special token embeddings: {e}")
                    print(f"     This will cause high initial loss in Stage 2!")
                    import traceback
                    traceback.print_exc()
        else:
            if self.is_main_process:
                print(f"\n  ⚠️  special_tokens.bin not found in {checkpoint_path}")
                print(f"     This checkpoint may be from an old version.")
                print(f"     Special token embeddings will use random initialization!")
                print(f"     Expected high initial loss - consider re-training Stage 1.")
        
        # 2. 加载 LoRA weights（如果存在且第一阶段使用了 LoRA）
        adapter_config_file = checkpoint_path / "adapter_config.json"
        adapter_model_file = checkpoint_path / "adapter_model.bin"
        
        if adapter_config_file.exists() and adapter_model_file.exists():
            if self.is_main_process:
                print(f"\n  Found LoRA weights from stage 1")
                print(f"  Note: Stage 2 does not use LoRA, but loading for reference...")
            
            try:
                # 读取 adapter_config 查看信息
                import json
                with open(adapter_config_file, 'r') as f:
                    adapter_config = json.load(f)
                    if self.is_main_process:
                        print(f"    LoRA rank: {adapter_config.get('r', 'unknown')}")
                        print(f"    LoRA alpha: {adapter_config.get('lora_alpha', 'unknown')}")
                        print(f"    Target modules: {adapter_config.get('target_modules', 'unknown')}")
                
                # 注意：第二阶段不使用 LoRA，所以不加载这些权重
                if self.is_main_process:
                    print(f"    ℹ️  LoRA weights not loaded (stage 2 trains lm_head only)")
            except Exception as e:
                if self.is_main_process:
                    print(f"    ⚠️  Could not read LoRA config: {e}")
        
        # 3. 加载训练状态信息（用于记录）
        training_state_file = checkpoint_path / "training_state.json"
        if training_state_file.exists():
            try:
                with open(training_state_file, 'r') as f:
                    training_state = json.load(f)
                    if self.is_main_process:
                        print(f"\n  Stage 1 Training Info:")
                        print(f"    Global step: {training_state.get('global_step', 'unknown')}")
                        print(f"    Best loss: {training_state.get('best_loss', 'unknown')}")
                        if 'use_lora' in training_state:
                            print(f"    Used LoRA: {training_state.get('use_lora', False)}")
            except Exception as e:
                if self.is_main_process:
                    print(f"    ⚠️  Could not read training state: {e}")
        
        # 4. 验证加载后的模型状态（简化版本，避免遍历所有参数导致内存问题）
        # 注意：在 DeepSpeed ZeRO 模式下，遍历所有参数可能导致内存问题
        final_lm_head_trainable = 0
        
        # 所有进程都需要检查，但只有主进程打印
        try:
            if hasattr(model, 'vision_encoder') and hasattr(model.vision_encoder, 'mllm_model'):
                if hasattr(model.vision_encoder.mllm_model, 'lm_head'):
                    lm_head = model.vision_encoder.mllm_model.lm_head
                    # 只检查参数数量，使用 list() 避免生成器问题
                    try:
                        lm_head_params = list(lm_head.parameters())
                        if lm_head_params:  # 确保有参数
                            lm_head_trainable = sum(p.numel() for p in lm_head_params if p.requires_grad)
                            final_lm_head_trainable = lm_head_trainable
                            
                            if self.is_main_process:
                                lm_head_total = sum(p.numel() for p in lm_head_params)
                                print(f"\n  {'='*56}")
                                print("  Verifying Model State for Stage 2 Training")
                                print(f"  {'='*56}")
                                print(f"\n  Direct lm_head check:")
                                print(f"    lm_head.trainable: {lm_head_trainable:,} / {lm_head_total:,}")
                                if lm_head_trainable > 0:
                                    print(f"    ✓ lm_head is trainable")
                                    print(f"\n  Stage 2 Requirements Check:")
                                    print(f"    ✓ lm_head is trainable ({lm_head_trainable:,} parameters)")
                                else:
                                    print(f"    ❌ lm_head is NOT trainable")
                                    print(f"\n  Stage 2 Requirements Check:")
                                    print(f"    ❌ lm_head is NOT trainable (ERROR!)")
                        else:
                            if self.is_main_process:
                                print(f"\n  {'='*56}")
                                print("  Verifying Model State for Stage 2 Training")
                                print(f"  {'='*56}")
                                print(f"    ⚠️  lm_head has no parameters")
                    except Exception as e:
                        if self.is_main_process:
                            print(f"    ⚠️  Error checking lm_head parameters: {e}")
                else:
                    if self.is_main_process:
                        print(f"\n  {'='*56}")
                        print("  Verifying Model State for Stage 2 Training")
                        print(f"  {'='*56}")
                        print(f"    ⚠️  Cannot find mllm_model.lm_head")
            else:
                if self.is_main_process:
                    print(f"\n  {'='*56}")
                    print("  Verifying Model State for Stage 2 Training")
                    print(f"  {'='*56}")
                    print(f"    ⚠️  Cannot find vision_encoder.mllm_model")
        except Exception as e:
            if self.is_main_process:
                print(f"    ⚠️  Error accessing lm_head: {e}")
        
        # 5. 如果 lm_head 不可训练，尝试重新启用（可能在 DeepSpeed 初始化后被重置）
        if self.config["training"].get("train_lm_head_only", False) and final_lm_head_trainable == 0:
            if self.is_main_process:
                print(f"\n  ⚠️  lm_head is not trainable, attempting to re-enable...")
            
            try:
                # 直接访问并重新启用 lm_head 的梯度
                if hasattr(model, 'vision_encoder') and hasattr(model.vision_encoder, 'mllm_model'):
                    if hasattr(model.vision_encoder.mllm_model, 'lm_head'):
                        lm_head = model.vision_encoder.mllm_model.lm_head
                        # 使用 list() 避免生成器问题
                        lm_head_params = list(lm_head.parameters())
                        for param in lm_head_params:
                            param.requires_grad = True
                        
                        # 验证（简化版本，避免重复遍历）
                        if lm_head_params:
                            recheck_trainable = sum(p.numel() for p in lm_head_params if p.requires_grad)
                            if self.is_main_process:
                                if recheck_trainable > 0:
                                    print(f"    ✓ Successfully re-enabled lm_head ({recheck_trainable:,} parameters)")
                                else:
                                    print(f"    ❌ Failed to re-enable lm_head")
            except Exception as e:
                if self.is_main_process:
                    print(f"    ⚠️  Error re-enabling lm_head: {e}")
        
        if self.is_main_process:
            print(f"{'='*60}\n")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点并恢复训练状态
        
        支持两种模式：
        1. Stage 1 断点续训：加载 projection_head + special_tokens + 训练状态
        2. Stage 2 断点续训：加载完整模型（DeepSpeed checkpoint 或 LoRA weights）
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        if self.is_main_process:
            print("\n" + "="*80)
            print("Resume Training from Checkpoint")
            print("="*80)
            print(f"Checkpoint path: {checkpoint_path}")
        
        # 检查训练阶段（stage1 or stage2）
        model_to_check = self.model.module if hasattr(self.model, "module") else self.model
        if self.model_engine is not None:
            model_to_check = self.model_engine.module
        is_stage1 = not getattr(model_to_check, 'stage2_mode', False)
        
        if self.is_main_process:
            print(f"Training stage: {'Stage 1' if is_stage1 else 'Stage 2'}")
        
        # Stage 1 断点续训
        if is_stage1:
            self._resume_stage1_training(checkpoint_path)
        # Stage 2 断点续训
        else:
            self._resume_stage2_training(checkpoint_path)
        
        if self.is_main_process:
            print("="*80)
            print(f"✓ Successfully resumed training from step {self.global_step}")
            print(f"  Best loss so far: {self.best_loss:.4f}")
            print("="*80 + "\n")
    
    def _resume_stage1_training(self, checkpoint_path: Path):
        """恢复 Stage 1 训练"""
        if self.is_main_process:
            print("\n📂 Loading Stage 1 checkpoint components...")
        
        # 获取实际模型（处理 DeepSpeed 包装）
        if self.use_deepspeed and self.model_engine is not None:
            model = self.model_engine.module
        else:
            model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 1. 加载训练状态（global_step, best_loss）
        state_file = checkpoint_path / "training_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                training_state = json.load(f)
                self.global_step = training_state.get("global_step", 0)
                self.best_loss = training_state.get("best_loss", float("inf"))
                if self.is_main_process:
                    print(f"  ✓ Training state loaded")
                    print(f"    - Global step: {self.global_step}")
                    print(f"    - Best loss: {self.best_loss:.4f}")
        else:
            if self.is_main_process:
                print(f"  ⚠️  training_state.json not found")
            
            # 尝试从 DeepSpeed checkpoint 目录名推断 global_step
            # 目录结构可能是 checkpoint_step_2000/global_step2000/ 或类似的
            # 或者更常见的是 checkpoint_path 本身包含 step 信息，如 checkpoint_step_2000
            try:
                # 1. 先尝试从 checkpoint_path 名字推断 (如 checkpoint_step_2000)
                step_match = re.search(r"checkpoint_step_(\d+)", checkpoint_path.name)
                if step_match:
                    self.global_step = int(step_match.group(1))
                    if self.is_main_process:
                        print(f"  ✓ Inferred global_step={self.global_step} from directory name '{checkpoint_path.name}'")
                else:
                    # 2. 尝试从内部的 global_step* 目录推断
                    ds_step_dirs = list(checkpoint_path.glob("global_step*"))
                    if ds_step_dirs:
                        # 取最大的数字
                        latest_dir = max(ds_step_dirs, key=lambda x: int(re.search(r"(\d+)", x.name).group(1) if re.search(r"(\d+)", x.name) else 0))
                        step_match = re.search(r"(\d+)", latest_dir.name)
                        if step_match:
                            # 注意：DeepSpeed 的 step 可能是 micro-step 还是 global-step 取决于配置
                            # 但通常可以直接用作 global_step
                            self.global_step = int(step_match.group(1))
                            if self.is_main_process:
                                print(f"  ✓ Inferred global_step={self.global_step} from DeepSpeed directory '{latest_dir.name}'")
            except Exception as e:
                if self.is_main_process:
                    print(f"  ⚠️  Failed to infer global_step: {e}")
                    print(f"     Starting from step 0")
        
        # 2. 加载 projection_head
        projection_file = checkpoint_path / "projection_head.bin"
        if projection_file.exists() and model.projection_head is not None:
            try:
                state_dict = torch.load(projection_file, map_location='cpu')
                model.projection_head.load_state_dict(state_dict)
                
                # 确保在正确的设备上
                if not self.use_deepspeed:
                    model.projection_head = model.projection_head.to(self.device)
                
                if self.is_main_process:
                    param_count = sum(p.numel() for p in model.projection_head.parameters())
                    print(f"  ✓ Projection head loaded ({param_count:,} parameters)")
            except Exception as e:
                if self.is_main_process:
                    print(f"  ❌ Failed to load projection_head: {e}")
                raise
        else:
            if self.is_main_process:
                print(f"  ⚠️  projection_head.bin not found")
        
        # 3. 加载 special token embeddings
        special_tokens_file = checkpoint_path / "special_tokens.bin"
        if special_tokens_file.exists():
            try:
                special_tokens_state = torch.load(special_tokens_file, map_location='cpu')
                embed_table = model.language_model.get_input_embeddings()
                
                if 'img_begin_emb' in special_tokens_state and hasattr(model, 'img_begin_token_id'):
                    embed_table.weight.data[model.img_begin_token_id].copy_(special_tokens_state['img_begin_emb'])
                    if self.is_main_process:
                        norm = embed_table.weight.data[model.img_begin_token_id].norm().item()
                        print(f"  ✓ <img_begin> embedding loaded (norm={norm:.4f})")
                
                if 'img_end_emb' in special_tokens_state and hasattr(model, 'img_end_token_id'):
                    embed_table.weight.data[model.img_end_token_id].copy_(special_tokens_state['img_end_emb'])
                    if self.is_main_process:
                        norm = embed_table.weight.data[model.img_end_token_id].norm().item()
                        print(f"  ✓ <img_end> embedding loaded (norm={norm:.4f})")
            except Exception as e:
                if self.is_main_process:
                    print(f"  ⚠️  Failed to load special token embeddings: {e}")
        else:
            if self.is_main_process:
                print(f"  ⚠️  special_tokens.bin not found")
        
        # 4. DeepSpeed 模式：恢复 optimizer 和 scheduler 状态
        if self.use_deepspeed and self.model_engine is not None:
            # 检查是否有 DeepSpeed checkpoint
            global_step_dirs = list(checkpoint_path.glob("global_step_*"))
            if global_step_dirs or (checkpoint_path / "latest").exists():
                if self.is_main_process:
                    print(f"\n  📦 Loading DeepSpeed optimizer/scheduler state...")
                try:
                    # 查找正确的 tag
                    if global_step_dirs:
                        latest_step_dir = max(global_step_dirs, key=lambda x: int(x.name.split("_")[-1]))
                        tag = latest_step_dir.name
                        load_path = str(checkpoint_path)
                    elif (checkpoint_path / "latest").exists():
                        with open(checkpoint_path / "latest", "r") as f:
                            tag = f.read().strip()
                        load_path = str(checkpoint_path)
                    else:
                        load_path = str(checkpoint_path)
                        tag = None
                    
                    # 加载 DeepSpeed checkpoint（会恢复 optimizer 和 scheduler）
                    _, client_state = self.model_engine.load_checkpoint(load_path, tag=tag)
                    
                    # 更新训练状态（如果 client_state 中有）
                    if client_state:
                        self.global_step = client_state.get("global_step", self.global_step)
                        self.best_loss = client_state.get("best_loss", self.best_loss)
                    
                    if self.is_main_process:
                        print(f"  ✓ DeepSpeed state loaded (optimizer + scheduler)")
                except Exception as e:
                    if self.is_main_process:
                        print(f"  ⚠️  Failed to load DeepSpeed state: {e}")
                        print(f"     Will start with fresh optimizer/scheduler")
            else:
                if self.is_main_process:
                    print(f"  ℹ️  No DeepSpeed checkpoint found, using fresh optimizer/scheduler")
        
        # 5. 非 DeepSpeed 模式：恢复 optimizer 和 scheduler
        else:
            optimizer_file = checkpoint_path / "optimizer.bin"
            if optimizer_file.exists() and self.optimizer is not None and self.scheduler is not None:
                try:
                    optimizer_state = torch.load(optimizer_file, map_location=self.device)
                    self.optimizer.load_state_dict(optimizer_state["optimizer_state_dict"])
                    self.scheduler.load_state_dict(optimizer_state["scheduler_state_dict"])
                    if self.is_main_process:
                        print(f"  ✓ Optimizer and scheduler state loaded")
                except Exception as e:
                    if self.is_main_process:
                        print(f"  ⚠️  Failed to load optimizer/scheduler: {e}")
            else:
                if self.is_main_process:
                    print(f"  ℹ️  No optimizer.bin found, using fresh optimizer/scheduler")
    
    def _resume_stage2_training(self, checkpoint_path: Path):
        """恢复 Stage 2 训练
        
        参考 evaluate.py 的加载逻辑，确保与推理时的加载方式一致
        """
        if self.is_main_process:
            print(f"\n📂 Resuming Stage 2 training from {checkpoint_path}")
        
        # 1. 尝试恢复训练状态 (global_step, best_loss)
        # 优先从 training_state.json 读取
        state_file = checkpoint_path / "training_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    training_state = json.load(f)
                    self.global_step = training_state.get("global_step", 0)
                    self.best_loss = training_state.get("best_loss", float("inf"))
                    if self.is_main_process:
                        print(f"  ✓ Loaded training state: step={self.global_step}, best_loss={self.best_loss:.4f}")
            except Exception as e:
                if self.is_main_process:
                    print(f"  ⚠️  Failed to load training_state.json: {e}")
        else:
            # 尝试从目录名推断 step
            try:
                global_step_dirs = sorted(checkpoint_path.glob("global_step*"))
                if global_step_dirs:
                    latest = global_step_dirs[-1]
                    step_str = latest.name.replace("global_step", "")
                    self.global_step = int(step_str)
                    if self.is_main_process:
                        print(f"  ⚠️  Inferred global_step={self.global_step} from directory name")
            except:
                pass

        # 2. [DeepSpeed] 加载 Optimizer 和 Scheduler 状态
        # 这一步也会尝试加载模型权重，但由于 Key Mismatch 问题，权重可能不正确
        # 我们稍后会手动覆盖权重
        if self.use_deepspeed and self.model_engine is not None:
            if self.is_main_process:
                print(f"  🔄 Calling model_engine.load_checkpoint()...")
            
            try:
                load_path = str(checkpoint_path)
                # load_checkpoint 返回 (load_path, client_state)
                _, client_state = self.model_engine.load_checkpoint(load_path)
                
                if self.is_main_process:
                    print(f"  ✓ DeepSpeed load_checkpoint completed (Optimizer/Scheduler loaded)")
                
                # 如果之前没读到 training_state，尝试从 client_state 读取
                if client_state and (not state_file.exists() or self.global_step == 0):
                    self.global_step = client_state.get("global_step", self.global_step)
                    self.best_loss = client_state.get("best_loss", self.best_loss)
                    if self.is_main_process:
                        print(f"  ✓ Loaded state from DeepSpeed client_state: step={self.global_step}")

            except Exception as e:
                if self.is_main_process:
                    print(f"  ❌ DeepSpeed load_checkpoint failed: {e}")
                    print(f"     This is critical for resuming optimizer state!")
                # 这里我们不 raise，尝试继续加载权重，也许能跑（虽然 optimizer 丢失）
                # 但通常应该 raise

        # 3. [关键修复] 手动加载 Stage 2 模型权重
        # 使用 evaluate.py 中的逻辑来处理 Key Mismatch
        if self.is_main_process:
            print(f"  🔧 Manually loading Stage 2 weights (fixing key mismatches)...")

        try:
            # 寻找 model states 文件
            ds_dirs = sorted(checkpoint_path.glob("global_step*"))
            latest_step_dir = None
            
            if ds_dirs:
                latest_step_dir = ds_dirs[-1]
            elif (checkpoint_path / "latest").exists():
                with open(checkpoint_path / "latest", "r") as f:
                    tag = f.read().strip()
                latest_step_dir = checkpoint_path / tag
            
            if latest_step_dir and (latest_step_dir / "mp_rank_00_model_states.pt").exists():
                model_states_file = latest_step_dir / "mp_rank_00_model_states.pt"
                if self.is_main_process:
                    print(f"  Loading weights from: {model_states_file}")
                
                # 加载 checkpoint
                checkpoint_state = torch.load(model_states_file, map_location="cpu")
                
                # 提取 state_dict
                state_dict = None
                for key in ["module", "model_state_dict", "model", "state_dict"]:
                    if key in checkpoint_state:
                        state_dict = checkpoint_state[key]
                        break
                if state_dict is None:
                    state_dict = checkpoint_state
                
                # 清理和标准化 Keys (参考 evaluate.py)
                normalized_state = {}
                keys_normalized = 0
                
                for key, value in state_dict.items():
                    clean_key = key
                    # 移除前缀
                    for prefix in ["module.", "model.", "vision_encoder.mllm_model."]:
                        if clean_key.startswith(prefix):
                            clean_key = clean_key[len(prefix):]
                            break
                    
                    # 跳过非 mllm_model 组件
                    if any(c in key for c in ["vision_encoder.ocr", "projection_head", "vision_encoder.clip", "text_renderer"]):
                        continue

                    # 标准化 logic
                    normalized_key = clean_key
                    # 情况1: 完全没有 base_model.model. 前缀
                    if not clean_key.startswith("base_model.model."):
                        if any(clean_key.startswith(p) for p in ["language_model.", "visual.", "llm_lm_head"]):
                            normalized_key = f"base_model.model.model.{clean_key}"
                            keys_normalized += 1
                    # 情况2: 有 base_model.model. 但缺少中间的 model.
                    elif clean_key.startswith("base_model.model.") and not clean_key.startswith("base_model.model.model."):
                        suffix = clean_key[len("base_model.model."):]
                        if any(suffix.startswith(p) for p in ["language_model.", "visual.", "llm_lm_head"]):
                            normalized_key = f"base_model.model.model.{suffix}"
                            keys_normalized += 1
                            
                    normalized_state[normalized_key] = value

                if self.is_main_process:
                    print(f"  Normalized {keys_normalized} keys. Total keys: {len(normalized_state)}")
                
                # 获取实际模型
                model_to_load = self.model.module if hasattr(self.model, "module") else self.model
                # 确保 vision_encoder 已加载
                if not hasattr(model_to_load, 'vision_encoder') or not hasattr(model_to_load.vision_encoder, 'mllm_model'):
                     print("  ⚠️  Model structure unexpected: missing vision_encoder.mllm_model")
                else:
                    mllm_model = model_to_load.vision_encoder.mllm_model
                    
                    # 加载到 mllm_model (Peft 或 Full)
                    missing, unexpected = mllm_model.load_state_dict(normalized_state, strict=False)
                    
                    if self.is_main_process:
                        # 过滤掉 expected missing keys
                        important_missing = [k for k in missing if not k.startswith("base_model.model.")]
                        if important_missing:
                             print(f"    ⚠️  Missing keys (subset): {important_missing[:5]}...")
                        print(f"  ✅ Manual weight loading completed")

            else:
                if self.is_main_process:
                    print(f"  ⚠️  Could not find mp_rank_00_model_states.pt in {latest_step_dir}")

        except Exception as e:
            if self.is_main_process:
                print(f"  ❌ Manual weight loading failed: {e}")
                import traceback
                traceback.print_exc()
                
        # 同步
        if self.use_deepspeed:
            torch.distributed.barrier()


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train CoT Compressor")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Config file path")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (override config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (override config)")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs (override config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (override config)")
    parser.add_argument("--deepspeed_config", type=str, default=None, help="DeepSpeed config file path")
    parser.add_argument(
        "--save_interval", type=int, default=None, help="Save checkpoint every N steps (override config)"
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume training from"
    )

    # DeepSpeed 和分布式训练参数（DeepSpeed 会自动传递这些参数）
    # 使用 parse_known_args 来忽略未知参数，避免 DeepSpeed 传递的参数导致错误
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training (set by DeepSpeed)"
    )

    # 使用 parse_known_args 来处理 DeepSpeed 可能传递的额外参数
    args, unknown_args = parser.parse_known_args()

    # 如果有未知参数，打印警告（但不报错）
    if unknown_args:
        if any("local_rank" in arg or "local-rank" in arg for arg in unknown_args):
            # DeepSpeed 传递的 local_rank 参数，忽略
            pass
        else:
            print(f"Warning: Unknown arguments ignored: {unknown_args}")

    # 如果提供了 DeepSpeed 配置，使用配置文件中的路径
    if args.deepspeed_config is None:
        args.deepspeed_config = None  # 将在 Trainer 中从 config 读取

    # 加载配置
    config = load_config(args.config)

    # 命令行参数覆盖
    if args.dataset:
        config["data"]["dataset_name"] = args.dataset
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.num_epochs:
        config["training"]["num_epochs"] = args.num_epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.save_interval:
        config["logging"]["save_interval"] = args.save_interval

    # 设置随机种子
    torch.manual_seed(config["misc"]["seed"])
    np.random.seed(config["misc"]["seed"])

    # 确定 DeepSpeed 配置文件路径
    deepspeed_config_path = args.deepspeed_config
    if deepspeed_config_path is None:
        deepspeed_config_path = config.get("deepspeed", {}).get("config_file", None)

    # 打印配置
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank <= 0:  # 只在主进程或单卡模式下打印
        print("=" * 80)
        print("Configuration")
        print("=" * 80)
        print(f"Dataset: {config['data']['dataset_name']}")
        print(f"Batch size: {config['training']['batch_size']}")
        print(f"Num epochs: {config['training']['num_epochs']}")
        print(f"Learning rate: {config['training']['learning_rate']}")
        print(f"Save interval: {config['logging']['save_interval']} steps")
        print(f"Local rank: {local_rank}")
        print(f"DeepSpeed config: {deepspeed_config_path}")
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print("=" * 80)

    # 训练
    trainer = Trainer(config, deepspeed_config_path=deepspeed_config_path)
    
    # 如果提供了恢复检查点路径，加载检查点
    if args.resume_from_checkpoint:
        trainer.load_checkpoint(args.resume_from_checkpoint)
    
    trainer.train()


if __name__ == "__main__":
    main()
