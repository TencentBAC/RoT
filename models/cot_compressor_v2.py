"""
CoT 压缩器模型
将文本 CoT 渲染成图像，然后用 OCR 编码器压缩
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any, List
from PIL import Image

from .cot_compressor import CoTCompressor
from .text_to_image import TextToImageRenderer
from .ocr_wrapper import OCRVisionEncoder
from .loss import StableLossFunctions


class SwiGLU(nn.Module):
    """
    SwiGLU 激活函数
    SwiGLU(x) = (Swish(W1 @ x) ⊙ (W2 @ x)) @ W3
    其中 Swish(x) = x * sigmoid(x) = SiLU(x)
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)  # gate projection
        self.w2 = nn.Linear(input_dim, hidden_dim, bias=False)  # value projection
        self.w3 = nn.Linear(hidden_dim, input_dim, bias=False)  # output projection

    def forward(self, x):
        # x: (..., input_dim)
        gate = F.silu(self.w1(x))      # (..., hidden_dim)
        value = self.w2(x)              # (..., hidden_dim)
        hidden = gate * value           # element-wise product
        return self.w3(hidden)          # (..., input_dim)


class LVRHead(nn.Module):
    """
        The simplest mlp w/o up_proj
    """
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x))
        return x


class CoTCompressorV2(CoTCompressor):
    def __init__(
        self,
        ocr_model_path: str = "path/to/xxx",
        llm_model_path: str = "path/to/xxx",
        image_size: int = 512,
        font_size: int = 16,
        device: str = "cuda",
        freeze_vision: bool = True,
        use_projection_head: bool = True,
        projection_hidden_dim: int = 2048,
        enable_lora: bool = True,
        use_lora: Optional[bool] = None,  # 向后兼容，如果提供则覆盖 enable_lora
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        use_uncertainty_weighting: bool = True,
        vision_loss_weight: float = 1.0,
        lm_loss_weight: float = 1.0,
        use_custom_llm: bool = True,
        loss_type: str = "mse_only",
        # 第二阶段训练参数
        stage2_mode: bool = False,
        train_lm_head_only: bool = False,
        freeze_projection_head: bool = False,
        include_img_end_loss: bool = False,
        include_vision_loss: bool = True,
        full_finetuning: bool = False,  # 新增：全参数微调模式
    ):
        # 注意：不调用 super().__init__()，因为我们要完全自定义初始化
        nn.Module.__init__(self)  # 调用 nn.Module 的初始化

        self.device = device
        self.image_size = image_size
        self.use_projection_head = use_projection_head
        
        self.enable_lora = enable_lora
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.vision_loss_weight = vision_loss_weight
        self.lm_loss_weight = lm_loss_weight
        
        # 第二阶段训练配置
        self.stage2_mode = stage2_mode
        self.train_lm_head_only = train_lm_head_only
        self.freeze_projection_head = freeze_projection_head
        self.include_img_end_loss = include_img_end_loss
        self.include_vision_loss = include_vision_loss
        self.full_finetuning = full_finetuning

        # 损失函数类型选择
        self.loss_type = getattr(self, 'loss_type', 'mse_only')
        self.loss_functions = StableLossFunctions(self.loss_type)

        # 文本渲染器
        self.text_renderer = TextToImageRenderer(
            image_size=image_size,
            font_size=font_size,
        )

        # OCR 视觉编码器（作为固定的 teacher）
        # freeze_vision=True 会冻结 SAM、CLIP 和 Projector
        # 确保训练目标稳定，遵循知识蒸馏原则
        self.vision_encoder = OCRVisionEncoder(
            ocr_model_path=ocr_model_path,
            llm_model_path=llm_model_path,
            device=device,
            freeze_vision=freeze_vision,
            use_custom_llm=use_custom_llm,
        )

        self.llm_loss_function = self.vision_encoder.mllm_model.loss_function
        self.llm_lm_head = self.vision_encoder.mllm_model.lm_head

        # 获取 tokenizer 和语言模型
        self.tokenizer = self.vision_encoder.get_tokenizer()
        self.language_model = self.vision_encoder.get_language_model()

        # 第一阶段训练：冻结整个 language model，严格只训练投影头
        if not self.stage2_mode:
            # 冻结完整模型的参数（包括 language_model 子模块）
            if self.language_model:
                for param in self.language_model.parameters():
                    param.requires_grad = False
            
            # Stage 1 强制禁用 LoRA，只训练投影头
            if self.enable_lora:
                # 强制禁用 LoRA（即使配置文件中启用了）
                self.enable_lora = False
        else:
            # 第二阶段训练：根据配置选择训练策略
            
            # 策略1：全参数微调（不使用 LoRA）
            if self.full_finetuning:
                # 解冻所有 language model 参数
                if self.language_model:
                    for param in self.language_model.parameters():
                        param.requires_grad = True
                
                # 强制禁用 LoRA（全参训练和 LoRA 不能同时启用）
                if self.enable_lora:
                    self.enable_lora = False
                
                self._print_trainable_parameters()
            
            # 策略2：LoRA 微调
            elif self.enable_lora:
                # 冻结完整模型的参数（包括 language_model 子模块）
                if self.language_model:
                    for param in self.language_model.parameters():
                        param.requires_grad = False
                
                self._apply_lora(
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=lora_target_modules
                )
                self._print_trainable_parameters()
            
            # 策略3：仅训练 lm_head（遗留模式）
            elif self.train_lm_head_only:
                # 冻结完整模型的参数（包括 language_model 子模块）
                if self.language_model:
                    for param in self.language_model.parameters():
                        param.requires_grad = False
                
                # 确保 lm_head 引用是最新的
                self.llm_lm_head = self.vision_encoder.mllm_model.lm_head
                # 启用 lm_head 的所有参数梯度
                for param in self.llm_lm_head.parameters():
                    param.requires_grad = True
                
                self._print_trainable_parameters()
            
            # 如果没有选择任何训练策略，报错
            else:
                raise ValueError(
                    "Stage 2 training requires one of the following:\n"
                    "  - full_finetuning=True (full parameter fine-tuning)\n"
                    "  - enable_lora=True (LoRA fine-tuning)\n"
                    "  - train_lm_head_only=True (legacy lm_head-only training)\n"
                    "Otherwise there are no trainable parameters!"
                )

        # 获取hidden dimension
        # For deepseek model
        # self.hidden_dim = self.language_model.config.hidden_size
        # For Qwen3 model
        self.hidden_dim = self.language_model.config.get_text_config().hidden_size

        # 添加 projection head（作为可训练的 student）
        # 功能：将 LM hidden states 投影到 vision embedding 空间 
        # 训练目标：学习预测 vision_encoder 生成的 target embeddings
        if use_projection_head:
            self.projection_head = nn.Sequential(
                nn.LayerNorm(self.hidden_dim, eps=1e-6),
                nn.Linear(self.hidden_dim, projection_hidden_dim),
                SwiGLU(input_dim=projection_hidden_dim, hidden_dim=projection_hidden_dim),
                nn.Linear(projection_hidden_dim, self.hidden_dim),
            )
            # self.projection_head = LVRHead(hidden_size=self.hidden_dim)
            # 确保projection_head与language_model使用相同的数据类型
            self.projection_head = self.projection_head.to(dtype=self.language_model.dtype)
            # for module in self.projection_head.modules():
            #     if isinstance(module, nn.Linear):
            #         # Xavier uniform 初始化，缩小初始值范围
            #         nn.init.xavier_uniform_(module.weight, gain=0.1)  # gain=0.1 让初始输出更小
            #         if module.bias is not None:
            #             nn.init.zeros_(module.bias)
            # print("✓ Projection head initialized with small weights (gain=0.1)")
        else:
            self.projection_head = None

        # 不确定性加权的可学习参数（log(σ²)）
        # 基于 Kendall et al. 2018: "Multi-Task Learning Using Uncertainty to Weigh Losses"
        if use_uncertainty_weighting:
            self.log_var_vision = nn.Parameter(torch.zeros(1))  # log(σ_vision²)
            self.log_var_lm = nn.Parameter(torch.zeros(1))      # log(σ_lm²)
        
        # 添加 <img_begin>, <img> 和 <img_end> token 到 tokenizer
        new_tokens = []
        if "<img_begin>" not in self.tokenizer.vocab:
            new_tokens.append("<img_begin>")
        if "<img>" not in self.tokenizer.vocab:
            new_tokens.append("<img>")
        if "<img_end>" not in self.tokenizer.vocab:
            new_tokens.append("<img_end>")
        
        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
            # 对完整的 mllm_model 调用 resize_token_embeddings
            # 注意：resize_token_embeddings 会重新创建 lm_head，需要重新获取引用
            if self.language_model:
                self.language_model.resize_token_embeddings(len(self.tokenizer))
                # 重新获取 lm_head 引用（resize_token_embeddings 可能替换了 lm_head）
                self.llm_lm_head = self.vision_encoder.mllm_model.lm_head
        
        self.img_begin_token_id = self.tokenizer.vocab["<img_begin>"]
        self.img_token_id = self.tokenizer.vocab["<img>"]  # 新增：vision token ID
        self.img_end_token_id = self.tokenizer.vocab["<img_end>"]
        
        # 初始化特殊 token 的 learnable embeddings (参考 LVR 实现)
        # 使用特定的范数初始化，帮助模型区分 latent reasoning 状态和 normal semantic 状态
        self._init_special_token_embeddings()
        
        # 第二阶段训练特殊处理
        if self.stage2_mode:
            # 冻结 projection_head（第一阶段已训练好）
            if self.freeze_projection_head and self.projection_head is not None:
                for param in self.projection_head.parameters():
                    param.requires_grad = False
            
            # Stage 2 训练策略说明
            if self.train_lm_head_only:
                # 兼容旧版本配置：只训练lm_head
                # 确保 lm_head 引用是最新的（resize_token_embeddings 后可能已替换）
                self.llm_lm_head = self.vision_encoder.mllm_model.lm_head
                # 启用 lm_head 的所有参数梯度
                for param in self.llm_lm_head.parameters():
                    param.requires_grad = True
            
            # 打印可训练参数统计
            self._print_trainable_parameters()

    def _apply_lora(
        self,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: Optional[List[str]]
    ):
        """
        应用 LoRA 到语言模型
        
        经典 LoRA 方案：
        1. 冻结所有原始模型参数（已在 __init__ 中完成）
        2. 通过低秩分解添加可训练的 LoRA 参数
        3. 只训练 LoRA 参数，保持原模型不变
        """
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError(
                "Please install peft library: pip install peft"
            )

        # 默认 target modules（针对常见的 Transformer 架构）
        # q_proj, k_proj, v_proj: attention 的 query, key, value 投影
        # o_proj: attention 的 output 投影
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        # 创建 LoRA 配置
        lora_config = LoraConfig(
            r=lora_r,                          # LoRA 秩（rank）
            lora_alpha=lora_alpha,             # LoRA 缩放因子
            target_modules=target_modules,     # 要应用 LoRA 的模块
            lora_dropout=lora_dropout,         # LoRA dropout
            bias="none",                       # 不训练 bias
            task_type=TaskType.CAUSAL_LM,      # 任务类型
        )

        # 应用 LoRA（peft 会自动标记 LoRA 参数为可训练）
        self.vision_encoder.mllm_model = get_peft_model(self.vision_encoder.mllm_model, lora_config)
    
    def _print_trainable_parameters(self):
        """打印可训练参数统计信息"""
        trainable_params = 0
        all_params = 0
        
        for name, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        trainable_percent = 100 * trainable_params / all_params if all_params > 0 else 0
