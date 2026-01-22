"""
DeepSeek-OCR 模型包装器
提供统一的接口来使用 OCR 模型的视觉编码器
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from PIL import Image, ImageOps
import sys
from pathlib import Path

# 添加 OCR 模型路径
sys.path.append("path/to/xxx")
sys.path.append("path/to/xxx")

from transformers import AutoModel, AutoTokenizer, Qwen3VLForConditionalGeneration, AutoModelForImageTextToText
from transformers import Qwen2_5_VLForConditionalGeneration


class OCRVisionEncoder(nn.Module):
    """
    DeepSeek-OCR / qwen3-vl 的视觉编码器包装
    """

    def __init__(
        self,
        ocr_model_path: str = "path/to/xxx",
        llm_model_path: str = "path/to/xxx",
        device: str = "cuda",
        freeze_vision: bool = True,
        use_custom_llm: bool = False,
    ):
        super().__init__()

        self.device = device
        self.freeze_vision = freeze_vision
        self.use_custom_llm = use_custom_llm

        self.ocr_model_path = ocr_model_path
        self.llm_model_path = llm_model_path

        if not self.use_custom_llm:
            # 加载 OCR 模型的视觉编码器（用于压缩 CoT 图像）
            self.ocr_model = AutoModel.from_pretrained(
                ocr_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
            ).to(device)
        else:
            # 加载自定义语言模型（Qwen3-VL-4B）
            # 自动检测模型类型
            if "Qwen2.5" in llm_model_path:
                self.model_class = Qwen2_5_VLForConditionalGeneration
            else:
                self.model_class = Qwen3VLForConditionalGeneration
            
            # 在多卡训练时，不使用 device_map="auto"，而是手动指定设备
            # 这样可以让 DeepSpeed 统一管理设备分配
            self.mllm_model = self.model_class.from_pretrained(
                llm_model_path,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map=None,
            ).to(device)

        # 冻结视觉编码器（只训练后续层）
        if freeze_vision:
            self._freeze_vision_encoders()

    def _freeze_vision_encoders(self):
        """冻结 SAM、CLIP 和 Projector

        原因：
        1. Target vision embeddings 在 torch.no_grad() 中计算，不应该接收梯度
        2. 遵循知识蒸馏原则：teacher (vision encoder) 冻结，student (projection head) 训练
        3. 保护预训练权重，避免破坏 OCR 模型的视觉-语言对齐
        4. 保持训练目标稳定，避免 target 分布漂移
        """
        if not self.use_custom_llm:
            # 实际的视觉模型在 self.ocr_model.model 中
            model = self.ocr_model.model if hasattr(self.ocr_model, "model") else self.ocr_model
            for param in model.sam_model.parameters():
                param.requires_grad = False
            for param in model.vision_model.parameters():
                param.requires_grad = False
            for param in model.projector.parameters():
                param.requires_grad = False
        else:
            model = self.mllm_model.visual if hasattr(self.mllm_model, "visual") else self.mllm_model
            for param in model.parameters():
                param.requires_grad = False

    def encode_image(self, image: Image.Image, image_size: int = 1024) -> torch.Tensor:
        """
        将图像编码为压缩的 vision embeddings (基于 Qwen3-VL)

        Args:
            image: PIL Image
            image_size: 图像尺寸

        Returns:
            vision_embeddings: [num_tokens, hidden_dim]
        """
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor

        # 使用 Qwen3-VL 的官方 processor 处理图像
        if not hasattr(self, "processor"):
            self.processor = AutoProcessor.from_pretrained(
                self.llm_model_path, trust_remote_code=True  # 使用与模型相同的路径
            )
        # Qwen3-VL 使用特定的消息格式处理图像
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,  # 直接传入 PIL Image
                    },
                ],
            }
        ]
        # 使用 process_vision_info 处理视觉信息
        # 这是 Qwen3-VL 的标准处理流程
        # Qwen2.5-VL 的初始处理流程类似
        image_inputs, video_inputs = process_vision_info(messages)
        # 应用 chat template 获取文本 prompt
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False  # 不添加生成提示，因为我们只需要视觉特征
        )
        # 使用 processor 将图像转换为 tensor
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )

        # 获取模型实际所在的设备
        # 在 DeepSpeed 包装的情况下，需要从模型参数获取设备
        model = self.mllm_model
        try:
            # 尝试从模型参数获取设备
            if hasattr(model, "visual"):
                vision_tower = model.visual
            elif hasattr(model, "model") and hasattr(model.model, "visual"):
                vision_tower = model.model.visual
            else:
                raise AttributeError("Cannot find visual encoder in the model")

            # 获取 vision_tower 的参数所在的设备
            vision_device = next(vision_tower.parameters()).device
        except (StopIteration, AttributeError):
            # 如果无法获取，尝试从整个模型获取
            try:
                vision_device = next(model.parameters()).device
            except (StopIteration, AttributeError):
                # 最后回退到 self.device
                if isinstance(self.device, str):
                    vision_device = torch.device(self.device)
                else:
                    vision_device = self.device

        # 将处理后的图像数据移到正确的设备和数据类型
        # 注意：processor 默认返回 CPU tensor，需要显式移动到 GPU
        pixel_values = inputs["pixel_values"]
        # print(pixel_values.shape)
        if pixel_values.device.type != vision_device.type:
            pixel_values = pixel_values.to(vision_device)
        pixel_values = pixel_values.to(torch.bfloat16)

        image_grid_thw = inputs.get("image_grid_thw", None)
        if image_grid_thw is not None:
            if image_grid_thw.device.type != vision_device.type:
                image_grid_thw = image_grid_thw.to(vision_device)

        with torch.no_grad():  # 推理模式，节省显存
            # Qwen3-VL 的视觉编码流程
            # 1. 通过 vision tower (ViT) 提取视觉特征

            # 处理图像 patches
            # Qwen3-VL 使用动态分辨率，将图像切分为 patches
            '''
            https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/qwenvl/train/train_qwen.py

            def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
                """
                Args:
                    hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                        The final hidden states of the model.
                    grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                        The temporal, height and width of feature shape of each image in LLM.

                Returns:
                    `torch.Tensor`: hidden_states.
                """
                hidden_states = self.patch_embed(hidden_states)

                pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
                hidden_states = hidden_states + pos_embeds

            '''
            vision_outputs = vision_tower(
                hidden_states=pixel_values,
                grid_thw=image_grid_thw,  # 传入视觉处理的元数据参数
            )

            # 获取视觉特征
            # Qwen3-VL 输出格式: [batch_size, num_patches, hidden_dim]
            # 注意Qwen3-VL的vision encoder返回一个tuple
            # vision_outputs[0]为hidden_state
            # vision_outputs[1]为长度3的deepstack feat list，具体是哪几层的特征抽头详见模型配置文件
            # 这里采用hidden state
            if self.model_class == Qwen2_5_VLForConditionalGeneration:
                vision_embeddings = vision_outputs
            elif self.model_class == Qwen3VLForConditionalGeneration:
                vision_embeddings = (
                    vision_outputs.last_hidden_state if hasattr(vision_outputs, "last_hidden_state") else vision_outputs[0]
                )
            else: raise ValueError(f"Unsupported model class: {self.model_class}")
            # 确保 vision_embeddings 在正确的设备上
            if vision_embeddings.device != vision_device:
                vision_embeddings = vision_embeddings.to(vision_device)

            # 移除 batch 维度 (假设 batch_size = 1)
            if vision_embeddings.dim() == 3:
                vision_embeddings = vision_embeddings.squeeze(0)  # [num_patches, hidden_dim]

        return vision_embeddings  # [num_tokens, hidden_dim]

    def get_mllm_model(self):
        if self.use_custom_llm:
            return self.mllm_model

    def get_language_model(self):
        """返回语言模型"""
        if self.use_custom_llm:
            return self.mllm_model.language_model
        else:
            return self.ocr_model

    def get_tokenizer(self):
        """返回 tokenizer"""
        if self.use_custom_llm:
            return AutoTokenizer.from_pretrained(self.llm_model_path, trust_remote_code=True)
        else:
            return AutoTokenizer.from_pretrained(self.ocr_model_path, trust_remote_code=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        批量处理图像（如果需要）

        Args:
            images: [batch_size, 3, H, W]

        Returns:
            embeddings: List of [num_tokens, hidden_dim]
        """
        # 暂时不支持批量处理，逐个处理
        raise NotImplementedError("Batch processing not yet implemented, use encode_image() for single images")


if __name__ == "__main__":
    # 测试
    encoder = OCRVisionEncoder()

    # 创建测试图像
    test_image = Image.new("RGB", (1024, 1024), color="white")

    # 编码
    embeddings = encoder.encode_image(test_image)

