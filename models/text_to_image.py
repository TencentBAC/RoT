"""
文本渲染成图像模块
"""

from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
import numpy as np


class TextToImageRenderer:
    """
    将文本渲染成单行细长图像（动态宽度）
    """

    def __init__(
        self,
        image_size: int = 512,
        font_size: int = 20,
        background_color: str = "white",
        text_color: str = "black",
        padding: int = 4,
        image_height: int = 32,
        add_thought_tokens: bool = True,
    ):
        self.font_size = font_size
        self.background_color = background_color
        self.text_color = text_color
        self.padding = padding
        self.image_height = image_height
        self.add_thought_tokens = add_thought_tokens

        # 加载字体
        self.font = self._load_font()

    def _load_font(self) -> ImageFont.FreeTypeFont:
        """加载字体，尝试多个路径"""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
        ]

        for font_path in font_paths:
            try:
                return ImageFont.truetype(font_path, self.font_size)
            except:
                continue

        # 如果都失败，使用默认字体
        return ImageFont.load_default()

    def render(self, text: str) -> Image.Image:
        """
        将文本渲染成单行细长图像（动态宽度，固定高度32px）

        Args:
            text: 要渲染的文本

        Returns:
            PIL Image
        """
        # 添加特殊 token（在token前后各加两个空格）
        if self.add_thought_tokens:
            text = f"<|begin_of_thought|>  {text}  <|end_of_thought|>"
        
        # 将换行符替换为空格以便在一行内渲染，同时保留所有内容
        render_text = text
        # 合并多个连续空格为一个
        # render_text = " ".join(render_text.split())
        
        # 创建临时 draw 对象用于测量文本宽度
        temp_img = Image.new("RGB", (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        
        # 测量文本的实际宽度
        bbox = temp_draw.textbbox((0, 0), render_text, font=self.font)
        text_width = bbox[2] - bbox[0]
        
        # 计算图像尺寸（宽度动态调整，高度固定为32px）
        image_width = text_width + 2 * self.padding
        
        # 创建实际图像（固定高度）
        img = Image.new("RGB", (image_width, self.image_height), color=self.background_color)
        draw = ImageDraw.Draw(img)
        
        # 绘制文本（垂直居中对齐，使用处理后的单行文本）
        y_position = (self.image_height - self.font_size) // 2
        draw.text((self.padding, y_position), render_text, fill=self.text_color, font=self.font)

        return img

    def _wrap_text(self, text: str, max_width: int, draw: ImageDraw) -> List[str]:
        """
        自动换行处理

        Args:
            text: 原始文本
            max_width: 最大宽度（像素）
            draw: ImageDraw 对象

        Returns:
            分行后的文本列表
        """
        lines = []
        paragraphs = text.split("\n")

        for paragraph in paragraphs:
            if not paragraph.strip():
                lines.append("")  # 保留空行
                continue

            words = paragraph.split()
            current_line = ""

            for word in words:
                test_line = current_line + word + " " if current_line else word + " "
                bbox = draw.textbbox((0, 0), test_line, font=self.font)
                width = bbox[2] - bbox[0]

                if width <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line.rstrip())
                    current_line = word + " "

            if current_line:
                lines.append(current_line.rstrip())

        return lines

    def estimate_image_size(self, text: str) -> Tuple[int, int]:
        """
        估计渲染后的图像尺寸（不实际渲染）

        Args:
            text: 文本

        Returns:
            (宽度, 高度) 像素，高度固定为32px
        """
        # 添加特殊 token（在token前后各加两个空格）
        if self.add_thought_tokens:
            text = f"<|begin_of_thought|>  {text}  <|end_of_thought|>"
        
        # 将换行符替换为空格以便在一行内渲染，同时保留所有内容
        render_text = text
        # 合并多个连续空格为一个
        # render_text = " ".join(render_text.split())
        
        # 创建临时 draw 对象用于测量
        temp_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(temp_img)
        
        # 测量文本宽度
        bbox = draw.textbbox((0, 0), render_text, font=self.font)
        text_width = bbox[2] - bbox[0]
        
        # 计算图像尺寸（宽度动态，高度固定）
        image_width = text_width + 2 * self.padding
        
        return image_width, self.image_height


if __name__ == "__main__":
    # 测试单行细长图像渲染（固定高度32px）
    renderer = TextToImageRenderer(font_size=20, padding=4, image_height=32)

    # 测试短文本
    short_text = "If there are $S$ students at Baker Middle School, then $\\frac{2}{3}S$ students take music.  Setting $\\frac{2}{3}S$ equal to 834 and multiplying both sides by $\\frac{3}{2}$, we find that there are $\\frac{3}{2}\\times 834=\\boxed{1251}$ students at the school."
    
    # 测试长文本（会被合并为一行）
    long_text = """Let me think step by step:
Step 1: Calculate total eggs laid per day - Janet's ducks lay 16 eggs per day.
Step 2: Calculate eggs consumed for breakfast - She eats 3 eggs for breakfast every morning.
Step 3: Calculate eggs used for baking - She uses 4 eggs to bake muffins for her friends.
Step 4: Calculate total eggs used - Total eggs used = 3 (breakfast) + 4 (muffins) = 7 eggs.
Step 5: Calculate remaining eggs for sale - Remaining eggs = 16 (total) - 7 (used) = 9 eggs.
Step 6: Calculate daily earnings - She sells each egg for $2, so daily earnings = 9 eggs × $2 = $18.
Therefore, Janet makes $18 every day at the farmers' market."""

    # 渲染短文本
    img_short = renderer.render(short_text)
    img_short.save("test_render_short.png")
    
    # 渲染长文本
    img_long = renderer.render(long_text)
    img_long.save("test_render_long.png")
