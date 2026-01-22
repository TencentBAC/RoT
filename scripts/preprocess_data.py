"""
数据预处理脚本

功能：
1. 读取原始数据集（gsm8k / math-500）
2. 统计分析 CoT 长度
3. 将 CoT 文本渲染成图像（可选保存）
4. 保存处理后的数据
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse
from tqdm import tqdm
import numpy as np
from collections import Counter

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
from models.text_to_image import TextToImageRenderer


class DataPreprocessor:
    """数据预处理器"""

    def __init__(
        self,
        dataset_name: str = "gsm8k",
        data_root: str = "path/to/xxx",
        output_dir: str = "data/processed",
        save_images: bool = False,
        max_save_images: int = 100,
        image_size: int = 1024,
        font_size: int = 20,
    ):
        self.dataset_name = dataset_name
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.save_images = save_images
        self.max_save_images = max_save_images
        self.font_size = font_size
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if save_images:
            self.images_dir = self.output_dir / "rendered_images" / dataset_name
            self.images_dir.mkdir(parents=True, exist_ok=True)

        # 初始化渲染器
        self.renderer = TextToImageRenderer(image_size=image_size, font_size=self.font_size)

    def load_raw_data(self, split: str = "train") -> List[Dict[str, Any]]:
        """
        加载原始数据

        Args:
            split: train 或 test

        Returns:
            数据列表
        """
        data_file = self.data_root / self.dataset_name / f"{split}.jsonl"

        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)

        return data

    def analyze_cot_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析 CoT 统计信息

        Args:
            data: 数据列表

        Returns:
            统计信息字典
        """
        # 统计 CoT 长度（字符数）
        cot_char_lengths = [len(item["cot"]) for item in data]

        # 统计 CoT 行数
        cot_line_counts = [item["cot"].count("\n") + 1 for item in data]

        # 统计单词数
        cot_word_counts = [len(item["cot"].split()) for item in data]

        # 统计是否能完整渲染
        rendering_stats = []
        for item in tqdm(data[:1000], desc="Checking renderability (sample)"):  # 采样 1000 个
            will_fit, num_lines = self.renderer.will_text_fit(item["cot"])
            rendering_stats.append({"will_fit": will_fit, "num_lines": num_lines})

        can_render_ratio = sum(1 for s in rendering_stats if s["will_fit"]) / len(rendering_stats)

        # 将 numpy 类型转换为 Python 原生类型，以便 JSON 序列化
        stats = {
            "total_samples": int(len(data)),
            "cot_char_length": {
                "mean": float(np.mean(cot_char_lengths)),
                "std": float(np.std(cot_char_lengths)),
                "min": int(np.min(cot_char_lengths)),
                "max": int(np.max(cot_char_lengths)),
                "median": float(np.median(cot_char_lengths)),
                "percentile_95": float(np.percentile(cot_char_lengths, 95)),
            },
            "cot_line_count": {
                "mean": float(np.mean(cot_line_counts)),
                "std": float(np.std(cot_line_counts)),
                "min": int(np.min(cot_line_counts)),
                "max": int(np.max(cot_line_counts)),
                "median": float(np.median(cot_line_counts)),
            },
            "cot_word_count": {
                "mean": float(np.mean(cot_word_counts)),
                "std": float(np.std(cot_word_counts)),
                "min": int(np.min(cot_word_counts)),
                "max": int(np.max(cot_word_counts)),
                "median": float(np.median(cot_word_counts)),
            },
            "rendering": {
                "can_render_ratio": float(can_render_ratio),
                "sampled_count": int(len(rendering_stats)),
            },
        }

        return stats

    def preprocess(self, split: str = "train") -> None:
        """
        预处理数据

        Args:
            split: train 或 test
        """
        print(f"\n{'='*80}")
        print(f"Preprocessing {self.dataset_name} - {split} split")
        print(f"{'='*80}")

        # 1. 加载原始数据
        data = self.load_raw_data(split)

        # 2. 分析统计信息
        stats = self.analyze_cot_statistics(data)

        # 3. 保存统计信息
        stats_file = self.output_dir / f"{self.dataset_name}_{split}_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        # 4. 处理每条数据
        processed_data = []
        saved_image_count = 0

        for idx, item in enumerate(tqdm(data, desc="Processing")):
            processed_item = {
                "id": idx,
                "question": item["question"],
                "cot": item["cot"],
                "answer": item["answer"],
                "cot_char_length": len(item["cot"]),
                "cot_word_count": len(item["cot"].split()),
            }

            # 检查是否能完整渲染
            will_fit, num_lines = self.renderer.will_text_fit(item["cot"])
            processed_item["can_render_completely"] = will_fit
            processed_item["estimated_lines"] = num_lines

            # 保存渲染的图像（可选）
            if self.save_images and saved_image_count < self.max_save_images:
                try:
                    img = self.renderer.render(item["cot"])
                    img_path = self.images_dir / f"{split}_{idx:05d}.png"
                    img.save(img_path)
                    processed_item["rendered_image_path"] = str(img_path)
                    saved_image_count += 1
                except Exception as e:
                    print(f"Warning: Failed to render image for sample {idx}: {e}")

            processed_data.append(processed_item)

        # 5. 保存处理后的数据
        output_file = self.output_dir / f"{self.dataset_name}_{split}_processed.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 6. 生成处理报告
        self._generate_report(processed_data, split)

    def _generate_report(self, data: List[Dict[str, Any]], split: str) -> None:
        """生成处理报告"""
        report_file = self.output_dir / f"{self.dataset_name}_{split}_report.txt"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"Data Preprocessing Report\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Split: {split}\n")
            f.write(f"Total samples: {len(data)}\n\n")

            # 统计可渲染比例
            can_render_count = sum(1 for item in data if item["can_render_completely"])
            f.write(
                f"Can render completely: {can_render_count}/{len(data)} ({can_render_count/len(data)*100:.1f}%)\n\n"
            )

            # 长度分布
            char_lengths = [item["cot_char_length"] for item in data]
            f.write(f"Character length distribution:\n")
            f.write(f"  Mean: {np.mean(char_lengths):.1f}\n")
            f.write(f"  Std:  {np.std(char_lengths):.1f}\n")
            f.write(f"  Min:  {np.min(char_lengths)}\n")
            f.write(f"  Max:  {np.max(char_lengths)}\n")
            f.write(f"  Median: {np.median(char_lengths):.1f}\n\n")

            # 行数分布
            line_counts = [item["estimated_lines"] for item in data]
            f.write(f"Estimated line count distribution:\n")
            f.write(f"  Mean: {np.mean(line_counts):.1f}\n")
            f.write(f"  Median: {np.median(line_counts):.1f}\n")
            f.write(f"  Max: {np.max(line_counts)}\n\n")

            f.write(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Preprocess CoT data for image compression")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "math-500"],
        help="Dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Data split",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="path/to/xxx",
        help="Root directory of datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save rendered images",
    )
    parser.add_argument(
        "--max_save_images",
        type=int,
        default=100000,
        help="Maximum number of images to save",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Image size for rendering",
    )
    parser.add_argument(
        "--font_size",
        type=int,
        default=20,
        help="Font size for rendering",
    )

    args = parser.parse_args()

    # 初始化预处理器
    preprocessor = DataPreprocessor(
        dataset_name=args.dataset,
        data_root=args.data_root,
        output_dir=args.output_dir,
        save_images=args.save_images,
        max_save_images=args.max_save_images,
        image_size=args.image_size,
    )

    # 执行预处理
    preprocessor.preprocess(split=args.split)


if __name__ == "__main__":
    main()
