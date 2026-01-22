<div align="center">
<h1>Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning</h1>
</div>

<div align="center">
  <a href='https://tencentbac.github.io/RoT/'><img src='https://img.shields.io/badge/Homepage-RoT-6c5ce7?logo=github&logoColor=white'></a>
  <a href='https://arxiv.org/abs/2601.14750'><img src='https://img.shields.io/badge/Paper-arXiv-d63031?logo=arxiv&logoColor=white'></a>
  <a href='https://huggingface.co/collections/TencentBAC/rot'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-0984e3'></a>
  <a href='https://github.com/TencentBAC/RoT'><img src='https://img.shields.io/badge/Code-GitHub-181717?logo=github'></a>
</div>

<br>

<p align="center">
  <b>Yifan Wang<sup>1,2</sup>, Shiyu Li<sup>1</sup>, Peiming Li<sup>1,3</sup>, Xiaochen Yang<sup>4</sup>, Yang Tang<sup>1,†,‡</sup>, Zheng Wei<sup>1,†</sup></b><br>
  <br>
  <sup>1</sup>Tencent BAC &nbsp;&nbsp; <sup>2</sup>Tsinghua University &nbsp;&nbsp; <sup>3</sup>Peking University &nbsp;&nbsp; <sup>4</sup>University of Glasgow<br>
  <br>
  <sup>†</sup>Corresponding Authors &nbsp;&nbsp; <sup>‡</sup>Project Lead<br>
  <br>
  <i>📧 {ethanntang, hemingwei}@tencent.com</i>
  <br>
  <br>
</p>

---

## 📌 Introduction

<div align="center">
  <img src="docs/static/images/overview.png" alt="Overview of Render-of-Thought" width="80%">
</div>

This repository hosts the official implementation of **Render-of-Thought (RoT)**, a novel framework that compresses textual Chain-of-Thought (CoT) reasoning by rendering it into images and leveraging visual latent spaces for efficient reasoning.

**Render-of-Thought** introduces a paradigm shift in latent reasoning by visualizing the reasoning chain. Instead of compressing CoT into opaque vectors, RoT renders textual reasoning steps into images and uses pre-trained vision encoders as semantic anchors to guide the reasoning process. This approach achieves:

- **3-4× token compression** compared to explicit CoT
- **Significant inference acceleration** while maintaining competitive accuracy
- **Interpretable reasoning** through visual representations
- **Plug-and-play implementation** without additional pre-training overhead

The key innovation lies in transforming intermediate reasoning paths into compact visual representations using a pre-trained vision encoder. During training, the framework aligns LLM-generated hidden states with visual features via a projection head, enabling the model to perform continuous reasoning within the visual latent space. At inference time, rendering and visual encoding are eliminated, requiring only a forward pass through the trained LLM backbone and visual projection head.

## Key Features

- 🎨 **Text-to-Image Rendering**: Converts textual CoT steps into compact single-line images
- 🔗 **Visual-Semantic Alignment**: Aligns LLM hidden states with visual embeddings via projection head
- 🚀 **Two-Stage Training**: 
  - Stage 1: Train projection head to align latent representations
  - Stage 2: Fine-tune language model head (with LoRA or full fine-tuning)

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-capable GPUs (recommended: 2+ GPUs for training)
- DeepSpeed >= 0.12.0

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TencentBAC/RoT.git
   cd RoT
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare model checkpoints**:
   - Download **Qwen3-VL-4B-Instruct** model and place it in `ckpt/base/`

## Data Preparation

### Supported Datasets

The project supports mathematical reasoning datasets such as:
- **GSM8K**: Grade school math problems
- **Math-500**: Mathematical reasoning problems
- **SVAMP**: Simple variations on arithmetic math problems

### Data Format

Each data sample should be in JSONL format with the following structure:
```json
{
  "id": xx,
  "question": "The problem statement",
  "cot": "Step-by-step reasoning chain",
  "answer": "Final answer"
}
```
An example dataset format is provided in the `data/GSM8k-Aug-NL` directory for reference.

## Training

### Stage 1: Projection Head Training

Stage 1 trains the projection head to align LLM hidden states with visual embeddings from rendered CoT images.

**Basic usage**:
```bash
bash run_train_stage1.sh
```

**With custom parameters**:
```bash
bash run_train_stage1.sh \
    --num_gpus 4 \
    --config configs/stage1_config_qwen3vl_4b.yaml \
    --dataset gsm8kaug \
    --batch_size 16 \
    --num_epochs 2 \
    --lr 2e-5 \
    --save_interval 200
```

**Key parameters**:
- `--num_gpus`: Number of GPUs to use (default: 2)
- `--config`: Path to training configuration file
- `--dataset`: Dataset name (default: gsm8kaug)
- `--batch_size`: Training batch size (default: 16)
- `--num_epochs`: Number of training epochs (default: 2)
- `--lr`: Learning rate (default: 2e-5)
- `--save_interval`: Steps between checkpoints (optional)
- `--resume_from_checkpoint`: Path to checkpoint for resuming training

**Stage 1 Configuration**:
- Freezes the entire language model
- Only trains the projection head
- Uses vision loss and language modeling loss
- Checkpoints are saved to `output/checkpoints/stage1/`

### Stage 2: Language Model Fine-tuning

Stage 2 fine-tunes the language model head (or uses LoRA) while keeping the projection head frozen.

**Basic usage**:
```bash
bash run_train_stage2.sh
```

**With custom parameters**:
```bash
bash run_train_stage2.sh \
    --num_gpus 4 \
    --config configs/stage2_config_qwen3vl_4b.yaml \
    --dataset gsm8kaug \
    --batch_size 16 \
    --num_epochs 2 \
    --lr 2e-5 \
    --save_interval 200 \
    --stage1_checkpoint output/checkpoints/stage1/checkpoint_epoch_2
```

**Key parameters**:
- Same as Stage 1, plus:
- `--stage1_checkpoint`: Path to Stage 1 checkpoint (required for loading projection head weights)

**Stage 2 Configuration**:
- Freezes vision encoder and projection head (from Stage 1)
- Fine-tunes language model using LoRA (default) or full fine-tuning
- Uses language modeling loss for answer generation
- Checkpoints are saved to `output/checkpoints/stage2/`

## Evaluation

The evaluation script supports two modes:
1. **Evaluate mode**: Computes accuracy and compression statistics
2. **Generate mode**: Only generates results and saves to JSONL file

### Basic Evaluation

```bash
bash run_evaluate.sh \
    --checkpoint output/checkpoints/stage2/checkpoint_step_16000 \
    --stage1_checkpoint output/checkpoints/stage1/checkpoint_epoch_2 \
    --dataset gsm8k \
    --split test
```

### Advanced Options

```bash
bash run_evaluate.sh \
    --checkpoint /path/to/stage2/checkpoint \
    --stage1_checkpoint /path/to/stage1/checkpoint \
    --config configs/stage2_config_qwen3vl_4b.yaml \
    --dataset gsm8k \
    --split test \
    --max_samples 100 \
    --max_new_tokens 256 \
    --temperature 0.0 \
    --num_vision_tokens 32 \
    --stop_threshold 0.01 \
    --model_type v2 \
    --mode evaluate \
    --output_file results/gsm8k_test.jsonl \
    --output_format jsonl
```

**Key parameters**:
- `--checkpoint`: Path to Stage 2 checkpoint (required)
- `--stage1_checkpoint`: Path to Stage 1 checkpoint (required for Stage 2 evaluation)
- `--dataset`: Dataset name (default: gsm8k)
- `--split`: Data split: train or test (default: test)
- `--max_samples`: Maximum number of samples to evaluate
- `--max_new_tokens`: Maximum tokens to generate (default: 64)
- `--temperature`: Generation temperature (default: 0.0)
- `--num_vision_tokens`: Maximum vision tokens for adaptive stopping
- `--stop_threshold`: Threshold for adaptive stopping
- `--model_type`: Model version: v1 or v2 (default: v2)
- `--mode`: evaluate or generate (default: evaluate)
- `--output_file`: Output file path
- `--output_format`: json or jsonl (default: json)

## Model Checkpoint Conversion

The repository provides two conversion scripts to convert between DeepSpeed checkpoint format and HuggingFace SafeTensors format. This is useful for model sharing, distribution, and compatibility with different frameworks.

### Use Cases

- **Model Sharing**: Convert DeepSpeed checkpoints to SafeTensors for safer and faster sharing
- **Cross-Framework Compatibility**: Use models in frameworks that prefer SafeTensors format
- **Storage Optimization**: SafeTensors format is more efficient for long-term storage
- **Evaluation Workflow**: Convert SafeTensors back to DeepSpeed format for evaluation with `evaluate.py`

### Convert DeepSpeed to SafeTensors

The `convert_to_safetensors.py` script converts DeepSpeed checkpoints (`.pt` files) to HuggingFace SafeTensors format with automatic sharding.

**Basic usage**:
```bash
python scripts/convert_to_safetensors.py \
    --input /path/to/mp_rank_00_model_states.pt \
    --output /path/to/output_dir \
    --max_shard_size 5GB \
    --model_name RoT
```

**Parameters**:
- `--input`: Path to DeepSpeed checkpoint file (`mp_rank_00_model_states.pt`) - **required**
- `--output`: Output directory for SafeTensors files - **required**
- `--max_shard_size`: Maximum size per shard (e.g., `5GB`, `2GB`, `500MB`) - default: `5GB`
- `--model_name`: Model name prefix for output files - default: `RoT`
- `--no_clean_keys`: Do not clean state_dict keys (keep original prefixes)
- `--verbose`: Print verbose output


**Output structure**:
```
ckpt/safetensors/global_step4000/
├── RoT-00001-of-00003.safetensors    # Shard 1
├── RoT-00002-of-00003.safetensors    # Shard 2
├── RoT-00003-of-00003.safetensors    # Shard 3
├── RoT.safetensors.index.json        # HuggingFace index file
└── conversion_metadata.json          # Conversion metadata
```

### Convert SafeTensors to DeepSpeed

The `convert_from_safetensors.py` script converts SafeTensors format back to DeepSpeed checkpoint format for use with `evaluate.py` or further training.

**Basic usage**:
```bash
python scripts/convert_from_safetensors.py \
    --input /path/to/safetensors_dir \
    --output /path/to/output.pt \
    --model_name RoT \
    --format module
```

**Parameters**:
- `--input`: Path to SafeTensors directory - **required**
- `--output`: Output path for DeepSpeed checkpoint (`.pt` file) - **required**
- `--model_name`: Model name prefix (should match the conversion) - default: `RoT`
- `--prefix`: Key prefix to restore (e.g., `module.`, `model.`)
- `--format`: Checkpoint format - choices: `module`, `model`, `state_dict`, `direct` - default: `module`
- `--restore_weight_tying`: Restore weight tying to save memory (optional)
- `--verbose`: Print verbose output


## Project Structure

```
RoT/
├── configs/                 # Configuration files
│   ├── stage1_config_qwen3vl_4b.yaml
│   ├── stage2_config_qwen3vl_4b.yaml
│   └── deepspeed_config.json
├── models/                  # Model implementations
│   ├── cot_compressor.py    # CoT compressor (v1)
│   ├── cot_compressor_v2.py # CoT compressor (v2)
│   ├── text_to_image.py     # Text-to-image renderer
│   ├── ocr_wrapper.py       # OCR vision encoder wrapper
│   └── loss.py              # Loss functions
├── scripts/                 # Training and evaluation scripts
│   ├── train.py            # Main training script
│   ├── evaluate.py         # Evaluation script
│   ├── preprocess_data.py  # Data preprocessing
│   ├── convert_to_safetensors.py    # Convert DeepSpeed → SafeTensors
│   └── convert_from_safetensors.py # Convert SafeTensors → DeepSpeed
├── data/                    # Data directory
├── ckpt/                    # Pre-trained model checkpoints
│   └── base/               # Base model (e.g., Qwen3-VL-4B-Instruct)
├── output/                  # Training outputs
│   ├── checkpoints/        # Model checkpoints
│   │   ├── stage1/         # Stage 1 checkpoints
│   │   └── stage2/         # Stage 2 checkpoints
│   └── logs/               # Training logs
├── run_train_stage1.sh     # Stage 1 training script
├── run_train_stage2.sh     # Stage 2 training script
├── run_evaluate.sh         # Evaluation script
└── requirements.txt         # Python dependencies
```


## Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2026rot,
  title={Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning},
  author={Yifan Wang and Shiyu Li and Peiming Li and Xiaochen Yang and Yang Tang and Zheng Wei},
  journal={arXiv preprint arXiv:2601.14750},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This repo benefits from the excellent work [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) and [DeepSpeed](https://github.com/microsoft/DeepSpeed).
