# SFT LoRA 微调训练工具

本项目提供了一套面向大语言模型的监督微调（SFT）与继续预训练（CPT）的完整流程，基于 LoRA（Low-Rank Adaptation）技术实现高效参数微调。同时附带丰富的数据预处理、格式转换、模型量化及辅助分析工具，便于快速构建和评估微调任务。

## 项目目的

- 实现**轻量化微调**：利用 LoRA 在保持基础模型参数不变的情况下，仅训练低秩矩阵，大幅降低显存和存储开销。
- 支持**继续预训练（CPT）**：在预训练基座基础上继续训练，适应领域词汇或风格，进一步提升模型性能。
- 提供**完备的训练控制**：支持早停、梯度累积、学习率调度、checkpoint 恢复等专业训练特性。
- 集成**数据与模型工具链**：覆盖从原始对话数据清洗、格式转换（JSON/TXT/Parquet）、分词计数到模型量化、秩扫描的全流程辅助功能。

## 上游技术框架

- **深度学习框架**：PyTorch（CUDA 加速）
- **微调核心**：
  - [PEFT](https://github.com/huggingface/peft) — LoRA 实现与管理
  - [Transformers](https://github.com/huggingface/transformers) — 模型加载、分词器与训练器
  - [Unsloth](https://github.com/unslothai/unsloth) — 优化训练速度与显存占用
- **数据处理**：
  - [Datasets](https://github.com/huggingface/datasets) — 数据集加载、缓存与划分
  - Pandas、PyArrow — Parquet 文件操作

## 主要实现功能

### 1. 训练核心（`main.py`）

| 功能模块 | 说明 |
|---------|------|
| **LoRA 微调** | 通过 `--lora_rank` 指定秩，支持从零训练或加载已有 LoRA 权重 |
| **继续预训练（CPT）** | `--continue_pretrain` 标志，在基座模型上继续语言建模任务 |
| **自动验证与早停** | 按 `--eval_steps` 间隔评估验证集损失，基于 `patience` 和 `threshold` 触发早停 |
| **Checkpoint 恢复** | `--resume` 从 `lora_cache_path` 恢复训练，支持中断后继续 |
| **CUDA 统一内存** | `--cuda_um` 使用统一内存管理，突破显存限制（适用于大模型） |
| **可调节超参数** | 学习率（`lr`）、权重衰减（`regularization`）、批次大小、梯度累积步数、梯度裁剪等 |

### 2. 数据预处理工具（`tools/`）

| 脚本 | 功能 |
|------|------|
| `auto_generate_conversation.py` | 自动生成对话格式训练数据 |
| `chatbox_to_conversations.py` | 将 Chatbox 导出文件转换为标准对话格式 |
| `json_to_parquet.py` / `txt_to_parquet.py` | 将 JSON/TXT 数据集转为高效的 Parquet 格式 |
| `split_parquet.py` | 划分 Parquet 文件为训练集/验证集 |
| `token_counter.py` | 统计数据集中 token 长度分布，辅助确定最大长度 |
| `log_to_excel.py` | 解析训练日志，生成 Excel 报表（损失、学习率曲线） |
| `quantization.py` | 对模型进行 8-bit 量化，压缩模型体积 |

### 3. 辅助脚本（`script/`）

- `rank_scan.bash`：批量测试不同 LoRA 秩（rank）对验证损失的影响，自动生成对比结果。

## 扩展实现功能

- **多格式数据集支持**：通过 `dataset_path` 可加载 JSON、Parquet 或 Hugging Face Dataset 格式，自动缓存至 `cache_dir`。
- **动态验证集划分**：通过 `--test_size` 从训练集中按比例划分验证集，无需提前准备。
- **中间 checkpoint 管理**：`lora_cache_path` 保存每个 epoch 或每 `eval_steps` 的临时权重，支持恢复。
- **量化与部署准备**：`quantization.py` 生成可直接用于推理的量化模型，降低部署门槛。

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt   # 或使用 uv: uv sync
```

### 基础微调命令

```bash
python main.py \
  --model_path /path/to/base_model \
  --dataset_path /path/to/train_data.json \
  --lora_rank 16 \
  --num_train_epochs 3 \
  --lr 2e-4 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --eval_steps 200 \
  --patience 3 \
  --lora_path ./output_lora
```

### 继续预训练（CPT）

```bash
python main.py --continue_pretrain --model_path ... --dataset_path ...
```

### 从 checkpoint 恢复

```bash
python main.py --resume --lora_cache_path ./checkpoints ...
```

### 使用 CUDA 统一内存（显存不足时）

```bash
python main.py --cuda_um ...
```

## 项目结构说明

```
.
├── main.py                     # 主训练入口
├── tools/                      # 数据与模型辅助工具集
├── script/                     # 批量实验脚本
├── model/                      # 存放原始模型（可选）
├── lora/                       # 最终保存的 LoRA 权重
├── lora_cache/                 # 训练中间 checkpoint
├── cache/                      # 数据集缓存
├── unsloth_model/              # 基于 Unsloth 优化的训练流程
├── pyproject.toml / uv.lock    # 依赖管理
└── README.md
```

## 注意事项

- 若使用 `--lora` 加载已经过训练的 LoRA 模型，需确保 `model_path` 指向对应的LoRA模型。
- CPT 模式下请勿同时指定 `--lora`，除非需要基于已微调的 LoRA 继续预训练（一般不建议）。
- 建议先运行 `token_counter.py` 了解数据集Token分布，确保训练数据质量