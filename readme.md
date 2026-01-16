# LoRA Instruction Fine-Tuning

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A parameter-efficient instruction fine-tuning pipeline using LoRA (Low-Rank Adaptation) to adapt the OPT-350M language model for code generation tasks, demonstrating significant improvements over the base model with minimal computational overhead.


## Problem Statement

Fine-tuning large language models is computationally expensive and memory-intensive. This project demonstrates how LoRA enables efficient instruction fine-tuning by training only a small fraction of parameters while achieving comparable performance to full fine-tuning.

## Features

- Parameter-efficient fine-tuning with LoRA (trains <1% of model parameters)
- Instruction-following capability on programming tasks
- Completion-only loss masking for efficient training
- Baseline vs. fine-tuned model comparison with SACREBLEU metrics
- Training visualization and evaluation pipeline

## Quick Start

### Prerequisites

```bash
pip install datasets==2.20.0 trl==0.9.6 transformers==4.42.3 peft==0.11.1 \
    evaluate sacrebleu torch matplotlib
```

### Dataset

The notebook automatically downloads the [CodeAlpaca-20k](https://github.com/sahil280114/codealpaca) dataset - a collection of 20,000 instruction-response pairs for code generation tasks.

### Run

Open `lora_instruction_fine-tuning.ipynb` in Jupyter and run all cells.

**Note**: Training requires a GPU with ~8GB VRAM and takes approximately 30-60 minutes.

## Model Architecture

| Component | Details |
|-----------|---------|
| **Base Model** | OPT-350M (Meta AI) |
| **Fine-Tuning** | LoRA with rank=16, alpha=32 |
| **Target Modules** | q_proj, v_proj (attention layers) |
| **Optimizer** | Adam (built into SFTTrainer) |
| **Training** | 10 epochs, batch size=2, FP16 mixed precision |
| **Evaluation** | SACREBLEU metric |

## LoRA Configuration

```python
LoraConfig(
    r=16,                              # Low-rank dimension
    lora_alpha=32,                     # Scaling factor (alpha/r = 2.0)
    target_modules=["q_proj", "v_proj"], # OPT attention modules
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM
)
```

**Trainable Parameters**: ~1.7M out of 350M total (0.49%)

## Project Structure

```
├── lora_instruction_fine-tuning.ipynb
├── README.md
├── CodeAlpaca-20k.json                    # Auto-downloaded
└── instruction_tuning_final_model_lora/   # Saved after training
```

## Results

The LoRA fine-tuned model demonstrates improved instruction-following capabilities compared to the base OPT-350M model. Training loss decreases consistently over 10 epochs, and SACREBLEU scores show measurable improvement on held-out test examples.

### Sample Comparison

**Instruction**: "Write a Python function to calculate factorial"

**Base Model**: [Generic or incomplete response]  
**LoRA Fine-Tuned**: [More accurate, complete code implementation]

See the notebook for complete training curves and evaluation metrics.

## Skills Demonstrated

- Parameter-efficient fine-tuning (PEFT) techniques
- Low-Rank Adaptation (LoRA) implementation
- Supervised fine-tuning for instruction-following
- Custom data formatting and loss masking
- Hugging Face Transformers & TRL libraries
- Training optimization (FP16, gradient checkpointing)
- Model evaluation and comparison

## Training Details

### Data Processing
- Filters dataset to instruction-only examples (no additional input context)
- Formats prompts with clear instruction/response delimiters
- Uses `DataCollatorForCompletionOnlyLM` to compute loss only on responses

### Training Configuration
- 80/20 train/test split
- FP16 mixed precision for efficiency
- Evaluation every epoch
- Saves checkpoints and final model

## Hardware Requirements

- **Minimum**: 8GB GPU (NVIDIA recommended)
- **Recommended**: 16GB+ GPU for larger batch sizes
- **CPU**: Supported but significantly slower (~10x training time)

## License

MIT

## Acknowledgments

- Completed as part of IBM AI Engineering Professional Certificate
- Dataset: [CodeAlpaca-20k](https://github.com/sahil280114/codealpaca) by Sahil Chaudhary
- Base Model: [OPT-350M](https://huggingface.co/facebook/opt-350m) by Meta AI
- Libraries: Hugging Face Transformers, PEFT, TRL


