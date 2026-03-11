# 🚀 Llama-3 Multi-Dataset LoRA Fine-Tuning Pipeline

A professional, modular framework for fine-tuning **Llama-3-8B** using **Unsloth**, **LoRA**, and **Weights & Biases**. This project represents a complete journey from basic notebook experiments to a production-grade, config-driven automated training pipeline.

> **"We taught it Language. The next step is teaching it Values."** — This project is the foundation for DPO Alignment (coming next).

---

## 🌟 Key Features
- **⚡ Unsloth Powered**: 2x faster training & 70% less VRAM via 4-bit quantization.
- **🛠️ Modular Architecture**: Config-driven design separating `configs/` (what) from `scripts/` (how).
- **📊 Professional Monitoring**: Full **WandB** integration for live loss curves, GPU stats & runtime analysis.
- **🧠 Multi-Dataset Intelligence**: Merged training on **Alpaca**, **Dolly 15k**, and **OpenAssistant**.
- **🔬 Automated Benchmarking**: Structured benchmark suite ready for LLM-as-a-Judge evaluation.

---

## 🏗️ Project Structure
```text
unsloth-multi-dataset-lora-finetuning/
├── configs/                # YAML configuration files (the "Brain")
│   ├── dataset_config.yaml # Dataset paths & merge settings
│   ├── model_config.yaml   # Hyperparameters (Rank, Alpha, LR, Steps)
│   └── questions.yaml      # Benchmark evaluation prompts
├── scripts/                # Modular Python scripts (the "Heart")
│   ├── dataset_merger.py   # Merges multiple datasets into one
│   └── train.py            # Professional WandB-tracked training engine
├── evaluation/             # Model assessment tools
│   └── benchmark.py        # Automated test-response generator for grading
├── inference/              # Local model testing
│   └── comparison.py       # Side-by-side Base vs LoRA evaluation
└── notebooks/              # Google Colab experiment logs
```

---

## 🛠️ Installation & Setup

This project uses **uv** for high-speed dependency management.

```bash
# Install uv
pip install uv

# Clone the repo, then sync all dependencies
git clone https://github.com/Edge-Explorer/unsloth-multi-dataset-lora-finetuning
cd unsloth-multi-dataset-lora-finetuning
uv sync
```

---

## 🚀 Execution Workflow

### Step 1: Prepare Your Data
Configure your dataset paths in `configs/dataset_config.yaml`, then run:
```bash
python scripts/dataset_merger.py
```

### Step 2: Launch Tracked Training
Set your hyperparameters in `configs/model_config.yaml`, then fire up the engine:
```bash
python scripts/train.py
```
> Make sure you are logged into WandB (`wandb login`) to see the live dashboard.

### Step 3: Benchmark the Model
Generate a structured set of model responses for evaluation:
```bash
python evaluation/benchmark.py
```
Upload the resulting `benchmark_results.json` to an LLM judge (e.g., Gemini, GPT-4) and ask it to rate responses 1-10.

---

## 📊 What WandB Tracks
| Metric | Description |
|---|---|
| `train/loss` | Real-time learning curve (should go down) |
| `train_loss` | Average loss across the full run |
| `train/samples_per_second` | GPU hardware throughput efficiency |
| GPU RAM | Live VRAM monitoring to detect OOM risk |

---

## 🎓 What This Project Teaches
1. **LoRA Mechanics**: Rank, Alpha, target_modules and why gradient checkpointing matters.
2. **YAML-Driven Pipelines**: Separating "code" from "configuration" like a real ML engineer.
3. **Professional Telemetry**: Moving from guessing to data-driven training decisions.
4. **Automated Evaluation**: Building infrastructure to systematically measure model quality.

---

## 📜 License
This project is licensed under the **MIT License**.

---
*Built with ❤️ by Karan Shelar*
