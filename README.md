# 🚀 Llama-3 Multi-Dataset LoRA Fine-Tuning Pipeline

A professional, modular framework for fine-tuning Llama-3-8B using **Unsloth**, **LoRA**, and **Weights & Biases**. This project transitioned from basic notebook experiments to a production-grade automated pipeline.

## 🌟 Key Features
- **Unsloth Powered**: 2x faster training and 70% less memory usage via 4-bit quantization.
- **Modular Architecture**: separate configurations (`configs/`) and processing logic (`scripts/`).
- **Professional Monitoring**: Full **Weights & Biases (WandB)** integration for live loss curves and GPU tracking.
- **Multi-Dataset IQ**: Merged training on **Alpaca**, **Dolly 15k**, and **OpenAssistant** datasets.
- **Automated Benchmarking**: Built-in evaluation suite for structured model testing.

---

## 🏗️ Project Structure
```text
unsloth-multi-dataset-lora-finetuning/
├── configs/                # YAML configuration files
│   ├── dataset_config.yaml # Dataset paths & merge settings
│   ├── model_config.yaml   # Hyperparameters (Rank, Alpha, LR)
│   └── questions.yaml      # Benchmark evaluation suite
├── scripts/                # Modular Python scripts
│   ├── dataset_merger.py   # Merges multiple sources into one
│   └── train.py            # Professional training engine
├── evaluation/             # Model assessment tools
│   └── benchmark.py        # Automated test-response generator
├── inference/              # Local testing
│   └── comparison.py       # Side-by-side Base vs LoRA evaluation
├── notebooks/              # Google Colab compatible logs
└── notebooks/              # Saved notebook logs
```

---

## 🛠️ Installation & Setup

This project uses **uv** for high-speed dependency management.

```bash
# Install uv
pip install uv

# Create virtual environment and install dependencies
uv sync
```

---

## 🚀 Execution Workflow

### 1. Data Preparation
Define your datasets in `configs/dataset_config.yaml` and run:
```bash
python scripts/dataset_merger.py
```

### 2. Tracked Training
Configure your model hyperparameters in `configs/model_config.yaml` and start the engine:
```bash
python scripts/train.py
```
*Note: Ensure you are logged into WandB to see live dashboards.*

### 3. Professional Evaluation
Run the automated benchmark suite to generate samples for grading:
```bash
python evaluation/benchmark.py
```

---

## 📊 Experiment Tracking
We use **Weights & Biases** to monitor:
- **Loss Curves**: Real-time visualization of learning performance.
- **VRAM Usage**: GPU memory monitoring to prevent OOM errors.
- **Hardware Efficiency**: Tokens per second and runtime stats.

---

## 🎓 Learning Milestones
1. **LoRA Mechanics**: Mastery of Rank/Alpha scaling and Adapter-Base interactions.
2. **Infrastructure**: Moving from monolithic notebooks to a modular YAML-driven pipeline.
3. **Telemetry**: Transitioning from "vibe-checks" to data-driven training decisions.
4. **Benchmarking**: Implementing automated evaluation frameworks.

---

## 📜 License
This project is licensed under the **MIT License**.

---
*Built with ❤️ Edge-Explorer*
