# Multi-Dataset Instruction Fine-Tuning with LoRA using Unsloth

## Overview

This project explores **instruction tuning of a Large Language Model (LLM)** using **LoRA (Low-Rank Adaptation)** with **Unsloth**. The objective is to study how combining multiple instruction datasets impacts the model’s ability to follow instructions and generate high-quality responses.

The model is fine-tuned using **parameter-efficient training**, allowing training to run on limited hardware such as **Google Colab GPUs**. Instead of training the full model, only **LoRA adapters** are trained, which significantly reduces memory usage and training time.

The project also performs **experiment tracking using different dataset combinations** to evaluate how dataset diversity affects model performance.

---

## Project Goals

* Perform **instruction tuning** on an open-source LLM.
* Combine **multiple instruction datasets** for training.
* Compare performance across **different dataset combinations**.
* Demonstrate **parameter-efficient fine-tuning with LoRA**.
* Train efficiently using **Unsloth on resource-limited GPUs**.

---

## Model

**Base Model:** Llama-3-8B-Instruct (4-bit quantized)
**Fine-Tuning Method:** LoRA (Low Rank Adaptation)
**Training Framework:** Unsloth
**Environment:** Google Colab GPU

The model is loaded using **4-bit quantization** to reduce GPU memory consumption.

---

## Experiment Design

The project trains multiple models using different dataset combinations.

### Experiment 1 — Baseline

Dataset used:

* Alpaca

Goal:
Establish baseline performance using a single instruction dataset.

---

### Experiment 2 — Multi-Dataset Training

Datasets used:

* Alpaca
* Dolly

Goal:
Evaluate improvement when combining two instruction datasets.

---

### Experiment 3 — Diverse Dataset Training

Datasets used:

* Alpaca
* Dolly
* OpenAssistant

Goal:
Analyze how dataset diversity impacts instruction-following ability.

---

## Dataset Sources

Datasets are **not included in this repository** because they exceed GitHub’s recommended file size limits.

Please download them directly from Hugging Face.

Alpaca Dataset
https://huggingface.co/datasets/tatsu-lab/alpaca

Dolly 15K Dataset
https://huggingface.co/datasets/databricks/databricks-dolly-15k

OpenAssistant Dataset
https://huggingface.co/datasets/OpenAssistant/oasst1

---

## Dataset Format

All datasets are converted into a unified format before training.

```
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

This ensures compatibility across all datasets during training.

---

## Project Structure

```
unsloth-multi-dataset-lora-finetuning/

data/
    alpaca_dataset.json
    dolly_dataset.jsonl
    openassistant_dataset.parquet

scripts/
    merge_datasets.py

training/
    train_unsloth.py

evaluation/
    benchmark.py

notebooks/
    training_colab.ipynb

README.md
pyproject.toml
```

---

## Environment Setup

This project uses **uv** for dependency management.

### Install uv

```
pip install uv
```

### Initialize the environment

```
uv init
uv venv
```

### Activate environment (Windows)

```
.venv\Scripts\activate
```

### Install dependencies

```
uv add transformers
uv add datasets
uv add accelerate
uv add peft
uv add trl
uv add bitsandbytes
uv add unsloth
uv add pandas
uv add scikit-learn
```

---

## Training Configuration

Typical training configuration used for LoRA fine-tuning:

```
LoRA Rank: 16
LoRA Alpha: 32
Dropout: 0.05
Batch Size: 2
Epochs: 1
Learning Rate: 2e-4
```

Training uses **4-bit quantization** to enable efficient GPU usage.

---

## Evaluation

The fine-tuned models are evaluated using a consistent set of prompts such as:

* Explain recursion
* Write Python code for Fibonacci
* Summarize climate change
* Translate English to French

Evaluation focuses on:

* Instruction following ability
* Response coherence
* Output quality

---

## Output

Instead of saving the full model, the project stores **LoRA adapter weights**.

Approximate size:

```
~100 MB
```

This allows efficient storage and reuse without distributing the entire model.

---

## Technologies Used

* Unsloth
* Hugging Face Transformers
* PEFT (LoRA)
* Hugging Face Datasets
* Google Colab
* Python

---

## Future Improvements

* Automated evaluation metrics
* Experiment tracking dashboards
* Training loss visualization
* Additional instruction datasets
* Advanced benchmarking

---

## License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this project under the terms of the MIT license.
