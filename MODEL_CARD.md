---
base_model: unsloth/llama-3-8b-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- sft
- unsloth
- llama3
- multi-dataset
license: mit
language:
- en
---

# Llama-3 8B Multi-Dataset SFT (LoRA)

This model is a fine-tuned version of **Llama-3-8B** aligned using Supervised Fine-Tuning (SFT) on a merged dataset of **Alpaca**, **Dolly 15k**, and **OpenAssistant**.

## 🚀 Key Features
- **Training Framework:** Unsloth (LoRA)
- **Datasets:** Merged Instruction tuning (Alpaca, Dolly, OASST)
- **Quantization:** 4-bit (bitsandbytes)
- **Monitoring:** Tracked via Weights & Biases

## 🔗 Project Links
- **Full Pipeline Code:** [GitHub Repository](https://github.com/Edge-Explorer/unsloth-multi-dataset-lora-finetuning)
- **Developer:** [Karan Shelar](https://github.com/Edge-Explorer)

## 🛠️ Usage
You can load this model using the `unsloth` library for ultra-fast inference:

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Karan6124/llama3-8b-multi-dataset-sft",
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# Test prompt
instruction = "Write a clear Python function to check if a string is a palindrome."
prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

## 📊 Training Details
- **Rank (r):** 16
- **Alpha:** 32
- **Learning Rate:** 2e-4
- **Optim:** adamw_8bit
