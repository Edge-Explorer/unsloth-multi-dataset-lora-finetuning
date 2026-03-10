import torch
import yaml
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import wandb

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train():
    # Fixed folder name from 'config' to 'configs'
    cfg = load_config("configs/model_config.yaml")

    print("🚀 Initializing WandB...")
    wandb.init(project="llama3-lora-finetuning", name="modular-script-run")

    print(f"📦 Loading Model: {cfg['model_name']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = cfg['model_name'],
        max_seq_length = cfg['max_seq_length'],
        dtype = None,
        load_in_4bit = True,
    )

    print("🛠️ Configuring LoRA Adapters...")
    model = FastLanguageModel.get_peft_model(model, **cfg['lora'])

    print(f"📖 Loading Merged Dataset: {cfg['training']['dataset_path']}...")
    dataset = load_dataset("json", data_files=cfg['training']['dataset_path'], split="train")

    # Initialize Trainer with settings from YAML
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = cfg['max_seq_length'],
        args = TrainingArguments(
            **{k: v for k, v in cfg['training'].items() if k != 'dataset_path'},
            report_to = "wandb"
        ),
    )

    print("🔥 Starting Training...")
    trainer.train()
    
    print("💾 Saving Model...")
    model.save_pretrained("lora_model_script")
    tokenizer.save_pretrained("lora_model_script")
    print("✅ Training Complete!")

if __name__ == "__main__":
    train()