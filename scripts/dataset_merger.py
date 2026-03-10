import yaml
from datasets import load_dataset, concatenate_datasets
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_merge():
    config = load_config("configs/dataset_config.yaml")
    all_ds = []

    print("🚀 Starting Dataset Merger...")
    
    for ds_info in config['datasets']:
        print(f"Loading {ds_info['name']}...")
        ds = load_dataset(ds_info['format'], data_files=ds_info['path'], split="train")
        
        # Ensure we only keep the 'text' column to make merging easy
        ds = ds.select_columns(["text"])
        all_ds.append(ds)

    print("Merging datasets into one...")
    merged_ds = concatenate_datasets(all_ds)
    
    # Save it so the trainer can pick it up
    merged_ds.to_json(config['output_file'])
    print(f"✅ Success! Merged dataset saved to: {config['output_file']}")

if __name__ == "__main__":
    run_merge()