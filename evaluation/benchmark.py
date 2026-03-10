import torch
import yaml
import json
import os
from unsloth import FastLanguageModel
from transformers import TextStreamer

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_benchmark(model_path="lora_model"):
    # Load questions from configs folder
    questions_cfg = load_config("configs/questions.yaml")
    results = []

    print(f"📦 Loading model for evaluation from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    print("📝 Running benchmark questions...")
    for q in questions_cfg['questions']:
        print(f"Evaluating Q{q['id']} [{q['category']}]...")
        
        prompt = f"### Instruction:\n{q['instruction']}\n\n### Input:\n\n### Response:\n"
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
        
        outputs = model.generate(**inputs, max_new_tokens = 256, temperature = 0.3)
        response = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
        
        # Extract only the response part
        clean_response = response.split("### Response:")[1].strip()
        
        results.append({
            "id": q["id"],
            "category": q["category"],
            "instruction": q["instruction"],
            "model_response": clean_response
        })

    # Save results for the Judge
    output_path = "evaluation/benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"✅ Benchmark finished! Results saved to {output_path}")
    print("\n--- NEXT STEP ---")
    print("You can now upload 'benchmark_results.json' to an LLM (like Gemini or GPT-4)")
    print("and ask: 'Grade these model responses from 1-10 based on accuracy and style.'")

if __name__ == "__main__":
    run_benchmark()
