import torch 
from unsloth import FastLanguageModel
import logging 

# Warning to keep the output clean
logging.getLogger("transformers").setLevel(logging.ERROR)

# Dual-Loading Function
def load_models(lora_path="lora_model"):
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True
    
    print("\n[1/2] Loading Fine-Tuned Model (LoRA)...")
    model_lora, tokenizer = FastLanguageModel.from_pretrained(
        model_name = lora_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model_lora)
    
    print("[2/2] Loading Base Model (Raw Llama-3)...")
    model_base, _ = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit
    )
    FastLanguageModel.for_inference(model_base)
    
    return model_base, model_lora, tokenizer

# The Comparison Logic
def compare(instruction, model_base, model_lora, tokenizer):
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n\n### Response:\n"
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    print(f"\nPROMPT: {instruction}\n" + "="*50)
    
    # Base Model Run
    out_base = model_base.generate(**inputs, max_new_tokens = 128, temperature = 0.5)
    ans_base = tokenizer.batch_decode(out_base, skip_special_tokens = True)[0]

    # LoRA Model Run
    out_lora = model_lora.generate(**inputs, max_new_tokens = 128, temperature = 0.5)
    ans_lora = tokenizer.batch_decode(out_lora, skip_special_tokens = True)[0]

    # Fixed the split logic and index
    print(f"RAW OUTPUT:\n{ans_base.split('### Response:')[1].strip()}")
    print(f"\nFINE-TUNED OUTPUT:\n{ans_lora.split('### Response:')[1].strip()}\n")

if __name__ == "__main__":
    base, lora, tk = load_models()
    
    test_queries = [
        "Explain Rayleigh scattering specifically for the color blue.",
        "Write a Python function for a binary search algorithm.",
        "How do I cook the perfect scrambled eggs?"
    ]
    
    for q in test_queries:
        compare(q, base, lora, tk)