import torch
from unsloth import FastLanguageModel
import logging

# 1. SETUP: Silence internal library warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

def load_base_and_lora(model_path="lora_model"):
    """
    Loads both the base model and the LoRA-adapter model for comparison.
    """
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    # Loading the Fine-tuned Model (LoRA)
    print("\n--- Loading Fine-tuned Model (LoRA) ---")
    model_lora, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model_lora)

    # Loading the Base Model (Raw Llama-3)
    # We load it without adapters to see how it originally behaved
    print("\n--- Loading Base Model (Raw Llama-3) ---")
    model_base, _ = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model_base)

    return model_base, model_lora, tokenizer

def run_comparison(instruction, model_base, model_lora, tokenizer):
    """
    Runs the same prompt through both models and prints the outputs side-by-side.
    """
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n\n\n### Response:\n"
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

    print(f"\n{'='*50}\nPROMPT: {instruction}\n{'='*50}")

    # Generate from Base
    outputs_base = model_base.generate(**inputs, max_new_tokens = 256, temperature = 0.7)
    answer_base = tokenizer.batch_decode(outputs_base, skip_special_tokens = True)[0]
    
    # Generate from LoRA
    outputs_lora = model_lora.generate(**inputs, max_new_tokens = 256, temperature = 0.7)
    answer_lora = tokenizer.batch_decode(outputs_lora, skip_special_tokens = True)[0]

    print("\n[--- RAW MODEL OUTPUT ---]")
    print(answer_base.split("### Response:")[1] if "### Response:" in answer_base else answer_base)

    print("\n[--- FINE-TUNED MODEL OUTPUT ---]")
    print(answer_lora.split("### Response:")[1] if "### Response:" in answer_lora else answer_lora)

if __name__ == "__main__":
    # Test Questions
    questions = [
        "Give me a short travel itinerary for a 3-day trip to Paris.",
        "Explain the concept of quantum entanglement to a 5-year-old.",
        "Write a Python function to check if a number is prime."
    ]

    base, lora, tk = load_base_and_lora()

    for q in questions:
        run_comparison(q, base, lora, tk)
