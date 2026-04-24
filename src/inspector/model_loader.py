"""
Model Loader for Llama 3.2 1B
Loads the model with full introspection flags for analysis.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B"


def pick_device():
    """Pick the best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_id=MODEL_ID, device=None):
    """
    Load model and tokenizer with introspection enabled.

    Returns:
        model: The loaded model with output_hidden_states and output_attentions enabled
        tokenizer: The tokenizer for the model
    """
    if device is None:
        device = pick_device()

    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading model from {model_id} (this may take a minute on first run)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        output_hidden_states=True,
        output_attentions=True,
    )
    model.to(device)
    model.eval()

    # Set pad token if not set (Llama doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded on {device} | Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0):
    """Simple text generation helper."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def get_model_info(model):
    """Print key model configuration details."""
    config = model.config
    info = {
        "Model": config._name_or_path,
        "Hidden Size": config.hidden_size,
        "Num Layers": config.num_hidden_layers,
        "Num Attention Heads": config.num_attention_heads,
        "Num KV Heads": config.num_key_value_heads,
        "Intermediate Size (MLP)": config.intermediate_size,
        "Vocab Size": config.vocab_size,
        "Max Position Embeddings": config.max_position_embeddings,
        "RoPE Theta": getattr(config, "rope_theta", "N/A"),
    }
    print("\n=== Model Configuration ===")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()
    return info


if __name__ == "__main__":
    model, tokenizer = load_model()
    get_model_info(model)
    print("--- Full Architecture ---")
    print(model)
    print("\n--- Test Generation ---")
    result = generate_text(model, tokenizer, "The capital of France is")
    print(f"Prompt: 'The capital of France is'")
    print(f"Output: {result}")
