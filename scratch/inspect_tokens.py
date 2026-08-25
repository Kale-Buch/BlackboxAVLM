import torch
from transformers import AutoProcessor

models = {
    "qwenVL": "Qwen/Qwen2-VL-2B-Instruct",
    "phi35": "microsoft/Phi-3.5-vision-instruct",
    "Llama32": "alpindale/Llama-3.2-11B-Vision-Instruct",
    "llava-hf": "llava-hf/llava-1.5-7b-hf",
}

for name, model_path in models.items():
    print(f"=== {name} ({model_path}) ===")
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        
        # Test targets
        targets = ["Sure", "sure, here it is", "Sure, here is"]
        for target in targets:
            # Let's see what tokens are produced
            ids_with_special = tokenizer(target, add_special_tokens=True).input_ids
            ids_no_special = tokenizer(target, add_special_tokens=False).input_ids
            tokens_with_special = [tokenizer.decode([i]) for i in ids_with_special]
            tokens_no_special = [tokenizer.decode([i]) for i in ids_no_special]
            
            print(f"Target: '{target}'")
            print(f"  add_special_tokens=True:  {ids_with_special} -> {tokens_with_special}")
            print(f"  add_special_tokens=False: {ids_no_special} -> {tokens_no_special}")
            
        # Inspect extra tokens
        if name == "Llama32":
            extra = "<|eot_id|>"
        elif name == "phi35":
            extra = "<|end|>\n"
        elif name == "qwenVL":
            extra = "<|im_end|>\n"
        elif name == "llava-hf":
            extra = "</s>"
        else:
            extra = ""
            
        print(f"Extra token: '{extra}'")
        extra_ids_true = tokenizer(extra, add_special_tokens=True).input_ids
        extra_ids_false = tokenizer(extra, add_special_tokens=False).input_ids
        print(f"  extra add_special_tokens=True:  {extra_ids_true}")
        print(f"  extra add_special_tokens=False: {extra_ids_false}")
        
        # Combined target + extra
        target_combined = "Sure" + extra
        comb_ids_false = tokenizer(target_combined, add_special_tokens=False).input_ids
        comb_tokens_false = [tokenizer.decode([i]) for i in comb_ids_false]
        print(f"Combined 'Sure' + extra (add_special_tokens=False):")
        print(f"  ids:    {comb_ids_false}")
        print(f"  tokens: {comb_tokens_false}")
        
    except Exception as e:
        print(f"Error: {e}")
    print()
