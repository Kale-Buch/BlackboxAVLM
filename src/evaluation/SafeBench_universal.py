import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

print(sys.path)

from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from src.processors import load_components
from src.processors.phi3processor import batch_processing

sfb_path = os.path.join(project_root, "SafeBench_Text")
dict_suf_modelname = {
    "qwenVL": "Qwen/Qwen2-VL-2B-Instruct",
    "phi35": "microsoft/Phi-3.5-vision-instruct",
    "Llama32": "alpindale/Llama-3.2-11B-Vision-Instruct",
    "llava-hf": "llava-hf/llava-1.5-7b-hf",
}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
######################################

def find_image_for_iteration(exp_dir, iteration):
    import glob
    import re
    # 1. If iteration is "final", return final.png if it exists
    if str(iteration).lower() == "final":
        final_path = os.path.join(exp_dir, "optimized_image_iter_final.png")
        if os.path.exists(final_path):
            return final_path

    # 2. Check direct iteration file
    direct_path = os.path.join(exp_dir, f"optimized_image_iter_{iteration}.png")
    if os.path.exists(direct_path):
        return direct_path

    # 3. Check direct iteration + 1 (legacy behavior when grad_accum_steps=1)
    try:
        plus_one_path = os.path.join(exp_dir, f"optimized_image_iter_{int(iteration) + 1}.png")
        if os.path.exists(plus_one_path):
            return plus_one_path
    except ValueError:
        pass

    # 4. Do index matching between saved CSV files and PNG files
    csv_pattern = os.path.join(exp_dir, "test_results_iter_*.csv")
    csv_files = glob.glob(csv_pattern)
    csv_iters = []
    for f in csv_files:
        match = re.search(r'test_results_iter_(\d+)\.csv', f)
        if match:
            csv_iters.append(int(match.group(1)))
    csv_iters = sorted(csv_iters)

    png_pattern = os.path.join(exp_dir, "optimized_image_iter_*.png")
    png_files = glob.glob(png_pattern)
    png_iters = []
    for f in png_files:
        match = re.search(r'optimized_image_iter_(\d+)\.png', f)
        if match:
            png_iters.append(int(match.group(1)))
    png_iters = sorted(png_iters)

    try:
        target_iter_int = int(iteration)
        if target_iter_int in csv_iters:
            idx = csv_iters.index(target_iter_int)
            if idx < len(png_iters):
                mapped_png_iter = png_iters[idx]
                mapped_path = os.path.join(exp_dir, f"optimized_image_iter_{mapped_png_iter}.png")
                if os.path.exists(mapped_path):
                    return mapped_path
    except Exception:
        pass

    # 5. Fallback: ratio-based estimation
    try:
        target_iter_int = int(iteration)
        if len(csv_iters) > 1 and len(png_iters) > 1:
            first_csv = [c for c in csv_iters if c > 0]
            first_png = [p for p in png_iters if p > 0]
            if first_csv and first_png:
                ratio = first_csv[0] / first_png[0]
                estimated_png_iter = round(target_iter_int / ratio)
                closest_png_iter = min(png_iters, key=lambda x: abs(x - estimated_png_iter))
                closest_path = os.path.join(exp_dir, f"optimized_image_iter_{closest_png_iter}.png")
                if os.path.exists(closest_path):
                    return closest_path
    except Exception:
        pass

    # 6. Fallback: closest match
    try:
        target_iter_int = int(iteration)
        if png_iters:
            closest_png_iter = min(png_iters, key=lambda x: abs(x - target_iter_int))
            closest_path = os.path.join(exp_dir, f"optimized_image_iter_{closest_png_iter}.png")
            if os.path.exists(closest_path):
                return closest_path
    except Exception:
        pass

    return os.path.join(exp_dir, f"optimized_image_iter_{iteration + 1}.png")

def main(exp, model_suf, device):
    assert model_suf in dict_suf_modelname, "Model suffix not found. Shall be one of: " + ", ".join(dict_suf_modelname.keys())
    model_name = dict_suf_modelname[model_suf]
    load_model_and_processor, AdvInputs, DIProcessor = load_components(model_name, )
    model, processor = load_model_and_processor(model_name, torch.device(device))
    paths = [os.path.join(PROJECT_ROOT, "tests", f"{exp[0]}_{exp[1]}", model_suf)]
    
    exp_dir = os.path.join(PROJECT_ROOT, "runs", exp[0])
    resolved_image = find_image_for_iteration(exp_dir, exp[1])
    images = [resolved_image]

    # Iterate over list of length 1 - from legacy code
    for path, image in zip(paths, images):
        if not os.path.exists(image):
            print("ERROR, image doesn't exist:", image)
        os.makedirs(path, exist_ok=True)
        
        save_path = path
        original_image = Image.open(image)
        
        for i in tqdm(range(1, 24)):
            sfb_i = pd.read_csv(f"{sfb_path}/{i}.csv", names=[0])
            generated_texts = []

            batch_size = 1
            batches = [sfb_i.iloc[j:j+batch_size] for j in range(0, len(sfb_i), batch_size)]

            for batch in batches:
                if model_suf == "phi35":
                    inference_prompts = [
                        f'<|user|>\n<|image_1|>\n{text}<|end|>\n<|assistant|>\n' for text in batch[0]
                    ]
                    inputs = batch_processing(
                        processor, 
                        inference_prompts, 
                        [original_image for _ in batch[0]]
                    ).to(device)
                    
                else:
                    if model_suf == "qwenVL":
                        if processor.chat_template is None:
                            processor.chat_template = processor.tokenizer.chat_template
                    inference_prompts = [processor.apply_chat_template([
                            {
                                "role": "user", 
                                "content": 
                                    [
                                        {"type": "image"}, 
                                        {"type": "text", "text": question}
                                    ]
                            },
                        ], add_generation_prompt=True) for question in batch[0]]

                    inputs = processor(
                            text=inference_prompts, 
                            images=[original_image for _ in batch[0]], 
                            return_tensors="pt", 
                            padding=True
                        ).to(device)

                outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False, min_new_tokens=32)
                for idx, output in enumerate(outputs):
                    ans = processor.decode(output[inputs["input_ids"].shape[1]:],  skip_special_tokens=True)
                    generated_texts.append(ans)
            sfb_answers = pd.DataFrame(generated_texts)
            sfb_answers.to_csv(os.path.join(save_path, f"{i}.csv"),index=False, header=False)

######################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Adversarial Model Testing")
    parser.add_argument("--exp", type=str, help="Experiment name")
    parser.add_argument("--iter", type=int, help="Iteration number")
    parser.add_argument("--model_suf", type=str, help="What model to use for inferencing the adversarial image")
    parser.add_argument("--cuda_num", type=int, help="Cuda number")

    args = parser.parse_args()
    exp = [args.exp, args.iter]
    main(exp, args.model_suf, args.cuda_num)
