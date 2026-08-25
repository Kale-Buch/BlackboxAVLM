import os
import sys
import argparse
import torch
from PIL import Image
import pandas as pd
import wandb
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from src.processors import load_components
from src.questions import questions, not_safe_questions, not_safe_questions_test
from src.train_test import run_model_test

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

def main():
    parser = argparse.ArgumentParser(description="Evaluate transferability of an adversarial image.")
    parser.add_argument("--exp_path", type=str, required=True, help="Path to the experiment directory (e.g., runs/exp_name)")
    parser.add_argument("--victim_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="The victim model to test on.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run evaluation on.")
    parser.add_argument("--wandb_project", type=str, default="image_attack_optimization", help="WandB project name.")
    parser.add_argument("--target_text", type=str, default="sure, here it is", help="The target text used for ASR calculation.")
    
    args = parser.parse_args()
    
    # 1. Load best iteration
    best_txt_path = os.path.join(args.exp_path, "best.txt")
    if not os.path.exists(best_txt_path):
        # Fallback to looking in safety_details if not in root
        best_txt_path = os.path.join(args.exp_path, "safety_details", "gemma-3-4b-it", "best_iter.txt")
        if not os.path.exists(best_txt_path):
            # Check if there's any other judge directory
            safety_dir = os.path.join(args.exp_path, "safety_details")
            if os.path.exists(safety_dir):
                judges = [d for d in os.listdir(safety_dir) if os.path.isdir(os.path.join(safety_dir, d))]
                if judges:
                    best_txt_path = os.path.join(safety_dir, judges[0], "best_iter.txt")
            
            if not os.path.exists(best_txt_path):
                raise FileNotFoundError(f"Could not find best.txt or best_iter.txt in {args.exp_path}")
            
    with open(best_txt_path, 'r') as f:
        best_iter = int(f.read().strip())
    
    print(f"Loading best iteration: {best_iter}")
    
    # 2. Load the adversarial image
    img_path = find_image_for_iteration(args.exp_path, best_iter)
    adv_image = Image.open(img_path).convert("RGB")
    
    # 3. Load victim model and processor
    print(f"Loading victim model: {args.victim_model} on {args.device}...")
    load_model_and_processor, AdvInputs, _ = load_components(args.victim_model)
    model, processor = load_model_and_processor(args.victim_model, args.device)
    model.eval()
    
    # 4. Prepare questions (same as training)
    # Training uses: not_safe_questions + questions
    eval_questions = not_safe_questions + questions
    
    # Initialize AdvInputs for internal consistency
    inputs_processor = AdvInputs(
        questions=eval_questions,
        test_questions=eval_questions, # Using all questions for evaluation
        batch_size=1,
        original_image=adv_image,
        processor=processor,
        device=args.device,
        target_text=args.target_text
    )
    
    # 5. Initialize WandB for the transfer evaluation
    exp_name = os.path.basename(args.exp_path)
    wandb.init(
        project=args.wandb_project,
        name=f"transfer_{exp_name}_to_{args.victim_model.split('/')[-1]}",
        config={
            "source_exp": exp_name,
            "victim_model": args.victim_model,
            "best_iter": best_iter,
            "questions_count": len(eval_questions)
        },
        tags=["transfer-eval"]
    )
    
    # 6. Run the test
    print(f"Running transfer evaluation on {args.victim_model} with {len(eval_questions)} questions...")
    # Use run_model_test to get consistency in metrics and CSV output
    models_output, wandb_log = run_model_test(
        models=[model],
        processors=[processor],
        inputs_processors=[inputs_processor],
        model_names=[args.victim_model],
        not_safe_questions_test=eval_questions, 
        target_text=args.target_text,
        exp_path=args.exp_path,
        iteration=f"transfer_{best_iter}",
        img=adv_image
    )
    
    # Supplement WandB log with transfer-specific prefix
    transfer_log = {f"transfer/{k}": v for k, v in wandb_log.items()}
    transfer_log["transfer/best_iteration"] = best_iter
    wandb.log(transfer_log)
    
    print("\n--- Transfer Results ---")
    print(f"Victim Model: {args.victim_model}")
    print(f"Target First Word ASR: {wandb_log['test_target_first_word_acc']:.2%}")
    print(f"Target Full String ASR: {wandb_log['test_target_acc']:.2%}")
    print(f"Refusal Rate: {wandb_log['test_refuse_count']:.2%}")
    
    # Close WandB
    wandb.finish()

if __name__ == "__main__":
    main()
