"""
Solution 1 test: victim-dominant hybrid.

Formula (inverted from Stage D):
    g_comb = g_victim + λ_surr * g_ens

instead of:
    g_comb = g_ens + λ_hybrid * g_victim

Tests three surrogate weights in sequence from the same starting point:
  --surr_lam 0.0   → pure victim (P-RGF only, no surrogate)
  --surr_lam 0.1   → victim-dominant with light surrogate regularization
  --surr_lam 0.5   → victim-dominant with moderate surrogate regularization

At each checkpoint logs: L_Phi, L_LLaMA, L_LLaVA, L_v, M(x), P(Yes), behavior.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.processors import load_components

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

TRAIN_Q = "Is this image colorful?"
TARGET  = "Yes"

CHECKPOINT_ITERS = {0, 25, 50, 75, 99}


def load_model_instance(name, raw_image, device, target_text, question):
    load_model, InputsProc, AdvProc = load_components(name)
    model, processor = load_model(name, device)
    model.eval()
    inputs_processor = InputsProc(
        questions=[question],
        test_questions=[question],
        batch_size=1,
        original_image=raw_image,
        processor=processor,
        device=device,
        target_text=target_text,
    )
    adv_processor = AdvProc(
        getattr(processor, "image_processor", processor),
        device,
    )
    return model, processor, adv_processor, inputs_processor


def victim_loss_point(v_model, v_adv_proc, v_inp_proc, v_inputs, x):
    proc = v_adv_proc.process(x)
    inp = {k: v for k, v in v_inputs.items()}
    for k, v in proc.items():
        inp[k] = v
    with torch.no_grad():
        out = v_model(**inp)
        loss = v_inp_proc.get_loss(out.logits[:, :-1, :])
    return loss.item()


def get_margin_and_behavior(v_model, v_processor, v_inp_proc, v_adv_proc, x, device):
    """Returns logit margin M(x)=logit(Yes)-logit(No), P(Yes), generated text."""
    img_pil = Image.fromarray(
        (x.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    )
    proc = v_adv_proc.process(x)
    inf_inputs = v_inp_proc.get_inputs_inference(img_pil, question=TRAIN_Q)
    for k, v in proc.items():
        inf_inputs[k] = v

    with torch.no_grad():
        out = v_model(**inf_inputs)
        next_logits = out.logits[0, -1, :]
        probs = F.softmax(next_logits, dim=-1)

    tok = v_processor.tokenizer
    yes_ids = set()
    for s in ("Yes", "yes", " Yes", " yes"):
        yes_ids.update(tok.encode(s, add_special_tokens=False))
    no_ids = set()
    for s in ("No", "no", " No", " no"):
        no_ids.update(tok.encode(s, add_special_tokens=False))

    logit_yes = next_logits[list(yes_ids)].max().item()
    logit_no  = next_logits[list(no_ids)].max().item()
    prob_yes  = probs[list(yes_ids)].sum().item()
    margin    = logit_yes - logit_no

    with torch.no_grad():
        gen = v_model.generate(**inf_inputs, max_new_tokens=20, do_sample=False)
    skip = inf_inputs["input_ids"].shape[1]
    gen_text = tok.decode(gen[0][skip:], skip_special_tokens=True).strip()

    return margin, prob_yes, gen_text


def compute_surrogate_grad(model, adv_proc, inp_proc, x, device):
    x_s = x.detach().to(device).requires_grad_(True)
    proc = adv_proc.process(x_s)
    sinputs = inp_proc.get_inputs_train()
    for k, v in proc.items():
        sinputs[k] = v
    out = model(**sinputs)
    loss = inp_proc.get_loss(out.logits[:, :-1, :])
    loss_val = loss.item()
    loss.backward()
    grad = x_s.grad.detach()
    del out, loss, sinputs, proc, x_s
    torch.cuda.empty_cache()
    return grad, loss_val


def sample_structured_basis(shape, patch_size=32, device="cuda:0"):
    """Generates smooth spatial basis perturbation."""
    C, H, W = shape
    h_p, w_p = max(1, H // patch_size), max(1, W // patch_size)
    low_res = torch.randint(0, 2, (C, h_p, w_p), device=device).float() * 2.0 - 1.0
    high_res = F.interpolate(low_res.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
    return high_res / (high_res.abs().max() + 1e-8)


def get_prgf_victim_gradient(v_model, v_adv_proc, v_inp_proc, v_inputs,
                              x_curr, g_ens, lambd=0.5, epsilon=0.05, num_dirs=30, patch_size=32):
    """
    Fixed P-RGF: Samples smooth low-frequency patch directions aligned with surrogate consensus.
    Uses proper L_inf exploration so perturbations are directly visible above float precision.
    """
    dev = x_curr.device
    v_norm = g_ens.to(dev)
    v_max = v_norm.abs().max()
    v_unit = v_norm / (v_max + 1e-8) if v_max > 1e-8 else torch.zeros_like(x_curr)

    deltas, dirs = [], []
    for _ in range(num_dirs):
        w = sample_structured_basis(x_curr.shape, patch_size=patch_size, device=dev)
        if v_max > 1e-8 and lambd > 0.0:
            u_raw = np.sqrt(lambd) * v_unit + np.sqrt(1.0 - lambd) * w
        else:
            u_raw = w
        u_i = u_raw / (u_raw.abs().max() + 1e-8)
        dirs.append(u_i)

        with torch.no_grad():
            l_pos = victim_loss_point(v_model, v_adv_proc, v_inp_proc, v_inputs,
                                      (x_curr + epsilon * u_i).clamp(0, 1))
            l_neg = victim_loss_point(v_model, v_adv_proc, v_inp_proc, v_inputs,
                                      (x_curr - epsilon * u_i).clamp(0, 1))
        deltas.append((l_pos - l_neg) / (2.0 * epsilon))

    g = torch.zeros_like(x_curr)
    for d, u in zip(deltas, dirs):
        g += d * u
    return g / num_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      default="images/gray.png")
    parser.add_argument("--num_iter",   type=int,   default=100)
    parser.add_argument("--step_size",  type=float, default=0.01)
    parser.add_argument("--epsilon",    type=float, default=0.05)
    parser.add_argument("--surr_lam",   type=float, default=0.1,
                        help="Weight for surrogate REGULARIZER (victim is base). "
                             "0 = pure victim, 0.1 = light surrogate, 0.5 = moderate.")
    parser.add_argument("--prgf_lam",   type=float, default=0.5)
    parser.add_argument("--num_dirs",   type=int,   default=3,
                        help="FD directions per victim query batch. 3=6 queries, 30=60 queries.")
    parser.add_argument("--patch_size",  type=int,   default=32,
                        help="Patch size for structured low-frequency basis (e.g., 16, 32).")
    parser.add_argument("--eps_ball",    type=float, default=0.05,
                        help="Maximum L_inf perturbation radius from initial image.")
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="Interval between full checkpoint evaluations.")
    parser.add_argument("--surr0_dev",  default="cuda:0")
    parser.add_argument("--surr1_dev",  default="cuda:1")
    parser.add_argument("--surr2_dev",  default="cuda:0")
    parser.add_argument("--surr_models",
                        default="microsoft/Phi-3.5-vision-instruct,"
                                "alpindale/Llama-3.2-11B-Vision-Instruct,"
                                "llava-hf/llava-1.5-7b-hf",
                        help="Comma-separated surrogate model names (victim auto-excluded)")
    parser.add_argument("--surr_devs",  default="cuda:0,cuda:1,cuda:0",
                        help="Comma-separated devices, one per surrogate model")
    parser.add_argument("--victim_dev", default="cuda:1")
    parser.add_argument("--victim_model", default="Qwen/Qwen2-VL-2B-Instruct",
                        help="HF model name for the black-box victim")
    parser.add_argument("--out_dir",    default="runs/victim_dominant")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    tag = f"surrlam_{args.surr_lam}"
    out_csv = os.path.join(args.out_dir, f"{tag}_trajectory.csv")

    print("=" * 80)
    print("VICTIM-DOMINANT HYBRID TEST (FIXED ZEROTH-ORDER)")
    print(f"  Formula    : g_victim + {args.surr_lam} * g_ens")
    print(f"  Victim     : {args.victim_model}")
    print(f"  Task       : {TRAIN_Q!r}  →  {TARGET!r}")
    print(f"  Iterations : {args.num_iter}")
    print(f"  Dirs/step  : {args.num_dirs}  ({args.num_dirs*2} victim queries/iter)")
    print(f"  Patch Size : {args.patch_size} (Spatial basis)")
    print(f"  Eps Ball   : {args.eps_ball}")
    print("=" * 80)

    raw_image  = Image.open(args.image).convert("RGB")
    master_dev = torch.device(args.surr0_dev)

    # Build surrogate pairs, auto-excluding the victim to prevent contamination
    surr_model_names = [s.strip() for s in args.surr_models.split(",")]
    surr_dev_names   = [d.strip() for d in args.surr_devs.split(",")]
    while len(surr_dev_names) < len(surr_model_names):
        surr_dev_names.append(surr_dev_names[-1])

    surr_configs = [(m, d) for m, d in zip(surr_model_names, surr_dev_names)
                    if m != args.victim_model]
    excluded = set(surr_model_names) - {m for m, _ in surr_configs}
    if excluded:
        print(f"[!] Auto-excluded from surrogates (== victim): {excluded}")

    surrogates = []
    print(f"[*] Loading {len(surr_configs)} surrogate(s): {[m.split('/')[-1] for m, _ in surr_configs]}")
    for name, dev in surr_configs:
        m, _, ap, ip = load_model_instance(name, raw_image, dev, TARGET, TRAIN_Q)
        surrogates.append((m, ap, ip, torch.device(dev)))

    master_dev = torch.device(surr_configs[0][1])

    print(f"[*] Loading victim ({args.victim_model}) on {args.victim_dev} ...")
    v_model, v_proc, v_adv_proc, v_inp_proc = load_model_instance(
        args.victim_model, raw_image, args.victim_dev,
        TARGET, TRAIN_Q
    )
    v_inputs = v_inp_proc.get_inputs_train()

    arr = np.array(raw_image).astype(np.float32) / 255.0
    x_init = torch.from_numpy(arr).permute(2, 0, 1).to(master_dev)
    x_curr = x_init.clone()
    total_queries = 0

    checkpoint_iters = set(range(0, args.num_iter + 1, args.eval_interval)) | {args.num_iter - 1}

    csv_f = open(out_csv, "w")
    csv_f.write("iter,loss_phi,loss_llama,loss_llava,surr_mean,victim_loss,margin,prob_yes,behavior\n")

    hdr = f"{'Iter':<5} | {'L_Phi':<8} | {'L_Llama':<8} | {'L_LLaVA':<8} | {'L_surr':<8} | {'L_v':<8} | {'M(x)':<8} | {'P(Yes)%':<8} | Behavior"
    print("\n" + hdr)
    print("-" * len(hdr))

    for t in range(args.num_iter):
        is_ckpt = (t in checkpoint_iters)

        # ── Surrogate gradients ───────────────────────────────────────────────
        surr_grads, surr_losses = [], []
        if args.surr_lam > 0.0 or args.prgf_lam > 0.0:
            for m, ap, ip, dev in surrogates:
                g, l = compute_surrogate_grad(m, ap, ip, x_curr, dev)
                surr_grads.append(g.to(master_dev))
                surr_losses.append(l)
            mean_surr = sum(surr_losses) / len(surr_losses) if surr_losses else 0.0
            norm_grads = [g / (g.norm() + 1e-8) for g in surr_grads]
            g_ens = torch.stack(norm_grads).mean(dim=0) if norm_grads else torch.zeros_like(x_curr)
        else:
            mean_surr = 0.0
            g_ens = torch.zeros_like(x_curr)

        # ── Victim P-RGF estimate ─────────────────────────────────────────────
        g_victim = get_prgf_victim_gradient(
            v_model, v_adv_proc, v_inp_proc, v_inputs,
            x_curr.to(args.victim_dev), g_ens.to(args.victim_dev),
            lambd=args.prgf_lam, epsilon=args.epsilon, num_dirs=args.num_dirs,
            patch_size=args.patch_size
        ).to(master_dev)
        total_queries += args.num_dirs * 2

        # ── Gradient combination & PGD update ─────────────────────────────────
        g_comb = g_victim + args.surr_lam * g_ens
        g_step = g_comb.sign() if args.surr_lam == 0.0 else g_comb / (g_comb.abs().max() + 1e-8)
        x_next = x_curr - args.step_size * g_step
        delta = (x_next - x_init).clamp(-args.eps_ball, args.eps_ball)
        x_curr = (x_init + delta).clamp(0.0, 1.0)

        # ── Checkpoint: measure victim loss + logit margin + behavior ──────────
        v_loss = margin = prob_yes = float("nan")
        behavior = ""
        if is_ckpt:
            v_loss = victim_loss_point(
                v_model, v_adv_proc, v_inp_proc, v_inputs, x_curr.to(args.victim_dev)
            )
            total_queries += 1
            margin, prob_yes, gen_text = get_margin_and_behavior(
                v_model, v_proc, v_inp_proc, v_adv_proc, x_curr.to(args.victim_dev), args.victim_dev
            )
            behavior = "YES" if gen_text.lower().startswith("yes") else "no"

        # Pad surrogate losses to 3 slots for consistent CSV columns
        sl = surr_losses + [float("nan")] * (3 - len(surr_losses))
        csv_f.write(
            f"{t},{sl[0]:.4f},{sl[1]:.4f},{sl[2]:.4f},"
            f"{mean_surr:.4f},{v_loss:.4f},{margin:.4f},"
            f"{prob_yes*100 if not np.isnan(prob_yes) else float('nan'):.2f},{behavior}\n"
        )
        csv_f.flush()

        if is_ckpt:
            surr_str = " | ".join(f"{l:<8.4f}" for l in surr_losses)
            print(
                f"{t:<5} | {surr_str} | {mean_surr:<8.4f} | {v_loss:<8.4f} | "
                f"{margin:<+8.3f} | {prob_yes*100:<8.2f} | {behavior} ({gen_text[:30]})"
            )

    csv_f.close()
    print(f"\n[*] Done. Total queries: {total_queries}. CSV: {out_csv}")


if __name__ == "__main__":
    main()
