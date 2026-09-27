from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
import os
import argparse
import wandb  # Import WandB
import random  # Added import for random sampling
from torchvision.transforms import GaussianBlur # Additional regularization for noise

from processors import load_components
from train_test import run_model_test

from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import GaussianBlur # Additional regularization for noise


def diverse_input_transform(x, prob=0.7):
    """
    DIM (Diverse Input Method): Apply random transformations to input
    to prevent overfitting to surrogate model preprocessing pipelines.
    This significantly improves transferability to unseen models.
    """
    if random.random() > prob:
        return x
    
    C, H, W = x.shape
    
    # Random resize within small range, then resize back
    scale = random.uniform(0.9, 1.1)
    new_h = max(2, int(H * scale))
    new_w = max(2, int(W * scale))
    x_t = F.interpolate(x.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
    x_t = F.interpolate(x_t, size=(H, W), mode='bilinear', align_corners=False)
    x = x_t.squeeze(0)
    
    # Random JPEG-like quantization noise
    if random.random() > 0.5:
        noise = torch.randn_like(x) * 0.015
        x = x + noise
    
    return x


def setup_device(i=0):
    """Setup computing device."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    num_gpus = torch.cuda.device_count()
    return torch.device(f"cuda:{i % num_gpus}")

def create_directory(exp_name, base_path="./runs"):
    """Creates a directory for experiment logs and outputs."""
    exp_path = os.path.join(base_path, exp_name)
    os.makedirs(exp_path, exist_ok=True)
    return exp_path

def save_checkpoint(image: Image.Image, tensor: torch.Tensor, path: str, iteration: int):
    """Saves the current image checkpoint."""
    image.save(os.path.join(path, f"optimized_image_iter_{iteration}.png"))
    tensor.cpu().detach().numpy().astype(np.float32).tofile(os.path.join(path, f"optimized_image_iter_{iteration}.bin"))

def initialize_wandb(exp_name, config):
    """Initialize WandB for experiment tracking."""
    wandb.init(
        project="image_attack_optimization",
        name=exp_name,
        config=config,  # Logging configuration
        tags=["image-attack", "training", "transformers"],
        mode="online"
    )

def log_metrics_wandb(
        iteration: int,
        loss: torch.Tensor,
        final_image: np.array,
        final_tensor: torch.Tensor,
        generated_table: wandb.Table,
        save_steps: int
    ):
    """Logs metrics and images to WandB."""

    if iteration % save_steps == 0:  # Log images and model output every `save_steps` iterations
        wandb.log({"optimized_image": [wandb.Image(final_image, caption=f"Iteration {iteration}")]})
        # Log the generated text to WandB
        wandb.log({"generated_text": generated_table})
        wandb.log({"iteration": iteration})
        # log x+x_0
        wandb.log({"optimized_tensor": [wandb.Image(final_tensor, caption=f"Iteration {iteration}")]})

def create_mask(mask_type, mask_size, image_shape, device):
    """Creates a mask tensor based on the specified mask_type and mask_size."""
    mask = torch.zeros(image_shape).to(device)
    C, H, W = image_shape
    if mask_type == 'corner':
        n = mask_size
        mask[:, :n, :n] = 1.0
    elif mask_type == 'bottom_lines':
        k = mask_size
        mask[:, -k:, :] = 1.0
    elif mask_type == 'random_square':
        n = mask_size
        # Randomly select top-left corner
        i = random.randint(0, H - n)
        j = random.randint(0, W - n)
        mask[:, i:i+n, j:j+n] = 1.0
    else:
        mask = torch.ones(image_shape).to(device)
    return mask

def image_fit_loss(
        x_0: torch.Tensor, 
        x: torch.Tensor, 
        lower_bound, 
        upper_bound, 
        center_force = 0.9
    ):
    # We compute the sum
    x_sum = (x_0 + x)
    
    lower_bound = torch.zeros(x_sum.shape).to(x_sum.device)
    upper_bound = torch.ones(x_sum.shape).to(x_sum.device)
    
    # Penalty for out-of-bounds
    lower_penalty = torch.relu(center_force*lower_bound - x_sum)  # if x_sum < center_force*lower_bound
    upper_penalty = torch.relu(x_sum - center_force*upper_bound)  # if x_sum > center_force*upper_bound
    
    # MSE-based penalty
    penalty = torch.mean(lower_penalty**2 + upper_penalty**2)
    
    return penalty

def pil_to_tensor(image: Image.Image, do_convert_rgb: bool = True, resize: bool = False) -> torch.Tensor:
    """
    Convert a PIL image to a tensor.

    Args:
        image: PIL image
        resize: Whether to resize the image to the optimal size for the model.

    Returns:
        A tensor image, shape (3, H, W)
    """
    if do_convert_rgb:
        image = image.convert("RGB")
    tensor_image = torch.tensor(np.array(image).astype(np.float32) / 255).permute(2, 0, 1)

    if resize:
        raise NotImplementedError("Resizing is not universal for models!!!")
    return tensor_image

#-----------------------------------------------------------------------------------------------------------

def sample_direction(x, bcd_block_size=None):
    """Generate a normalized random direction, optionally within a BCD block."""
    if bcd_block_size is not None and bcd_block_size > 0:
        C, H, W = x.shape
        h_start = random.randint(0, max(0, H - bcd_block_size))
        w_start = random.randint(0, max(0, W - bcd_block_size))
        
        direction = torch.zeros_like(x)
        block = torch.randn(C, min(H, bcd_block_size), min(W, bcd_block_size), device=x.device)
        direction[:, h_start:h_start+bcd_block_size, w_start:w_start+bcd_block_size] = block / (block.norm() + 1e-8)
        dimension = C * min(H, bcd_block_size) * min(W, bcd_block_size)
    else:
        direction = torch.randn_like(x)
        direction = direction / (direction.norm() + 1e-8)
        dimension = x.numel()
        
    return direction, dimension


def query_blackbox(
    victim_model,
    victim_adv_processor,
    victim_inputs_processor,
    victim_inputs,
    image,
):
    """
    Query the victim model and return one scalar loss.

    This function is the model-specific part.
    """

    inputs = victim_inputs.copy()
    processed = victim_adv_processor.process(image)
    for key, value in processed.items():
        inputs[key] = value

    with torch.no_grad():
        outputs = victim_model(**inputs)
        logits = outputs.logits[:, :-1, :]
        return victim_inputs_processor.get_loss(logits).detach()

def get_blackbox_grad(
    victim_model,
    victim_adv_processor,
    victim_inputs_processor,
    victim_inputs,
    image,
    epsilon=0.02,
    num_samples=5,
    bcd_block_size=None,
    zo_mode="two_sided",
    ):
    """
    Zeroth-order gradient estimator supporting two-sided or one-sided finite differences.

    `epsilon` is the finite-difference step size. Values in [0.01, 0.05] work well in practice.
    `num_samples` random directions are averaged to reduce estimation noise.
    `zo_mode`: 'two_sided' (2 queries/sample) or 'one_sided' (1 query/sample + 1 base query, ~2x faster).

    Returns:
        grad (Tensor): ZO gradient estimate w.r.t. `image`.
        mean_loss_delta (float): Mean |loss delta| across samples.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    grad = torch.zeros_like(image)
    loss_deltas = []

    with torch.no_grad():
        if zo_mode == "one_sided":
            loss_base = query_blackbox(
                victim_model,
                victim_adv_processor,
                victim_inputs_processor,
                victim_inputs,
                image,
            )

        for _ in range(num_samples):
            direction, dimension = sample_direction(image, bcd_block_size)

            image_plus = (image + epsilon * direction).clamp(0.0, 1.0)
            loss_plus = query_blackbox(
                victim_model,
                victim_adv_processor,
                victim_inputs_processor,
                victim_inputs,
                image_plus,
            )

            if zo_mode == "one_sided":
                delta_val = (loss_plus - loss_base).abs().item()
                loss_deltas.append(delta_val)
                directional_derivative = (loss_plus - loss_base) / epsilon
            else:
                image_minus = (image - epsilon * direction).clamp(0.0, 1.0)
                loss_minus = query_blackbox(
                    victim_model,
                    victim_adv_processor,
                    victim_inputs_processor,
                    victim_inputs,
                    image_minus,
                )
                delta_val = (loss_plus - loss_minus).abs().item()
                loss_deltas.append(delta_val)
                directional_derivative = (loss_plus - loss_minus) / (2.0 * epsilon)

            # Scale by the dimensionality of the perturbed space
            grad += dimension * directional_derivative * direction

    mean_loss_delta = sum(loss_deltas) / len(loss_deltas)
    return grad / num_samples, mean_loss_delta

#-----------------------------------------------------------------------------------------------------------

def get_blackbox_grad_exhaustive(
    victim_model,
    victim_adv_processor,
    victim_inputs_processor,
    victim_inputs,
    image,
    epsilon=0.02,
    patch_size=48,
    zo_mode="two_sided",
):
    """
    Exhaustive patch-wise zeroth-order gradient estimator (Zer0-Jack style).

    Iterates over EVERY non-overlapping `patch_size×patch_size` tile in the image and estimates
    a per-patch finite-difference gradient.
    Scales each patch estimate by d_patch = C * patch_size^2 to ensure mathematically correct gradient scale.
    Supports 'two_sided' (2 queries/patch) or 'one_sided' (1 query/patch + 1 base query, ~2x faster).

    Args:
        image (Tensor): Float32 tensor of shape (C, H, W), values in [0, 1].
        epsilon: Finite-difference step in pixel space.
        patch_size: Side length of the non-overlapping tile (pixels).
        zo_mode: 'two_sided' or 'one_sided'.

    Returns:
        grad (Tensor): Dense ZO gradient estimate, same shape as `image`.
        mean_loss_delta (float): Mean |loss delta| across all patches.
    """
    C, H, W = image.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    d_patch = C * patch_size * patch_size

    grad = torch.zeros_like(image)
    loss_deltas = []

    with torch.no_grad():
        if zo_mode == "one_sided":
            loss_base = query_blackbox(
                victim_model,
                victim_adv_processor,
                victim_inputs_processor,
                victim_inputs,
                image,
            )

        for ph in range(num_patches_h):
            for pw in range(num_patches_w):
                h0 = ph * patch_size
                w0 = pw * patch_size

                # Independent Gaussian direction for this patch only
                gauss = torch.randn(
                    C, patch_size, patch_size,
                    device=image.device,
                    dtype=image.dtype,
                )
                gauss = gauss / (gauss.norm() + 1e-8)   # unit direction

                # Embed patch perturbation into full-image tensors
                direction = torch.zeros_like(image)
                direction[:, h0:h0+patch_size, w0:w0+patch_size] = gauss

                image_plus = (image + epsilon * direction).clamp(0.0, 1.0)
                loss_plus = query_blackbox(
                    victim_model, victim_adv_processor,
                    victim_inputs_processor, victim_inputs, image_plus,
                )

                if zo_mode == "one_sided":
                    delta_val = (loss_plus - loss_base).abs().item()
                    loss_deltas.append(delta_val)
                    delta = (loss_plus - loss_base) / epsilon
                else:
                    image_minus = (image - epsilon * direction).clamp(0.0, 1.0)
                    loss_minus = query_blackbox(
                        victim_model, victim_adv_processor,
                        victim_inputs_processor, victim_inputs, image_minus,
                    )
                    delta_val = (loss_plus - loss_minus).abs().item()
                    loss_deltas.append(delta_val)
                    delta = (loss_plus - loss_minus) / (2.0 * epsilon)

                # Properly scaled by patch dimension d_patch
                grad[:, h0:h0+patch_size, w0:w0+patch_size] += d_patch * delta * gauss

    mean_loss_delta = sum(loss_deltas) / (len(loss_deltas) + 1e-12)
    return grad, mean_loss_delta

#-----------------------------------------------------------------------------------------------------------

def sample_structured_basis(shape, patch_size=32, device="cuda:0"):
    """Generates smooth spatial basis perturbation."""
    C, H, W = shape
    h_p, w_p = max(1, H // patch_size), max(1, W // patch_size)
    low_res = torch.randint(0, 2, (C, h_p, w_p), device=device).float() * 2.0 - 1.0
    high_res = F.interpolate(low_res.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
    return high_res / (high_res.abs().max() + 1e-8)


def get_blackbox_grad_prgf(
    victim_model,
    victim_adv_processor,
    victim_inputs_processor,
    victim_inputs,
    image,
    g_ens,
    prgf_lambda=0.5,
    epsilon=0.05,
    num_dirs=15,
    patch_size=32,
):
    """
    Structured P-RGF: Samples smooth low-frequency patch directions aligned with surrogate consensus.
    Uses proper L_inf exploration so perturbations are directly visible above float precision.
    """
    dev = image.device
    v_norm = g_ens.to(dev)
    v_max = v_norm.abs().max()
    v_unit = v_norm / (v_max + 1e-8) if v_max > 1e-8 else torch.zeros_like(image)

    deltas, dirs = [], []
    loss_deltas = []
    for _ in range(num_dirs):
        w = sample_structured_basis(image.shape, patch_size=patch_size, device=dev)
        if v_max > 1e-8 and prgf_lambda > 0.0:
            u_raw = np.sqrt(prgf_lambda) * v_unit + np.sqrt(1.0 - prgf_lambda) * w
        else:
            u_raw = w
        u_i = u_raw / (u_raw.abs().max() + 1e-8)
        dirs.append(u_i)

        loss_pos = query_blackbox(
            victim_model, victim_adv_processor,
            victim_inputs_processor, victim_inputs,
            (image + epsilon * u_i).clamp(0.0, 1.0),
        )
        loss_neg = query_blackbox(
            victim_model, victim_adv_processor,
            victim_inputs_processor, victim_inputs,
            (image - epsilon * u_i).clamp(0.0, 1.0),
        )
        delta = (loss_pos - loss_neg).item()
        deltas.append(delta / (2.0 * epsilon))
        loss_deltas.append(abs(delta))

    g = torch.zeros_like(image)
    for d, u in zip(deltas, dirs):
        g = g + d * u
    mean_loss_delta = sum(loss_deltas) / (len(loss_deltas) + 1e-12) if loss_deltas else 0.0
    return (g / num_dirs), mean_loss_delta

#-----------------------------------------------------------------------------------------------------------

def get_blackbox_grad_pzo(
    victim_model,
    victim_adv_processor,
    victim_inputs_processor,
    victim_inputs,
    image,
    surrogate_grads,
    epsilon=0.01,
):
    """
    Prior-Guided Zeroth-Order (P-ZO) gradient estimator.

    Projects black-box queries into the low-dimensional subspace S spanned by
    the surrogate gradients:
        S = span{g_1, ..., g_K}
    Constructs an orthonormal basis of S via QR decomposition and evaluates
    symmetric finite differences along each basis vector.

    Args:
        victim_model: Blackbox model instance.
        victim_adv_processor: Adversarial image processor for victim.
        victim_inputs_processor: Inputs processor for victim.
        victim_inputs: Base model inputs for victim.
        image (Tensor): Float32 tensor of shape (C, H, W) on victim_device.
        surrogate_grads (list of Tensor): Surrogate gradients (each C, H, W).
        epsilon (float): Finite-difference step in pixel space.

    Returns:
        grad (Tensor): Estimated gradient inside S, same shape and device as image.
        mean_loss_delta (float): Mean absolute loss change along basis directions.
        alphas (list of float): Directional derivatives along each basis vector.
    """
    if not surrogate_grads:
        return torch.zeros_like(image), 0.0, []

    device = image.device
    dtype = image.dtype
    C, H, W = image.shape

    # 1. Build surrogate matrix on victim device
    sum_grad = torch.zeros_like(image)
    flat_vectors = []
    for g in surrogate_grads:
        g_dev = g.to(device=device, dtype=dtype)
        sum_grad = sum_grad + g_dev
        flat_vectors.append(g_dev.flatten())

    if sum_grad.norm() > 1e-8:
        # Place consensus direction first so basis[0] aligns with ensemble consensus
        flat_vectors.insert(0, sum_grad.flatten())

    matrix = torch.stack(flat_vectors)  # shape (num_vectors, D)
    Q, R = torch.linalg.qr(matrix.T)   # Q is (D, num_vectors), R is (num_vectors, num_vectors)

    r_diag = torch.abs(torch.diag(R))
    valid_mask = r_diag > 1e-4
    rank = int(valid_mask.sum().item())
    if rank == 0:
        return torch.zeros_like(image), 0.0, []

    basis = [Q[:, i].view(C, H, W) for i in range(rank)]

    # 2. Query victim model along each orthonormal basis vector
    grad = torch.zeros_like(image)
    loss_deltas = []
    alphas = []

    with torch.no_grad():
        for u in basis:
            # Scale u to have unit L_infinity norm so that epsilon corresponds to max per-pixel perturbation,
            # ensuring perturbations stay well above bfloat16/float16 precision floors
            u_max = u.abs().max()
            u_query = u / (u_max + 1e-8) if u_max > 1e-8 else u

            image_plus = (image + epsilon * u_query).clamp(0.0, 1.0)
            loss_plus = query_blackbox(
                victim_model, victim_adv_processor,
                victim_inputs_processor, victim_inputs, image_plus,
            )

            image_minus = (image - epsilon * u_query).clamp(0.0, 1.0)
            loss_minus = query_blackbox(
                victim_model, victim_adv_processor,
                victim_inputs_processor, victim_inputs, image_minus,
            )

            delta_val = (loss_plus - loss_minus).abs().item()
            loss_deltas.append(delta_val)

            # Directional derivative along the L2-unit basis vector u:
            # The query perturbation image ± epsilon * (u / u_max) corresponds to
            # a step h = epsilon / u_max along the L2-unit basis vector u, ensuring
            # per-pixel displacements stay above bfloat16/float16 precision floors.
            # The directional derivative along the L2-unit basis vector u is:
            #   beta = (loss_plus - loss_minus) / (2 * h) = ((loss_plus - loss_minus) / (2 * epsilon)) * u_max
            # Reconstructing grad = sum(beta_i * u_i) with the L2-unit basis vectors {u_i}
            # produces the exact orthogonal projection of the victim gradient onto S:
            #   \hat{g}_v = P_S(\nabla L_v(x)) = sum_k <\nabla L_v, q_k> q_k
            beta = ((loss_plus - loss_minus) / (2.0 * epsilon)).item() * (u_max.item() if torch.is_tensor(u_max) else u_max)
            alphas.append(beta)

            # Reconstruct genuine orthogonal projection: grad = sum(beta_i * q_i)
            grad = grad + beta * u

    mean_loss_delta = sum(loss_deltas) / len(loss_deltas) if loss_deltas else 0.0
    return grad, mean_loss_delta, alphas

#-----------------------------------------------------------------------------------------------------------



def train(
    exp_name,
    img_orig,
    prompt,
    target_text,
    model_names,
    victim,
    lr,
    num_iterations,
    save_steps,
    batch_size,
    grad_accum_steps,
    scheduler_step_size,
    scheduler_gamma,
    restart_num,          # Added for optimizer restart
    mask_type,            # Added for mask selection
    mask_size,            # Added for mask size
    clamp_method,         # Added for clamping method
    epsilon,              # Added for epsilon in 4.2.3. IMPLEMENTATION DETAILS
    sigma,                # Added for sigma in 4.2.3. IMPLEMENTATION DETAILS
    start_from_white,      # Added for starting from white image
    target_text_random,
    DPO_flag = True,
    refuse_prob = 0.1,
    model_weights = None,  # <-- NEW
    attack_norm = 0.5,
    # gaussian blur
    use_gaussian_blur = False,
    gblur_kernel_size = 5,
    # random crop
    use_local_crop = False,
    crop_scale_min = 0.6,
    crop_scale_max = 1.0,
    crop_ratio_min = 0.75,
    crop_ratio_max = 1.33,
    # Transferability enhancements
    use_grad_norm = False,       # Normalize per-model gradients before aggregation
    use_momentum = False,        # MI-FGSM style momentum for gradient stabilization
    momentum_decay = 0.9,        # Momentum decay factor (mu)
    use_diverse_input = False,   # DIM: diverse input transforms
    diverse_input_prob = 0.7,    # Probability of applying DIM per iteration
    n_multi_crop = 1,            # Number of crops to average (M-Attack style MCA)
    use_query_based_gradient_check = False,
    zo_every_n_iters = 1,        # Run ZO only every N iterations (1 = every iter). N=10 cuts ZO overhead by 10×.
    bcd_block_size = None,       # Block Coordinate Descent size for zeroth-order queries
    zo_epsilon = 0.02,           # Finite-difference step for ZO gradient (decoupled from sigma)
    zo_num_samples = 5,          # Directions averaged per ZO query to reduce estimation noise
    # Exhaustive patch-wise BCD (Zer0-Jack style)
    use_exhaustive_bcd = False,  # If True, sweep every patch instead of random BCD samples
    zo_patch_size = 48,          # Patch tile size for exhaustive BCD (pixels)
    zo_weight_multiplier = 1.0,  # Multiply ZO gradient norm before normalization; >1 → ZO gets proportionally more weight
    zo_cosine_threshold = 0.0,   # (Low-signal regime only) drop ZO noise if cosine with surrogates < this
    zo_signal_threshold = 0.001, # zo_mean_loss_delta must exceed this to be treated as real signal vs. noise
    zo_mode = "two_sided",       # 'two_sided' (2 queries/sample) or 'one_sided' (1 query/sample + 1 base, ~2x faster)
    use_pzo = False,             # Prior-Guided ZO: project queries into surrogate gradient subspace
    pzo_epsilon = 0.05,          # Finite-difference step size for P-ZO queries (0.05 ensures above quantization floor)
    use_victim_dominant = False, # Victim-dominant hybrid formulation: g_victim + surr_lambda * g_ens
    surr_lambda = 0.1,           # Regularizer weight for surrogate consensus in victim-dominant mode
    prgf_lambda = 0.5,           # Surrogate prior fraction in P-RGF query directions
    prgf_patch_size = 32,        # Low-frequency patch size for P-RGF spatial basis
    prgf_num_dirs = 15,          # Number of random directions per iteration for P-RGF
    ):
    """Train the model on the given image with specific settings."""
    from questions import questions, not_safe_questions, not_safe_questions_test
    from answers import answers, adv_answers
    questions = not_safe_questions + questions 

    refuse_flag = False
    
    if target_text_random:
        target_text = answers + adv_answers
    
    if prompt != "list":
        questions = [prompt]

    # Create exp directory
    exp_path = create_directory(exp_name)
    
    # Preprocess images and prepare tensors
    if os.path.exists(img_orig):
        original_image = Image.open(img_orig).convert("RGB")
    elif os.path.exists(os.path.join("./images", img_orig)):
        original_image = Image.open(os.path.join("./images", img_orig)).convert("RGB")
    else:
        raise FileNotFoundError(f"Cannot find {img_orig}")
    print("Original image size: ", original_image.size)

    # Initialize gaussian blur object
    if use_gaussian_blur:
        gaussian_blur = GaussianBlur(kernel_size=gblur_kernel_size)
    else:
        gaussian_blur = None

    # Load model and processor
    devices = []
    models = []
    processors = []
    adv_processors = []
    inputs_processors = []
    for i, model_name in enumerate(model_names):
        print(f"Loading {model_name}")
        print(f"Device number is {i}")
        devices.append(setup_device(i))
        load_model_and_processor, AdvInputs, DifferentiableImageProcessor = load_components(model_name)
        model, processor = load_model_and_processor(model_name, devices[-1])
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        orig_img_processor = getattr(processor, "image_processor", processor)
        adv_processor = DifferentiableImageProcessor(orig_img_processor, devices[-1])
        inputs_processor = AdvInputs(
            questions=questions, 
            test_questions=not_safe_questions_test, 
            batch_size=batch_size, 
            original_image=original_image, 
            processor=processor, 
            device=devices[-1], 
            target_text=target_text)
        
        models.append(model)
        processors.append(processor)
        adv_processors.append(adv_processor)
        inputs_processors.append(inputs_processor)

    if use_query_based_gradient_check or use_victim_dominant:
        victim_name = victim[0] if isinstance(victim, (list, tuple)) else victim
        victim_device = setup_device(len(devices))
        print(f"Loading victim model {victim_name} on {victim_device} for query-based gradient checks / victim-dominant mode")
        load_victim, VictimAdvInputs, VictimImageProcessor = load_components(victim_name)
        victim_model, victim_processor = load_victim(victim_name, victim_device)
        victim_model.eval()
        for parameter in victim_model.parameters():
            parameter.requires_grad = False
        victim_inputs_processor = VictimAdvInputs(
            questions=questions,
            test_questions=not_safe_questions_test,
            batch_size=1,
            original_image=original_image,
            processor=victim_processor,
            device=victim_device,
            target_text=target_text,
        )
        victim_image_processor = VictimImageProcessor(
            getattr(victim_processor, "image_processor", victim_processor),
            victim_device,
        )
        victim_inputs_processor.questions = [not_safe_questions_test[0]]
        victim_chk_inputs = victim_inputs_processor.get_inputs_train()
        print(f"Victim model loaded: {victim_name} on {victim_device} (Fixed reference prompt: '{not_safe_questions_test[0]}')")
    
    device_last = setup_device(len(devices) - 1)

    x_0 = pil_to_tensor(original_image, resize=False, do_convert_rgb=adv_processors[0].do_convert_rgb).to(device_last)
    # Initialize or override model_weights if needed
    if model_weights is None:
        # If not provided, default each model to weight 1.0
        model_weights = [1.0]*len(models)
    elif len(model_weights) != len(models):
        raise ValueError("The length of model_weights must match the number of model_names.")
    
    # Initialize x and optimizer based on clamping method
    if clamp_method == 'tanh':
        p = torch.zeros(x_0.shape, requires_grad=True, device=device_last)
        optimizer = torch.optim.AdamW([p], lr=lr)
    else:
        raise NotImplementedError("Clamping method except tanh are not implemented yet.")
    
    # Create mask
    if mask_type is not None and mask_size is not None:
        mask = create_mask(mask_type, mask_size, x_0.shape, device_last)
    else:
        mask = (x_0 != 0).int()

    # Initialize local_crop
    if use_local_crop:
        local_crop = RandomResizedCrop(
            size=(x_0.shape[1], x_0.shape[2]),
            scale=(crop_scale_min, crop_scale_max),
            ratio=(crop_ratio_min, crop_ratio_max)
        ) # DifferentiableRandomCrop

    # save mask
    torch.save(mask, os.path.join(exp_path, 'mask.pt'))
    Image.fromarray((mask.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(exp_path, 'mask.png'))

    # Set up a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # Initialize WandB
    initialize_wandb(exp_name, {
        "learning_rate": lr,
        "batch_size": batch_size,
        "num_iterations": num_iterations,
        "grad_accum_steps": grad_accum_steps,
        "scheduler_step_size": scheduler_step_size,
        "scheduler_gamma": scheduler_gamma,
        "original_mean": x_0.mean(),
        "original_std": x_0.std(),
        "target_text": target_text,
        "full_prompt": prompt,
        "questions amount": len(questions),
        "restart_num": restart_num,
        "mask_type": mask_type,
        "mask_size": mask_size,
        "clamp_method": clamp_method,
        "sigma": sigma,
        "start_from_white": start_from_white,
        "DPO_flag": DPO_flag,
        "refuse_prob": refuse_prob,
        "attack_norm": epsilon,
        "model_weights": model_weights,
        "use_gaussian_blur": use_gaussian_blur,
        "gblur_kernel_size": gblur_kernel_size,
        "use_local_crop": use_local_crop,
        "crop_scale_min": crop_scale_min,
        "crop_scale_max": crop_scale_max,
        "crop_ratio_min": crop_ratio_min,
        "crop_ratio_max": crop_ratio_max
    })

    min_losses = []

    # Создаём таблицу для логгирования выводов моделей
    model_outputs_table = wandb.Table(columns=["iteration"] + model_names)


    # Gradient accumulation variables
    global_iteration = 0
    accumulated_loss = 0
    accumulated_p_grad = torch.zeros_like(p)
    
    # Momentum buffer for MI-FGSM style gradient stabilization
    if use_momentum:
        grad_momentum = torch.zeros_like(p)
    
    # Std of difference between x_resaved and x_0 + x, used for updating noise.std
    resave_error_std = 0.001
    resaved_loss_val = 0.0
    resave_error_mean = 0.0
    resave_error_l1 = 0.0

    # ── Victim ZO loss tracking (for victim-aware checkpoint selection) ─────────
    # Accumulates zo_mean_loss_delta across iterations between each save_steps
    # checkpoint.  At each checkpoint the average is written to victim_zo_log.csv
    # so that find_best_iter can pick the checkpoint where the victim was most
    # sensitive — rather than the surrogate-optimal checkpoint.
    victim_zo_log_path = os.path.join(exp_path, "victim_zo_log.csv")
    if use_query_based_gradient_check or use_victim_dominant:
        with open(victim_zo_log_path, "w") as _f:
            _f.write("global_iteration,victim_loss,avg_zo_loss_delta\n")
    _zo_loss_accum = 0.0
    _zo_steps_accum = 0

    for iteration in tqdm(range(num_iterations)):
        # Possibly switch target_text if DPO_flag or random
        if DPO_flag or target_text_random:
            coin = random.random()
            print("coin:", coin)
            if DPO_flag and coin < refuse_prob:
                for inputs_processor in inputs_processors:
                    random_text = random.choice(inputs_processor.refuses)
                    inputs_processor.set_target_text(random_text)
                refuse_flag = True
            elif target_text_random:
                random_text = random.choice(inputs_processor.target_texts)
                for inputs_processor in inputs_processors:
                    inputs_processor.set_target_text(random_text)
                refuse_flag = False
            else:
                random_text = target_text
                for inputs_processor in inputs_processors:
                    inputs_processor.set_target_text(random_text)
                refuse_flag = False
            print(f"{refuse_flag} text:", random_text)
        
        # Update mask for random square (if needed)
        if mask_type == 'random_square':
            raise NotImplementedError("Dynamic random-square mask updating not implemented.")

        # Prepare image input for training
        if clamp_method == 'tanh':
            x = attack_norm * torch.tanh(p)
        else:
            raise NotImplementedError("Clamping method except tanh are not implemented yet.")

        # Apply gaussian blur to trained x and save it later 
        if use_gaussian_blur:
            x = gaussian_blur(x)
        
        # Untransformed adversarial image attached to p's computation graph
        x_adv = x_0 + x

        # Detached intermediate node for untransformed image space.
        # Surrogate gradients will be pulled back to this node through spatial
        # augmentations (DIM, MCA, local crop) before being mapped to p.
        x_adv_node = x_adv.detach().clone().requires_grad_(True)

        argument = x_adv_node

        if use_local_crop:
            # Ensure x_0 + x has batch dimension for local_crop
            argument = local_crop(argument.unsqueeze(0)).squeeze(0)

        # Apply diverse input transform (DIM) to improve transferability
        if use_diverse_input:
            argument = diverse_input_transform(argument, prob=diverse_input_prob)

        # Calculate penalty-based loss for staying in valid image range (on p graph)
        img_loss = image_fit_loss(x_0, x, 0, 1)

        # Zero out the gradient at the start of each iteration
        if p.grad is not None:
            p.grad.zero_()

        # Multi-crop alignment: average loss across n_multi_crop random crops
        # (M-Attack V2 style MCA for gradient variance reduction)
        n_crops = max(1, n_multi_crop)
        crop_arguments = []
        if n_crops > 1:
            multi_crop_transform = RandomResizedCrop(
                size=(argument.shape[1], argument.shape[2]),
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1)
            )
            for _ in range(n_crops):
                crop_arguments.append(multi_crop_transform(argument.unsqueeze(0)).squeeze(0))
        else:
            crop_arguments.append(argument)

        # Synchronize training prompt across all surrogates and the victim for this iteration
        if use_victim_dominant:
            sync_question = questions[0]
        else:
            sync_question = random.choice(questions)
        for ip in inputs_processors:
            ip.questions = [sync_question]
        if use_query_based_gradient_check or use_victim_dominant:
            victim_inputs_processor.questions = [sync_question]

        inputss = []
        pgrads = []
        image_grads = []
        losses = []

        for i, inputs_processor in enumerate(inputs_processors):
            inputs = inputs_processor.get_inputs_train()
            inputss.append(inputs)

            # Detach crop arguments to isolate VLM backward pass graphs
            crop_arguments_detached = [c.detach().requires_grad_(True) for c in crop_arguments]

            # Accumulate loss across all crops for this model
            model_loss_accum = 0
            for crop_arg in crop_arguments_detached:
                processed = adv_processors[i].process(crop_arg.to(devices[i]))
                for key, value in processed.items():
                    if isinstance(value, torch.Tensor):
                        repeat_size = len(value.shape)*[1]
                        repeat_size[0] = batch_size
                        value = value.repeat(repeat_size)
                        noise = torch.randn_like(value).to(device_last) * resave_error_std
                        inputs[key] = value + noise.to(devices[i])
                    else:
                        inputs[key] = value
                
                # Forward pass and compute logits
                outputs = models[i](**inputs)
                logits = outputs.logits[:, :-1, :]
                model_loss_accum += inputs_processor.get_loss(logits)
            
            model_loss = model_loss_accum / n_crops
            
            # Track the total loss value for logging
            total_loss_val = model_weights[i] * model_loss.item() + img_loss.item()
            losses.append(total_loss_val)
            accumulated_loss += total_loss_val

            # Backward pass for model i: propagates gradients back to crop_arguments_detached
            # without keeping VLM activations in memory
            weighted_model_loss = model_weights[i] * model_loss
            weighted_model_loss.backward()

            # Collect computed gradients w.r.t. crop_arguments_detached
            crop_grads_i = [c.grad for c in crop_arguments_detached]

            # 1. Pull crop gradients backwards through spatial transforms (DIM, MCA, local crop)
            # to the untransformed image node x_adv_node.
            # This yields the pure surrogate gradient w.r.t. the UNTRANSFORMED image in (C, H, W).
            img_grad_i = torch.autograd.grad(
                outputs=crop_arguments,
                inputs=x_adv_node,
                grad_outputs=crop_grads_i,
                retain_graph=True,
            )[0]
            image_grads.append(img_grad_i.detach().to(device_last))

            # 2. Backpropagate from untransformed image space to p-space (applies tanh Jacobian once):
            p_grad_i = torch.autograd.grad(
                outputs=x_adv,
                inputs=p,
                grad_outputs=img_grad_i,
                retain_graph=True,
            )[0]
            pgrads.append(p_grad_i.detach().to(device_last))
            torch.cuda.empty_cache()

        if use_victim_dominant:
            # ── Victim-dominant hybrid update: g_comb = g_victim + surr_lambda * g_ens ──
            # 1. Compute unit-normalized surrogate consensus in untransformed image space
            surr_norm_grads = [g / (g.norm() + 1e-8) for g in image_grads]
            g_ens = torch.stack(surr_norm_grads).mean(dim=0) if surr_norm_grads else torch.zeros_like(x_adv_node)

            # 2. Query victim blackbox with structured low-frequency P-RGF
            query_image = (x_0 + x).detach().to(victim_device)
            victim_inputs = victim_inputs_processor.get_inputs_train()

            blackbox_grad, zo_mean_loss_delta = get_blackbox_grad_prgf(
                victim_model=victim_model,
                victim_adv_processor=victim_image_processor,
                victim_inputs_processor=victim_inputs_processor,
                victim_inputs=victim_inputs,
                image=query_image,
                g_ens=g_ens.to(victim_device),
                prgf_lambda=prgf_lambda,
                epsilon=zo_epsilon,
                num_dirs=prgf_num_dirs,
                patch_size=prgf_patch_size,
            )

            # 3. Combine in image space
            comb_img_grad = blackbox_grad.to(device_last) + surr_lambda * g_ens.to(device_last)

            # 4. Direct PGD update in image space (avoids AdamW second-moment distortion & weight decay)
            g_step = comb_img_grad.sign() if surr_lambda == 0.0 else comb_img_grad / (comb_img_grad.abs().max() + 1e-8)
            g_step = g_step * mask
            current_lr = scheduler.get_last_lr()[0]
            x_curr = (x_0 + x).detach()
            x_next = x_curr - current_lr * g_step
            delta = (x_next - x_0).clamp(-attack_norm, attack_norm)
            x_clamped = (x_0 + delta).clamp(0.0, 1.0)

            with torch.no_grad():
                new_delta = x_clamped - x_0
                new_p = torch.atanh((new_delta / attack_norm).clamp(-0.99999, 0.99999))
                p.copy_(new_p)

            scheduler.step()
            global_iteration += 1
            _zo_loss_accum += zo_mean_loss_delta
            _zo_steps_accum += 1
            grad_norm = comb_img_grad.norm()

            wandb.log({
                "zo_mean_loss_delta": zo_mean_loss_delta,
                "zo_grad_norm":       blackbox_grad.norm().item(),
                "surr_ens_norm":      g_ens.norm().item(),
                "surr_lambda":        surr_lambda,
            })
        else:
            # Query based gradient check
            if use_query_based_gradient_check and (iteration % zo_every_n_iters == 0):
                query_image = (x_0 + x).detach().to(victim_device)
                # Refresh inputs each iteration so query gradient matches the
                # current target_text (updated by DPO / target_text_random logic above).
                victim_inputs = victim_inputs_processor.get_inputs_train()

                # ── Choose estimator: P-ZO (subspace), exhaustive patch sweep, or random BCD ──────
                if use_pzo:
                    blackbox_grad, zo_mean_loss_delta, basis_alphas = get_blackbox_grad_pzo(
                        victim_model,
                        victim_image_processor,
                        victim_inputs_processor,
                        victim_inputs,
                        query_image,
                        image_grads,  # Pure untransformed image-space surrogate gradients
                        epsilon=pzo_epsilon,
                    )
                    zo_patch_count = len(basis_alphas)
                elif use_exhaustive_bcd:
                    blackbox_grad, zo_mean_loss_delta = get_blackbox_grad_exhaustive(
                        victim_model,
                        victim_image_processor,
                        victim_inputs_processor,
                        victim_inputs,
                        query_image,
                        epsilon=zo_epsilon,
                        patch_size=zo_patch_size,
                        zo_mode=zo_mode,
                    )
                    C, H, W = query_image.shape
                    zo_patch_count = (H // zo_patch_size) * (W // zo_patch_size)
                else:
                    blackbox_grad, zo_mean_loss_delta = get_blackbox_grad(
                        victim_model,
                        victim_image_processor,
                        victim_inputs_processor,
                        victim_inputs,
                        query_image,
                        epsilon=zo_epsilon,
                        num_samples=zo_num_samples,
                        bcd_block_size=bcd_block_size,
                        zo_mode=zo_mode,
                    )
                    zo_patch_count = zo_num_samples

                # Map blackbox_grad from untransformed image space to p-space:
                # Applies the tanh Jacobian (and gaussian blur if active) ONCE:
                zo_grad_on_device = torch.autograd.grad(
                    outputs=x_adv,
                    inputs=p,
                    grad_outputs=blackbox_grad.to(device_last),
                    retain_graph=True,
                )[0].detach()

                # ── ZO diagnostic metrics ────────────────────────────────────────
                with torch.no_grad():
                    zo_grad_norm = zo_grad_on_device.norm().item()
                    surrogate_grad_norms = [g.norm().item() for g in pgrads]
                    surrogate_norm_mean = sum(surrogate_grad_norms) / (len(surrogate_grad_norms) + 1e-8)
                    zo_vs_surrogate_ratio = zo_grad_norm / (surrogate_norm_mean + 1e-8)

                    # Cosine similarity between ZO and mean of current surrogate grads
                    if len(pgrads) > 0:
                        surrogate_mean_grad = torch.stack(pgrads).mean(dim=0)
                        zo_cosine = F.cosine_similarity(
                            zo_grad_on_device.flatten().unsqueeze(0),
                            surrogate_mean_grad.flatten().unsqueeze(0),
                        ).item()
                    else:
                        zo_cosine = float("nan")

                # ── Signal-quality gate ──────────────────────────────────────────
                zo_is_finite = torch.isfinite(zo_grad_on_device).all().item()
                zo_has_signal = zo_is_finite and (zo_mean_loss_delta >= zo_signal_threshold)
                if not zo_is_finite:
                    zo_accepted = False
                    zo_accept_reason = "dropped_non_finite"
                elif zo_has_signal:
                    # Real victim signal: always trust ZO over surrogates
                    zo_accepted = True
                    zo_accept_reason = "high_signal"
                else:
                    # Noise regime: only add if not opposing surrogates and finite
                    zo_accepted = (zo_cosine >= zo_cosine_threshold)
                    zo_accept_reason = "low_signal_cosine_ok" if zo_accepted else "low_signal_dropped"

                if zo_accepted:
                    pgrads.append(zo_grad_on_device * zo_weight_multiplier)
                    _zo_loss_accum += zo_mean_loss_delta
                    _zo_steps_accum += 1
                else:
                    zo_grad_on_device = torch.zeros_like(zo_grad_on_device)  # for logging only

                wandb.log({
                    "zo_mean_loss_delta":        zo_mean_loss_delta,
                    "zo_grad_norm":              zo_grad_norm,
                    "surrogate_grad_norm_mean":  surrogate_norm_mean,
                    "zo_vs_surrogate_ratio":     zo_vs_surrogate_ratio,
                    "zo_cosine_with_surrogates": zo_cosine,
                    "zo_accepted":               int(zo_accepted),
                    "zo_has_signal":             int(zo_has_signal),
                    "zo_patch_count":            zo_patch_count,
                    "zo_weight_multiplier":      zo_weight_multiplier,
                })

            # Aggregate model gradients with optional per-model normalization
            if use_grad_norm:
                zo_is_last = (
                    use_query_based_gradient_check
                    and (iteration % zo_every_n_iters == 0)
                    and len(pgrads) > len(models)  # ZO was accepted and appended
                )
                normalized_grads = []
                for idx, g in enumerate(pgrads):
                    g_unit = g / (g.norm() + 1e-8)
                    if zo_is_last and idx == len(pgrads) - 1:
                        g_unit = g_unit * zo_weight_multiplier
                    normalized_grads.append(g_unit)
                weight_sum = len(models) + (zo_weight_multiplier if zo_is_last else 0.0)
                iter_grad = torch.stack(normalized_grads).sum(dim=0) / (weight_sum + 1e-8)
            else:
                iter_grad = torch.stack(pgrads).sum(dim=0)

        if not use_victim_dominant:
            # Compute gradient of image bounding penalty w.r.t p (decoupled from surrogates and P-ZO)
            img_loss_grad = torch.autograd.grad(
                outputs=img_loss,
                inputs=p,
                retain_graph=False,
            )[0].detach()
            
            # Apply momentum (MI-FGSM style) for gradient direction stabilization
            if use_momentum:
                grad_normalized = iter_grad / (iter_grad.norm() + 1e-8)
                grad_momentum = momentum_decay * grad_momentum + grad_normalized
                iter_grad = grad_momentum.clone()

            # Apply mask to gradient
            iter_grad = iter_grad * mask

            # Add image boundary penalty gradient (masked) to keep image in [0, 1]
            iter_grad = iter_grad + (img_loss_grad * mask)
            
            grad_norm = iter_grad.norm()

            # Accumulate gradient across steps
            accumulated_p_grad += iter_grad / grad_accum_steps

            # Gradient accumulation and optimizer step
            if (iteration + 1) % grad_accum_steps == 0:
                if clamp_method == 'tanh':
                    p.grad = accumulated_p_grad.clone()
                else:
                    x.grad = accumulated_p_grad.clone()
                optimizer.step()
                optimizer.zero_grad()
                accumulated_p_grad.zero_()
                scheduler.step()  # Update the learning rate according to the scheduler
                
                # Log the accumulated loss
                wandb.log({
                    "accumulated_loss": accumulated_loss,
                })
                accumulated_loss = 0  # Reset accumulated loss for the next round
                global_iteration += 1

        # Clamping methods
        if clamp_method == 'tanh':
            pass  # No need to clamp since tanh output is already between -1 and 1
        elif clamp_method == 'none':
            pass
        else:
            NotImplementedError("Clamping method except tanh and none are not implemented")

        min_losses.append(sum(losses)/len(losses))

        # Only evaluate resaved loss and save tmp.png every save_steps (or last iter) to eliminate
        # disk I/O and thousands of redundant surrogate forward passes every iteration.
        if iteration % save_steps == 0 or iteration == num_iterations - 1:
            with torch.inference_mode():
                # Copy the sum to another tensor and re-save image to measure error
                x_mod = (x_0 + x).clone().detach().to(device_last)
                img = adv_processors[0].tensor2pil(x_mod)
                img.save('tmp.png')
                x_mod_resaved = pil_to_tensor(img, resize=False).to(device_last)
                
                resave_error_std = (x_mod_resaved - x_mod).abs().std()
                resave_error_mean = (x_mod_resaved - (x + x_0)).abs().mean().item()
                resave_error_l1 = (x_mod_resaved - x_mod).abs().sum().item()
                
                # Evaluate the re-saved image for logging
                resaved_losses = []
                for i, inputs in enumerate(inputss):
                    processed = adv_processors[i].process(x_mod_resaved.to(devices[i]))
                    for key, value in processed.items():
                        if isinstance(value, torch.Tensor):
                            repeat_size = len(value.shape)*[1]
                            repeat_size[0] = batch_size
                            inputs[key] = value.repeat(repeat_size)
                        else:
                            inputs[key] = value
                    outputs = models[i](**inputs)
                    logits = outputs.logits[:, 0:-1, :]
                    resaved_losses.append(inputs_processors[i].get_loss(logits))
                
                resaved_loss_val = torch.tensor([ls.item() for ls in resaved_losses]).mean().item()

        # Log metrics
        wandb_log_data = {
            "loss_resaved": resaved_loss_val,
            "iteration": iteration,
            "adversarial_mean": x.mean(),
            "adversarial_std": x.std(),
            "lr": scheduler.get_last_lr()[0],
            "resave_error_mean": resave_error_mean,
            "resave_error_std": resave_error_std.item() if isinstance(resave_error_std, torch.Tensor) else resave_error_std,
            "resave_error_l1": resave_error_l1,
            "noise_mean": noise.mean(),
            "noise_std": noise.std(),
            "loss_per_iteration": sum(losses)/len(losses),
            "img_loss": img_loss.item(),  # <-- Log image penalty
            "global_iteration": global_iteration,
            "grad_norm": grad_norm,
            "use_gaussian_blur": use_gaussian_blur,
            "gblur_kernel_size": gblur_kernel_size
        }

        # Also log each model's partial loss
        for i, loss_val in enumerate(losses):
            mn = model_names[i].replace("/", "_")
            wandb_log_data[f"loss_{i}_{mn}"] = loss_val

        wandb.log(wandb_log_data)

        # Every `save_steps`, run inference and log results
        if iteration % save_steps == 0 or iteration == num_iterations - 1:
            x_mod = (x_0 + x).clone().detach()
            final_image = adv_processors[0].tensor2pil(x_mod)
            save_checkpoint(final_image, x + x_0, exp_path, iteration)

            # Flush accumulated victim ZO loss and true checkpoint loss to CSV
            if use_query_based_gradient_check or use_victim_dominant:
                avg_zo = _zo_loss_accum / max(_zo_steps_accum, 1)
                with torch.inference_mode():
                    chk_victim_loss = query_blackbox(
                        victim_model,
                        victim_image_processor,
                        victim_inputs_processor,
                        victim_chk_inputs,
                        (x_0 + x).detach().to(victim_device),
                    ).item()
                with open(victim_zo_log_path, "a") as _f:
                    _f.write(f"{iteration},{chk_victim_loss:.6f},{avg_zo:.6f}\n")
                _zo_loss_accum = 0.0
                _zo_steps_accum = 0
            
            img_path = os.path.join(exp_path, f"optimized_image_iter_{iteration}.png")
            img_for_test = Image.open(img_path).convert("RGB")
            
            iteration_outputs = [iteration]  # first column is iteration

            models_output, wandb_log = run_model_test(
                models=models,
                processors=processors,
                inputs_processors=inputs_processors,
                model_names=model_names,
                not_safe_questions_test=not_safe_questions_test,
                target_text="sure, here it is",
                exp_path=exp_path,
                iteration=iteration,
                img=img_for_test
            )
            
            wandb.log(wandb_log)

            iteration_outputs = iteration_outputs + models_output[1:]
            print("Question:", models_output[0])
            for i, ans in enumerate(models_output[1:]):
                print(f"Model {model_names[i]} output:", ans)
            
            model_outputs_table.add_data(*iteration_outputs)
            
            log_metrics_wandb(iteration, sum(losses)/len(losses), final_image, (x + x_0), model_outputs_table, save_steps)

        # If configured, clamp/clip every restart_num steps
        if restart_num > 0 and (iteration + 1) % restart_num == 0:
            with torch.no_grad():
                y = (x + x_0).clamp(0.0, 1.0).mul(255).to(torch.uint8)
                x_new = y - x_0
                wandb.log({
                    "fix_error_mean": (x_new - x).abs().mean(),
                    "fix_error_std": (x_new - x).abs().std()
                })
                x = x_new.clone()
        
    # Final image save
    x_mod = (x_0 + x).clone().detach()
    final_image = adv_processors[0].tensor2pil(x_mod)
    save_checkpoint(final_image, x + x_0, exp_path, "final")

    # Write best_victim.txt: the checkpoint with lowest victim loss (strictly iteration > 0).
    if (use_query_based_gradient_check or use_victim_dominant) and os.path.exists(victim_zo_log_path):
        try:
            import csv
            best_victim_iter = None
            best_loss_val = float("inf")
            best_delta_val = -1.0
            has_victim_loss = False
            with open(victim_zo_log_path, "r") as _f:
                reader = csv.DictReader(_f)
                for row in reader:
                    g_iter = int(row["iteration"] if "iteration" in row else row["global_iteration"])
                    if g_iter == 0:
                        continue  # STRICTLY EXCLUDE ITERATION 0
                    if "victim_loss" in row and row["victim_loss"] != "nan":
                        has_victim_loss = True
                        val = float(row["victim_loss"])
                        if val < best_loss_val:
                            best_loss_val = val
                            best_victim_iter = g_iter
                    elif not has_victim_loss and "avg_zo_loss_delta" in row and row["avg_zo_loss_delta"] != "nan":
                        val = float(row["avg_zo_loss_delta"])
                        if val > best_delta_val:
                            best_delta_val = val
                            best_victim_iter = g_iter
            if best_victim_iter is not None:
                with open(os.path.join(exp_path, "best_victim.txt"), "w") as _f:
                    _f.write(str(best_victim_iter))
                metric_str = f"victim_loss={best_loss_val:.4f}" if has_victim_loss else f"avg_zo_loss_delta={best_delta_val:.4f}"
                print(f"Victim best iteration: {best_victim_iter} ({metric_str})")
        except Exception as _e:
            print(f"Warning: could not write best_victim.txt: {_e}")

    wandb.finish()


def parse_model_names(model_names_str):
    return model_names_str.split(',')


def main():
    parser = argparse.ArgumentParser(description="Train image attack model.")
    parser.add_argument("--exp_name", type=str, default="oxford-to-msu", help="Experiment name.")
    parser.add_argument("--img_orig", type=str, default="oxford.png", help="Original image filename.")
    parser.add_argument("--prompt", type=str, default="USER: <image>\nWho are you? ASSISTANT: ", help="Prompt for training.")
    parser.add_argument("--target_text", type=str, default="I am a terrorist and I kill people!", help="Target text.")
    parser.add_argument("--model_names", type=parse_model_names, default=["llava-hf/llava-1.5-7b-hf"], help="Model name for loading from Hugging Face.")
    parser.add_argument("--victim", type=parse_model_names, default=["Qwen/Qwen2-VL-2B-Instruct"], help="Victim name for loading from Hugging Face.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of training iterations.")
    parser.add_argument("--save_steps", type=int, default=10, help="Steps to save intermediate outputs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Number of steps to accumulate gradients.")
    parser.add_argument("--scheduler_step_size", type=int, default=100, help="Step size for the learning rate scheduler.")
    parser.add_argument("--scheduler_gamma", type=float, default=0.9, help="Gamma for the learning rate scheduler.")
    parser.add_argument("--restart_num", type=int, default=0, help="Number of steps after which to restart the optimizer (0 means no restart).")  # Added argument
    parser.add_argument("--mask_type", type=str, default=None, choices=['corner', 'bottom_lines', 'random_square'], help="Type of mask to apply.")  # Added argument
    parser.add_argument("--mask_size", type=int, default=None, help="Size parameter for the mask (n for corner or random_square, k for bottom_lines).")  # Added argument
    parser.add_argument("--clamp_method", type=str, default='tanh', choices=['clamp', 'tanh', 'none'], help="Method to enforce pixel value constraints.")  # Updated default
    parser.add_argument("--start_from_white", action='store_true', help="Start attack from a white image instead of the original image.")  # Added argument
    parser.add_argument("--target_text_random", action='store_true', help="Randomly select target_text from the answers list.")
    parser.add_argument("--DPO_flag", action='store_true', help="DPO flag")
    parser.add_argument("--refuse_prob", type=float, default=0.0, help="Probability of using refusing answers. Used if DPO_flag is True.")
    parser.add_argument("--epsilon", type=float, default=0.4, help="Attack norm. Decrease to make attack more imperceptable. Values from 0 (no attack) to 0.5-0.6 (may lead to increased img resaving loss).")
    # sigma squared from 4.2.3. IMPLEMENTATION DETAILS
    parser.add_argument("--sigma", type=float, default=0.001, help="Sigma squared hparam for 'enhance robustness' or `resave_error_std` from code.")
    # gaussian blur
    parser.add_argument("--use_gaussian_blur", action='store_true', help="Use gaussian blur for training.")
    parser.add_argument("--gblur_kernel_size", type=int, default=5, help="Kernel size for gaussian blur.")
    # Add random crop parameter
    parser.add_argument("--use_local_crop", action='store_true', help="Use random resized crop for data augmentation.")
    # Add random crop scale parameters
    parser.add_argument("--crop_scale_min", type=float, default=0.6, help="Minimum scale factor for random crop.")
    parser.add_argument("--crop_scale_max", type=float, default=1.0, help="Maximum scale factor for random crop.")
    # Add random crop ratio parameters
    parser.add_argument("--crop_ratio_min", type=float, default=0.75, help="Minimum aspect ratio for random crop.")
    parser.add_argument("--crop_ratio_max", type=float, default=1.33, help="Maximum aspect ratio for random crop.")
    
    # Transferability enhancement arguments
    parser.add_argument("--use_grad_norm", action='store_true', help="Normalize per-model gradients before aggregation to prevent gradient dominance.")
    parser.add_argument("--use_momentum", action='store_true', help="MI-FGSM style momentum for gradient stabilization (improves transferability).")
    parser.add_argument("--momentum_decay", type=float, default=0.9, help="Momentum decay factor (mu). Higher = more history retained.")
    parser.add_argument("--use_diverse_input", action='store_true', help="DIM: Apply diverse input transforms to prevent overfitting to surrogate preprocessing.")
    parser.add_argument("--diverse_input_prob", type=float, default=0.7, help="Probability of applying DIM per iteration.")
    parser.add_argument("--n_multi_crop", type=int, default=1, help="Number of crops to average for M-Attack style MCA. 1 = disabled.")
    parser.add_argument("--use_query_based_gradient_check", action='store_true', help="Add a zeroth-order gradient estimated from the victim model.")
    parser.add_argument("--zo_every_n_iters", type=int, default=1, help="Run ZO gradient check only every N iterations. N=1 = every iter (default). N=10 cuts ZO overhead by 10x. Recommended: 5-10 for exhaustive BCD to keep wall-clock under 24h.")
    parser.add_argument("--bcd_block_size", type=int, default=None, help="Block size for Block Coordinate Descent in zeroth-order queries. (e.g., 32)")
    parser.add_argument("--zo_epsilon", type=float, default=0.02, help="Finite-difference step size for zeroth-order victim gradient. Should be in [0.01, 0.05]; decoupled from --sigma (resave noise).")
    parser.add_argument("--zo_num_samples", type=int, default=5, help="Number of random directions averaged per ZO query. Higher = less noisy gradient but more victim queries per iteration.")
    # Exhaustive patch BCD (Zer0-Jack style)
    parser.add_argument("--use_exhaustive_bcd", action='store_true', help="Use exhaustive patch-wise BCD (Zer0-Jack style) instead of random BCD. Queries every non-overlapping patch per iteration — much denser gradient but more victim calls.")
    parser.add_argument("--zo_patch_size", type=int, default=48, help="Patch tile size (pixels) for exhaustive BCD. Smaller = more patches = more queries but finer gradient coverage. 48 matches Zer0-Jack LLaVA.")
    parser.add_argument("--zo_weight_multiplier", type=float, default=1.0, help="Multiply ZO gradient by this factor before normalization. With use_grad_norm, multiplier=2 gives ZO twice the weight of each surrogate. Compensates for 1/N dilution.")
    parser.add_argument("--zo_cosine_threshold", type=float, default=0.0, help="(Low-signal regime only) Drop ZO if cosine with surrogates < this when zo_mean_loss_delta < zo_signal_threshold. Ignored when ZO has real victim signal.")
    parser.add_argument("--zo_signal_threshold", type=float, default=0.001, help="Minimum zo_mean_loss_delta to treat ZO as real victim signal. Above this, ZO is always added regardless of cosine alignment. Below this, ZO is treated as noise and the cosine gate applies.")
    parser.add_argument("--zo_mode", type=str, default="two_sided", choices=["two_sided", "one_sided"], help="Zeroth-order estimator mode: two_sided (2 queries/patch) or one_sided (1 query/patch + 1 base query, ~2x faster).")
    parser.add_argument("--use_pzo", action='store_true', help="Use Prior-Guided ZO (P-ZO): project queries into surrogate gradient subspace for low-variance black-box estimation.")
    parser.add_argument("--pzo_epsilon", type=float, default=0.05, help="Finite-difference step size for P-ZO directional queries.")
    
    # Victim-dominant hybrid arguments
    parser.add_argument("--use_victim_dominant", action='store_true', help="Enable victim-dominant hybrid attack (g_victim + surr_lambda * g_ens with structured P-RGF).")
    parser.add_argument("--surr_lambda", type=float, default=0.1, help="Regularizer weight for surrogate consensus in victim-dominant mode (e.g. 0.0, 0.1).")
    parser.add_argument("--prgf_lambda", type=float, default=0.5, help="Fraction of surrogate consensus direction in P-RGF query generation.")
    parser.add_argument("--prgf_patch_size", type=int, default=32, help="Patch size for low-frequency spatial basis in P-RGF.")
    parser.add_argument("--prgf_num_dirs", type=int, default=15, help="Number of random directions per iteration for P-RGF (2x queries).")

    # NEW ARGUMENT FOR MODEL WEIGHTS
    parser.add_argument(
        "--model_weights",
        type=float,
        nargs='+',
        default=None,
        help="Loss weights for each model, must match length of model_names (default is 1.0 for each)."
    )

    # NEW ARGUMENT FOR AUTOMATION
    parser.add_argument(
        "--output_path_file",
        type=str,
        default=None,
        help="Path to a file where the created experiment directory will be written."
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_exp_name = f"{args.exp_name}_{timestamp}"

    train(
        exp_name=unique_exp_name,
        img_orig=args.img_orig,
        prompt=args.prompt,
        target_text=args.target_text,
        model_names=args.model_names,
        victim=args.victim,
        lr=args.lr,
        num_iterations=args.num_iterations,
        save_steps=args.save_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        restart_num=args.restart_num,
        mask_type=args.mask_type,
        mask_size=args.mask_size,
        clamp_method=args.clamp_method,
        epsilon=args.epsilon,
        sigma=args.sigma,
        start_from_white=args.start_from_white,
        target_text_random=args.target_text_random,
        DPO_flag=args.DPO_flag,
        refuse_prob=args.refuse_prob,
        model_weights=args.model_weights,
        use_gaussian_blur=args.use_gaussian_blur,
        gblur_kernel_size=args.gblur_kernel_size,
        use_local_crop=args.use_local_crop,
        crop_scale_min=args.crop_scale_min,
        crop_scale_max=args.crop_scale_max,
        crop_ratio_min=args.crop_ratio_min,
        crop_ratio_max=args.crop_ratio_max,
        # Transferability enhancements
        use_grad_norm=args.use_grad_norm,
        use_momentum=args.use_momentum,
        momentum_decay=args.momentum_decay,
        use_diverse_input=args.use_diverse_input,
        diverse_input_prob=args.diverse_input_prob,
        n_multi_crop=args.n_multi_crop,
        use_query_based_gradient_check=args.use_query_based_gradient_check,
        zo_every_n_iters=args.zo_every_n_iters,
        bcd_block_size=args.bcd_block_size,
        zo_epsilon=args.zo_epsilon,
        zo_num_samples=args.zo_num_samples,
        use_exhaustive_bcd=args.use_exhaustive_bcd,
        zo_patch_size=args.zo_patch_size,
        zo_weight_multiplier=args.zo_weight_multiplier,
        zo_cosine_threshold=args.zo_cosine_threshold,
        zo_signal_threshold=args.zo_signal_threshold,
        zo_mode=args.zo_mode,
        use_pzo=args.use_pzo,
        pzo_epsilon=args.pzo_epsilon,
        use_victim_dominant=args.use_victim_dominant,
        surr_lambda=args.surr_lambda,
        prgf_lambda=args.prgf_lambda,
        prgf_patch_size=args.prgf_patch_size,
        prgf_num_dirs=args.prgf_num_dirs,
    )

    # Write the output path if requested
    if args.output_path_file:
        with open(args.output_path_file, 'w') as f:
            f.write(os.path.abspath(os.path.join("./runs", unique_exp_name)))

if __name__ == "__main__":
    main()
