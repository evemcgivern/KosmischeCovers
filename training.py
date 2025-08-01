# training.py
# This script fine-tunes the Stable Diffusion v1.5 model using LoRA (Low-Rank Adaptation)
# on a custom album cover dataset.
# It is designed to work across multiple platforms - CUDA GPUs, Apple Silicon, and CPU.
#
# Steps:
# 1. Detects available hardware and configures optimizations for the specific platform
# 2. Loads environment variables (including Hugging Face token) from .env
# 3. Loads the Stable Diffusion VAE and UNet with LoRA adapters for efficient training
# 4. Sets up optimizers and training parameters based on the detected hardware
# 5. Runs a training loop optimized for the detected hardware
# 6. Saves LoRA weights and pushes to Hugging Face Hub
#
# End result:
# - Fine-tuned LoRA weights saved locally and on Hugging Face Hub
# - Optimized performance based on user's hardware

import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
# import numpy as np
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.loaders import AttnProcsLayers
import gc
import time
# from PIL import Image
import platform

# Import data loaders from our dataloader.py
from dataloader import train_loader, val_loader

# Enhanced hardware detection and optimization
print("\n=== Detecting Hardware ===")
device_name = "Unknown"
gpu_memory = None
recommended_batch_size = 4  # Default

# Check for NVIDIA GPU (CUDA)
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = "NVIDIA GPU"
    n_gpus = torch.cuda.device_count()
    current_gpu = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_gpu)
    gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / (1024**3)  # GB

    print(f"✓ Using CUDA with {n_gpus} GPU(s)")
    print(f"✓ Current GPU: {gpu_name}")
    print(f"✓ GPU memory: {gpu_memory:.2f} GB")

    # Memory-based batch size recommendation
    if gpu_memory > 20:  # High-end GPU (>20GB VRAM)
        recommended_batch_size = 8
    elif gpu_memory > 12:  # Mid-range GPU (12-20GB VRAM)
        recommended_batch_size = 4
    elif gpu_memory > 6:  # Lower-end GPU (6-12GB VRAM)
        recommended_batch_size = 2
    else:  # Very limited VRAM (<6GB)
        recommended_batch_size = 1

    # CUDA-specific optimizations
    torch.backends.cudnn.benchmark = True
    print("✓ CUDA optimizations: Enabled cuDNN benchmark")
    use_mixed_precision = True

# Check for Apple Silicon (MPS)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "Apple Silicon"
    mac_model = platform.machine()

    # Detect Apple Silicon chip generation
    if mac_model == "arm64":
        import subprocess
        model_info = subprocess.check_output(['sysctl', 'hw.model']).decode().strip()
        if 'Mac' in model_info:
            print(f"✓ Mac model: {model_info}")
            # Rough estimate based on Mac model
            if "Mac14," in model_info:  # M3 series
                recommended_batch_size = 4
            elif "Mac13," in model_info:  # M2 series
                recommended_batch_size = 3
            else:  # M1 series or unrecognized
                recommended_batch_size = 2

    print(f"✓ Using MPS ({device_name} GPU)")
    print("✓ MPS optimizations: Aggressive memory management")
    use_mixed_precision = False  # MPS doesn't fully support mixed precision yet

# Check for Intel Mac (non-Silicon)
elif platform.system() == "Darwin" and "Intel" in platform.processor():
    device = torch.device("cpu")
    device_name = "Intel Mac CPU"
    cpu_count = os.cpu_count()
    print(f"✓ Using CPU (Intel Mac) with {cpu_count} cores")

    # Set reasonable number of threads for Intel Mac
    recommended_threads = min(cpu_count, 8) if cpu_count else 4
    torch.set_num_threads(recommended_threads)
    print(f"✓ CPU optimizations: Using {recommended_threads} threads")

    # Check if MKL is available for additional Intel optimizations
    try:
        import torch.utils.mkldnn as mkldnn
        has_mkl = mkldnn.is_available()
        if has_mkl:
            print("✓ Intel MKL optimizations available")
    except:
        pass

    recommended_batch_size = 2  # Conservative for CPU
    use_mixed_precision = False  # Not applicable for CPU

# Fallback to generic CPU
else:
    device = torch.device("cpu")
    cpu_count = os.cpu_count()

    # Try to detect CPU architecture for better optimization
    import re
    cpu_info = platform.processor()
    cpu_arch = "Unknown"

    # Try to identify modern CPU architectures
    if re.search(r'AMD Ryzen', cpu_info, re.IGNORECASE):
        cpu_arch = "AMD Ryzen"
        recommended_batch_size = 2
    elif re.search(r'Intel.*i[579]', cpu_info, re.IGNORECASE):
        cpu_arch = "Intel Core i-series"
        recommended_batch_size = 2
    else:
        recommended_batch_size = 1

    device_name = f"{cpu_arch} CPU"
    print(f"✓ Using CPU ({cpu_arch}) with {cpu_count} cores")

    # Set reasonable number of threads
    recommended_threads = min(cpu_count, 8) if cpu_count else 4
    torch.set_num_threads(recommended_threads)
    print(f"✓ CPU optimizations: Using {recommended_threads} threads")
    use_mixed_precision = False  # Not applicable for CPU

print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ Recommended batch size for your hardware: {recommended_batch_size}")

# Hugging Face token setup: loads from .env or environment variable
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

from diffusers import DiffusionPipeline

if not HF_TOKEN:
    print("\n⚠️  No Hugging Face token found. Some models might not be accessible.")
    print("To set your token, either:")
    print("1. Set the HF_TOKEN environment variable, or")
    print("2. Create a .env file with HF_TOKEN=your_token, or")
    print("3. Edit this script to set HF_TOKEN directly\n")
    print("Tip: Install python-dotenv to use .env files: pip install python-dotenv")

# Configuration with hardware-specific adjustments
model_id = "runwayml/stable-diffusion-v1-5"
output_dir = "models/stoner-psych-lora"
num_epochs = 10
learning_rate = 1e-4
log_every = 20

# Adjust batch size based on hardware detection (user can override)
print(f"\nCurrent batch size: {train_loader.batch_size}")
print(f"Recommended batch size for your {device_name}: {recommended_batch_size}")
print("Note: You can adjust batch_size in dataloader.py if needed")

# Set up mixed precision training if supported
if use_mixed_precision and device.type == "cuda":
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    print("✓ Mixed precision training enabled (faster training, less memory)")
else:
    autocast = lambda: nullcontext()  # No-op if mixed precision is not available
    from contextlib import nullcontext

# Create output directory for model weights and samples
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)

# Load Stable Diffusion VAE and UNet (no text encoder for memory efficiency)
print(f"Loading {model_id}...")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", use_auth_token=HF_TOKEN)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", use_auth_token=HF_TOKEN)

# Ensure vae and unet are model instances, then move to device (MPS or CPU)
if isinstance(vae, tuple):
    vae = vae[0]
if isinstance(unet, tuple):
    unet = unet[0]
vae = vae.to(device)
unet = unet.to(device)

# Set up LoRA for efficient fine-tuning (adapts only attention layers)
lora_rank = 4      # Lower rank = fewer parameters
lora_alpha = 32    # Higher alpha = stronger adaptation

# Train using a simpler approach - direct training without LoRA
print("Setting up direct fine-tuning (skipping LoRA due to compatibility issues)...")

# Freeze all UNet parameters
for param in unet.parameters():
    param.requires_grad = False

# Only enable training for cross-attention layers (similar to what LoRA would target)
for name, param in unet.named_parameters():
    if "attn2" in name:  # Cross-attention layers
        param.requires_grad = True

# Count trainable parameters
trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
print(f"✓ Cross-attention fine-tuning enabled")
print(f"✓ Training {trainable_params:,} parameters")

# Set up optimizer (AdamW) targeting only trainable parameters
optimizer = AdamW([p for p in unet.parameters() if p.requires_grad], lr=learning_rate)

# Fix multiprocessing issues with DataLoader on MPS
# Reload data with appropriate num_workers setting
if device.type == "mps":
    # On Apple Silicon, set num_workers=0 to avoid multiprocessing issues
    from dataloader import AlbumCoversDataset, common_tfms, TRAIN_DIR, VAL_DIR

    print("⚠️ Setting num_workers=0 for MPS compatibility")
    train_ds = AlbumCoversDataset(TRAIN_DIR, transform=common_tfms)
    val_ds = AlbumCoversDataset(VAL_DIR, transform=common_tfms)

    # Recreate dataloaders without multiprocessing
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=train_loader.batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=train_loader.batch_size, shuffle=False, num_workers=0
    )

# Total training steps and learning rate scheduler setup
total_steps = len(train_loader) * num_epochs
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps
)

# Set up noise scheduler for diffusion process
from diffusers import DDPMScheduler
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Create empty/dummy encoder hidden states for unconditional training
# The shape is [batch_size, seq_len, hidden_size]
dummy_encoder_hidden_states = torch.zeros(
    (train_loader.batch_size, 77, 768),  # Standard dimensions for SD v1.5
    device=device
)

# Helper function for batch processing that handles encoder_hidden_states
def process_batch(model, latents, timesteps, noise=None, train=True):
    # Create encoder_hidden_states of the right batch size
    batch_size = latents.shape[0]
    if batch_size != dummy_encoder_hidden_states.shape[0]:
        encoder_hidden_states = torch.zeros(
            (batch_size, 77, 768), device=device
        )
    else:
        encoder_hidden_states = dummy_encoder_hidden_states

    # Forward pass
    if train:
        noise_pred = model(
            latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )[0]
    else:
        with torch.no_grad():
            noise_pred = model(
                latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]

    return noise_pred

# Helper function: encode images to latent space using VAE
def encode_images(images):
    with torch.no_grad():
        images = 2 * images - 1  # Scale to [-1, 1]
        vae_model = vae[0] if isinstance(vae, tuple) else vae
        if use_mixed_precision and device.type == "cuda":
            with autocast():
                latents = vae_model.encode(images).latent_dist.sample()
        else:
            latents = vae_model.encode(images).latent_dist.sample()
        latents = latents * 0.18215  # VAE scaling factor
    return latents

# Helper function: memory cleanup optimized for device type
def cleanup_memory():
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    # Print memory stats for CUDA devices
    if device.type == "cuda":
        used_mem = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU Memory in use: {used_mem:.1f}MB")

# Helper function: generate and save a sample image using the current model
def save_sample(epoch, unet, vae, device, output_dir):
    sample_pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        unet=unet,
        vae=vae,
        safety_checker=None,
        use_auth_token=HF_TOKEN,
    )
    sample_pipeline.to(device)
    prompt = "psychedelic stoner rock album cover, trippy, cosmic, detailed artwork"
    with torch.no_grad():
        image = sample_pipeline(prompt=prompt, num_inference_steps=30).images[0]
    image_path = os.path.join(output_dir, "samples", f"sample_epoch_{epoch}.png")
    image.save(image_path)
    del sample_pipeline
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print(f"✓ Saved sample to {image_path}")

# Check for existing checkpoints to resume training
start_epoch = 0
checkpoint_files = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
if checkpoint_files:
    # Find the highest epoch checkpoint
    latest_epoch = max([int(d.split("-")[1]) for d in checkpoint_files])
    checkpoint_path = os.path.join(output_dir, f"checkpoint-{latest_epoch}", "model_checkpoint.pt")

    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint for epoch {latest_epoch}. Resuming training...")
        checkpoint = torch.load(checkpoint_path)
        unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']  # This was the completed epoch
        print(f"Resuming from epoch {start_epoch+1}")

# === Training Loop ===
print("\n=== Starting Training ===")
print(f"Training on: {device_name}")
start_time = time.time()

for epoch in range(start_epoch, num_epochs):  # Changed to start from last completed epoch
    epoch_start = time.time()
    unet.train()
    epoch_loss = 0.0
    step = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (images, _) in enumerate(progress_bar):
        images = images.to(device)

        # Encode images to latents
        latents = encode_images(images)

        # Add noise to latents (forward diffusion step)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise using UNet (with mixed precision if available)
        if use_mixed_precision and device.type == "cuda":
            with autocast():
                noise_pred = process_batch(unet, noisy_latents, timesteps, noise=noise)
                loss = F.mse_loss(noise_pred, noise)
            # Update with gradient scaling for mixed precision
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
        else:
            # Standard training path for CPU and MPS
            noise_pred = process_batch(unet, noisy_latents, timesteps, noise=noise)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        # Track and log loss
        epoch_loss += loss.item()
        step += 1

        if batch_idx % log_every == 0:
            avg_loss = epoch_loss / step if step > 0 else 0
            progress_bar.set_postfix({"loss": avg_loss})

            # Device-specific memory cleanup
            del noise_pred, loss, latents, noisy_latents
            cleanup_memory()

    # End of epoch: log, save sample, and run validation
    avg_epoch_loss = epoch_loss / step if step > 0 else 0
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s. Average loss: {avg_epoch_loss:.6f}")

    # Generate and save sample image at end of epoch
    save_sample(epoch+1, unet, vae, device, output_dir)

    # Validation (calculate loss, no backprop)
    unet.eval()
    val_loss = 0.0
    val_steps = 0
    print("Running validation...")
    with torch.no_grad():
        for val_images, _ in val_loader:
            val_images = val_images.to(device)
            val_latents = encode_images(val_images)
            val_noise = torch.randn_like(val_latents)
            val_timesteps = torch.randint(0, 1000, (val_latents.shape[0],), device=device)
            val_noisy_latents = val_noise
            val_noise_pred = process_batch(unet, val_noisy_latents, val_timesteps, train=False)
            val_batch_loss = F.mse_loss(val_noise_pred, val_noise)
            val_loss += val_batch_loss.item()
            val_steps += 1

            # Clean up validation tensors
            del val_noise_pred, val_batch_loss, val_latents, val_noisy_latents
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
    print(f"Validation loss: {avg_val_loss:.6f}")

    # Save checkpoint after each epoch - using standard PyTorch saving
    checkpoint_path = os.path.join(output_dir, f"checkpoint-{epoch+1}")
    os.makedirs(checkpoint_path, exist_ok=True)
    # Save the model weights
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_val_loss,
    }, os.path.join(checkpoint_path, "model_checkpoint.pt"))
    print(f"✓ Saved checkpoint for epoch {epoch+1}")

# Save the final model - using standard PyTorch saving
print(f"\nSaving final model to {output_dir}...")
torch.save(unet.state_dict(), os.path.join(output_dir, "unet_model.pt"))

# Create a README file with model information
with open(os.path.join(output_dir, "README.md"), "w") as f:
    f.write(f"# Stoner Psych Album Cover Model\n\n")
    f.write(f"Fine-tuned UNet for Stable Diffusion v1.5 focused on psychedelic album covers.\n\n")
    f.write(f"- **Base model**: {model_id}\n")
    f.write(f"- **Training approach**: Direct fine-tuning of cross-attention layers\n")
    f.write(f"- **Training device**: {device_name}\n")
    f.write(f"- **Parameters trained**: {trainable_params:,}\n")
    f.write(f"- **Training completed**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Publish to Hugging Face Hub with custom loading approach
print("Note: Since we used direct fine-tuning instead of LoRA, we need a custom loading approach.")
print("Pushing to Hub with the full fine-tuned UNet...")

# Create a pipeline for testing and pushing
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    unet=unet,  # Use our fine-tuned UNet directly
    safety_checker=None,
    # use_auth_token parameter removed as it's deprecated
)

# Update push_to_hub call to use current API (remove use_temp_dir parameter)
pipe.push_to_hub("kilbey1/stoner-psych-lora", token=HF_TOKEN)
print("🚀 Model pushed to: https://huggingface.co/kilbey1/stoner-psych-lora")

# Training summary
total_time = time.time() - start_time
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"\n✅ Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s!")
print(f"✅ Model saved to: {output_dir}")
print(f"✅ Sample images saved to: {os.path.join(output_dir, 'samples')}")

# Fix for ImportError: cannot import name 'cached_download' from 'huggingface_hub'
# This is caused by an incompatible version of huggingface_hub and diffusers.
# Solution: Upgrade huggingface_hub to >=0.16.4 and diffusers to >=0.19.3

# You can fix this by running:
#   pip install --upgrade huggingface_hub diffusers
# You can fix this by running:
#   pip install --upgrade huggingface_hub diffusers
