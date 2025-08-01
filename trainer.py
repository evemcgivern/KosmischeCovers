from PIL import Image, ImageEnhance
import os
import random
import multiprocessing
from tqdm import tqdm
import platform
import psutil

# This script performs data augmentation on album cover images for LLM (Large Language Model) or ML training.
# It generates multiple augmented versions of each input image to increase dataset diversity.

# Configuration
INPUT_DIR = "data/album_covers"        # Directory containing original album cover images
OUTPUT_DIR = "data/augment/train"      # Directory to save augmented images
AUG_PER_IMAGE = 5                      # Number of augmentations per original image

# Hardware detection for optimization
def detect_hardware():
    """Detect system hardware for optimization purposes"""
    system = platform.system()
    processor = platform.processor()
    cpu_count = os.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)

    print("\n=== Hardware Detection ===")
    print(f"Operating System: {system}")
    print(f"Processor: {processor}")
    print(f"CPU cores: {cpu_count}")
    print(f"System memory: {memory_gb:.1f} GB")

    # Determine optimal number of processes based on CPU cores
    # Use 75% of available cores to avoid overloading the system
    optimal_processes = max(1, int(cpu_count * 0.75)) if cpu_count else 1

    # For older systems with limited RAM, reduce parallelism
    if memory_gb < 4:  # Less than 4GB RAM
        optimal_processes = 1

    print(f"Using {optimal_processes} parallel processes for image augmentation")
    return optimal_processes

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def augment_image(img: Image.Image) -> Image.Image:
    """
    Apply random augmentations to an image:
    - Random horizontal flip (50% chance)
    - Random rotation between -15 and 15 degrees
    - Random color jitter (brightness, contrast, saturation)
    Returns the augmented image.
    """
    # Random horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    # Random rotation between -15 and 15 degrees
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=True).resize((img.width, img.height))
    # Color jitter: brightness, contrast, and saturation
    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        enhancer = enhancer_cls(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    return img

def process_image(filename):
    """Process a single image and its augmentations"""
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        base, ext = os.path.splitext(filename)
        try:
            img = Image.open(os.path.join(INPUT_DIR, filename)).convert("RGB")
            # Save original
            img.save(os.path.join(OUTPUT_DIR, f"{base}_orig{ext}"))
            # Save augmentations
            for i in range(AUG_PER_IMAGE):
                aug_img = augment_image(img)
                aug_img.save(os.path.join(OUTPUT_DIR, f"{base}_aug{i+1}{ext}"))
            return AUG_PER_IMAGE + 1  # Count original + augmentations
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return 0
    return 0

def main():
    # Detect hardware and determine optimal parallelism
    num_processes = detect_hardware()

    # Get all image files
    image_files = [f for f in os.listdir(INPUT_DIR)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    total_images = len(image_files)

    if total_images == 0:
        print(f"No images found in '{INPUT_DIR}'")
        return

    print(f"\nFound {total_images} images to process")
    print(f"Will generate {total_images * (AUG_PER_IMAGE + 1)} total images")

    # Process images in parallel if multiple cores available
    total_generated = 0
    if num_processes > 1:
        print("Using parallel processing...")
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_image, image_files),
                total=total_images,
                desc="Augmenting images"
            ))
            total_generated = sum(results)
    else:
        print("Using single-process mode...")
        for filename in tqdm(image_files, desc="Augmenting images"):
            total_generated += process_image(filename)

    print(f"\n✓ Augmentation complete. Generated {total_generated} images in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
