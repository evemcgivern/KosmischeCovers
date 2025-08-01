import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import gc
import sys

# This script sets up data loading pipelines for training vision models or multi-modal LLMs
# that work with album cover images. It handles efficient loading, preprocessing, and batching.

# Custom dataset class that works with a flat directory of images (no class subdirectories)


class AlbumCoversDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Return image and dummy label 0 (since we're not doing classification)
        return image, 0


# Paths to the augmented dataset directories
DATA_DIR = "data/augment"
TRAIN_DIR = os.path.join(DATA_DIR, "train")      # Training images directory (input)
VAL_DIR = os.path.join(DATA_DIR, "validation")   # Validation images directory (input)

# Hyperparameters for loading
BATCH_SIZE = 2   # bump up if you have RAM/GPU to spare
NUM_WORKERS = 2  # parallel file loading - increases throughput

# Image transformations applied to all images:
# 1. Center crop to create square 512x512 images (standard size for many vision models)
# 2. Convert to PyTorch tensors (this is where tensors are created)
# 3. Normalize pixel values to [-1,1] range (common for GANs and transformers)
common_tfms = transforms.Compose([
    transforms.CenterCrop(512),
    transforms.ToTensor(),  # <-- This line converts PIL images to PyTorch tensors
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# Move initialization code into a function to avoid worker process re-initialization
def get_data_loaders():
    """Create and return the train and validation dataloaders."""
    print(f"📁 Scanning training directory (input): {TRAIN_DIR}")
    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")
    train_ds = AlbumCoversDataset(TRAIN_DIR, transform=common_tfms)
    print(f"✓ Found {len(train_ds)} training images (input)")

    print(f"📁 Scanning validation directory (input): {VAL_DIR}")
    if not os.path.exists(VAL_DIR):
        raise FileNotFoundError(f"Validation directory not found: {VAL_DIR}")
    val_ds = AlbumCoversDataset(VAL_DIR, transform=common_tfms)
    print(f"✓ Found {len(val_ds)} validation images (input)")

    print(f"🔄 Creating data loaders with batch size {BATCH_SIZE} and {NUM_WORKERS} workers")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print("✓ Created data loaders successfully (output: PyTorch batches for training/validation)")

    return train_loader, val_loader

# Initialize loaders only in the main process
train_loader, val_loader = None, None

# Sanity check
if __name__ == "__main__":
    # Only initialize loaders in the main script, not in worker processes
    train_loader, val_loader = get_data_loaders()

    print("\n=== 🧪 Running sanity checks ===")
    print(f"📊 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    try:
        print("📥 Loading first training batch...")
        imgs, _ = next(iter(train_loader))
        print(f"✓ Batch loaded successfully! Shape: {imgs.shape}")
        print(f"📝 Image stats - Min: {imgs.min():.4f}, Max: {imgs.max():.4f}, Mean: {imgs.mean():.4f}")

        if imgs.shape[0] != BATCH_SIZE:
            print(f"⚠️ Warning: Expected batch size {BATCH_SIZE} but got {imgs.shape[0]}")
        if imgs.shape[1] != 3:
            print(f"⚠️ Warning: Expected 3 channels but got {imgs.shape[1]}")
        if imgs.shape[2] != 512 or imgs.shape[3] != 512:
            print(f"⚠️ Warning: Expected 512x512 images but got {imgs.shape[2]}x{imgs.shape[3]}")
    except KeyboardInterrupt:
        print("⚠️ Interrupted by user; shutting down data loaders.")
    except Exception as e:
        print(f"❌ Error loading batch: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up GPU/MPS memory
        try:
            del imgs
        except Exception:
            pass
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("✅ Cleared MPS cache and garbage collected.")
        # Extra cleanup for DataLoader workers and debuggers
        import multiprocessing
        multiprocessing.active_children()  # Touch to trigger cleanup
        if hasattr(sys, 'gettrace') and sys.gettrace():
            print("⚠️ Debugger detected. Some debuggers may prevent full process exit.")
        # Force exit (should work unless a debugger or non-daemon thread is blocking)
        print("Exiting now")
        os._exit(0)
else:
    # When imported as a module, initialize without printing messages
    class SilentAlbumCoversDataset(AlbumCoversDataset):
        """A version of AlbumCoversDataset that doesn't print messages."""
        pass

    def get_data_loaders_silently():
        """Initialize data loaders without printing messages."""
        if not os.path.exists(TRAIN_DIR):
            raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")
        train_ds = SilentAlbumCoversDataset(TRAIN_DIR, transform=common_tfms)

        if not os.path.exists(VAL_DIR):
            raise FileNotFoundError(f"Validation directory not found: {VAL_DIR}")
        val_ds = SilentAlbumCoversDataset(VAL_DIR, transform=common_tfms)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        return train_loader, val_loader

    train_loader, val_loader = get_data_loaders_silently()

# Note: The tensors are not stored on disk. They are created in memory by the DataLoader
# as batches for each training/validation step. When you iterate over train_loader or val_loader,
# each batch is a tensor (or batch of tensors) held in RAM, used directly by your model.
# If you want to save tensors to disk, you would need to add code to do so (e.g., torch.save()).

# Note: By default, DataLoader and transforms keep tensors on CPU (RAM), not GPU.
# Tensors are only moved to GPU when you explicitly call .to("cuda") or .cuda() on them.
# If your GPU is maxed out, check your training loop or model code for any .to("cuda")/.cuda() calls,
# and ensure you are properly deleting tensors and calling torch.cuda.empty_cache() if needed.
# On Apple Silicon (M1/M2/M3/Max), PyTorch uses the CPU unless you use torch.device("mps").
# If you are using MPS (Apple GPU), use torch.mps.empty_cache() to release memory if needed.

# Note: This script does NOT use torch.device, .to("cuda"), or .cuda().
# All tensors remain on CPU unless you explicitly move them to GPU in your training code.
