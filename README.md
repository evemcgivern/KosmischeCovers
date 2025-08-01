# KosmischeCovers Project Guide

## Project Overview

KosmischeCovers is a machine-learning project for generating psychedelic album cover art in the style of 1970s German “kosmische” rock. It includes scripts for data preparation, augmentation, model training, and inference.

## Associated Model

- **Live Model on Hugging Face:**
  [kilbey1/stoner-psych-lora](https://huggingface.co/kilbey1/stoner-psych-lora)

## Script Descriptions

### Core Scripts

- **dataloader.py**
  Sets up data loading pipelines for training vision models with album cover images.
- **trainer.py**
  Performs data augmentation on album cover images to increase dataset diversity.
- **training.py**
  Fine-tunes the Stable Diffusion v1.5 model using cross-attention training.
- **test.py**
  Utility script to verify image files and display their properties.
- **api_test.py**
  Tests connection to the Hugging Face API using your credentials.

### Auxiliary Scripts

- **downloader.py** (OPTIONAL)
  Downloads album cover images from sources like Discogs and MusicBrainz based on the provided `krautrock_list.csv` file. This script is only needed if you want to automatically build your dataset from the included album list rather than using your own album cover collection. If you already have album covers, you can skip this step entirely.

## Step-by-Step Setup Instructions

1. **Clone the Repository**

   ```bash

   git clone https://github.com/your-org/KosmischeCovers.git
   cd KosmischeCovers
```

2. ***Set Up Environment***

```python
# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. ***Create Required Directories***

```bash
mkdir -p data/covers
mkdir -p data/album_covers
mkdir -p data/augment/train
mkdir -p data/augment/validation
```

4. ***Configure API Access (Optional)***

Create a .env file in the project root with your Hugging Face token:

```bash
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

5. ***Run the Scripts in Order***

## Step 1: Prepare your album cover images

Place your album cover images in the album_covers directory.

## Step 2: Run data augmentation

```python
python trainer.py
```

## Step 3: Create validation set

```text
# Manually move ~20% of the images from data/augment/train to data/augment/validation
```

## Step 4: Verify your dataset

```bash
python test.py
```

## Step 5: Train the model

```bash
python training.py
```

## User-Configurable Settings

### In `trainer.py`

- **`INPUT_DIR`**: Location of original album cover images
  *Default:* `"data/album_covers"`
- **`OUTPUT_DIR`**: Directory to save augmented images
  *Default:* `"data/augment/train"`
- **`AUG_PER_IMAGE`**: Number of augmentations to generate per original image
  *Default:* `5`

```python
# Example of changing configuration
INPUT_DIR = "your_custom_input_path"
OUTPUT_DIR = "your_custom_output_path"
AUG_PER_IMAGE = 10                          # Increase for more variety
```

### In `trainer.py`

- **`INPUT_DIR`**: Location of original album cover images
  *Default:* `"data/album_covers"`
- **`OUTPUT_DIR`**: Directory to save augmented images
  *Default:* `"data/augment/train"`
- **`AUG_PER_IMAGE`**: Number of augmentations to generate per original image
  *Default:* `5`

```python
# Example of changing configuration
INPUT_DIR = "your_custom_input_path"
OUTPUT_DIR = "your_custom_output_path"
AUG_PER_IMAGE = 10                          # Increase for more variety
```

### In `dataloader.py`

- **`BATCH_SIZE`**: Number of images processed at once
  *Default:* `2`
- **`NUM_WORKERS`**: Parallel file-loading threads
  *Default:* `2`
- **Image transformations**: Adjust crop size, normalization, and other preprocessing parameters as needed.

### In 'training.py'

- **`model_id`: Base model to fine-tune (default: "runwayml/stable-diffusion-v1-5")
- **`output_dir`: Where to save the trained model (default: "models/stoner-psych-lora")
- **`num_epochs`: Number of training epochs (default: 10)
- **`learning_rate`: Learning rate for training (default: 1e-4)
- **`lora_rank & lora_alpha`: LoRA parameters that control adaptation strength

## Hardware Considerations

- **NVIDIA GPUs (CUDA):** Ensure CUDA is properly installed for GPU acceleration.
- **Apple Silicon (MPS):** Use PyTorch ≥ 2.0 to enable MPS support and aggressive memory management.
- **CPU (fallback):** Training will work but expect significantly slower performance.

## Using the Trained Model

After training completes, you can use the model for inference:

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
  "models/stoner-psych-lora",
  safety_checker=None
).to("cuda")  # or "mps" / "cpu"

image = pipe(
  "cosmic stoner rock album cover",
  num_inference_steps=25
).images[0]
image.save("generated_cover.png")
```

## Troubleshooting

- **Memory errors**: Reduce BATCH_SIZE in dataloader.py
- **Slow training**: Check if hardware acceleration is acMissing files: Ensure all required directories are created
- **API errors**: Verify your Hugging Face token in the .env file
