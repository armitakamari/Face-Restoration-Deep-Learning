# Face Restoration & Quality Enhancement

Restores degraded facial images (low resolution, Gaussian noise, motion blur) using a 3-level U-Net architecture, trained with a 2-phase pipeline combining L1 reconstruction loss and identity perceptual loss (InceptionResnetV1 / VGGFace2 embeddings).

## Pipeline

1. **Face Alignment:** Detect & crop aligned faces via MTCNN (160x160).
2. **Paired Augmentation & Degradation:** Downsampling, Gaussian noise (σ ∈ [0.2, 0.5]), and motion blur.
3. **2-Phase Training:**
   - **Phase 1:** L1 warm-up.
   - **Phase 2:** Combined L1 + 0.1 × Identity Embedding Loss.
4. **Evaluation:** PSNR, SSIM, and FaceNet embedding distance.

## Project Structure
Face-Restoration-Deep-Learning/
├── src/ # Alignment, model, dataset utilities
├── train.py # Training entry point
├── evaluate.py # Evaluation entry point
├── requirements.txt
├── Face_Restoration_Run.ipynb # End-to-end notebook (extract → align → train → evaluate)
└── dataset.zip # (not tracked) place your image archive here


## Installation

```bash
git clone https://github.com/armitakamari/Face-Restoration-Deep-Learning.git
cd Face-Restoration-Deep-Learning
pip install -r requirements.txt
```

## Usage

### Option 1: Run the notebook
Place your image archive as `dataset.zip` in the project root, then open and run `Face_Restoration_Run.ipynb` from top to bottom. It will:
1. Extract and validate the dataset,
2. Align faces and split into train/val/test,
3. Train the U-Net (2-phase),
4. Evaluate and save metrics/plots.

### Option 2: Run scripts directly
```bash
python train.py
python evaluate.py
```

## Results

Example run (512 images, 10 total epochs, batch size 4):

| Metric | Value |
|---|---|
| PSNR | 27.84 dB |
| SSIM | 0.836 |
| Embedding distance | 0.00047 |

Plots (training curve, qualitative comparisons, predictions) are saved under `results/`.

## Requirements
See `requirements.txt`. Training was run on both CPU/GPU/MPS backends.
 
