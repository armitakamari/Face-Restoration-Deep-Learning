# Face Restoration & Quality Enhancement

Restores degraded facial images (low resolution, Gaussian noise, motion blur) using a 3-level U-Net architecture, trained with a 2-phase pipeline combining L1 reconstruction loss and identity perceptual loss (InceptionResnetV1 / VGGFace2 embeddings).

## Pipeline
1. **Face Alignment:** Detect & crop aligned faces via MTCNN (160x160).
2. **Paired Augmentation & Degradation:** Downsampling, Gaussian noise ($\sigma \in [0.2, 0.5]$), and motion blur.
3. **2-Phase Training:**
   - **Phase 1:** L1 warm-up (5 epochs).
   - **Phase 2:** Combined L1 + 0.1 * Identity Embedding Loss (30 epochs).
4. **Evaluation:** PSNR, SSIM, and FaceNet embedding distance.

## Installation
```bash
git clone https://github.com/armitakamari/Face-Restoration-UNet.git
cd Face-Restoration-UNet
pip install -r requirements.txt
