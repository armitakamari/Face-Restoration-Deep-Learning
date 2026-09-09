# Face Restoration & Quality Enhancement — CelebA

A U-Net-based encoder–decoder that restores degraded face images
(low resolution, Gaussian noise, motion blur).

## Pipeline
1. **Face detection & alignment** — MTCNN, 5-point landmarks
2. **Data split** — train / val / test
3. **Augmentation & degradation** — downsampling (×1.2 / ×1.4), Gaussian noise (σ 0.2–0.5), motion blur
4. **Model** — 3-level U-Net
5. **Training** — Phase 1: L1 restoration loss only → Phase 2: L1 + identity loss (pretrained face embeddings)
6. **Evaluation** — PSNR, SSIM, embedding distance
7. **Generalization test** — 3-level U-Net
5. **Training** — Phase 1: L1 restoration loss only → Phase 2: L1 + identity loss (pretrained face embeddings)
6. **Evaluation** — PSNR, SSIM, embedding distance
7. **Generalization test** — on a second, unseen face dataset
, matplotlib
