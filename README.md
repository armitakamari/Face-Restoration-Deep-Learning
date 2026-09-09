# Face Restoration & Quality Enhancement (CelebA)

Restores degraded face images (low resolution, Gaussian noise, motion blur) using a U-Net
encoder–decoder, trained with a combined **restoration loss** and **identity loss**.

Built for: مبانی یادگیری ماشین (ML Fundamentals), Fall 1404 — Shahid Beheshti University.

## Pipeline

1. **Align faces** — MTCNN (5-point landmark alignment)
2. **Split** — 75% train / 15% val / 10% test
3. **Augment** — flip, brightness/contrast, small rotation/shift (applied once, before degradation)
4. **Degrade** — random per sample: downsample (×1.2/×1.4), Gaussian noise (σ 0.2–0.5), or motion blur
5. **Model** — 3-level U-Net
6. **Train** — Phase 1: L1 loss warm-up → Phase 2: L1 + identity loss (`InceptionResnetV1`, VGGFace2)
7. **Evaluate** — PSNR, SSIM, embedding distance on the CelebA test set
8. **Generalize** — same metrics on a second, unseen face dataset
9. **Visualize** — degraded vs. restored vs. original, side by side

## Two fixes in this version

- **Normalization:** images are scaled to `[0,1]` before the model (it outputs `sigmoid`, so
  `[0,255]` targets silently break the loss and PSNR/SSIM).
- **Paired augmentation:** the clean and degraded images now share one augmentation draw. Augmenting
  them separately (as in earlier drafts) gave each a different random flip/rotation, misaligning
  input and target.

## Setup

Edit these paths in Section 1, then run all cells top to bottom (GPU recommended):

```python
SRC_FOLDER = "./images"                  # CelebA images for train/val/test
GENERALIZATION_FOLDER = "./other_faces"  # a DIFFERENT face dataset for the generalization test
```

## Outputs

`face_restoration_unet.pth`, a training curve plot, PSNR/SSIM/embedding metrics (test +
generalization sets), and a qualitative comparison grid.

## Report structure

Problem & dataset → preprocessing pipeline → architecture → loss functions → training details →
quantitative results (test set) → generalization results → qualitative examples → limitations.
