import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from src.model import UNet
from src.dataset import FaceDataset

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("./results", exist_ok=True)

    model = UNet().to(device)
    model.load_state_dict(torch.load("./checkpoints/face_restoration_unet.pth", map_location=device))
    model.eval()

    face_embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    test_loader = DataLoader(FaceDataset("./data/test", is_train=False), batch_size=8, shuffle=False)

    psnr_list, ssim_list, embed_list = [], [], []

    print("Evaluating model on test dataset...")
    with torch.no_grad():
        for bad, clean in test_loader:
            bad, clean = bad.to(device), clean.to(device)
            pred = model(bad)

            for p, c in zip(pred.cpu().numpy(), clean.cpu().numpy()):
                p = p.transpose(1, 2, 0)
                c = c.transpose(1, 2, 0)
                psnr_list.append(psnr(c, p, data_range=1))
                ssim_list.append(ssim(c, p, channel_axis=2, data_range=1))

            e1 = face_embedder(pred)
            e2 = face_embedder(clean)
            embed_list.append(torch.mean((e1 - e2) ** 2).item())

    print("\n--- Test Set Evaluation Metrics ---")
    print(f"PSNR: {np.mean(psnr_list):.4f}")
    print(f"SSIM: {np.mean(ssim_list):.4f}")
    print(f"Embedding Distance: {np.mean(embed_list):.4f}")

    # Visualize test results
    bad_batch, clean_batch = next(iter(test_loader))
    bad_batch, clean_batch = bad_batch.to(device), clean_batch.to(device)
    with torch.no_grad():
        restored_batch = model(bad_batch)

    n_show = min(5, bad_batch.size(0))
    fig, axes = plt.subplots(3, n_show, figsize=(3 * n_show, 9))
    labels = ["Degraded", "Restored", "Original"]
    imgs = [
        bad_batch.cpu().permute(0, 2, 3, 1).numpy().clip(0, 1),
        restored_batch.cpu().permute(0, 2, 3, 1).numpy().clip(0, 1),
        clean_batch.cpu().permute(0, 2, 3, 1).numpy().clip(0, 1),
    ]

    for r in range(3):
        for i in range(n_show):
            ax = axes[r, i] if n_show > 1 else axes[r]
            ax.imshow(imgs[r][i])
            ax.set_xticks([])
            ax.set_yticks([])
        (axes[r, 0] if n_show > 1 else axes[r]).set_ylabel(labels[r], fontsize=12)

    plt.suptitle("Face Restoration: Degraded vs Restored vs Original")
    plt.tight_layout()
    plt.savefig("./results/qualitative_comparison.png")
    print("Visual comparison saved to results/qualitative_comparison.png")

if __name__ == "__main__":
    evaluate()
