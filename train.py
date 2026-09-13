import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm

from src.model import UNet
from src.dataset import FaceDataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    train_loader = DataLoader(FaceDataset("./data/train", is_train=True), batch_size=8, shuffle=True)
    
    model = UNet().to(device)
    l1_loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

   
    print("\n>>> Phase 1: L1 Loss Warm-up (5 Epochs)")
    phase1_losses = []
    for epoch in range(5):
        model.train()
        train_loss = 0.0
        for bad, clean in tqdm(train_loader, desc=f"Phase 1 Epoch {epoch+1}/5"):
            bad, clean = bad.to(device), clean.to(device)
            pred = model(bad)
            loss = l1_loss(pred, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        epoch_loss = train_loss / len(train_loader)
        phase1_losses.append(epoch_loss)
        print(f"Phase 1 - Epoch [{epoch+1}/5] Loss: {epoch_loss:.5f}")

    
    print("\n>>> Phase 2: L1 + Identity Loss (30 Epochs)")
    face_embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    def recognition_loss(pred, target):
        e1 = face_embedder(pred)
        e2 = face_embedder(target)
        return torch.mean((e1 - e2) ** 2)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    phase2_losses = []

    for epoch in range(30):
        model.train()
        total_loss = 0.0
        for bad, clean in tqdm(train_loader, desc=f"Phase 2 Epoch {epoch+1}/30"):
            bad, clean = bad.to(device), clean.to(device)
            pred = model(bad)
            loss_r = l1_loss(pred, clean)
            loss_id = recognition_loss(pred, clean)
            loss = loss_r + 0.1 * loss_id

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        phase2_losses.append(epoch_loss)
        print(f"Phase 2 - Epoch [{epoch+1}/30] Loss: {epoch_loss:.5f}")

 
    save_path = "./checkpoints/face_restoration_unet.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved successfully to {save_path}")

   
    all_losses = phase1_losses + phase2_losses
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(all_losses) + 1), all_losses, marker='o', color='b')
    plt.axvline(x=len(phase1_losses) + 0.5, linestyle='--', color='red', label='Phase 2 Start')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("./results/training_curve.png")
    print("Training curve saved to results/training_curve.png")

if __name__ == "__main__":
    train()
