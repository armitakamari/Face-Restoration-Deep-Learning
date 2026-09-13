import os
import cv2
import random
import numpy as np
import torch
from shutil import copy2
from facenet_pytorch import MTCNN

def run_alignment_and_split(src_folder="./data/raw", aligned_folder="./data/aligned", output_dir="./data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=20, device=device)

    os.makedirs(aligned_folder, exist_ok=True)
    print(f"Aligning raw images from {src_folder}...")

    for name in os.listdir(src_folder):
        try:
            path = os.path.join(src_folder, name)
            img = cv2.imread(path)
            if img is None:
                continue
            img = img[:, :, ::-1]
            face, prob = mtcnn(img, return_prob=True)
            if face is None:
                continue
            face = (face.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(aligned_folder, name), face[:, :, ::-1])
        except Exception as e:
            print("Error processing:", name, e)

    print("Alignment done.")

     
    all_images = os.listdir(aligned_folder)
    random.shuffle(all_images)
    n = len(all_images)
    train_end = int(n * 0.75)
    val_end = int(n * 0.90)

    splits = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:]
    }

    for split, img_list in splits.items():
        folder = os.path.join(output_dir, split)
        os.makedirs(folder, exist_ok=True)
        for img_name in img_list:
            copy2(os.path.join(aligned_folder, img_name), folder)

    print(f"Data split completed: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

if __name__ == "__main__":
    run_alignment_and_split()
