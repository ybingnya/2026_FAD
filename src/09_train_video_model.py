import os
import random
import numpy as np
import pandas as pd
from PIL import Image

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# =========================
# 기본 설정
# =========================
PROJECT_ROOT = "/Users/youbin/Desktop/replay_pad"
META_PATH = os.path.join(PROJECT_ROOT, "metadata", "video_clips.csv")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

BATCH_SIZE = 2
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
NUM_WORKERS = 0

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Dataset
# =========================
class VideoClipDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip_path = row["clip_path"]
        label = int(row["label_binary"])

        clip = np.load(clip_path)  # (T, H, W, C), uint8
        frames = []

        for t in range(clip.shape[0]):
            frame = clip[t]
            frame = Image.fromarray(frame)

            if self.transform is not None:
                frame = self.transform(frame)

            frames.append(frame)

        # (T, C, H, W)
        frames = torch.stack(frames, dim=0)

        return frames, torch.tensor(label, dtype=torch.long)


# =========================
# Model
# =========================
class ResNet18LSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=1, num_classes=2, bidirectional=False):
        super().__init__()

        backbone = models.resnet18(weights=None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_out_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)         # (B*T, F)
        feats = feats.view(B, T, -1)     # (B, T, F)

        lstm_out, (h_n, c_n) = self.lstm(feats)

        # 마지막 timestep 출력 사용
        last_out = lstm_out[:, -1, :]    # (B, hidden_dim)
        logits = self.classifier(last_out)

        return logits


# =========================
# Utils
# =========================
def get_dataloaders():
    df = pd.read_csv(META_PATH)

    train_df = df[df["split"] == "train"].copy()
    devel_df = df[df["split"] == "devel"].copy()

    print("[INFO] train clips:", len(train_df))
    print("[INFO] devel clips:", len(devel_df))
    print("[INFO] train label counts:")
    print(train_df["label_binary"].value_counts())
    print("[INFO] devel label counts:")
    print(devel_df["label_binary"].value_counts())

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = VideoClipDataset(train_df, transform=train_transform)
    devel_dataset = VideoClipDataset(devel_df, transform=valid_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    devel_loader = DataLoader(
        devel_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, devel_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for clips, labels in tqdm(loader, desc="Train", leave=False):
        clips = clips.to(device)    # (B, T, C, H, W)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(clips)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for clips, labels in tqdm(loader, desc="Eval", leave=False):
        clips = clips.to(device)
        labels = labels.to(device)

        logits = model(clips)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def main():
    set_seed(SEED)

    print("[INFO] Device:", DEVICE)

    train_loader, devel_loader = get_dataloaders()

    model = ResNet18LSTM(
        hidden_dim=256,
        num_layers=1,
        num_classes=2,
        bidirectional=False
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_devel_acc = 0.0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} =====")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )

        devel_loss, devel_acc = evaluate(
            model, devel_loader, criterion, DEVICE
        )

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
        print(f"[Epoch {epoch}] devel_loss={devel_loss:.4f} devel_acc={devel_acc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "devel_loss": devel_loss,
            "devel_acc": devel_acc
        })

        if devel_acc > best_devel_acc:
            best_devel_acc = devel_acc

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                "best_video_model_resnet18_lstm.pth"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_devel_acc": best_devel_acc
            }, checkpoint_path)

            print(f"[INFO] Best model saved -> {checkpoint_path}")

    history_df = pd.DataFrame(history)
    history_path = os.path.join(LOG_DIR, "video_model_train_history.csv")
    history_df.to_csv(history_path, index=False)

    print("\n[INFO] Training finished")
    print(f"[INFO] Best devel acc: {best_devel_acc:.4f}")
    print(f"[INFO] History saved -> {history_path}")


if __name__ == "__main__":
    main()