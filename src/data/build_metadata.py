import os
import csv
from pathlib import Path

ROOT = Path("/home/saslab01/Desktop/replay_pad")
DATA_DIR = ROOT / "data"
META_DIR = ROOT / "metadata"
META_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "devel", "test"]

def get_label_and_attack_type(path_str):
    path_str = path_str.lower()
    if "/real/" in path_str:
        return 0, "real"
    elif "/attack/fixed/" in path_str:
        return 1, "fixed"
    elif "/attack/hand/" in path_str:
        return 1, "hand"
    return None, None

def collect_videos():
    rows = []
    for split in SPLITS:
        split_dir = DATA_DIR / split
        for ext in ["*.mov", "*.mp4", "*.avi"]:
            for video_path in split_dir.rglob(ext):
                label, attack_type = get_label_and_attack_type(str(video_path))
                if label is None:
                    continue

                video_id = video_path.stem
                rows.append({
                    "split": split,
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "label": label,
                    "attack_type": attack_type
                })
    return rows

def save_csv(rows, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "video_id", "video_path", "label", "attack_type"]
        )
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    rows = collect_videos()
    save_path = META_DIR / "metadata_all.csv"
    save_csv(rows, save_path)
    print(f"[INFO] saved: {save_path}")
    print(f"[INFO] total videos: {len(rows)}")