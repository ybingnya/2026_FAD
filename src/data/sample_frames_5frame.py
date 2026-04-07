import cv2
import pandas as pd
from pathlib import Path

ROOT = Path("/home/saslab01/Desktop/replay_pad")
META_PATH = ROOT / "metadata" / "metadata_all.csv"
SAVE_ROOT = ROOT / "data" / "frames_5frame"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(META_PATH)
rows = []

for _, row in df.iterrows():
    video_path = row["video_path"]
    split = row["split"]
    video_id = row["video_id"]
    label = row["label"]
    attack_type = row["attack_type"]

    save_dir = SAVE_ROOT / split / attack_type / video_id
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        continue

    indices = []
    for i in range(5):
        idx = int((i + 1) * total_frames / 6)
        indices.append(idx)

    for frame_idx, target_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        save_path = save_dir / f"{video_id}_f{frame_idx}.jpg"
        cv2.imwrite(str(save_path), frame)

        rows.append({
            "split": split,
            "video_id": video_id,
            "frame_path": str(save_path),
            "label": label,
            "attack_type": attack_type,
            "frame_idx": frame_idx
        })

    cap.release()

save_csv_path = ROOT / "metadata" / "frames_5frame.csv"
pd.DataFrame(rows).to_csv(save_csv_path, index=False)
print(f"[INFO] saved: {save_csv_path}")
print(f"[INFO] total frames: {len(rows)}")