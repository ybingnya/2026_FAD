import cv2
import pandas as pd
from pathlib import Path

ROOT = Path("/home/saslab01/Desktop/replay_pad")
META_PATH = ROOT / "metadata" / "metadata_all.csv"
SAVE_ROOT = ROOT / "data" / "clips_10frame"
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

    frame_paths = []
    indices = [int((i + 1) * total_frames / 11) for i in range(10)]

    for frame_idx, target_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            frame_paths = []
            break

        save_path = save_dir / f"{video_id}_f{frame_idx}.jpg"
        cv2.imwrite(str(save_path), frame)
        frame_paths.append(str(save_path))

    cap.release()

    if len(frame_paths) == 10:
        item = {
            "split": split,
            "video_id": video_id,
            "label": label,
            "attack_type": attack_type,
        }
        for i in range(10):
            item[f"frame_{i}"] = frame_paths[i]
        rows.append(item)

save_csv_path = ROOT / "metadata" / "clips_10frame.csv"
pd.DataFrame(rows).to_csv(save_csv_path, index=False)
print(f"[INFO] saved: {save_csv_path}")
print(f"[INFO] total clips: {len(rows)}")