import os
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================
# 기본 경로 설정
# =========================
PROJECT_ROOT = "/Users/youbin/Desktop/replay_pad"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
META_ROOT = os.path.join(PROJECT_ROOT, "metadata")
CLIP_ROOT = os.path.join(PROJECT_ROOT, "interim", "video_clips")

# =========================
# clip 생성 설정
# =========================
CLIP_LENGTH = 16
STRIDE = 8
IMG_SIZE = 224

VIDEO_EXTENSIONS = ["*.mov", "*.mp4", "*.avi", "*.m4v"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_all_videos():
    video_paths = []
    for split in ["train", "devel", "test"]:
        split_root = os.path.join(DATA_ROOT, split)
        for ext in VIDEO_EXTENSIONS:
            video_paths.extend(
                glob.glob(os.path.join(split_root, "**", ext), recursive=True)
            )
    return sorted(video_paths)


def infer_split(video_path):
    video_path = video_path.replace("\\", "/")
    if "/train/" in video_path:
        return "train"
    elif "/devel/" in video_path:
        return "devel"
    elif "/test/" in video_path:
        return "test"
    else:
        raise ValueError(f"split 추론 실패: {video_path}")


def infer_label_and_attack(video_path):
    p = video_path.lower().replace("\\", "/")

    if "/real/" in p:
        return 0, "real"   # live
    elif "/attack/fixed/" in p:
        return 1, "fixed"
    elif "/attack/hand/" in p:
        return 1, "hand"
    else:
        return 1, "attack"


def read_video_frames(video_path, img_size=224):
    cap = cv2.VideoCapture(video_path)

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)

    cap.release()
    return frames, fps, total_frames


def make_clips(frames, clip_length=16, stride=8):
    """
    frames: list of HWC RGB np.array
    return: list of (start_idx, end_idx, clip_array)
            clip_array shape = (T, H, W, C)
    """
    clips = []
    n = len(frames)

    if n == 0:
        return clips

    # 짧은 영상이면 마지막 프레임 반복해서 16장 맞춤
    if n < clip_length:
        padded = frames.copy()
        while len(padded) < clip_length:
            padded.append(padded[-1].copy())
        clip = np.stack(padded, axis=0)
        clips.append((0, n - 1, clip))
        return clips

    # 일반적인 경우 sliding window
    for start in range(0, n - clip_length + 1, stride):
        end = start + clip_length - 1
        clip = np.stack(frames[start:start + clip_length], axis=0)
        clips.append((start, end, clip))

    # 마지막 tail이 아예 버려지지 않게 보정
    last_start = n - clip_length
    last_end = n - 1
    if len(clips) == 0 or clips[-1][0] != last_start:
        clip = np.stack(frames[last_start:last_start + clip_length], axis=0)
        clips.append((last_start, last_end, clip))

    return clips


def make_safe_video_id(video_path):
    rel = os.path.relpath(video_path, DATA_ROOT)
    rel = rel.replace("\\", "/")
    no_ext = os.path.splitext(rel)[0]
    safe_id = no_ext.replace("/", "__")
    return safe_id


def main():
    ensure_dir(META_ROOT)
    for split in ["train", "devel", "test"]:
        ensure_dir(os.path.join(CLIP_ROOT, split))

    video_paths = find_all_videos()
    print(f"[INFO] 찾은 비디오 수: {len(video_paths)}")

    rows = []

    for video_path in tqdm(video_paths, desc="Making clips"):
        split = infer_split(video_path)
        label_binary, attack_type = infer_label_and_attack(video_path)

        frames, fps, total_frames = read_video_frames(video_path, img_size=IMG_SIZE)

        if len(frames) == 0:
            print(f"[WARN] 프레임이 0개인 비디오 스킵: {video_path}")
            continue

        clips = make_clips(
            frames,
            clip_length=CLIP_LENGTH,
            stride=STRIDE
        )

        safe_video_id = make_safe_video_id(video_path)
        save_dir = os.path.join(CLIP_ROOT, split, safe_video_id)
        ensure_dir(save_dir)

        for clip_idx, (start_idx, end_idx, clip_array) in enumerate(clips):
            clip_name = f"clip_{clip_idx:03d}.npy"
            clip_path = os.path.join(save_dir, clip_name)

            np.save(clip_path, clip_array)

            rows.append({
                "split": split,
                "video_path": video_path,
                "clip_path": clip_path,
                "label_binary": label_binary,
                "attack_type": attack_type,
                "clip_length": CLIP_LENGTH,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "fps": fps,
                "num_frames": total_frames
            })

    df = pd.DataFrame(rows)

    save_csv_path = os.path.join(META_ROOT, "video_clips.csv")
    df.to_csv(save_csv_path, index=False)

    print(f"\n[INFO] 저장 완료: {save_csv_path}")
    print(f"[INFO] 총 clip 수: {len(df)}")
    print("[INFO] split별 clip 수:")
    print(df["split"].value_counts())
    print("\n[INFO] label별 clip 수:")
    print(df["label_binary"].value_counts())


if __name__ == "__main__":
    main()