"""
setup_rafdb_only.py
-------------------
Creates a clean RAF-DB-only dataset at:
    data_rafdb/train/<emotion>/
    data_rafdb/test/<emotion>/

This does NOT touch or delete your existing merged data/ folder.
After running this, set DATASET_MODE = "rafdb" in config.py.

Run:
    python setup_rafdb_only.py
"""

import os
import shutil

# --- Paths -------------------------------------------------------------------

RAFDB_DIR      = r"C:\Users\kunal\Desktop\rafdb\DATASET"
OUTPUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_rafdb")

# RAF-DB numeric label -> FER2013-compatible folder name
LABEL_MAP = {
    "1": "surprise",
    "2": "fear",
    "3": "disgust",
    "4": "happy",
    "5": "sad",
    "6": "angry",
    "7": "neutral",
}

SPLITS = ["train", "test"]

# --- Main -------------------------------------------------------------------

def setup_rafdb_only():
    print("=" * 50)
    print("  RAF-DB Only Dataset Setup")
    print("=" * 50)

    if not os.path.exists(RAFDB_DIR):
        print(f"\n[ERROR] RAF-DB source not found at:\n  {RAFDB_DIR}")
        print("\nMake sure you have RAF-DB extracted to your Desktop.")
        return

    total_copied = 0

    for split in SPLITS:
        print(f"\n--- Processing split: {split} ---")
        raf_split_dir = os.path.join(RAFDB_DIR, split)

        if not os.path.exists(raf_split_dir):
            print(f"  [SKIP] RAF-DB {split} not found at {raf_split_dir}")
            continue

        for raf_label, emotion_name in LABEL_MAP.items():
            src_dir = os.path.join(raf_split_dir, raf_label)
            dst_dir = os.path.join(OUTPUT_DIR, split, emotion_name)

            if not os.path.exists(src_dir):
                print(f"  [SKIP] {split}/{raf_label} not found")
                continue

            os.makedirs(dst_dir, exist_ok=True)

            files = [f for f in os.listdir(src_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            count = 0

            for f in files:
                src_file = os.path.join(src_dir, f)
                dst_file = os.path.join(dst_dir, f)
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                    count += 1

            total_copied += count
            print(f"  [{emotion_name:8s}] Copied {count:>4} images  ->  {dst_dir}")

    print(f"\n[SUCCESS] Total images copied: {total_copied}")
    print(f"[SUCCESS] RAF-DB-only dataset ready at:\n  {OUTPUT_DIR}")
    print("\nNext step: set DATASET_MODE = \"rafdb\" in config.py and run  python train.py  to train on this dataset.")

    # Print folder stats
    print("\n--- Dataset Summary ---")
    for split in SPLITS:
        split_dir = os.path.join(OUTPUT_DIR, split)
        if not os.path.exists(split_dir):
            continue
        print(f"\n  {split}/")
        total = 0
        for emotion in sorted(os.listdir(split_dir)):
            emo_dir = os.path.join(split_dir, emotion)
            if os.path.isdir(emo_dir):
                n = len(os.listdir(emo_dir))
                total += n
                print(f"    {emotion:10s}: {n:>5} images")
        print(f"    {'TOTAL':10s}: {total:>5} images")


if __name__ == "__main__":
    setup_rafdb_only()
