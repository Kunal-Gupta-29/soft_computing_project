import os
import shutil

# Correct paths based on your desktop extraction
RAFDB_DIR = r"C:\Users\kunal\Desktop\rafdb\DATASET"
FER_DIR = r"C:\Users\kunal\Desktop\soft computing\soft_computing_project\data"

# RAF-DB mapping
# 1: Surprise, 2: Fear, 3: Disgust, 4: Happiness, 5: Sadness, 6: Anger, 7: Neutral
LABEL_MAP = {
    "1": "surprise",
    "2": "fear",
    "3": "disgust",
    "4": "happy",
    "5": "sad",
    "6": "angry",
    "7": "neutral"
}

def merge_split(split="train"):
    print(f"\n--- Scanning {split} datasets ---")
    raf_split_dir = os.path.join(RAFDB_DIR, split)
    fer_split_dir = os.path.join(FER_DIR, split)
    
    if not os.path.exists(raf_split_dir):
        print(f"Error: RAF-DB {split} directory not found at {raf_split_dir}")
        return
        
    for raf_label, fer_label in LABEL_MAP.items():
        src_dir = os.path.join(raf_split_dir, raf_label)
        dst_dir = os.path.join(fer_split_dir, fer_label)
        
        if not os.path.exists(src_dir):
            continue
            
        os.makedirs(dst_dir, exist_ok=True)
        files = os.listdir(src_dir)
        count = 0
        
        for f in files:
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')): 
                continue
                
            src_file = os.path.join(src_dir, f)
            # Prefix with rafdb_ to strictly protect existing FER2013 files
            dst_file = os.path.join(dst_dir, f"rafdb_{split}_{raf_label}_{f}")
            
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)
                count += 1
                
        print(f"Copied {count} images from RAF-DB [{split}/{raf_label}] -> FER [{fer_label}]")

if __name__ == "__main__":
    print("====================================")
    print("  FER + RAF-DB Dataset Merger Tool  ")
    print("====================================")
    merge_split("train")
    merge_split("test")
    print("\n[SUCCESS] Merge Complete! You now have a massive robust dataset.")
