import os
import sys
import json
import shutil
import random
from pathlib import Path

# ป้องกันปัญหา UnicodeEncodeError ใน Windows Terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Paths
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw_images"
OUTPUT_DIR = BASE_DIR / "dataset"

# คลาสที่รองรับ
CLASS_MAP = {
    "license_plate": 0
}

def setup_directories():
    for split in ["train", "val"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

def json_to_yolo_obb(json_path, target_txt_path):
    """แปลงไฟล์ .json จาก X-AnyLabeling เป็นไฟล์ .txt สำหรับ YOLO-OBB"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data.get("imageWidth")
    img_h = data.get("imageHeight")

    if not img_w or not img_h:
        return False

    lines = []
    for shape in data.get("shapes", []):
        label = shape.get("label", "license_plate")
        cls_id = CLASS_MAP.get(label, 0)
        points = shape.get("points", [])

        # ต้องการจุดพิกัด 4 จุดสำหรับ OBB
        if len(points) == 4:
            norm_points = []
            for pt in points:
                nx = max(0.0, min(1.0, pt[0] / img_w))
                ny = max(0.0, min(1.0, pt[1] / img_h))
                norm_points.extend([f"{nx:.6f}", f"{ny:.6f}"])
            
            lines.append(f"{cls_id} " + " ".join(norm_points))

    if lines:
        with open(target_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return True
    return False

def split_data(train_ratio=0.8, seed=42):
    random.seed(seed)
    setup_directories()

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    # ตรวจหาคู่ไฟล์รูปภาพกับ .json หรือ .txt
    valid_items = []
    for img_path in RAW_DIR.iterdir():
        if img_path.suffix.lower() in image_extensions:
            json_path = img_path.with_suffix(".json")
            txt_path = img_path.with_suffix(".txt")
            
            if json_path.exists():
                valid_items.append((img_path, json_path, "json"))
            elif txt_path.exists():
                valid_items.append((img_path, txt_path, "txt"))

    if not valid_items:
        print(f"[!] ยังไม่พบไฟล์ Label (.json หรือ .txt) ใน {RAW_DIR}")
        return

    random.shuffle(valid_items)
    train_count = max(1, int(len(valid_items) * train_ratio))
    train_items = valid_items[:train_count]
    val_items = valid_items[train_count:] if len(valid_items) > 1 else valid_items

    # Copy files & convert
    for split_name, items in [("train", train_items), ("val", val_items)]:
        for img_path, label_path, label_type in items:
            dest_img = OUTPUT_DIR / split_name / "images" / img_path.name
            dest_txt = OUTPUT_DIR / split_name / "labels" / (img_path.stem + ".txt")

            shutil.copy2(img_path, dest_img)

            if label_type == "json":
                json_to_yolo_obb(label_path, dest_txt)
            else:
                shutil.copy2(label_path, dest_txt)

    print(f"[OK] แปลงและแบ่งข้อมูลสำเร็จ:")
    print(f"   - พบข้อมูลที่วาดแล้ว: {len(valid_items)} รูป")
    print(f"   - Train: {len(train_items)} รูป -> {OUTPUT_DIR / 'train'}")
    print(f"   - Val: {len(val_items)} รูป -> {OUTPUT_DIR / 'val'}")

    # Generate data.yaml for YOLO OBB training
    yaml_content = f"""path: {OUTPUT_DIR.as_posix()}
train: train/images
val: val/images

names:
  0: license_plate
"""
    yaml_path = BASE_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[OK] สร้างไฟล์ตั้งค่า {yaml_path} เรียบร้อย พร้อมเทรน!")

if __name__ == "__main__":
    split_data()
