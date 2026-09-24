"""
สคริปต์เทรนโมเดล YOLO11-OBB สำหรับตรวจจับป้ายทะเบียนมุมเอียง/มุมแปลก
"""
import sys
import torch
from ultralytics import YOLO

# ป้องกันปัญหา UnicodeEncodeError ใน Windows Terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def train():
    # ตรวจสอบการ์ดจอ
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[*] กำลังเริ่มเทรนบนอุปกรณ์: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    # ใช้โมเดลเริ่มต้น YOLO11 Nano OBB (เบา เร็ว และแม่นยำสูง)
    model = YOLO("yolo11n-obb.pt")

    # เริ่มกระบวนการเทรน
    results = model.train(
        data="data.yaml",       # ไฟล์ config ชุดข้อมูล
        epochs=100,             # จำนวนรอบการเทรน (100 epochs)
        imgsz=640,              # ขนาดภาพ (640x640)
        batch=16,               # ปรับตามขนาด VRAM (RTX 3080 10GB ใส่ 16-32 ได้สบาย)
        device=device,
        save=True,
        project="runs/detect_plate_obb",
        name="experiment_1",
        # Data Augmentation เพื่อช่วยรับมือกับมุมแปลกๆ และสภาพแสงของ CCTV
        degrees=15.0,           # สุ่มหมุนภาพ +/- 15 องศา
        perspective=0.001,      # บิดมุมมอง Perspective
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    print("[OK] เทรนเสร็จสมบูรณ์! โมเดลที่ดีที่สุดถูกบันทึกไว้ที่ runs/detect_plate_obb/experiment_1/weights/best.pt")

if __name__ == "__main__":
    train()
