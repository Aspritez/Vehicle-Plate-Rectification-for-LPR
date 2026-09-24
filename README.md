# Vehicle-Plate-Rectification-for-LPR (Branch: Phet)

ระบบตรวจจับและปรับระนาบป้ายทะเบียนรถยนต์จากกล้องวงจรปิด CCTV ด้วยเทคโนโลยี **YOLOv8-OBB** และทฤษฎีเรขาคณิตการแปลงพิกัด **Geometric Homography Transformation (3-DOF, 4-DOF, 6-DOF, 8-DOF)**

---

## 🌟 ฟีเจอร์หลัก (Key Features)

1. **AI License Plate Detection (YOLO-OBB):**
   - ตรวจจับป้ายทะเบียนรถยนต์ที่เอียงในมุมมองต่าง ๆ จากภาพกล้องวงจรปิด (CCTV) ด้วยความแม่นยำสูง
   - โมเดลตรวจจับแบบหมุนตามทิศทางของวัตถุ (Oriented Bounding Box)
2. **Geometric Homography Rectification:**
   - ดึงภาพป้ายทะเบียนที่ถ่ายจากมุมเฉียง/เอียงรุนแรง (Perspective Foreshortening) ให้กลับมาเป็นภาพหน้าตรงแนวนอน 100% ขนาดมาตรฐาน 340x150 px
   - รองรับทฤษฎีเรขาคณิต 4 ระดับ:
     - **8-DOF Projective (Homography):** แปลงระนาบ 3 มิติ แก้ Perspective และจุดลู่เข้า Vanishing Point
     - **6-DOF Affine:** รักษาเส้นขนาน และแก้แรงเฉือน (Shear)
     - **4-DOF Similarity:** ปรับการหมุน เลื่อนตำแหน่ง และย่อขยายแบบสมมาตร
     - **3-DOF Euclidean (Rigid):** หมุนและเลื่อนตำแหน่งอย่างเดียว
3. **Interactive Testing GUI (`test_gui.py`):**
   - หน้าต่างโปรแกรมทดสอบ Dark Theme พร้อมการแสดงผลเปรียบเทียบแบบ Side-by-Side:
     - ฝั่งซ้าย: ภาพ CCTV เต็มพร้อมกรอบตรวจจับและจุดพิกัด 4 มุม
     - ฝั่งขวา: ภาพตัดตามมุมกล้องเดิม (Raw Crop) เทียบกับภาพที่ดึงตรงแล้ว (Rectified)
   - สลับระหว่าง **`🎯 ใช้เฉลย (.JSON)`** และ **`🤖 ตรวจจับด้วย AI (OBB)`** ได้ทันที
   - ปุ่มลัด **`🏷️ รูปที่มีเฉลย`** สำหรับกระโดดข้ามไปยังรูปที่มีการทำเฉลยไว้
   - บันทึกภาพป้ายทะเบียนที่ดึงตรงแล้วทั้งหมดลงในโฟลเดอร์ `predictions/cropped_plates/`

---

## 🚀 วิธีการติดตั้งและรันโปรแกรม (Getting Started)

### 1. ติดตั้ง Dependencies
```bash
pip install ultralytics opencv-python pillow numpy
```

### 2. รันโปรแกรม GUI สำหรับทดสอบ (Interactive GUI)
```bash
python test_gui.py
```
หรือดับเบิลคลิกไฟล์ `Start_Test_GUI.bat`

### 3. รันการทดสอบผ่าน Command Line
```bash
python test_detect.py
```

### 4. การจัดการชุดข้อมูลและการเทรนโมเดล
- แปลงไฟล์เฉลย AnyLabeling (`.json`) เป็นรูปแบบ YOLO-OBB:
  ```bash
  python split_dataset.py
  ```
- สั่งเทรนโมเดล YOLO-OBB:
  ```bash
  python train_obb.py
  ```

---

## 📁 โครงสร้างไฟล์ในโฟลเดอร์ (Repository Structure)

```text
├── best.pt                       # โมเดล YOLO-OBB ล่าสุดที่พร้อมใช้งาน
├── test_gui.py                   # โปรแกรม GUI หลักสำหรับทดสอบและดึงระนาบ Homography
├── test_detect.py                # สคริปต์ทดสอบผ่าน Command-line
├── train_obb.py                  # สคริปต์เทรนโมเดล YOLO-OBB
├── split_dataset.py              # สคริปต์แปลงเฉลย JSON เป็น YOLO-OBB
├── context.md                    # บันทึกประวัติและสรุปการดำเนินงานโครงการอย่างละเอียด
├── data.yaml                     # คอนฟิกชุดข้อมูลสำหรับเทรน YOLO
├── data/
│   └── raw_images/               # ภาพ CCTV และไฟล์เฉลย JSON
├── dataset/                      # ชุดข้อมูล Train/Val
└── predictions/                  # ผลลัพธ์ภาพที่ผ่านการปรับระนาบ
```

---

*พัฒนาโดย Phet (ทีม Vehicle-Plate-Rectification-for-LPR)*
