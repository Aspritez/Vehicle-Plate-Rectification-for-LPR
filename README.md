# Thai License Plate Deskew

เว็บแอป Streamlit สำหรับเลือกตำแหน่งป้ายทะเบียน 4 จุด ปรับมุมภาพป้ายด้วย Perspective Transform และเพิ่มความคมชัดด้วย OpenCV

แอปไม่มีฐานข้อมูล ไม่เก็บไฟล์อัปโหลด และเก็บจุดที่ผู้ใช้เลือกไว้เฉพาะระหว่างเปิดหน้าเว็บนั้น

## ความสามารถ

- อัปโหลดภาพ JPG หรือ PNG ขนาดไม่เกิน 10 MB
- ค้นหาตำแหน่งป้ายอัตโนมัติ หรือคลิกเลือกมุมป้าย 4 จุดด้วยตนเอง
- ป้องกันจุดซ้ำ รูปสี่เหลี่ยมผิดรูป และการตัดภาพที่ไม่ถูกต้อง
- แสดงภาพต้นฉบับพร้อมกรอบ และภาพป้ายที่ปรับมุมแล้ว
- ดาวน์โหลดภาพผลลัพธ์เป็น PNG

## ขั้นตอนตรวจจับอัตโนมัติ

`ภาพรถ → Gray / Blur → Canny / Threshold → Contours → คะแนน 6 ด้าน → จัดอันดับ Candidate → ROI → 4 Corners → Homography`

ระบบให้คะแนนทุก contour ที่ผ่านเงื่อนไขพื้นฐาน แทนการเลือกสี่เหลี่ยมที่ใหญ่ที่สุดก่อน:

| Feature | ใช้วัด | น้ำหนัก |
| --- | --- | --- |
| Area | พื้นที่เหมาะกับขนาดภาพและค่าพื้นที่ขั้นต่ำ | 10% |
| Aspect ratio | สัดส่วนกว้าง/สูงของกรอบที่หมุนตามวัตถุ | 15% |
| Rectangularity | พื้นที่ contour / พื้นที่กรอบหมุนที่ครอบ contour | 15% |
| Solidity | พื้นที่ contour / พื้นที่ convex hull | 10% |
| Contour approximation | จำนวนจุดจาก `approxPolyDP` ใกล้ 4 จุดเพียงใด | 15% |
| Edge density | สัดส่วนเส้นขอบภายใน contour โดยตัดขอบนอกออก | 35% |

แต่ละ feature แปลงเป็นคะแนน 0–1 แล้วรวมแบบถ่วงน้ำหนัก คะแนนนี้ใช้จัดอันดับ ไม่ใช่ความน่าจะเป็นหรือเปอร์เซ็นต์ความแม่นยำ ดูสูตรและค่าปรับแต่งได้ใน `plate_detection.py`

- ค้นหา contour จาก Canny, การเชื่อมขอบด้วย morphology, adaptive threshold และ Otsu พร้อมลดกรอบซ้ำ
- พิจารณา candidate ตามคะแนนจากมากไปน้อย ต้องได้คะแนนรวมอย่างน้อย 0.62 และ edge density อย่างน้อย 0.012
- หาก contour แรกให้สี่มุมนูนที่เข้ากับรูปร่างได้ดี ใช้มุมนั้นได้ทันที
- หากยังหาสี่มุมที่เหมาะสมไม่ได้ ให้ตัด ROI พร้อมขอบเผื่อ แล้วหา contour ของขอบป้ายซ้ำด้วย Canny / Otsu และ `approxPolyDP` ก่อนคืนพิกัดสู่ภาพเต็ม
- ถ้าหามุมไม่ได้ ให้ข้ามไป candidate ถัดไป โดยจำกัดการลองไว้ 40 candidate ที่ผ่านเกณฑ์ หากไม่มีผลให้ผู้ใช้เลือกจุดเอง
- ใช้มุมที่ตรวจสอบแล้วกับ `getPerspectiveTransform` และ `warpPerspective` บนภาพต้นฉบับ ไม่ใช้มุมของ bounding box แทนขอบป้าย
- หลังค้นหา แสดงตารางคะแนนของ candidate 20 อันดับแรก และผลที่เลือกหากอยู่นอก 20 อันดับ พร้อมคะแนนรวมและวิธีหามุม

ยังเป็นวิธีเชิง heuristic: ไฟท้าย ลวดลายพื้น หรือกรอบอื่นอาจได้คะแนนสูง และป้ายที่ขอบไม่ชัดอาจตรวจไม่พบ ต้องตรวจกรอบก่อนดาวน์โหลด ยังไม่มีชุดข้อมูลที่มีมุมป้ายกำกับสำหรับวัดความแม่นของ Homography

## ผลปรับค่ากับ dataset v23

ทดลอง 27 ชุดพารามิเตอร์กับภาพ 456 ภาพ โดยแยกภาพดัดแปลงของต้นฉบับเดียวกันให้อยู่ในชุดเดียวกัน ผลบน test ที่กันไว้ 93 ภาพ: ค่าเดิมตรงกรอบที่ IoU ≥ 0.5 จำนวน 39 ภาพ (41.9%) ส่วนค่าทดลองได้ 40 ภาพ (43.0%) แต่คืนกรอบคลาดเคลื่อนเพิ่มจาก 27 เป็น 48 ภาพ

ตั้ง preset **“ทดลอง dataset v23” เป็นค่าเริ่มต้นตามที่ผู้ใช้เลือก** อัปโหลดภาพแล้วกด **“ค้นหาป้ายอัตโนมัติ”** จะใช้ blur 3, Canny 20/80, พื้นที่ขั้นต่ำ 150 และเกณฑ์สัดส่วน/มุมของ dataset v23 ทันที สามารถเลือก preset อื่นได้ อ่านวิธีแบ่งข้อมูล ค่าทั้งหมด และข้อจำกัดได้ใน [รายงานการทดลอง](reports/plate_tuning/REPORT.md)

## ทดสอบ

```powershell
python -m unittest discover -s tests -v
```

ครอบคลุมการเลือกป้ายที่มีตัวอักษรแทนกรอบว่างขนาดใหญ่ ภาพว่าง ป้ายเอียง การคืนพิกัดจาก ROI และการตรวจความถูกต้องของมุม

## รันบนเครื่องเอง (Local)

วิธีที่ง่ายที่สุดคือดับเบิลคลิก `run_local.bat` หรือรันใน PowerShell:

```powershell
.\run_local.ps1
```

สคริปต์จะสร้าง `.venv` อัตโนมัติ ติดตั้งไลบรารีจาก `requirements-local.txt` แล้วเปิดแอปที่ `http://localhost:8501`

หรือจะรันแบบ manual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-local.txt
streamlit run app.py
```

> **หมายเหตุ OpenCV**: `requirements-local.txt` ใช้ `opencv-python` (full build, มี GUI helper) แทน `opencv-python-headless` ที่ใช้บน Cloud ทั้งสองแพ็กเกจทำงานกับ Streamlit ได้เหมือนกัน แต่ติดตั้งพร้อมกันไม่ได้ — pip จะแทนที่ให้อัตโนมัติ

## Deploy บน Streamlit Community Cloud

1. Push โปรเจกต์นี้ขึ้น GitHub
2. เข้า [Streamlit Community Cloud](https://share.streamlit.io/) และเชื่อมบัญชี GitHub
3. เลือก repository, branch และไฟล์หลัก `app.py`
4. กด **Deploy**

Community Cloud จะติดตั้งไลบรารีจาก `requirements.txt` ให้อัตโนมัติ (ใช้ `opencv-python-headless` ที่เหมาะกับ server)

## ไฟล์ที่จำเป็น

```text
app.py                  Streamlit application
_environment.py         ตรวจสอบ runtime (local / Community Cloud)
plate_detection.py      candidate scoring and contour / ROI corner detection
tests/                  synthetic image regression tests
requirements.txt        dependencies สำหรับ Streamlit Community Cloud (headless OpenCV)
requirements-local.txt  dependencies สำหรับ local (full OpenCV)
run_local.ps1           PowerShell launcher — สร้าง venv และรันแอปอัตโนมัติ
run_local.bat           CMD wrapper สำหรับดับเบิลคลิกใน Windows Explorer
.streamlit/config.toml  Streamlit config สำหรับ local (ปิด telemetry, เปิด browser อัตโนมัติ)
```
