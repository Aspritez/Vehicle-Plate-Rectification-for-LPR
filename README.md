# PlateLab — Vehicle Plate Rectification for LPR

เว็บแอป Streamlit สำหรับปรับมุมป้ายทะเบียนและเตรียมภาพสำหรับ OCR ในหัวข้อ CP461
ไม่มีโมเดล YOLO และไม่มีขั้น train; ใช้ชุดข้อมูลรูปแบบ YOLOv8 ประเมินและปรับค่า detector โดยไม่รวมภาพใน repository

## สถานะและขอบเขต

- รับ JPG/PNG ครั้งละหนึ่งป้าย สูงสุด 10 MB และ 24 ล้านพิกเซล
- **ภาพเดียว:** กด **ค้นหาและปรับป้าย** → หาขอบและกลุ่มตัวอักษร → ตรวจมุมจากภาพต้นฉบับ → Homography → ผลลัพธ์ทันที หากหาไม่ได้/เลือกผิดให้ระบุมุมบนภาพเต็ม
- **มี reference:** ป้ายเอียงและภาพด้านหน้าของป้ายเดียวกัน → SIFT/ORB → KNN → Lowe’s ratio test → RANSAC → Homography
- แสดงจุด matching, inliers, reprojection error, ก่อน–หลัง และส่งออก PNG/JSON
- เตรียม OCR ด้วยภาพสี/Grayscale/Adaptive threshold พร้อมเปิด–ปิด CLAHE และ denoise
- ยังไม่มี OCR อ่านตัวอักษร วิดีโอ ระบบ train หรือฐานข้อมูลป้าย
- ค่าเริ่มต้น Auto ปรับจากป้ายจริง 36 ภาพแล้ว ผลนี้เป็นชุดปรับค่า ยังไม่ยืนยันกับภาพใหม่ ส่วน matching และ OCR ยังไม่ประเมินด้วยชุดนี้
- Source พร้อม deploy แต่การมี repository นี้ไม่ได้หมายความว่าเผยแพร่ public URL แล้ว

## ติดตั้งและรัน

แนะนำ Python **3.13** (ใช้ Python 3.13.2 ในการพัฒนา) ติดตั้งจากโฟลเดอร์รากของโปรเจกต์
ใช้ virtual environment ใหม่เพื่อไม่ให้ OpenCV หลายแพ็กเกจปะปนกัน

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

macOS / Linux:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

เปิด `http://localhost:8501` แล้วอัปโหลดภาพของคุณ ไม่ต้องดาวน์โหลดโมเดลหรือใส่ API key

## วิธีใช้: ภาพเดียว

1. เลือก **ภาพเดียว · Auto / เลือกมุม** และอัปโหลดภาพ
2. กด **ค้นหาและปรับป้าย** ระบบหา candidate ตรวจมุม และแสดงภาพ deskew ต่อในครั้งเดียว
3. หากไม่มีตัวใดผ่าน หรือมีหลายบริเวณคะแนนใกล้กัน ระบบแจ้งเหตุผลและเปิดการเลือกมุมบนภาพเต็ม
4. หากผลอัตโนมัติเลือกผิด ให้กด **เลือกผิดป้าย / แก้มุมด้วยตนเอง** ใต้ผลลัพธ์
5. คลิก **เลือกใหม่** เพื่อล้างมุมเดิมแล้วเลือกป้ายจริง หรือ **ย้อนหนึ่งจุด** เพื่อแก้ทีละจุด
6. คลิกมุมป้าย 4 จุดบนภาพเต็ม หรือกรอก `x, y` หนึ่งจุดต่อบรรทัด (พิกัดภาพเต็มหลังปรับ EXIF)
7. กด **ปรับภาพจากมุมที่เลือก** เพื่อแสดงผลใหม่ วิธีที่ใช้จะเปลี่ยนเป็น manual อย่างชัดเจน
8. ตั้งความกว้าง/สัดส่วนใน **สัดส่วนผลลัพธ์และค่าค้นหาขั้นสูง** ถ้าภาพยืดหรือบีบ แล้วประมวลผลใหม่
9. เลือกการเตรียม OCR และดาวน์โหลดภาพหรือรายละเอียดการแปลง

เข้าโหมด manual ได้ทันทีด้วย **ระบุมุมเองบนภาพเต็ม** โดยไม่ต้องลอง auto ก่อน
การเปลี่ยนภาพ/มุม/สัดส่วนล้างผลเก่า การค้นหาใหม่ที่ล้มเหลวไม่แสดงภาพสำเร็จจากครั้งก่อน
ค่าอัตราส่วน 2.0 ไม่ใช่มาตรฐานป้ายทุกประเภท ถ้าด้านกว้าง/สูงสลับให้เลือก **สลับแนวมุม 90°**

โหมดนี้คำนวณ `getPerspectiveTransform` จากมุม 4 จุด ไม่มี descriptor matching และไม่กล่าวอ้างว่าใช้ RANSAC
การประมาณมุมจาก contour เป็น heuristic อาจต้องแก้มุมด้วยมือ

### Auto detector ทำอะไรบ้าง

หน้าเว็บใช้ค่าล่าสุดจาก `web_detector_config()` ใน `src/detection.py` ตรงกับ `evaluation/tuned_config.json`:
ตรวจโครงสร้างข้อความก่อนจัดอันดับ candidate, เพิ่มการหาพื้นหลังสีเป็นกลาง, ตรวจละเอียด 24 บริเวณจาก pool สูงสุด 250,
ยอมรับ apparent aspect ขั้นต่ำ 0.25 และทดลองตรวจข้อความบนภาพสัดส่วน 2:1
คะแนน appearance มีน้ำหนัก 25%, contrast ขั้นต่ำ 55, neutral fraction ขั้นต่ำ 0.25,
ความสูงกลุ่มอักษรหลักขั้นต่ำ 0.22 และคะแนนรวมขั้นต่ำ **0.90**; Canny ยังคง 40/140
ผู้ใช้ยังปรับ Canny และสัดส่วน output ได้ การเปลี่ยนค่าจะล้างผลเก่า

ผลล่าสุดในชุดปรับค่า 36 ภาพ: รับผล 17 ภาพและผ่านเกณฑ์ตำแหน่ง IoU ≥ 0.5 ทั้ง 17 ภาพ อีก 19 ภาพให้เลือกมุมเอง
เป็นผลการปรับค่าภายในชุดข้อมูล ไม่ใช่การรับรอง deskew/OCR หรือความแม่นยำบนข้อมูลใหม่ ยังไม่ได้ประเมินค่าล่าสุดบนชุดแยก 16 ภาพ

ขั้นตอนด้านล่างอธิบายฐานอัลกอริทึมและค่าพื้นฐาน `DetectorConfig()`; หน้าเว็บแทนค่าด้วยโปรไฟล์ข้างต้น:

1. สอง pass ที่จำกัดขนาด: ภาพ grayscale สูงสุด 1000 px และภาพปรับ CLAHE สูงสุด 1400 px
2. เส้นทางขอบใช้ Canny, Closing, Adaptive threshold; เส้นทางข้อความใช้ Black-hat สองขนาด kernel ตามความกว้างภาพแล้วเชื่อมกลุ่ม
3. แต่ละ mask พิจารณา contour ไม่เกิน 600 รายการ รวม candidate ที่ซ้ำและตรวจละเอียดไม่เกิน 12 บริเวณ บริเวณละไม่เกิน 16 รูปสี่เหลี่ยม
4. กล่องกลุ่มข้อความใช้เป็นพื้นที่ค้นหาเท่านั้น **ไม่ใช้มุมกล่องข้อความเป็นมุมป้าย**
5. ครอปบริเวณพร้อมขอบเผื่อจากต้นฉบับ แล้วหา contour/เส้นขอบซ้ำ (จำกัด ROI ที่ใหญ่เกิน 2000 px ต่อด้าน)
6. ใช้ polygon approximation และ robust line fitting จากพิกเซลขอบใกล้เส้นเดิม ตรวจขอบจริงทั้งสี่ด้าน; `minAreaRect` ใช้คิดคะแนนเท่านั้น
7. ทดลอง warp ขนาดเล็กเพื่อตรวจ connected components ที่มีขนาดและแนวคล้ายข้อความหนึ่งหรือสองแถว ทั้งสอง polarity ไม่อ่านตัวอักษรหรือบังคับจำนวนตัวอักษรตายตัว
8. รวมคะแนน shape 22%, text 40%, edge support 28%, agreement 10% (agreement ไม่ใช่หลักฐานอิสระ)
9. เกณฑ์เริ่มต้น: ชิ้นข้อความหลักอย่างน้อย 4, text score ≥ 0.46, mean edge support ≥ 0.55, weakest edge ≥ 0.30, score ≥ 0.64
10. รวมผลที่หามุมแล้วซ้ำกัน หากอันดับหนึ่งและสองที่ผ่านมีคะแนนห่างน้อยกว่า 0.045 จะไม่เลือกเอง
11. นำมุมกลับไปยังภาพเต็มแล้ว warp จากต้นฉบับตามสัดส่วนที่ผู้ใช้เลือก

ดู candidate ทุกอันดับ เหตุผลที่ปฏิเสธ และภาพ Canny/Black-hat ได้ใน **รายละเอียดการค้นหาอัตโนมัติ**
ข้อความบนป้ายที่ขาด/ติดกัน ป้ายไม่มีกรอบ แสงสะท้อน และลวดลายที่คล้ายข้อความยังเป็นข้อจำกัด
ผลจากชุดปรับค่ามีขอบเขตจำกัด ต้องประเมินด้วยภาพใหม่ก่อนสรุปความแม่นยำทั่วไป

## วิธีใช้: SIFT / ORB กับ reference

1. เลือก **มี reference · SIFT / ORB**
2. อัปโหลดภาพเอียงและ reference ที่เป็นภาพด้านหน้าของ **ป้ายเดียวกัน** ไม่ใช่ป้ายเปล่าหรือป้ายคนละเลข
3. เลือกบริเวณป้ายทั้งสองภาพให้แน่นพอ เหลือรายละเอียดทั่วป้ายและไม่ตัดขอบออก
4. เลือก SIFT หรือ ORB แล้วกด **ปรับป้ายให้ตรง**
5. ตรวจภาพและข้อมูลการจับคู่ ผลลัพธ์จัดแนวตาม reference; ถ้า reference เอียง ผลก็ไม่ได้กลายเป็นมุมด้านหน้าเอง
6. หากจับคู่ไม่ได้ แก้บริเวณ เปลี่ยน reference หรือสลับไปโหมดมุม 4 จุดด้วยตนเอง

SIFT ใช้ L2 distance, ORB ใช้ Hamming distance, KNN ใช้ k=2 และ Lowe’s ratio test
ระบบคัดคู่ที่ซ้ำ reference keypoint ก่อน RANSAC ต้องมีอย่างน้อย 8 คู่ตามเกณฑ์ของแอป
ภาพสำหรับ matching ย่อไม่เกิน 1200 px แล้วแปลง Homography กลับไปใช้ภาพต้นฉบับก่อน warp
ผลลัพธ์ reference ย่อไม่เกิน 1600 px ต่อด้าน

### เกณฑ์ตรวจคุณภาพเริ่มต้น

- inliers อย่างน้อย 8 (ปรับได้ใน UI), สัดส่วนอย่างน้อย 0.45
- convex hull ของ inliers ครอบคลุมอย่างน้อย 3% ในทั้งสองบริเวณภาพ
- median reprojection error ไม่เกิน RANSAC threshold
- ปฏิเสธรูปทรงกลับด้าน/ไม่นูน การแปลงเสื่อม พื้นที่นอกภาพมาก และภาพปลายทางที่มีข้อมูลจริงน้อยกว่า 90%
- Threshold และ error วัดในพิกเซลของ reference ที่ย่อเพื่อ matching ไม่ใช่พิกเซลภาพต้นฉบับ

เกณฑ์เหล่านี้ไม่รับประกันว่าจับคู่ป้ายถูกต้อง เป็นมาตรการกรองที่ต้องประเมินกับข้อมูลจริง
inliers สูงไม่เท่ากับความแม่นยำ OCR และการป้อนภาพเดียวกันทั้งสองช่องไม่พิสูจน์การแก้ perspective

## Flow

```text
Upload → ตรวจไฟล์/EXIF
 ├─ Auto: edges + text groups → รวม candidate → ตรวจข้อความ/ขอบ → หามุมละเอียด → Homography
 │    └─ ไม่ผ่าน/เลือกผิด: ระบุมุมบนภาพเต็ม → Homography
 ├─ Manual: ระบุมุมบนภาพเต็มทันที → Homography
 └─ มี reference: SIFT/ORB → KNN → ratio test → RANSAC → Homography
       ↓
ตรวจคุณภาพ → Warp จากภาพต้นฉบับ → เตรียม OCR → แสดง/ดาวน์โหลด
```

JSON ระบุเมทริกซ์ `homography_source_to_output` จาก **พิกัดภาพเต็มหลังแก้ EXIF** ไปพิกัด output
รวม source ROI, reference ROI, มุม, ขนาดผลลัพธ์, settings และ metrics โดยไม่บรรจุภาพอัปโหลด

## การทดสอบโค้ด (ไม่ใช่การประเมิน dataset)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

ทดสอบไฟล์เสีย/EXIF, พิกัดภาพย่อ, มุมผิดรูป, Homography ที่รู้คำตอบ, RANSAC กับ outliers,
SIFT/ORB บนรูปทรงเรขาคณิตที่สร้างชั่วคราวในหน่วยความจำ, ภาพว่าง, OCR preparation และ Streamlit UI
ไม่มีการสร้างชุดภาพป้ายตัวอย่างหรือเก็บภาพ fixture ไว้ในโปรเจกต์
การทดสอบเหล่านี้ยืนยันการทำงานของโค้ด ไม่ใช่อัตราตรวจจับป้ายหรือความแม่นยำ LPR

เมื่อมีข้อมูลจริง ให้แยกชุดปรับค่ากับชุดทดสอบ บันทึกมุมอ้างอิง และวัดอัตราสำเร็จ/ปฏิเสธผิด
ความคลาดเคลื่อนมุมและเวลา โดยแยกผลอัตโนมัติกับผลที่แก้มุมด้วยมือ

## Deploy บน Streamlit Community Cloud

1. สร้าง GitHub repository แล้วนำ source ขึ้นไป โดยไม่รวม `.venv`, `tmp`, ไฟล์ส่วนตัวหรือภาพอัปโหลด
2. ที่ `https://share.streamlit.io` เชื่อม GitHub แล้วเลือก **Create app**
3. เลือก repository, branch และ entrypoint `app.py`
4. ใน Advanced settings เลือก Python **3.13** ให้ตรงกับที่ทดสอบ
5. Deploy แล้วตรวจ build logs ว่าติดตั้งจาก `requirements.txt` สำเร็จ
6. ตั้งให้ผู้ชมทั่วไปเข้าถึง แล้วเปิด public URL ในหน้าต่างที่ไม่ได้เข้าสู่ระบบ
7. ตรวจอัปโหลด เลือกมุม ทั้งสองโหมด กรณีไม่สำเร็จ และดาวน์โหลดบน deployment จริง

ใช้ `opencv-python-headless` เท่านั้น ไม่ติดตั้ง `opencv-python`/`opencv-contrib-python` เพิ่มพร้อมกัน
ไม่มี `cv2.imshow`, camera capture, GUI desktop หรือ system package ที่จำเป็นใน pipeline นี้
ส่วนเลือกมุมใช้ `streamlit-image-coordinates`; ถ้าคอมโพเนนต์โหลดไม่ได้ ยังระบุพิกัดเป็นตัวเลขได้

เอกสารทางการ:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization
- https://docs.streamlit.io/knowledge-base/dependencies/libgl
- https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html

## โครงสร้าง

```text
app.py                    UI, session state, uploads, exports
src/imaging.py            decode, resize, OCR preparation
src/detection.py          multi-pass proposals, text evidence, border refinement, auto rejection/warp
src/geometry.py           corner validation, projection, four-point warp
src/matching.py           SIFT / ORB, KNN, RANSAC, quality checks
tests/                    procedural checks and UI smoke tests
.streamlit/config.toml    theme, upload size, server settings
requirements.txt          pinned runtime dependencies
requirements-dev.txt      test dependencies
```

## การจัดการข้อมูลและข้อจำกัด

ภาพอัปโหลดถูกส่งไปยังเซิร์ฟเวอร์และเก็บในหน่วยความจำ session; แอปไม่เขียนภาพลงดิสก์
กด **ล้างงานทั้งหมด** เพื่อเริ่มงานใหม่ ไม่มี global cache สำหรับภาพของผู้ใช้
ภาพเล็ก เบลอ สะท้อน ป้ายโค้ง หรือไม่มีขอบชัดอาจล้มเหลว การ warp ไม่สร้างรายละเอียดตัวอักษรที่หายไป
ระบบตรวจรูปทรงไม่ใช่โมเดลที่ยืนยันว่าบริเวณนั้นเป็นป้าย ต้องตรวจผลด้วยสายตา

## สิ่งที่ต้องเตรียมเพิ่มก่อนส่งงาน CP461

- ชุดภาพทดสอบจริงและ reference ของป้ายเดียวกัน พร้อมผลประเมิน
- Public deployment URL ที่เข้าถึงได้
- วิดีโอ **10 นาที** มีเสียงอธิบายและ live edge-case demo พร้อมสมาชิกครบ 5 คน
- Repository สะอาด, requirements และคำสั่งรันที่ทดสอบแล้ว

โหมด feature-based ครอบคลุมวิธีหลักของโจทย์; โหมดมุม 4 จุดเป็นอีกวิธีและไม่ใช้แทนข้อกำหนด feature matching
