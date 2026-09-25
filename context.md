# CCTV SIFT method — context

เอกสารสรุปสถานะโปรเจกต์สำหรับคนที่กลับมาทำต่อ (รวมถึง Claude session ถัดไป) อ่านไฟล์นี้ก่อนแก้โค้ด
อัปเดตล่าสุด: 2026-09-24 · ภาษาที่ใช้คุยกับผู้ใช้: ไทย (ศัพท์เทคนิคใช้อังกฤษ)

## 1. โปรเจกต์นี้คืออะไร

ระบบตรวจจับและอ่านป้ายทะเบียนรถไทยจาก **ภาพเดี่ยว** (ภาพ CCTV ความชัดต่ำเป็นเป้าหมายหลัก)
หน้าเว็บ + FastAPI backend รันในเครื่อง ขั้นตอน: หาป้าย → เกาะขอบ 4 มุม → ดัด perspective → ปรับความเอียงจากตัวอักษร → (super-resolution ถ้าจำเป็น) → OCR

สิ่งที่ผู้ใช้ต้องการตั้งแต่ต้น: ใช้ **SIFT เป็นแกนหลัก** ("อยากใช้แค่ SIFT") ดังนั้นตัวหาป้ายใช้ SIFT descriptor (ไม่ใช่โมเดล detector ภายนอก) ผู้ใช้เคยบอกว่ามีโมเดลที่เทรนเองที่ตรวจจับ 4 มุม แต่ **ยังไม่ได้ส่งไฟล์มา** (ไม่มีไฟล์โมเดลในโฟลเดอร์)

## 2. รันยังไง

```bash
# จากโฟลเดอร์โปรเจกต์ (มี run_backend.bat ที่ทำ pip install แล้วรันให้)
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

- เปิดหน้าเว็บ: http://localhost:8000 (backend เสิร์ฟ `frontend/` เอง) · เช็กสถานะ: `/api/v1/health`
  → ต้องได้ `locator_loaded: true` และ `super_resolution_available: true`
- **ไม่มี auto-reload** แก้โค้ด backend แล้วต้องรีสตาร์ต (แก้ frontend แค่กด Ctrl+F5)
- ตอนทำงานกับผู้ใช้ Claude เป็นคนสตาร์ตเซิร์ฟเวอร์แบบซ่อนหน้าต่างด้วย PowerShell `Start-Process ... -WindowStyle Hidden` โดยเขียน log ที่ `server.log` / `server.log.err` ในโฟลเดอร์โปรเจกต์ (ลบทิ้งได้เมื่อปิดเซิร์ฟเวอร์) ถ้าพอร์ต 8000 ชน ให้ปิด process เก่าก่อน (`Get-NetTCPConnection -LocalPort 8000`)
- Dependencies: [backend/requirements.txt](backend/requirements.txt) (มี `scikit-learn`, `joblib` เพิ่มแล้ว; `torch` มากับ easyocr) Python 3.13 บน Windows 11, เครื่องมี GPU (CUDA) ใช้กับ SR อัตโนมัติ, EasyOCR รันบน CPU
- ไฟล์โมเดลที่ **ต้องมี** ใน `backend/app/models/`:
  - `plate_locator.joblib` (~150 MB) — เทรนด้วย `python train_locator.py` (ดูหัวข้อ 6)
  - `RealESRGAN_x4plus.pth` (67,040,989 ไบต์, จาก github.com/xinntao/Real-ESRGAN release v0.1.0) — ถ้าไม่มี ระบบข้าม super-resolution เอง
- **โปรเจกต์นี้ไม่ใช่ git repo** (มี `.gitignore` แต่ไม่มี `.git`) ไม่มีประวัติ/ที่กู้คืน ระวังเวลาลบไฟล์

## 3. โครงสร้างและ flow

```
frontend/  index.html (หน้าเดียวเต็มจอ 3 คอลัมน์) · css/style.css · js/{app,api_client,canvas_viewer}.js
backend/app/
  main.py                    FastAPI + CORS + เสิร์ฟ frontend
  api/routes.py              endpoints + ตัวประกอบ pipeline (read_corners, run_recognition, snap_corners)
  core/plate_locator.py      หาป้ายด้วย SIFT + ExtraTrees (ไม่ใช้ template)
  core/plate_refiner.py      เกาะขอบ 4 มุมของกรอบหยาบให้พอดีขอบป้าย
  core/plate_from_box.py     หาป้ายในสี่เหลี่ยมที่ผู้ใช้วาด (GrabCut → สี่เหลี่ยมคางหมู)
  core/deskew.py             เรียงมุม, perspective warp, straighten_text (หมุน+เฉือนจากตัวอักษร)
  core/preprocessor.py       enhance_for_ocr (เทา + ขยาย 2x + CLAHE, ไม่ binarize)
  core/ocr_engine.py         EasyOCR แยกบรรทัดเลขทะเบียน/จังหวัด + ตรวจรูปแบบ + จับคู่จังหวัด
  core/super_resolution.py   Real-ESRGAN x4 (สถาปัตยกรรมเขียนเอง โหลด weights แบบ weights_only)
  core/thai_provinces.py     77 จังหวัด + พยัญชนะที่ใช้บนป้าย (ตัด ฃ ฅ)
  core/sift_detector.py      (legacy ไม่ใช้แล้ว — แบบ template matching เดิม)
backend/train_locator.py     เทรน/ทำ cross-validation ตัวหาป้ายจาก LabelMe
backend/test_pipeline.py     (legacy ของเดิม ไม่ได้อัปเดต)
```

**Flow อัตโนมัติ** (`POST /api/v1/recognize`): `locator.locate` → `snap_corners` (`refine_quad`) → `run_recognition`:
1. `read_corners`: `arrange_corners` → `deskew_plate` → `straighten_text` → `enhance_for_ocr` → `ocr_engine.recognize_plate`
2. ถ้าอ่านไม่ตรงรูปแบบป้าย ลองหมุนมุมเริ่มต้นอีก 3 แบบ และ **ใช้ทิศใหม่เฉพาะเมื่อ** valid + confidence ≥ 0.4 และดีกว่าเดิม ≥ 0.15 (กันป้ายต่างประเทศถูกหมุนกลับหัว)
3. ถ้ายังไม่ "มั่นใจ" (valid + conf ≥ 0.9 + มีจังหวัด) ลองอ่านซ้ำจากสำเนา SR ของบริเวณป้าย ใช้เมื่อ quality ดีกว่า > 0.15 → response มี `super_resolution_used: true` และหน้าเว็บขึ้นข้อความเตือน

**Endpoints:** `POST /recognize` (อัตโนมัติ) · `POST /recognize-box` (ผู้ใช้วาดกรอบ `[x0,y0,x1,y1]`) · `POST /interactive-deskew` (ผู้ใช้ลาก 4 มุม + `snap` เลือกได้) · `GET /health` · ประวัติการตรวจจับ **ถูกเอาออกแล้ว** (ทั้ง UI และ API)
ถ้าหาป้ายไม่เจอ backend ตอบ `no_plate_detected` + `suggested_corners` (กล่องกลางภาพ) ให้ผู้ใช้ลาก/วาดเอง ไม่มี alert แล้ว

**UI:** หน้าเดียวไม่ต้องเลื่อนที่ ≥ 1000px กว้าง (ตรวจแล้วที่ 1440×900 และ 1280×720) ดีไซน์ตามภาพอ้างอิงที่ผู้ใช้ให้ (โทนเทาอมฟ้า การ์ดโค้ง แคปซูล) คอลัมน์: Source+Corners | Pipeline | Result
ในช่อง Corners: ลากจุดมุม 1→4 (ตามเข็มนาฬิกาเริ่มซ้ายบน) หรือ **ลากบนพื้นที่ว่างเพื่อวาดกรอบ** → เรียก `/recognize-box` ทันที · ช่องติ๊ก "Snap to plate edges" มีผลตอนกด Apply

## 4. ข้อมูล

- `data/raw_images/`: ภาพ 441 ภาพ (`Car433–437` = ป้ายไทย 5 ภาพ, `Cars*` = ป้ายหลายประเทศ) + ไฟล์ LabelMe `.json` ~44 ไฟล์ (label `license_plate`, polygon 4 จุด; 2 ไฟล์ 5 จุด) ตัวโหลดใช้ได้ 42 ภาพ ตอนเทรนโมเดลปัจจุบัน มี 39 ภาพ (**เทรนใหม่ได้เลยเพื่อรวมภาพที่เพิ่ม**)
- **ลำดับมุมในเฉลยไม่คงที่** (เริ่มมุมไหนก็ได้ เช่น Car436/437 ป้ายตะแคง, Cars28) อย่าตีความว่า มุมแรก = ซ้ายบน
- **ไม่มีเฉลยตัวอักษร** (ข้อความทะเบียน) ในชุดข้อมูลเลย ป้ายไทยที่มีมีแค่ 5 ใบ
- ผู้ใช้มี **X-AnyLabeling** (`X-AnyLabeling/`, `X-AnyLabeling-CUDA13.exe`) ไว้ label เพิ่ม

## 5. ผลที่วัดได้ (ตัวเลขและเหตุผลของการตัดสินใจ)

หาป้ายอัตโนมัติ (cross-validation 5 fold, นับถูกเมื่อ IoU > 0.5):
| วิธี | ถูก |
|---|---|
| SIFT จับคู่ template เดียว / คลัง template จากป้ายจริง | ~0% / 5% |
| contour 4 มุม / morphology (ไม่ใช้ SIFT) | 23% / 10% |
| **SIFT descriptor + ExtraTrees (ใช้อยู่)** | **36%** (IoU>0.3: 51%) |

- ขนาดภาพก่อนหา SIFT: ด้านยาว 480px ดีที่สุด (320: 26%, 640: 26%, 960: 21%) → **ขยายภาพก่อนเข้า SIFT ไม่ช่วย**
- เส้นโค้งการเรียนรู้: สอน 8 → 31 ภาพ ได้ 18% → 33% ยังขึ้นอยู่ แต่ผันผวน; คาดว่าต้อง label 100–300 ภาพ (ประมาณของผม ไม่ใช่ค่าที่วัด) หรือใช้ detector จริงถึงจะ > 90%
- ใช้ SIFT เชิงลึกเพิ่มไม่ได้เพราะ: ป้ายเรียบ ตัวอักษรต่างกันทุกป้าย, ภาพเดี่ยว (ไม่มีหลายเฟรมให้จัดเรียง)

ปรับมุม/ความตรง (เทียบเฉลยมุม):
- กรอบจากตัวตรวจจับคลาดจากป้ายจริงค่ากลาง ~13% ของความกว้างป้าย → หลัง `refine_quad` ~10% (ชุด held-out 12.9→9.6%); ขอบแนวนอนเอียง 3.0°→1.1°, แนวตั้ง 5.7°→3.8°
- `refine_quad` ทำให้กรอบ "แน่นเกิน" (ตัดตัวอักษรขาด) จึงเผื่อ `OUTPUT_MARGIN = 1.1` **ห้ามเอาออก**; ใช้กับผลตัวตรวจจับเสมอ ส่วนตอนลากมุมเองให้ผู้ใช้เลือก (ลากแม่นอยู่แล้ว การเกาะขอบบางภาพแย่ลง)
- `refine_quad` เกาะจากกรอบที่หลวมมากไม่ได้ (เผื่อ 15%+ ก็แทบไม่ดีขึ้น) → เป็นเหตุผลของโหมดวาดกรอบ
- `straighten_text`: ประมาณมุมหมุนคลาดค่ากลาง 0.16°, มุมเฉือน 0.8° (บนป้ายสังเคราะห์ที่ใส่มุมรู้ค่า)

โหมดวาดกรอบ (`plate_from_box`): จำลองกรอบหลวม 15–35% → คลาดมุมจาก ~33% เหลือ ~9%, ขนาดจาก 2.75× เหลือ ~1.2× GrabCut เริ่มจาก "กลางกรอบน่าจะเป็นป้าย" (ไม่งั้นจับตัวรถ), ถ้าล้มเหลวใช้แบบ rect-init สำรอง, hull → 4 จุด (คางหมู, ป้ายเอียงต้องใช้ ไม่ใช่สี่เหลี่ยมหมุน) แล้วขยาย 14%
ป้ายไทยจริง (กรอบหลวม 8 แบบต่อป้าย): Car434 อ่านถูก 8/8, Car435 (เอียง) 7/8 (แบบสี่เหลี่ยมหมุน 0/8), Car433 (มืดบนกระจังหน้าสีเข้ม) หาเจอ 4/8
**ทดสอบแล้วไม่ดีกว่า:** contour score (ผู้สมัครกรอบจากเส้นขอบ + ให้คะแนน) ได้ความคลาด ~7% เท่ากัน แต่ตัดป้ายขาดมากกว่า; เพดาน (oracle) ของผู้สมัครเองแค่ 5–6% → **ไม่ได้ใช้** เฉลยที่วาดมือมีความคลาดในตัว ~2–4% ความต่างเล็กกว่านั้นแยกไม่ออก

OCR (วัดบนป้ายไทยสังเคราะห์ 100 ใบ ที่ทำให้เบลอ/เล็กแบบ CCTV; ไม่มีเฉลยจริง):
- ก่อนจูน (Otsu + EasyOCR รวมทั้งป้าย): เลขถูก ~74%, จังหวัด 5–15%, ถูกทั้งป้าย ~2%
- หลังจูน: เลข 84% (ป้ายชัด 97%), เลขทะเบียนถูกทั้งหมด 32%, จังหวัดถูก ~50% (แสดงเฉพาะ score ≥ 0.6 → ที่แสดงถูก 100%), ตรงรูปแบบ 72%
- ตัวชี้ขาด: ไม่ binarize · แยกบรรทัดบน/ล่างตามช่องว่างแนวตั้ง · อ่านแยก "ตัวอักษรนำหน้า (ไม่มี 0)" กับ "ตัวเลข (0–9 ล้วน)" · allowlist ไม่มีละติน/ฃ/ฅ · จับคู่ 77 จังหวัดด้วย difflib
- ที่ยังผิด: พยัญชนะหน้าตาคล้ายกัน (ษ/บ, ฆ/ข, ฎ/ภ) เป็นขีดจำกัดของโมเดลไทยใน EasyOCR การปรับค่าไม่ช่วย ต้อง fine-tune ตัวอ่านด้วยข้อมูลจริง (ต้องมีเฉลยตัวอักษร)
- ขยาย 3× แย่กว่า 2×; CLAHE/unsharp/beam search อยู่ในระดับสัญญาณรบกวน

Super-resolution (Real-ESRGAN x4plus): ป้ายสังเคราะห์ อ่านทะเบียนถูก 32% → 49% แต่ป้ายเสียหนัก "ดูถูกรูปแบบแต่ผิด" เพิ่ม (41% → 62%) เพราะโมเดล **แต่งตัวอักษร** ป้ายจริง: Car433 ได้จังหวัดเพชรบุรีถูก, ป้าย BMW (ตัวอักษรสูง ~12px, ภาพ 8.png) ไม่ช่วย (แต่ง glyph ขึ้นมาเอง) จึงใช้แบบมีตัวกัน (ข้อ 3 ใน flow) เสมอ และห้ามเชื่อผลจากป้ายเสียหนักโดยไม่ตรวจ

## 6. คำสั่งที่ใช้บ่อย

```bash
cd backend
python train_locator.py            # cross-validate แล้วเทรนใหม่จาก ../data/raw_images → app/models/plate_locator.joblib
python train_locator.py --no-cv    # ข้าม cross-validation
```
- พารามิเตอร์หลักของตัวหาป้าย: `plate_locator.py` (`TARGET_LONG_SIDE=480`, `sigma_frac .04`, `keep .5`, `radius_frac .12`, `grow 1.3`) จูนด้วยชุดข้อมูลเดียวกัน (มีโอกาส overfit เล็กน้อย)
- ค่าปรับที่อื่น: `plate_refiner.py` (WINDOW .15, LAMBDA_*, OUTPUT_MARGIN 1.1) · `ocr_engine.py` (`MIN_PROVINCE_SCORE=0.6`, `ROW_GAP_RATIO`) · `routes.py` (`MIN_ROTATED_CONFIDENCE`, `SR_*`) · `plate_from_box.py` (`GROW 1.14`)

## 7. ข้อควรระวัง (gotchas)

- ลำดับมุมที่ `deskew_plate` ใช้ต้องเป็น **ซ้ายบน→ขวาบน→ขวาล่าง→ซ้ายล่าง ตามเข็มนาฬิกา** `arrange_corners` เชื่อลำดับที่ให้มาถ้าเป็นสี่เหลี่ยมนูนตามเข็ม ไม่งั้นเรียงใหม่ด้วยมุมรอบจุดกลาง (สูตร sum/diff เดิมพังกับป้ายเอียงมาก — เคยทำให้ภาพดำ)
- บั๊กเดิมที่แก้แล้ว (กันไม่ให้กลับมา): `enhance_for_ocr` เรียก `denoise(h=..)` ผิด + รับค่า Otsu ผิดลำดับ (OCR ล้มเงียบ) · `OCREngine.recognize_plate` กลืน exception (ตอนนี้ route ส่ง error จริงออกมา) · `/interactive-deskew` ประกาศ body ผิดแบบ (422) · `@router.on_event("startup")` ใช้ไม่ได้กับ FastAPI/Starlette ใหม่ · canvas ถูกลบตอนอัปโหลดภาพที่สอง · พิกัดเมาส์ไม่ชดเชยการย่อ canvas
- เอกสารเก่าในโฟลเดอร์ (`IMPLEMENTATION_*.md`, `PROJECT_SUMMARY.md`, `DEVELOPMENT.md`, `QUICKSTART.md`, `RUN_ME.md`, `TROUBLESHOOTING.md`, `START.*`) เขียนก่อนออกแบบใหม่ ยังอ้าง flow แบบ template SIFT **อย่าเชื่อ** ใช้ไฟล์นี้กับ README เป็นหลัก (README แก้แค่เรื่องประวัติออกแล้ว ส่วนอื่นอาจยังเก่า)
- Windows: รันสคริปต์ที่พิมพ์ภาษาไทยให้ตั้ง `PYTHONIOENCODING=utf-8` (ข้อมูล JSON ปกติ แค่ console แสดงเพี้ยน) · มี opencv หลายแพ็กเกจติดตั้งซ้อนกัน (5.0.0.93) `cv2.ml.TrainData_create` ไม่มี ใช้ sklearn แทน
- เบราว์เซอร์ในแอป Claude เข้า `localhost:8000` ไม่ได้ (ถูกปฏิเสธ) ตอนทดสอบ UI ใช้พอร์ตอื่น (`python -m http.server` ที่ `frontend/` + CORS server ของภาพตัวอย่าง) หรือ Edge headless: `msedge --headless=new --screenshot=... --window-size=1440,900 <url>`
- ตอนวัดผลให้แยกชุดสอน/ทดสอบเสมอ (มีเฉลยแค่ ~42 ภาพ ตัวเลขผันผวน)

## 8. ของที่เคยทดลองแต่ **ไม่อยู่ในโปรเจกต์** (อยู่ในโฟลเดอร์ scratchpad ชั่วคราวของ session ซึ่งอาจถูกลบ)

ตัวสร้างป้ายไทยสังเคราะห์ + ชุดวัด OCR (`synth.py`, `ocr_eval.py`, `sr_eval.py`), ตัววัด deskew/refine/box (`deskew_eval.py`, `refine*.py`, `box_eval.py`, `gc3.py`, `contour_score.py`), ตัวรัน Edge headless ถ้าต้องวัดซ้ำและหาไฟล์ไม่เจอ ต้องเขียนใหม่ (หลักการอยู่ในหัวข้อ 5) หรือขอให้ผู้ช่วยคัดลอกไว้ที่ `backend/tools/`

## 9. ทางที่ยังไม่ได้ทำ (เรียงตามผลที่คาดว่าจะได้)

1. **label ภาพเพิ่ม** (ภาพรออยู่ ~400 ภาพ) แล้วเทรนใหม่ — ทางที่ผลชัดที่สุดสำหรับตัวหาป้าย; เสนอไว้: ปุ่ม "บันทึกมุมเป็น LabelMe" ในโหมดวาดกรอบ (ได้ 4 มุมที่แม่นระดับหนึ่งแล้วแก้เล็กน้อย) หรือใช้ X-AnyLabeling
2. **เปลี่ยนตัวหาป้ายเป็น detector มาตรฐาน** (YOLO-pose ฯลฯ) หรือเสียบโมเดล 4 มุมที่ผู้ใช้เทรนเอง (ต้องได้ไฟล์ก่อน) ถ้าเสียบ ให้แทน `SIFTPlateLocator.locate` โดยคืนมุมรูปแบบเดิม `[[x,y]×4]`
3. **เทรนตัวปรับมุมที่เรียนรู้จากข้อมูล** (ทำนายส่วนต่างระหว่างมุมจาก GrabCut กับเฉลย) — ทุกวิธีคลาสสิกติดเพดาน ~5–7%
4. **fine-tune ตัวอ่านตัวอักษรไทย** ด้วยป้ายจริง + สังเคราะห์ ต้องการเฉลยตัวอักษรอย่างน้อยร้อยกว่าใบ (ไฟล์ CSV `ชื่อภาพ, ทะเบียน, จังหวัด` ก็พอ)
5. ถ้ามีวิดีโอ/หลายเฟรม: จัดเรียงเฟรมด้วย SIFT/ECC แล้วซ้อนเพื่อเพิ่มความคมก่อน OCR (ผู้ใช้บอกว่าตอนนี้เป็นภาพเดี่ยว)
6. เก็บกวาด: ลบ `sift_detector.py`/`test_pipeline.py`/เอกสารเก่า (ยังไม่ได้ลบ เพราะไม่มี git กู้คืนไม่ได้ — ให้ถามผู้ใช้ก่อน)

## 10. Deploy และ Streamlit (อัปเดต 2026-09-25)

**Branch ที่เกี่ยวข้อง:** `phet` (ขึ้น GitHub แล้ว; `main` บน GitHub รับรุ่นแรกไปแล้วผ่าน PR #7 ส่วน Streamlit ตามไปทาง PR ใหม่จาก `phet`) · `hf` (เตรียม Docker/Hugging Face: `Dockerfile`, `DEPLOY_HF.md`, ใช้ Git LFS) · `streamlit` (แอป Streamlit + `pipeline.py`) · `streamlit-clean` (สแนปช็อตของ `streamlit` 1 commit ไม่มีประวัติ ไว้ push ขึ้น repo ใหม่ ประวัติของ branch อื่นมี dataset 205 MB ปนอยู่)

**เหตุผลที่เลือก Streamlit Cloud:** Hugging Face Spaces แบบ Docker/Gradio ต้องแผนเสียเงิน (PRO $9/เดือน) บัญชีฟรีสร้างได้แค่ Static (ไม่มี backend) · Azure for Students ($100 เครดิต ไม่ต้องใช้บัตร ต้องอีเมลมหาวิทยาลัย) เป็นอีกทางถ้า Streamlit ไม่พอ · Streamlit Community Cloud: RAM สูงสุด 2.7 GB, CPU 2 คอร์, หลับหลังไม่มีคนเข้า 12 ชม. (ตัวเลขนี้ผมเคยบอกผิดว่า ~1 GB)

**โครงสร้างโค้ดหลังแยก:** ตรรกะอ่านป้ายทั้งหมดอยู่ที่ [backend/app/pipeline.py](backend/app/pipeline.py) (`PlateReader`: `recognize`, `recognize_box`, `recognize_corners`, มี lock ประมวลผลทีละคำขอ) · FastAPI ([routes.py](backend/app/api/routes.py)) กับ Streamlit ([streamlit_app.py](streamlit_app.py)) เป็นแค่ตัวห่อ · ผลลัพธ์ของ pipeline เป็น numpy (`result["images"]`) routes แปลงเป็น base64 เอง

**ตัวหาป้ายลดขนาด:** `train_locator.py --trees 100` (ค่าเริ่มต้น) ไฟล์ 84 MB (จาก 143 MB) ความแม่น CV 32% เทียบ 34% (ในระดับสัญญาณรบกวน, 44 ภาพ) ไฟล์ < 100 MB จึงขึ้น GitHub ตรงๆ ได้ (ไม่ต้อง LFS ซึ่ง Streamlit Cloud รองรับไม่ดี)

**RAM (CPU ล้วน, Windows):** โหลดโมเดล ~1.4 GB → ใช้งานต่อเนื่อง ~2.2 GB คงที่ (ไม่ leak) → รวม Streamlit peak ~2.35 GB (เพดาน 2.7 GB) `PLATE_DISABLE_SR=1` ปิด Super-resolution ประหยัด ~0.5 GB

**Streamlit app:** อัปโหลด/ภาพตัวอย่าง (`samples/`) → หาป้ายอัตโนมัติ → วาดกรอบด้วย `streamlit-image-coordinates` (`click_and_drag=True`, `width="stretch"`; พิกัดที่คืนมาต้องชดเชยขนาดที่แสดง) หรือพิมพ์ 4 มุม ไม่มีการลากมุมแบบหน้าเว็บเดิม · `streamlit-drawable-canvas` ตัวดั้งเดิมพังกับ Streamlit รุ่นใหม่ จึงไม่ใช้ · ทดสอบแล้ว: Car433 → `กด 3789` เพชรบุรี, Car435 วาดกรอบ → `กฎ 4644`
- ต้องรัน Streamlit ด้วย `--server.address=127.0.0.1` ตอนทดสอบ ไม่งั้นเปิดให้ทั้งเครือข่ายเข้าได้

**Docker:** image ของ Space (`Dockerfile`) build สำเร็จและรันผ่านจริง (health, อ่านป้าย, RAM 1.2 GB) · เหตุการณ์ 2026-09-25: ไดรฟ์ C: เต็ม 100% จากพื้นที่ Docker 15 GB ทำให้ Docker Desktop พัง (`input/output error`) ผู้ใช้ล้างพื้นที่แล้ว แต่ engine ยังตอบ 500 ต้องรีสตาร์ต/Purge เอง · **ก่อน build image ใหญ่ให้เช็กพื้นที่ว่างก่อน**

**ยังไม่ได้ทำ/ยังไม่ยืนยัน:** deploy จริงบน Streamlit Cloud (ยังไม่ทราบว่าตัวติดตั้งจะเลือก torch แบบ CPU ตาม `--extra-index-url` หรือไม่ ดูวิธีแก้ใน [DEPLOY_STREAMLIT.md](DEPLOY_STREAMLIT.md)) · ทดสอบใน Docker ที่จำกัด RAM 2.7 GB (Docker เสีย)
