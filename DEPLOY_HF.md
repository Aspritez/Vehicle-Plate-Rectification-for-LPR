# Deploy บน Hugging Face Spaces

branch `hf` เตรียมไว้ให้ push ขึ้น Space แบบ Docker แล้ว (ไม่ต้องเขียนหน้าเว็บใหม่: FastAPI เสิร์ฟหน้าเว็บให้เอง)

## สิ่งที่อยู่ใน branch นี้เพิ่มจาก `phet`
| ไฟล์ | หน้าที่ |
|---|---|
| `Dockerfile` | python 3.13 + torch แบบ CPU + แพ็กเกจที่ล็อกเวอร์ชัน + ดาวน์โหลดโมเดล EasyOCR ตอน build, รันที่พอร์ต 7860 |
| `backend/requirements-docker.txt` | เวอร์ชันแพ็กเกจที่ล็อก (ตรงกับที่ใช้เทรน/บันทึกโมเดล) |
| `README.md` | มี YAML front matter ด้านบนที่ Spaces ต้องการ (`sdk: docker`, `app_port: 7860`) |
| `.gitattributes` | ให้ไฟล์ `*.joblib`, `*.pth` (โมเดล 143 MB + 64 MB) ไปกับ Git LFS |
| `.dockerignore` | ไม่เอา `data/`, `X-AnyLabeling/`, log ฯลฯ เข้า image |
| `frontend/js/api_client.js` | เรียก API แบบ relative (เดิมตายตัวที่ `localhost:8000`) |
| `backend/app/api/routes.py` | จำกัดไฟล์ 10 MB / 40 ล้านพิกเซล, endpoint ไม่บล็อก event loop, ประมวลผลทีละคำขอ (มีคิว) |

## ขั้นตอน (ทำครั้งเดียว)
1. huggingface.co → **New Space** → SDK **Docker** (Blank) → Hardware **CPU basic** → ตั้ง Public/Private
2. Settings → Access Tokens → สร้าง token แบบ **Write** (ใช้เป็นรหัสผ่านตอน push อย่าแชร์ให้ใคร)
3. ในโฟลเดอร์โปรเจกต์ที่ branch `hf`:
   ```bash
   git remote add space https://huggingface.co/spaces/<ชื่อผู้ใช้>/<ชื่อ-space>
   git push space hf:main
   ```
   ตอนถามรหัส: Username = ชื่อ HF ของคุณ, Password = token
4. เปิดแท็บ **Logs** ของ Space รอ build (ครั้งแรกหลายนาที) จนสถานะเป็น Running แล้วเปิดใช้ได้เลย

อัปเดตภายหลัง: commit บน `hf` แล้ว `git push space hf:main` อีกครั้ง

## ควรรู้
- ฟรี (CPU) ไม่มี GPU: ต่อภาพราว 3–6 วินาที, Super-resolution ~2 วินาทีต่อภาพที่ต้องใช้
- แอปหลับเมื่อไม่มีคนใช้นาน เข้าใหม่ต้องรอเปิด~1–2 นาที
- ระบบไม่เก็บภาพที่อัปโหลด (ประมวลผลในหน่วยความจำแล้วทิ้ง)
- ยังไม่มีรหัสผ่านหน้าเว็บ: ถ้าไม่อยากให้ใครก็เข้าได้ ตั้ง Space เป็น **Private**
- ยังไม่ได้ทดสอบ `docker build` บนเครื่องนี้ (Docker Desktop ปิดอยู่) รันโค้ดเดียวกันบนพอร์ต 7860 ผ่านแล้ว ถ้า build บน Spaces ล้ม ดูข้อความใน Logs

## ทดสอบในเครื่อง (ถ้ามี Docker Desktop)
```bash
docker build -t plate-space .
docker run --rm -p 7860:7860 plate-space
```
แล้วเปิด http://localhost:7860
