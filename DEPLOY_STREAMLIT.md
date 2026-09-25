# Deploy บน Streamlit Community Cloud (ฟรี)

## ไฟล์ที่เกี่ยวข้อง
| ไฟล์ | หน้าที่ |
|---|---|
| `streamlit_app.py` | หน้าแอป: อัปโหลด/ภาพตัวอย่าง → หาป้ายอัตโนมัติ → วาดกรอบรอบป้าย (หรือพิมพ์ 4 มุม) → ผลลัพธ์ |
| `backend/app/pipeline.py` | ตรรกะอ่านป้าย (ใช้ร่วมกับ FastAPI `routes.py`) |
| `requirements.txt`, `packages.txt` | แพ็กเกจ Python (torch แบบ CPU) และแพ็กเกจระบบ |
| `.streamlit/config.toml` | จำกัดอัปโหลด 10 MB, ปิดการส่งสถิติ |
| `samples/` | ภาพตัวอย่างป้ายไทย 3 ภาพ (ให้ผู้ตรวจกดลองได้เลย) |
| `backend/app/models/` | `plate_locator.joblib` (84 MB, 100 ต้นไม้) และ `RealESRGAN_x4plus.pth` (67 MB) เป็นไฟล์ git ธรรมดา **ไม่ใช้ Git LFS** |

## รันในเครื่องก่อน
```bash
pip install streamlit==1.64.0 streamlit-image-coordinates==0.4.1
streamlit run streamlit_app.py
```
(เครื่องที่มีแพ็กเกจของโปรเจกต์ครบอยู่แล้ว ไม่ต้องติดตั้ง `requirements.txt` ทั้งไฟล์)

## ทรัพยากรที่วัดได้ (CPU ล้วน, Windows)
| | RAM |
|---|---|
| โหลดโมเดลเสร็จ | ~1.4 GB |
| ใช้งานต่อเนื่อง (รวม Super-resolution) | ~2.2 GB คงที่ ไม่โตตามจำนวนภาพ |
| รวมส่วนของ Streamlit (peak) | **~2.35 GB** |
| เพดานของ Community Cloud | 2.7 GB (CPU 2 คอร์) |

เหลือที่ไม่มาก ถ้าบน cloud ขึ้น error "resource limits" ให้ **ปิด Super-resolution**: ในหน้า Manage app → Settings → Secrets ใส่
```toml
PLATE_DISABLE_SR = "1"
```
(ประหยัดราว 0.4–0.5 GB แต่ป้ายเบลอปานกลางจะอ่านได้แย่ลง) ความเร็วต่อภาพบน CPU ราว 3–10 วินาที

## ขั้นตอน deploy (repo ของทีม: `Aspritez/Vehicle-Plate-Rectification-for-LPR`)
Streamlit Cloud deploy จาก repo ที่ผู้ deploy เป็นเจ้าของหรือมีสิทธิ์ ในที่นี้คือ **เจ้าของ repo (Aspritez)** เป็นคนกด deploy

1. **เปิด Pull Request จาก `phet` เข้า `main`** บน GitHub (ตัว `main` มีงานรุ่นแรกอยู่แล้ว PR นี้เพิ่มเฉพาะส่วน Streamlit) แล้วให้เจ้าของ repo/ผู้มีสิทธิ์ **Merge**
2. เจ้าของ repo เข้า https://share.streamlit.io ด้วยบัญชี GitHub ของตัวเอง → **Create app** (Deploy a public app from GitHub)
3. เลือก Repository `Aspritez/Vehicle-Plate-Rectification-for-LPR`, Branch `main`, Main file path `streamlit_app.py`
4. **Advanced settings → Python version = 3.13** (แพ็กเกจที่ล็อกไว้ต้องใช้รุ่นนี้) แล้วกด **Deploy**
5. รอ build (ครั้งแรกหลายนาที) เปิดแอปครั้งแรกต้องรอโหลดโมเดล ~1–2 นาที แอปจะหลับถ้าไม่มีใครเข้า 12 ชั่วโมง (กดปลุกได้)

ถ้าไม่ใช้ repo ของทีม ก็ push branch `streamlit-clean` (สแนปช็อตไม่มีประวัติ ~144 MB) ขึ้น repo ใหม่ของตัวเอง แล้วทำข้อ 2–5 เหมือนกัน

**หมายเหตุเรื่องขนาด repo:** commit นี้เพิ่มไฟล์โมเดล 2 ไฟล์ (84 MB + 67 MB) เข้า git ถาวร (อยู่ในประวัติแม้จะลบทีหลัง) หลังจากนี้ **อย่า commit ไฟล์โมเดลซ้ำถ้าไม่จำเป็น** (ทุกครั้งที่เทรนใหม่จะเพิ่มอีก ~84 MB)

## ถ้า build มีปัญหา
- **Log แสดงการดาวน์โหลด `nvidia-*` / torch หลาย GB:** ตัวติดตั้งดึง torch รุ่น CUDA แทน CPU ให้แทนบรรทัด `torch==...` และ `torchvision==...` ใน `requirements.txt` ด้วย
  ```
  torch @ https://download.pytorch.org/whl/cpu/torch-2.11.0%2Bcpu-cp313-cp313-manylinux_2_28_x86_64.whl
  torchvision @ https://download.pytorch.org/whl/cpu/torchvision-0.26.0%2Bcpu-cp313-cp313-manylinux_2_28_x86_64.whl
  ```
- **Error "resource limits" / แอปรีสตาร์ตเอง:** ปิด Super-resolution ตามด้านบน
- **อ่านตัวหาป้ายไม่ขึ้น:** ตรวจว่าไฟล์ `backend/app/models/plate_locator.joblib` ขึ้นไปครบ (84 MB) และใช้ Python 3.13 (ไฟล์บันทึกด้วย scikit-learn 1.9.1)

## ข้อจำกัดเทียบกับหน้าเว็บ FastAPI เดิม
- ไม่มีการลากจุดมุมทั้ง 4 (ใช้ "วาดกรอบ" หรือพิมพ์พิกัดมุมแทน)
- ช้ากว่า (CPU ล้วน) และแอปจะหลับเมื่อไม่มีคนใช้
- ระบบไม่เก็บภาพที่อัปโหลด แต่แอปเปิดสาธารณะ ใครก็อัปโหลดได้ ถ้าไม่ต้องการ ให้ตั้ง repo/แอปเป็น private (ต้องอนุญาตให้ Streamlit เข้าถึง repo)
