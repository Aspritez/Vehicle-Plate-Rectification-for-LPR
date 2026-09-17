# รันบนเครื่อง (Local)

## วิธีที่ง่ายที่สุด

ดับเบิลคลิก `run_local.bat` หรือรันใน PowerShell:

```powershell
.\run_local.ps1
```

สคริปต์จะสร้าง `.venv`, ติดตั้ง dependencies และเปิดแอปที่ `http://localhost:8501` ให้อัตโนมัติ

---

## รันแบบ manual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-local.txt
streamlit run app.py
```

จากนั้นเปิด `http://localhost:8501`

---

## หมายเหตุ

| ไฟล์ | ใช้สำหรับ |
|------|-----------|
| `requirements.txt` | Streamlit Community Cloud (headless OpenCV) |
| `requirements-local.txt` | Local development (full OpenCV) |
| `run_local.ps1` | PowerShell launcher |
| `run_local.bat` | CMD / Explorer double-click wrapper |
