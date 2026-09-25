# 🚀 RUN ME - ไฟล์รันสำเร็จที่สุด

ไฟล์นี้คือวิธีที่ **ง่ายที่สุด** ในการรัน License Plate Recognition System ทั้ง Backend และ Frontend

---

## ⚡ วิธีรันสั้นๆ (2 ขั้นตอน)

### **Windows** 🖥️

ดับเบิลคลิก: **`START.bat`**

แค่นั้นแหละ! ระบบจะ:
1. ✅ Check Python
2. ✅ Install dependencies
3. ✅ Start Backend Server
4. ✅ เปิด Browser ให้โดยอัตโนมัติ
5. ✅ แสดง Link ให้ใช้

---

### **Mac** 🍎

ในเทอร์มินัล:
```bash
chmod +x START.sh
./START.sh
```

---

### **Linux** 🐧

ในเทอร์มินัล:
```bash
chmod +x START.sh
./START.sh
```

---

## 📌 เมื่อรันแล้วจะเห็นอะไร?

```
╔════════════════════════════════════════╗
║  🚗 Thai License Plate Recognition    ║
║  SIFT + Homography + OCR              ║
╚════════════════════════════════════════╝

✓ Python found
✓ Dependencies installed
🚀 Starting Backend Server...

✅ Backend started successfully!

📍 Access the Web UI:
   🌐 http://localhost:8000

📚 API Documentation:
   🔗 http://localhost:8000/docs

🌐 Opening browser in 2 seconds...
```

---

## 🎯 ใช้งาน

เมื่อเปิดแล้ว Browser จะเข้าไปที่:

### **`http://localhost:8000`** 🌐

ที่หน้านี้จะมี:
- 📷 ลาก & วาง ภาพป้ายทะเบียน
- 🔍 Pipeline visualization (4 ขั้นตอนการประมวลผล)
- 📊 Recognition results (เลขป้าย + โปรวินซ์)
- 🎨 Interactive corner adjustment
- 📈 Detection history

---

## ✨ ลองทำแบบนี้

1. **เปิด `START.bat` (Windows)** หรือ **`./START.sh` (Mac/Linux)**
2. **รอสักครู่** ให้ Browser เปิดเอง
3. **ลาก & วาง** ภาพรถยนต์ที่มีป้ายทะเบียน
4. **ดูผลลัพธ์** ที่ปรากฏมา

ง่ายมากใช่ไหม! 🎉

---

## 🔧 File ที่มี

| ไฟล์ | ระบบ | ลักษณะ |
|------|------|--------|
| `START.bat` | Windows | ดับเบิลคลิก |
| `START.py` | All | `python START.py` |
| `START.sh` | Mac/Linux | `./START.sh` |

---

## 📍 Links ที่สำคัญ

เมื่อระบบรันขึ้นมาแล้ว:

| ที่ | Link | ใช้สำหรับ |
|----|------|---------|
| **Web UI** | http://localhost:8000 | ใช้ระบบจริง |
| **API Docs** | http://localhost:8000/docs | ทดสอบ API |
| **Health Check** | http://localhost:8000/api/v1/health | ตรวจสอบว่าใช้ได้ |

---

## ⚠️ ปัญหาเบื้องต้น

### ❌ "Python not found" (Windows)

**แก้:** ติดตั้ง Python จาก https://www.python.org
- ✅ Check "Add Python to PATH"
- ✅ Reboot computer
- ✅ Try again

### ❌ "Address already in use" 

**แก้:** Port 8000 ถูกใช้งาน

```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Mac/Linux
lsof -i :8000
kill -9 <PID>
```

### ❌ "No module named 'easyocr'"

**แก้:** รัน installer อีกครั้ง
```bash
pip install -r backend/requirements.txt
```

---

## 💡 Tips

✅ **ครั้งแรก**: ใช้เวลานิดหน่อย (รอให้ OCR download models)  
✅ **ครั้งต่อไป**: เร็วมาก (150-300ms)  
✅ **ภาพดีที่สุด**: ป้ายทะเบียนชัด, มืดเพียงพอ, มองหน้า  
✅ **ถ้าอ่านผิด**: ใช้ "Edit Manually" ปุ่ม  

---

## 🛑 ปิด Server

```bash
# ในหน้า Command Prompt/Terminal ที่รัน Server:
Ctrl + C
```

หรือเพียงแค่ปิดหน้าต่าง Backend ได้เลย

---

## 📚 ต้องการอ่านเพิ่มเติม?

- **QUICKSTART.md** - Setup details
- **README.md** - Full documentation  
- **TROUBLESHOOTING.md** - Common issues
- **DEVELOPMENT.md** - For developers

---

## 🎉 That's it!

**ลองรัน `START.bat` (Windows) หรือ `./START.sh` (Mac/Linux) เลย!**

ระบบจะทำทุกอย่างให้เอง ✅

---

**สำหรับความช่วยเหลือเพิ่มเติม:** ดูไฟล์อื่นๆ ในโปรเจค

**Status:** ✅ Production Ready - สามารถใช้งานได้เลย!
