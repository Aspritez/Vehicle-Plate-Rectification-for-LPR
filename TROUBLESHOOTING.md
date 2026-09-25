# Troubleshooting Guide

## Common Issues & Solutions

---

## 🔴 Backend Issues

### Issue: "ModuleNotFoundError: No module named 'cv2'"

**Cause**: OpenCV not installed or Python environment wrong

**Solution**:
```bash
pip install -r backend/requirements.txt
```

If that fails:
```bash
pip install opencv-contrib-python --upgrade
```

### Issue: "Address already in use" on port 8000

**Cause**: Port 8000 is occupied by another application

**Solution**:
```bash
# Windows - Find and kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8000
kill -9 <PID>

# Or use different port
python -m uvicorn backend.app.main:app --port 8001 --reload
```

### Issue: "Template not loaded" on startup

**Cause**: Template image file missing

**Solution**:
The system creates a sample template automatically. For better results:
1. Find a clear Thai license plate image
2. Save it as: `backend/app/templates/thai_plate_standard.png`
3. Restart the backend

### Issue: "ImportError: cannot import name 'easyocr'"

**Cause**: EasyOCR not installed

**Solution**:
```bash
pip install easyocr
# First run will download language models (takes 1-2 min)
```

### Issue: Slow startup (first run takes 2+ minutes)

**Cause**: EasyOCR downloading language models

**Solution**:
- First run only - subsequent runs are fast
- Or pre-download models:
  ```python
  import easyocr
  reader = easyocr.Reader(['th', 'en'])
  ```

### Issue: "Connection refused" when accessing API

**Cause**: Backend server not running

**Solution**:
1. Ensure `run_backend.bat/sh` is running
2. Check for error messages in terminal
3. Try starting manually:
   ```bash
   python -m uvicorn backend.app.main:app --reload
   ```

---

## 🔴 Frontend Issues

### Issue: Blank page or "Cannot GET /"

**Cause**: Frontend not being served by backend

**Solution**:
1. Ensure backend is running on http://localhost:8000
2. Check browser console (F12) for errors
3. Try accessing Swagger docs: http://localhost:8000/docs

### Issue: File upload not working

**Cause**: Wrong file format or backend not responding

**Solution**:
1. Check file is JPG, PNG, or WEBP
2. File should be <10MB
3. Check browser console (F12) for errors
4. Verify backend is running

### Issue: Results not displaying

**Cause**: Recognition failed or API error

**Solution**:
1. Open browser console (F12)
2. Check Network tab for API response
3. Try simpler image (clear, front-facing plate)
4. Check backend logs for errors

### Issue: Pipeline images not showing

**Cause**: API not returning base64 images

**Solution**:
1. Check API response in Network tab
2. Verify `deskewed_plate_base64` is present
3. Try uploading different image
4. Check backend processing

### Issue: Corner adjustment not working

**Cause**: Canvas visualization not loaded

**Solution**:
1. Verify image was uploaded successfully
2. Check browser console for JavaScript errors
3. Try refreshing page (Ctrl+R)
4. Check that corners are detected

---

## 🔴 Recognition Issues

### Issue: "No license plate detected"

**Cause**: SIFT cannot find template pattern in image

**Solutions**:
1. **Ensure clear image**: Try well-lit, high-contrast photos
2. **Front-facing angle**: Plate should face camera directly
3. **Improve template**: Place better template in `backend/app/templates/`
4. **Adjust SIFT parameters**: Edit `backend/app/core/config.py`:
   ```python
   SIFT_RATIO_TEST = 0.8        # More lenient (default 0.7)
   RANSAC_THRESHOLD = 10.0      # More forgiving
   ```

### Issue: Poor OCR accuracy (wrong characters)

**Cause**: Image quality too low

**Solutions**:
1. **Use clearer image**: Avoid motion blur, shadows
2. **Increase contrast**: Adjust CLAHE in `config.py`:
   ```python
   CLAHE_CLIP_LIMIT = 5.0       # Higher = more contrast
   ```
3. **Manual correction**: Click "Edit Manually" to fix text
4. **Good lighting**: Ensure plate is well-lit
5. **Adjust angle**: Try different viewing angles

### Issue: Wrong confidence score

**Cause**: OCR confidence calculation difference

**Solutions**:
1. Confidence shows per-character average (0-1)
2. 0.7+ is generally reliable
3. <0.5 may indicate poor image quality
4. Manual correction available for low confidence

### Issue: Province not recognized

**Cause**: Province name not in mapping

**Solutions**:
1. Check spelling in detection results
2. Add custom provinces in `ocr_engine.py`:
   ```python
   THAI_PROVINCES = {
       "your_province": "full_name",
       # ...
   }
   ```
3. Some abbreviations auto-expand

### Issue: Deskewing looks wrong

**Cause**: Corner detection not optimal

**Solutions**:
1. Use **Interactive Corner Adjustment**:
   - Drag the 4 corner pins to correct position
   - Click "Apply & Re-recognize"
2. Different template might help:
   - Try another clear Thai plate image as template
3. Check image angle:
   - Extreme angles may not work well

---

## 🔴 API Issues

### Issue: 400 Bad Request

**Cause**: Invalid file format or missing parameters

**Solution**:
```bash
# Correct usage
curl -X POST -F "file=@image.jpg" http://localhost:8000/api/v1/recognize

# Check response for detailed error
```

### Issue: 500 Internal Server Error

**Cause**: Backend processing error

**Solutions**:
1. Check backend console for error message
2. Try with different image
3. Verify all dependencies installed
4. Restart backend server

### Issue: CORS errors in browser console

**Cause**: Cross-origin request blocked

**Solution**:
CORS is configured in `backend/app/main.py`. For production, adjust:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Change this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🟡 Performance Issues

### Issue: Slow recognition (>1 second)

**Cause**: Large image or slow hardware

**Solutions**:
1. **Downscale image**: Reduce resolution before upload
2. **Adjust settings**: Reduce CLAHE iterations
3. **Faster CPU needed**: Some operations are CPU-intensive
4. **Check system load**: Close other applications

### Issue: Memory usage increasing

**Cause**: Long-running session with many detections

**Solutions**:
1. Clear history: Click "Clear History" button
2. Restart backend: Fresh Python process
3. Limited history: 100 records max in memory

### Issue: First image takes longer

**Cause**: EasyOCR model loading on first use

**Solution**:
- Expected on first run (~2-3 seconds)
- Subsequent images are fast (~150-300ms)

---

## 🟡 Image Quality Issues

### Issue: Image too blurry

**Cause**: Camera motion or focus

**Solution**:
- Capture stable, high-quality images
- Good lighting essential
- Try different angle

### Issue: Image too dark/bright

**Cause**: Lighting conditions

**Solution**:
1. CLAHE helps but can't fix extremely dark
2. Try manual adjustment:
   ```python
   CLAHE_CLIP_LIMIT = 4.0  # Increase for dark images
   ```
3. Use image editor to pre-process

### Issue: Glare/reflection on plate

**Cause**: Light reflection

**Solution**:
- Avoid direct light
- Try different angle
- Remove glare with image editor

---

## 🟡 Configuration Issues

### Issue: Changes to config.py not taking effect

**Cause**: Python bytecode cached

**Solution**:
1. Delete `__pycache__` directories:
   ```bash
   find . -type d -name __pycache__ -exec rm -r {} +
   ```
2. Restart backend server

### Issue: Template changes not applied

**Cause**: Template cached in memory

**Solution**:
1. Restart backend server
2. Or modify `load_template()` to force reload

---

## 🟢 Verification Steps

### If everything seems broken:

1. **Check Python version**:
   ```bash
   python --version  # Should be 3.10+
   ```

2. **Verify dependencies**:
   ```bash
   pip list | grep -E "fastapi|opencv|easyocr"
   ```

3. **Test backend only**:
   ```bash
   python backend/test_pipeline.py
   ```

4. **Test API endpoint manually**:
   ```bash
   curl -X GET http://localhost:8000/api/v1/health
   ```

5. **Check browser console**:
   - Press F12
   - Look for JavaScript errors
   - Check Network tab for failed requests

6. **Test with simple image**:
   - Use the sample template as test image
   - If that works, your setup is correct

---

## 📋 Diagnostic Checklist

Before reporting issues, verify:

- [ ] Python 3.10+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Backend running (`python -m uvicorn backend.app.main:app --reload`)
- [ ] Port 8000 available
- [ ] Browser can reach http://localhost:8000
- [ ] Frontend loads without console errors
- [ ] At least one image uploaded for testing
- [ ] No antivirus blocking file operations

---

## 🆘 Getting Help

### Resources
1. **Quick Start**: See `QUICKSTART.md`
2. **Documentation**: See `README.md`
3. **Developer Guide**: See `DEVELOPMENT.md`
4. **API Docs**: Visit http://localhost:8000/docs

### Debug Information to Collect
When reporting issues, include:
- Python version: `python --version`
- OS: Windows/Mac/Linux
- Error message (exact text)
- Browser console errors (F12)
- Backend console output
- Steps to reproduce

---

## 🔄 Resetting Everything

If configuration is broken:

```bash
# 1. Clear cache
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# 2. Reinstall dependencies
pip uninstall -y -r backend/requirements.txt
pip install -r backend/requirements.txt

# 3. Clear browser data
# Open DevTools (F12) > Application > Clear site data

# 4. Delete template cache (optional)
rm -rf backend/app/templates/*.png

# 5. Restart everything
run_backend.bat  # Windows
./run_backend.sh # Linux/Mac
```

---

## 💡 Tips & Tricks

### Speed Up Detection
```python
# In config.py
SIFT_RATIO_TEST = 0.8  # Less strict matching
DEFAULT_DOWNSCALE_WIDTH = 800  # More aggressive downscaling
```

### Improve Accuracy
```python
# In config.py
SIFT_RATIO_TEST = 0.6  # Stricter matching
CLAHE_CLIP_LIMIT = 4.0  # More contrast enhancement
RANSAC_THRESHOLD = 2.0  # Stricter geometry
```

### Debug Mode
```python
# In backend code, add logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test Image Generation
```bash
python backend/test_pipeline.py
# Generates test images for debugging
```

---

**Last Updated**: September 24, 2026  
**For more help**: Check README.md or DEVELOPMENT.md
