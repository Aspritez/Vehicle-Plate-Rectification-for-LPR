# Quick Start Guide

## 5-Minute Setup

### Windows Users

1. **Open Command Prompt** in the project directory
2. **Run the startup script:**
   ```bash
   run_backend.bat
   ```
3. **Open Browser** and go to: `http://localhost:8000`

### Mac/Linux Users

1. **Open Terminal** in the project directory
2. **Make script executable:**
   ```bash
   chmod +x run_backend.sh
   ```
3. **Run the startup script:**
   ```bash
   ./run_backend.sh
   ```
4. **Open Browser** and go to: `http://localhost:8000`

## Usage

### Basic Flow

1. **Upload an Image**
   - Drag & drop a car image with a license plate
   - Or click to browse for a file
   - Supported: JPG, PNG, WEBP

2. **View Results**
   - License plate text displays in the center
   - Confidence score shows accuracy
   - Province name is automatically detected

3. **Adjust if Needed**
   - Use the "Visual Pipeline Inspector" to see each processing step
   - Drag the corner pins to manually adjust the plate boundary
   - Click "Apply & Re-recognize" to update OCR

4. **Export or Copy**
   - Copy the plate text to clipboard
   - View detection history in the table
   - Clear history as needed

## Testing POC

Run the complete algorithm test:

```bash
python backend/test_pipeline.py
```

This generates:
- `output_deskewed.jpg` - Corrected plate image
- `output_enhanced.jpg` - Enhanced image for OCR

## API Documentation

Access interactive API docs at: `http://localhost:8000/docs`

## Troubleshooting

### Issue: "Connection refused" error
- **Solution**: Make sure `run_backend.bat/sh` is running
- Check if port 8000 is available

### Issue: Poor OCR accuracy
- **Solution**: Ensure the license plate is clearly visible
- Try different angles/lighting
- Use the manual corner adjustment

### Issue: Template not loading
- **Solution**: First run creates a sample template
- For better results, place a clear Thai plate image in:
  `backend/app/templates/thai_plate_standard.png`

## Key Files to Know

| File | Purpose |
|------|---------|
| `backend/app/core/sift_detector.py` | SIFT plate detection |
| `backend/app/core/deskew.py` | Perspective correction |
| `backend/app/core/ocr_engine.py` | Text recognition |
| `backend/app/api/routes.py` | REST API endpoints |
| `frontend/index.html` | Web UI |
| `backend/test_pipeline.py` | Standalone test script |

## Performance Tips

- Keep images under 4MB for faster processing
- Clear, front-facing plates work best
- Good lighting improves OCR accuracy

## Next Steps

- Read `README.md` for full documentation
- Check `IMPLEMENTATION_PLAN.md` for technical details
- Review API endpoints in Swagger: `/docs`

## Need Help?

1. Check the error messages in the browser console
2. Review logs in the terminal where backend is running
3. Ensure all dependencies installed: `pip install -r backend/requirements.txt`
