# License Plate Recognition System

> **Streamlit version:** `streamlit run streamlit_app.py` · deployment guide: [DEPLOY_STREAMLIT.md](DEPLOY_STREAMLIT.md) · how it works: [HOW_IT_WORKS.md](HOW_IT_WORKS.md)

Thai License Plate Detection & Recognition using SIFT + Homography + OCR

## Features

- **SIFT-based Plate Detection**: Robust feature matching using SIFT algorithm
- **Perspective Correction**: Automatic homography-based deskewing
- **OCR Recognition**: Thai + English text recognition with EasyOCR
- **Interactive Adjustment**: Manual corner adjustment for fine-tuning
- **Visual Pipeline**: Step-by-step visualization of processing pipeline
- **Modern Web UI**: Dark-themed responsive dashboard
- **Real-time Processing**: Fast inference with downscaling optimization

## Quick Start

### Requirements

- Python 3.10+
- Node.js/npm (optional, for frontend development)
- Windows/Mac/Linux

### Installation

1. **Clone/Download the project**
   ```bash
   cd "C:\CCTV SIFT method"
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

### Running the System

#### Option 1: Full Stack (Backend + Frontend)

1. **Start the Backend Server**
   ```bash
   python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Open Frontend in Browser**
   Navigate to: `http://localhost:8000`

#### Option 2: Test POC Script Only

```bash
python backend/test_pipeline.py
```

This runs a complete pipeline test:
- SIFT detection on sample image
- Perspective correction
- Image enhancement
- OCR recognition

## Project Structure

```
CCTV SIFT method/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI application
│   │   ├── api/routes.py           # API endpoints
│   │   └── core/
│   │       ├── sift_detector.py    # SIFT matching
│   │       ├── deskew.py           # Perspective transform
│   │       ├── preprocessor.py     # Image enhancement
│   │       └── ocr_engine.py       # OCR wrapper
│   ├── test_pipeline.py            # Phase 1 POC script
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── app.js
│       ├── api_client.js
│       └── canvas_viewer.js
└── IMPLEMENTATION_PLAN.md
```

## API Endpoints

### POST /api/v1/recognize
Complete license plate recognition

**Input**: Multipart form with image file

**Response**:
```json
{
  "status": "success",
  "plate_found": true,
  "corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
  "deskewed_plate_base64": "data:image/jpeg;base64,...",
  "enhanced_plate_base64": "data:image/jpeg;base64,...",
  "visualization_base64": "data:image/jpeg;base64,...",
  "ocr_result": {
    "text": "1กข 9999",
    "province": "กรุงเทพมหานคร",
    "confidence": 0.94,
    "is_valid": true
  },
  "processing_time_ms": 240
}
```

### POST /api/v1/interactive-deskew
Manual corner adjustment and re-recognition

**Input**:
```json
{
  "image_base64": "data:image/jpeg;base64,...",
  "corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
}
```


### GET /api/v1/health
Health check endpoint

## Features Breakdown

### Phase 1: Proof of Concept ✓
- SIFT keypoint detection and matching
- Template matching with FLANN
- Homography estimation with RANSAC
- Complete algorithm testing script

### Phase 2: Backend Development ✓
- FastAPI REST API
- Image preprocessing pipeline
- OCR integration (EasyOCR)

### Phase 3: Frontend Web UI ✓
- Modern dark-themed dashboard
- Drag-and-drop image upload
- Visual pipeline inspector
- Interactive corner adjustment
- Webcam support

### Phase 4: Robustness & Optimization ✓
- Automatic image downscaling
- CLAHE contrast enhancement
- Bilateral filtering for noise reduction
- Otsu's thresholding
- Morphological operations
- Province mapping and validation

### Phase 5: Extended Features
- Real-time CCTV stream support
- CSV/Excel export
- Batch processing
- Advanced ML models

## Configuration

### Customize API Base URL
Edit `frontend/js/api_client.js`:
```javascript
const apiClient = new APIClient('http://your-backend-url:8000');
```

### Adjust Detection Parameters
Edit `backend/app/core/sift_detector.py`:
- `RANSAC` threshold
- Feature matcher parameters
- Ratio test threshold (Lowe's)

### Image Enhancement Settings
Edit `backend/app/core/preprocessor.py`:
- CLAHE `clip_limit`
- Bilateral filter parameters
- Thresholding method

## Troubleshooting

### "Template not found" error
The system creates a sample template on first run. For better accuracy:
1. Prepare a clear Thai license plate image
2. Place it in `backend/app/templates/thai_plate_standard.png`

### Poor OCR accuracy
- Try adjusting image preprocessing in `preprocessor.py`
- Increase `CLAHE clip_limit` for low-contrast images
- Ensure good lighting on the license plate

### Backend not responding
- Ensure port 8000 is available
- Check Python version (3.10+ required)
- Run `pip install -r backend/requirements.txt` again

## Performance Tips

1. **Faster Detection**: Reduce template size or downscale input images
2. **Better OCR**: Use clearer images, avoid motion blur
3. **Real-time**: Cache template SIFT descriptors (already implemented)

## Future Improvements

- GPU acceleration with CUDA
- Multi-plate detection in single image
- Video stream processing
- Database integration
- Mobile app support
- Advanced plate format support (different countries)

## License

This project is provided as-is for educational and authorized testing purposes.

## Support

For issues or questions, refer to the IMPLEMENTATION_PLAN.md for technical details.
