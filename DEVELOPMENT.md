# Development Guide

## Project Architecture

### Backend (FastAPI)

```
backend/
├── app/
│   ├── main.py              # FastAPI app initialization
│   ├── api/
│   │   └── routes.py        # API endpoint handlers
│   └── core/
│       ├── config.py        # Configuration constants
│       ├── sift_detector.py # SIFT plate detection
│       ├── deskew.py        # Perspective transformation
│       ├── preprocessor.py  # Image enhancement
│       └── ocr_engine.py    # OCR text recognition
├── templates/               # Template images for SIFT
├── test_sample/            # Test images
├── test_pipeline.py        # Standalone test script
└── requirements.txt        # Python dependencies
```

### Frontend (Vanilla JS)

```
frontend/
├── index.html             # Main HTML page
├── css/
│   └── style.css         # Dark theme styling
└── js/
    ├── app.js            # Main application logic
    ├── api_client.js     # Backend API client
    └── canvas_viewer.js  # Canvas and visualization
```

## Key Algorithms

### 1. SIFT Detection (`sift_detector.py`)

**Purpose**: Detect license plate location in an image

**Algorithm**:
1. Load template image (Thai plate standard)
2. Extract SIFT keypoints from template and input image
3. Match keypoints using FLANN matcher
4. Filter matches with Lowe's ratio test (threshold: 0.7)
5. Estimate homography matrix using RANSAC
6. Project template corners onto input image to get plate boundary

**Parameters** (from `config.py`):
- `SIFT_KEYPOINT_THRESHOLD`: Min keypoints required (default: 4)
- `SIFT_RATIO_TEST`: Lowe's ratio test (default: 0.7)
- `RANSAC_THRESHOLD`: RANSAC error (default: 5.0)
- `FEATURE_MATCH_THRESHOLD`: Min matches required (default: 4)

### 2. Perspective Correction (`deskew.py`)

**Purpose**: Correct skewed/perspective-distorted plates

**Algorithm**:
1. Order 4 corner points (top-left, top-right, bottom-right, bottom-left)
2. Define target dimensions (400x177 for Thai plates)
3. Compute perspective transform matrix
4. Warp image using the matrix
5. Optional: Auto-rotate using Hough lines if needed

**Parameters**:
- `PLATE_STANDARD_RATIO`: Width:height ratio (default: 2.26)
- `DESKEWED_PLATE_WIDTH`: Target width (default: 400px)

### 3. Image Enhancement (`preprocessor.py`)

**Purpose**: Improve image quality for OCR

**Pipeline**:
1. Convert to grayscale
2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. Bilateral filtering (denoise while preserving edges)
4. Gaussian blur
5. Otsu's thresholding
6. Morphological closing

**Parameters**:
- `CLAHE_CLIP_LIMIT`: Clip limit (default: 3.0)
- `CLAHE_TILE_SIZE`: Tile grid size (default: 8)
- `BILATERAL_FILTER_D`: Filter diameter (default: 9)
- `BILATERAL_FILTER_SIGMA`: Sigma values (default: 15)

### 4. OCR Recognition (`ocr_engine.py`)

**Purpose**: Extract text from enhanced plate image

**Features**:
- Uses EasyOCR for Thai/English recognition
- Validates against Thai plate format regex
- Maps abbreviated province names to full names
- Returns confidence scores

**Validation**:
- Pattern: `[1-9]?[ก-ฮ]{2}\s?[0-9]{1,4}`
- Supports 77 Thai provinces

## API Endpoints

### POST /api/v1/recognize

Complete end-to-end recognition pipeline.

**Request**:
```python
files = {'file': open('plate.jpg', 'rb')}
response = requests.post('http://localhost:8000/api/v1/recognize', files=files)
```

**Response Structure**:
```python
{
    "status": "success|no_plate_detected|deskew_failed|error",
    "processing_time_ms": 240,
    "plate_found": True,
    "corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    "deskewed_plate_base64": "data:image/jpeg;base64,...",
    "enhanced_plate_base64": "data:image/jpeg;base64,...",
    "visualization_base64": "data:image/jpeg;base64,...",
    "ocr_result": {
        "text": "1กข 9999",
        "province": "กรุงเทพมหานคร",
        "confidence": 0.94,
        "is_valid": True
    }
}
```

### POST /api/v1/interactive-deskew

Manual corner adjustment and re-recognition.

### GET /api/v1/history

Retrieve detection history.

### DELETE /api/v1/history

Clear detection history.

### GET /api/v1/health

Health check endpoint.

## Frontend Architecture

### App State Management

```javascript
class LicensePlateApp {
    currentImage: File           // Uploaded image
    currentImageBase64: string   // Base64 encoded image
    currentResult: object        // Recognition result
    corners: array              // Plate corner coordinates
    canvasViewer: object        // Canvas visualization
    history: array              // Detection history
}
```

### Pipeline Viewer

Shows 4 stages of image processing:
1. **Original**: Input image with detected bounds
2. **Detected**: Plate region highlighted
3. **Deskewed**: Corrected perspective
4. **Enhanced**: Processed for OCR

### Canvas Viewer

Interactive corner adjustment:
- Click and drag corner pins to adjust
- Real-time visualization
- Submit changes for re-recognition

## Extending the System

### Adding New OCR Language

1. Update `backend/app/core/config.py`:
```python
OCR_LANGUAGES = ['th', 'en', 'xx']  # Add 'xx'
```

2. Update EasyOCR reader in `ocr_engine.py`:
```python
self.reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
```

### Custom Validation Rules

Modify `_validate_and_clean()` in `ocr_engine.py`:

```python
def _validate_and_clean(self, text: str) -> tuple:
    # Add your custom validation logic
    is_valid = your_validation_function(text)
    return cleaned_text, province, is_valid
```

### Adjusting Detection Sensitivity

Edit `backend/app/core/config.py`:

```python
SIFT_RATIO_TEST = 0.7        # Lower = stricter matching
RANSAC_THRESHOLD = 5.0       # Lower = stricter geometry
FEATURE_MATCH_THRESHOLD = 4  # Higher = require more matches
```

### Adding GPU Support

1. Install CUDA-enabled OpenCV:
```bash
pip install opencv-contrib-python-headless-cuda
```

2. Update `config.py`:
```python
USE_GPU = True
```

3. Modify SIFT detector and FLANN matcher initialization

## Testing

### Unit Testing

Create tests in `backend/tests/`:

```python
import unittest
from app.core.sift_detector import SIFTDetector

class TestSIFTDetector(unittest.TestCase):
    def setUp(self):
        self.detector = SIFTDetector()
        
    def test_template_loading(self):
        # Test template loading
        pass
```

### Integration Testing

Run the complete pipeline:
```bash
python backend/test_pipeline.py
```

### Load Testing

Use `locust` for API stress testing:
```bash
pip install locust
locust -f locustfile.py --host=http://localhost:8000
```

## Performance Optimization

### Current Optimizations

1. **Downscaling**: Reduce image size before SIFT (faster detection)
2. **Template Caching**: Pre-compute SIFT descriptors
3. **FLANN Matcher**: Faster than BFMatcher for large datasets

### Future Optimizations

1. **GPU Acceleration**: CUDA/OpenCL for image processing
2. **Model Quantization**: Reduce OCR model size
3. **Batch Processing**: Process multiple images simultaneously
4. **Caching**: Cache recognition results for identical inputs
5. **Worker Pool**: Distribute heavy operations across workers

## Deployment

### Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t lpr-system .
docker run -p 8000:8000 lpr-system
```

### Production Server

Use Gunicorn with Uvicorn workers:
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.app.main:app
```

## Monitoring & Logging

Add logging to track system performance:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In API routes
logger.info(f"Processing time: {processing_time}ms")
```

## Contributing

1. Follow PEP 8 style guide
2. Add docstrings to functions
3. Update config.py for tunable parameters
4. Test changes with test_pipeline.py
5. Document new features

## Troubleshooting Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Template not loaded" | File path incorrect | Check `config.py` or manually place template |
| Poor OCR accuracy | Low contrast image | Increase CLAHE_CLIP_LIMIT |
| Slow detection | High-res image | Enable downscaling or reduce resolution |
| False positives | SIFT too lenient | Lower SIFT_RATIO_TEST or increase RANSAC_THRESHOLD |

## References

- [SIFT Algorithm Paper](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform)
- [OpenCV Documentation](https://docs.opencv.org/)
- [EasyOCR Documentation](https://github.com/JaidedAI/EasyOCR)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
