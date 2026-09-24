# Vehicle Plate Rectification for LPR

A Streamlit web application that deskews license plates captured from angled CCTV footage, making them suitable for Optical Character Recognition (OCR).

## Features

- **Auto-detection** of license plate corners using edge detection, contour analysis, and SIFT/ORB keypoint clustering
- **Interactive corner adjustment** – select a corner and click to reposition, or type exact coordinates
- **Dual rectification methods:**
  - Perspective Transform (`cv2.getPerspectiveTransform`)
  - Homography with RANSAC (`cv2.findHomography` + `cv2.RANSAC`)
- **Post-processing enhancements:** CLAHE contrast, unsharp-mask sharpening, non-local-means denoising
- **Geometric validation** – convexity, aspect-ratio, and area checks before processing
- **One-click download** of the rectified plate image

## Computer Vision Pipeline

| Step | Technique | OpenCV API |
|------|-----------|------------|
| Edge detection | Canny + bilateral filter | `cv2.Canny`, `cv2.bilateralFilter` |
| Contour analysis | Approx polygon | `cv2.findContours`, `cv2.approxPolyDP` |
| Keypoint extraction | SIFT (preferred), ORB fallback | `cv2.SIFT_create`, `cv2.ORB_create` |
| Geometric transform | Perspective / Homography | `cv2.getPerspectiveTransform`, `cv2.findHomography` |
| Robust estimation | RANSAC | `cv2.findHomography(method=cv2.RANSAC)` |
| Enhancement | CLAHE, unsharp mask, NLM denoise | `cv2.createCLAHE`, `cv2.fastNlMeansDenoisingColored` |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

## How to Use

1. **Upload** a JPG/PNG image containing a vehicle with a visible license plate.
2. The app **auto-detects** the four plate corners (shown as coloured dots on the image).
3. **Adjust** corners if needed:
   - Select a corner button (TL / TR / BR / BL), then click on the image to move it.
   - Or type exact pixel coordinates in the input fields.
4. Choose **rectification method** and **enhancement** options.
5. Click **Rectify Plate** to process.
6. **Download** the result.

## Project Structure

```
├── app.py              # Streamlit front-end
├── cv_pipeline.py      # Computer vision back-end
├── requirements.txt    # Production dependencies
├── requirements-local.txt  # Dev dependencies (full OpenCV)
└── .streamlit/
    └── config.toml     # Streamlit server config
```

## Privacy

No data leaves your machine. Images are processed entirely in-browser/in-process and are never stored on disk.