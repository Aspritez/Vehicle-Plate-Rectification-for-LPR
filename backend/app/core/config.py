"""Configuration settings for the license plate recognition system."""

# SIFT Detection Settings
SIFT_KEYPOINT_THRESHOLD = 4  # Minimum keypoints required
SIFT_RATIO_TEST = 0.7  # Lowe's ratio test threshold
RANSAC_THRESHOLD = 5.0  # RANSAC error threshold
FEATURE_MATCH_THRESHOLD = 4  # Minimum good matches required

# Image Processing Settings
DEFAULT_DOWNSCALE_WIDTH = 1024  # Max width for downscaling
CLAHE_CLIP_LIMIT = 3.0  # CLAHE clip limit for histogram equalization
CLAHE_TILE_SIZE = 8  # CLAHE tile grid size
BILATERAL_FILTER_D = 9  # Bilateral filter diameter
BILATERAL_FILTER_SIGMA = 15  # Bilateral filter sigma values

# Plate Detection Settings
PLATE_STANDARD_RATIO = 2.26  # Thai plate width:height ratio
DESKEWED_PLATE_WIDTH = 400  # Target width for deskewed plate
DESKEWED_PLATE_HEIGHT = int(400 / PLATE_STANDARD_RATIO)  # Target height

# OCR Settings
OCR_LANGUAGES = ['th', 'en']  # Supported languages
OCR_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for OCR results
USE_GPU = False  # Use GPU for OCR (if available)

# Validation Settings
MIN_THAI_PLATE_CONFIDENCE = 0.5  # Minimum confidence for valid plate

# API Settings
MAX_IMAGE_SIZE_MB = 10  # Maximum image upload size
SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'webp']

# Performance Settings
ENABLE_CACHING = True  # Cache template descriptors
DETECTION_HISTORY_LIMIT = 1000  # Maximum history records to keep

# File Paths
TEMPLATE_DIR = "backend/app/templates"
DEFAULT_TEMPLATE = "thai_plate_standard.png"
