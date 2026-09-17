import os
import sys
import time
import json
import random
import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# Reconfigure stdout for Windows utf-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw_images"
CROPS_SAVE_DIR = BASE_DIR / "predictions" / "cropped_plates"
CROPS_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Modern Dark Theme Colors
BG_DARK = "#14151D"
BG_CARD = "#1F212E"
BG_PANEL = "#181924"
BG_CARD_INNER = "#26293A"
ACCENT_GREEN = "#06D6A0"
ACCENT_BLUE = "#4CC9F0"
ACCENT_YELLOW = "#FFD166"
ACCENT_RED = "#EF476F"
ACCENT_PURPLE = "#A06CD5"
TEXT_WHITE = "#FFFFFF"
TEXT_MUTED = "#8D99AE"
BORDER_COLOR = "#323548"

def find_all_trained_models():
    """ค้นหาไฟล์โมเดล best.pt ทั้งหมดในโฟลเดอร์ runs และ root directory แล้วเรียงตามเวลาแก้ไขล่าสุด"""
    models = list(BASE_DIR.glob("runs/**/best.pt"))
    root_best = BASE_DIR / "best.pt"
    if root_best.exists() and root_best not in models:
        models.append(root_best)
    
    models.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return models

def get_latest_model_path():
    models = find_all_trained_models()
    if models:
        return models[0]
    return BASE_DIR / "best.pt"

def get_line_angle(p1, p2):
    """คำนวณมุมเอียงของเส้นเมื่อเทียบกับแกนแนวนอน [0, 90 องศา]"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = abs(np.degrees(np.arctan2(dy, dx))) % 180.0
    if angle > 90.0:
        angle = 180.0 - angle
    return angle

def order_plate_corners_physical(pts):
    """
    จัดเรียงจุด 4 จุดตามหลักเรขาคณิตและมุมมองจริงของป้ายทะเบียน:
    1. วิเคราะห์คู่เส้นตรง: คู่เส้นที่มีมุมเอียงเข้าใกล้แนวนอนมากกว่า (< 45°) คือขอบบน-ล่าง (Top/Bottom)
       แม้ว่ามุมกล้องด้านข้างจะทำให้ความกว้างในภาพหดสั้นกว่าความสูง (Extreme Perspective Foreshortening)
    2. ขอบบน (Top Edge) จะอยู่สูงกว่าในภาพ (ค่า y เฉลี่ยน้อยกว่า)
    3. เรียงพิกัดตามลำดับแน่นอน: Top-Left (TL), Top-Right (TR), Bottom-Right (BR), Bottom-Left (BL)
    """
    pts = np.array(pts, dtype="float32")

    # 2 คู่ของเส้นขอบตรงข้าม:
    # Pair A: ด้าน 0 (P0-P1) และ 2 (P2-P3)
    # Pair B: ด้าน 1 (P1-P2) และ 3 (P3-P0)
    angle_A = (get_line_angle(pts[0], pts[1]) + get_line_angle(pts[2], pts[3])) / 2.0
    angle_B = (get_line_angle(pts[1], pts[2]) + get_line_angle(pts[3], pts[0])) / 2.0

    # คู่ที่มุมใกล้เคียงแนวนอนมากกว่าคือขอบบน/ล่างของป้าย
    if angle_A <= angle_B:
        edge_1 = (pts[0], pts[1])
        edge_2 = (pts[3], pts[2])
    else:
        edge_1 = (pts[1], pts[2])
        edge_2 = (pts[0], pts[3])

    # ขอบบนคือขอบที่อยู่สูงกว่าในภาพ (ค่า y เฉลี่ยน้อยกว่า)
    y_1 = (edge_1[0][1] + edge_1[1][1]) / 2.0
    y_2 = (edge_2[0][1] + edge_2[1][1]) / 2.0

    top_edge = edge_1 if y_1 < y_2 else edge_2
    bottom_edge = edge_2 if y_1 < y_2 else edge_1

    # ขอบบน: x น้อยคือ TL, x มากคือ TR
    tl = top_edge[0] if top_edge[0][0] < top_edge[1][0] else top_edge[1]
    tr = top_edge[1] if top_edge[0][0] < top_edge[1][0] else top_edge[0]

    # ขอบล่าง: x น้อยคือ BL, x มากคือ BR
    bl = bottom_edge[0] if bottom_edge[0][0] < bottom_edge[1][0] else bottom_edge[1]
    br = bottom_edge[1] if bottom_edge[0][0] < bottom_edge[1][0] else bottom_edge[0]

    return np.array([tl, tr, br, bl], dtype="float32")

def analyze_dof_and_rectify(image, pts, target_size=(340, 150), mode="Auto"):
    """
    วิเคราะห์เวกเตอร์ของเส้นขอบตามทฤษฎีเรขาคณิตการแปลงภาพ (Geometric Transformation Hierarchy):
    - 8-DOF (Projective / Homography): เส้นขอบบน-ล่าง ลู่เข้าหาจุด Vanishing Point พร้อมแก้ระนาบให้ตรงสมบูรณ์
    - 6-DOF (Affine): เส้นขอบขนานคงที่ แต่มีแรงเฉือน (Shear เป็นสี่เหลี่ยมด้านขนาน) และมาตรส่วนอิสระ
    - 4-DOF (Similarity): เส้นขนานและมุมฉาก 90° คงที่ (Rotation + Translation + Uniform Scale)
    - 3-DOF (Euclidean / Rigid): หมุนและเลื่อนตำแหน่งอย่างเดียว (Scale = 1)
    """
def analyze_dof_and_rectify(image, pts, mode="Auto", target_size=(340, 150), is_gt=False):
    """
    คำนวณและปรับระนาบของป้ายทะเบียนตามทฤษฎีเรขาคณิต Geometric Transformations (Homography):
    - 8-DOF: Projective (Homography) - ปรับระนาบ 3D จากมุมมองกล้องให้แบนราบตั้งตรง 100%
    - 6-DOF: Affine - รักษาเส้นขนาน แก้แรงเฉือน Shear
    - 4-DOF: Similarity - ย่อ/ขยาย + หมุน + เลื่อน
    - 3-DOF: Euclidean - หมุน + เลื่อนอย่างเดียว
    """
    rect = order_plate_corners_physical(pts)
    (tl, tr, br, bl) = rect

    # ตัดภาพ Raw Crop
    x_min = max(0, int(np.min(pts[:, 0])))
    y_min = max(0, int(np.min(pts[:, 1])))
    x_max = min(image.shape[1], int(np.max(pts[:, 0])))
    y_max = min(image.shape[0], int(np.max(pts[:, 1])))
    raw_crop = image[y_min:y_max, x_min:x_max].copy() if (x_max > x_min and y_max > y_min) else None

    target_w, target_h = target_size
    dst_pts = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype="float32")

    # วิเคราะห์เวกเตอร์ของขอบป้าย
    v_top = tr - tl
    v_bot = br - bl
    v_left = bl - tl
    v_right = br - tr

    # 1. เช็คความเป็นเส้นขนาน (Parallelism Error)
    cos_h = abs(np.dot(v_top, v_bot)) / (np.linalg.norm(v_top) * np.linalg.norm(v_bot) + 1e-6)
    delta_h = np.degrees(np.arccos(np.clip(cos_h, 0.0, 1.0)))
    cos_v = abs(np.dot(v_left, v_right)) / (np.linalg.norm(v_left) * np.linalg.norm(v_right) + 1e-6)
    delta_v = np.degrees(np.arccos(np.clip(cos_v, 0.0, 1.0)))
    max_par_err = max(delta_h, delta_v)

    # 2. เช็คความเป็นมุมฉาก (Orthogonality Error จาก 90 องศา)
    cos_corner = abs(np.dot(v_top, v_left)) / (np.linalg.norm(v_top) * np.linalg.norm(v_left) + 1e-6)
    ortho_err = abs(np.degrees(np.arccos(np.clip(cos_corner, 0.0, 1.0))) - 90.0)

    # คำนวณ Full 8-DOF Homography Matrix
    H = cv2.getPerspectiveTransform(rect, dst_pts)

    # 3. คำนวณภาพตามระดับ DOF ที่เลือกหรือ Auto
    if "8-DOF" in mode or ("Auto" in mode and (is_gt or max_par_err > 2.0 or ortho_err > 4.0)):
        dof_name = "8-DOF (Projective / Homography)"
        if is_gt or max_par_err > 2.0:
            dof_desc = f"ลู่เข้า Vanishing Point (ลู่เข้า={max_par_err:.1f}°, เฉือน={ortho_err:.1f}°) แก้ระนาบตั้งตรง 100%"
        else:
            dof_desc = "แปลงพิกัดมุมมอง 3 มิติกลับระนาบตรงหน้า (Full 3x3 Projective H)"
        rectified = cv2.warpPerspective(image, H, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    elif "6-DOF" in mode:
        dof_name = "6-DOF (Affine Transformation)"
        dof_desc = f"เส้นขนานคงที่ (err={max_par_err:.1f}°), มีแรงเฉือน Shear={ortho_err:.1f}°"
        M_aff = cv2.getAffineTransform(rect[:3], dst_pts[:3])
        H = np.vstack([M_aff, [0.0, 0.0, 1.0]])
        rectified = cv2.warpAffine(image, M_aff, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    elif "3-DOF" in mode:
        dof_name = "3-DOF (Euclidean / Rigid)"
        dof_desc = "หมุนและเลื่อนตำแหน่งอย่างเดียว (มุมและขนาดคงที่ 100%)"
        M_sim, _ = cv2.estimateAffinePartial2D(rect, dst_pts)
        if M_sim is not None:
            H = np.vstack([M_sim, [0.0, 0.0, 1.0]])
            rectified = cv2.warpAffine(image, M_sim, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            rectified = cv2.warpPerspective(image, H, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    else: # 4-DOF Similarity
        dof_name = "4-DOF (Similarity Transformation)"
        dof_desc = f"เส้นขนาน & มุมฉากคงที่ (par_err={max_par_err:.1f}°, ortho_err={ortho_err:.1f}°)"
        M_sim, _ = cv2.estimateAffinePartial2D(rect, dst_pts)
        if M_sim is not None:
            H = np.vstack([M_sim, [0.0, 0.0, 1.0]])
            rectified = cv2.warpAffine(image, M_sim, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            rectified = cv2.warpPerspective(image, H, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 4. กรณีมาจาก AI Bounding Box (ไม่ใช่ Ground Truth) ที่กล่องเดิมเป็นสี่เหลี่ยมมุมฉาก OBB
    # ตรวจสอบการเอียงภายในของตัวหนังสือและปรับให้ระนาบตัวหนังสือตรงระดับแนวนอนเป๊ะ
    if not is_gt and max_par_err < 1.5 and ortho_err < 1.5:
        try:
            gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=35, minLineLength=35, maxLineGap=10)
            if lines is not None:
                angles = []
                for l in lines:
                    x1, y1, x2, y2 = l.ravel()
                    deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if -30 < deg < 30:
                        angles.append(deg)
                if len(angles) >= 4:
                    deskew = np.median(angles)
                    if abs(deskew) > 1.5:
                        M_rot = cv2.getRotationMatrix2D((target_w/2, target_h/2), deskew, 1.0)
                        rectified = cv2.warpAffine(rectified, M_rot, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            pass

    # 5. ตรวจสอบว่าภาพอยู่ในแนวนอนหรือไม่ (ถ้าแนวตั้ง หมุน 90 องศา)
    if rectified.shape[0] > rectified.shape[1]:
        rectified = cv2.rotate(rectified, cv2.ROTATE_90_CLOCKWISE)

    return rectified, raw_crop, H, dof_name, dof_desc, max_par_err, ortho_err

class LicensePlateTestApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🔍 CCTV License Plate Detection - AI Test & Geometric DOF Homography Rectifier")
        self.geometry("1460x900")
        self.minsize(1160, 720)
        self.configure(bg=BG_DARK)

        # Style config
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.configure_styles()

        # State
        self.model = None
        self.available_models = []
        self.model_path = get_latest_model_path()
        self.current_img_path = None
        self.cv_image = None
        self.annotated_cv_image = None
        self.detected_plates = []
        self.image_list = []
        self.image_index = -1

        # Cache images for Tkinter
        self.tk_main_img = None
        self.crop_tk_widgets = []

        # Build UI
        self.create_header()
        self.create_main_layout()

        # Keybindings
        self.bind("<Left>", lambda e: self.navigate_image(-1))
        self.bind("<Right>", lambda e: self.navigate_image(1))

        # Initialize
        self.init_image_list()
        self.refresh_model_list()

        # Auto-load first image
        if self.image_list:
            self.image_index = 0
            self.load_and_predict(self.image_list[0])

    def configure_styles(self):
        self.style.configure(".", background=BG_DARK, foreground=TEXT_WHITE, font=("Segoe UI", 10))
        self.style.configure("TFrame", background=BG_DARK)
        self.style.configure("Card.TFrame", background=BG_CARD)
        self.style.configure("TLabel", background=BG_DARK, foreground=TEXT_WHITE)
        self.style.configure("Card.TLabel", background=BG_CARD, foreground=TEXT_WHITE)
        self.style.configure("TScale", background=BG_CARD, troughcolor=BG_PANEL)

    def create_header(self):
        header = tk.Frame(self, bg=BG_CARD, padx=20, pady=10, relief="solid", bd=1)
        header.pack(fill=tk.X, side=tk.TOP, padx=12, pady=(12, 6))

        # Title & Subtitle
        title_box = tk.Frame(header, bg=BG_CARD)
        title_box.pack(side=tk.LEFT)
        lbl_title = tk.Label(title_box, text="🔍 CCTV License Plate - AI Test & Geometric Homography",
                             font=("Segoe UI", 14, "bold"), fg=TEXT_WHITE, bg=BG_CARD)
        lbl_title.pack(anchor="w")

        self.lbl_model_info = tk.Label(title_box, text="กำลังโหลดโมเดล...", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD)
        self.lbl_model_info.pack(anchor="w")

        # Right Metrics
        metrics_box = tk.Frame(header, bg=BG_CARD)
        metrics_box.pack(side=tk.RIGHT)

        self.lbl_speed = self.add_metric_card(metrics_box, "ความเร็ว (Inference)", "-- ms", ACCENT_BLUE)
        self.lbl_homography_badge = self.add_metric_card(metrics_box, "ทฤษฎีเรขาคณิต", "Auto DOF", ACCENT_PURPLE)
        self.lbl_plate_count = self.add_metric_card(metrics_box, "ป้ายที่ตรวจพบ", "0 ป้าย", ACCENT_GREEN)

    def add_metric_card(self, parent, title, val, color):
        box = tk.Frame(parent, bg=BG_PANEL, padx=15, pady=5, relief="ridge", bd=1)
        box.pack(side=tk.LEFT, padx=6)
        lbl_v = tk.Label(box, text=val, font=("Segoe UI", 13, "bold"), fg=color, bg=BG_PANEL)
        lbl_v.pack()
        lbl_t = tk.Label(box, text=title, font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_PANEL)
        lbl_t.pack()
        return lbl_v

    def create_main_layout(self):
        # Toolbar Bar
        toolbar = tk.Frame(self, bg=BG_CARD, padx=15, pady=8)
        toolbar.pack(fill=tk.X, padx=12, pady=(0, 6))

        btn_open = tk.Button(toolbar, text="📂 เปิดรูป...", bg=ACCENT_BLUE, fg="#000000",
                             font=("Segoe UI", 9, "bold"), bd=0, padx=10, pady=5, command=self.open_image_dialog)
        btn_open.pack(side=tk.LEFT, padx=3)

        btn_random = tk.Button(toolbar, text="🎲 สุ่มภาพ CCTV", bg=BG_PANEL, fg=TEXT_WHITE,
                               font=("Segoe UI", 9, "bold"), bd=1, relief="ridge", padx=10, pady=5, command=self.pick_random_image)
        btn_random.pack(side=tk.LEFT, padx=3)

        btn_prev = tk.Button(toolbar, text="◀", bg=BG_PANEL, fg=TEXT_WHITE,
                             font=("Segoe UI", 9, "bold"), bd=1, relief="ridge", padx=8, pady=5, command=lambda: self.navigate_image(-1))
        btn_prev.pack(side=tk.LEFT, padx=2)

        btn_next = tk.Button(toolbar, text="▶", bg=BG_PANEL, fg=TEXT_WHITE,
                             font=("Segoe UI", 9, "bold"), bd=1, relief="ridge", padx=8, pady=5, command=lambda: self.navigate_image(1))
        btn_next.pack(side=tk.LEFT, padx=2)

        # Model Selector Dropdown
        model_frame = tk.Frame(toolbar, bg=BG_CARD)
        model_frame.pack(side=tk.LEFT, padx=8)

        tk.Label(model_frame, text="🤖 โมเดล:", font=("Segoe UI", 9, "bold"), fg=ACCENT_GREEN, bg=BG_CARD).pack(side=tk.LEFT, padx=(0, 3))
        self.combo_models = ttk.Combobox(model_frame, width=28, state="readonly", font=("Segoe UI", 9))
        self.combo_models.pack(side=tk.LEFT, padx=2)
        self.combo_models.bind("<<ComboboxSelected>>", self.on_model_selected)

        # DOF Selector Dropdown
        dof_frame = tk.Frame(toolbar, bg=BG_CARD)
        dof_frame.pack(side=tk.LEFT, padx=6)

        tk.Label(dof_frame, text="📐 ทฤษฎี DOF:", font=("Segoe UI", 9, "bold"), fg=ACCENT_PURPLE, bg=BG_CARD).pack(side=tk.LEFT, padx=(0, 3))
        self.combo_dof = ttk.Combobox(
            dof_frame,
            values=["Auto (วิเคราะห์จากเส้น)", "8-DOF (Projective)", "6-DOF (Affine)", "4-DOF (Similarity)", "3-DOF (Euclidean)"],
            width=20,
            state="readonly",
            font=("Segoe UI", 9)
        )
        self.combo_dof.current(0)
        self.combo_dof.pack(side=tk.LEFT, padx=2)
        self.combo_dof.bind("<<ComboboxSelected>>", lambda e: self.run_prediction())

        # Ground Truth Toggle & Navigation Button
        gt_frame = tk.Frame(toolbar, bg=BG_CARD)
        gt_frame.pack(side=tk.LEFT, padx=6)
        self.use_gt_var = tk.BooleanVar(value=True)
        self.chk_gt = tk.Checkbutton(
            gt_frame,
            text="🎯 ใช้เฉลย (.JSON) ถ้ามี",
            variable=self.use_gt_var,
            bg=BG_CARD,
            fg="#00e5ff",
            selectcolor=BG_PANEL,
            activebackground=BG_CARD,
            activeforeground="#00e5ff",
            font=("Segoe UI", 9, "bold"),
            command=self.run_prediction
        )
        self.chk_gt.pack(side=tk.LEFT)

        btn_next_gt = tk.Button(
            toolbar,
            text="🏷️ รูปที่มีเฉลย",
            bg=BG_PANEL,
            fg="#00e5ff",
            font=("Segoe UI", 9, "bold"),
            bd=1,
            relief="ridge",
            padx=6,
            pady=4,
            command=self.next_annotated_image
        )
        btn_next_gt.pack(side=tk.LEFT, padx=2)

        self.lbl_img_status = tk.Label(toolbar, text="ไฟล์: -", fg=TEXT_MUTED, bg=BG_CARD, font=("Segoe UI", 9))
        self.lbl_img_status.pack(side=tk.LEFT, padx=8)

        # Confidence Slider
        conf_box = tk.Frame(toolbar, bg=BG_CARD)
        conf_box.pack(side=tk.RIGHT, padx=5)

        self.lbl_conf_val = tk.Label(conf_box, text="Confidence: 25%", fg=ACCENT_YELLOW, bg=BG_CARD, font=("Segoe UI", 9, "bold"))
        self.lbl_conf_val.pack(side=tk.LEFT, padx=5)

        self.conf_slider = ttk.Scale(conf_box, from_=0.10, to=0.95, value=0.25, length=100, command=self.on_slider_change)
        self.conf_slider.pack(side=tk.LEFT)

        # Split Content Area
        split_panes = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=BG_DARK, bd=0, sashwidth=6)
        split_panes.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        # ================= LEFT: MAIN IMAGE CANVAS =================
        left_box = tk.Frame(split_panes, bg=BG_CARD, padx=12, pady=12)
        split_panes.add(left_box, minsize=540)

        header_left = tk.Frame(left_box, bg=BG_CARD)
        header_left.pack(fill=tk.X, pady=(0, 6))
        tk.Label(header_left, text="📷 ภาพตรวจจับมุมเอียง & เส้นขอบป้าย (Detected Quadrilateral)",
                 font=("Segoe UI", 11, "bold"), fg=TEXT_WHITE, bg=BG_CARD).pack(side=tk.LEFT)

        self.canvas_main = tk.Canvas(left_box, bg="#101117", highlightthickness=1, highlightbackground=BORDER_COLOR)
        self.canvas_main.pack(fill=tk.BOTH, expand=True)
        self.canvas_main.bind("<Configure>", lambda e: self.render_main_image())

        # ================= RIGHT: HOMOGRAPHY RECTIFIED CROPS =================
        right_box = tk.Frame(split_panes, bg=BG_CARD, padx=12, pady=12, width=580)
        split_panes.add(right_box, minsize=500)

        header_right = tk.Frame(right_box, bg=BG_CARD)
        header_right.pack(fill=tk.X, pady=(0, 6))
        tk.Label(header_right, text="📐 เปรียบเทียบ: ตัดตามมุมกล้อง vs ดึงกลับตรงตามทฤษฎี Homography",
                 font=("Segoe UI", 11, "bold"), fg=ACCENT_GREEN, bg=BG_CARD).pack(side=tk.LEFT)

        btn_save_all = tk.Button(header_right, text="💾 บันทึกรูปป้ายทั้งหมด", bg=BG_PANEL, fg=TEXT_WHITE,
                                 font=("Segoe UI", 8), bd=1, relief="ridge", padx=8, pady=2, command=self.save_all_crops)
        btn_save_all.pack(side=tk.RIGHT)

        # Scrollable container for crops
        self.crop_container = tk.Canvas(right_box, bg=BG_PANEL, highlightthickness=0)
        self.crop_scrollbar = tk.Scrollbar(right_box, orient=tk.VERTICAL, command=self.crop_container.yview, bg=BG_PANEL)
        self.crop_scroll_frame = tk.Frame(self.crop_container, bg=BG_PANEL)

        self.crop_scroll_frame.bind("<Configure>", lambda e: self.crop_container.configure(scrollregion=self.crop_container.bbox("all")))
        self.crop_container.create_window((0, 0), window=self.crop_scroll_frame, anchor="nw")
        self.crop_container.configure(yscrollcommand=self.crop_scrollbar.set)

        self.crop_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.crop_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def init_image_list(self):
        if RAW_DIR.exists():
            exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            self.image_list = sorted([p for p in RAW_DIR.iterdir() if p.suffix.lower() in exts], key=lambda x: x.name.lower())

    def refresh_model_list(self):
        self.available_models = find_all_trained_models()
        if not self.available_models:
            self.lbl_model_info.config(text="❌ ไม่พบไฟล์โมเดล best.pt ในระบบ", fg=ACCENT_RED)
            return

        display_options = []
        for idx, p in enumerate(self.available_models):
            mtime = time.strftime('%H:%M น.', time.localtime(p.stat().st_mtime))
            exp_name = p.parent.parent.name if p.parent.name == "weights" else p.stem
            is_latest = "⭐ [ล่าสุด]" if idx == 0 else ""
            display_options.append(f"{is_latest} {exp_name} ({mtime})")

        self.combo_models["values"] = display_options
        self.combo_models.current(0)
        self.load_yolo_model(self.available_models[0])

    def on_model_selected(self, event):
        idx = self.combo_models.current()
        if 0 <= idx < len(self.available_models):
            self.load_yolo_model(self.available_models[idx])
            if self.current_img_path:
                self.run_prediction()

    def load_yolo_model(self, path=None):
        target_path = path if path else self.model_path
        if not target_path.exists():
            self.lbl_model_info.config(text=f"❌ ไม่พบไฟล์: {target_path.name}", fg=ACCENT_RED)
            return

        try:
            self.lbl_model_info.config(text=f"กำลังโหลดโมเดล: {target_path.name}...")
            self.update_idletasks()
            self.model = YOLO(str(target_path))
            self.model_path = target_path

            exp_name = target_path.parent.parent.name if target_path.parent.name == "weights" else target_path.name
            device_name = "NVIDIA GeForce RTX 3080" if self.model.device.type != "cpu" else "CPU"
            self.lbl_model_info.config(text=f"✅ ใช้งาน: {exp_name} | อุปกรณ์: {device_name}", fg=ACCENT_GREEN)
        except Exception as e:
            self.lbl_model_info.config(text=f"❌ โหลดโมเดลล้มเหลว: {e}", fg=ACCENT_RED)

    def open_image_dialog(self):
        filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")]
        init_dir = str(RAW_DIR) if RAW_DIR.exists() else str(BASE_DIR)
        path = filedialog.askopenfilename(title="เลือกรูปภาพสำหรับทดสอบ", initialdir=init_dir, filetypes=filetypes)
        if path:
            p = Path(path)
            if p in self.image_list:
                self.image_index = self.image_list.index(p)
            self.load_and_predict(p)

    def pick_random_image(self):
        if not self.image_list:
            messagebox.showinfo("แจ้งเตือน", "ไม่พบรูปภาพในโฟลเดอร์ data/raw_images")
            return
        self.image_index = random.randint(0, len(self.image_list) - 1)
        self.load_and_predict(self.image_list[self.image_index])

    def navigate_image(self, step):
        if not self.image_list:
            return
        self.image_index = (self.image_index + step) % len(self.image_list)
        self.load_and_predict(self.image_list[self.image_index])

    def on_slider_change(self, val):
        conf = float(val)
        self.lbl_conf_val.config(text=f"Confidence: {int(conf*100)}%")
        if self.current_img_path:
            self.run_prediction()

    def next_annotated_image(self):
        if not self.image_list:
            return
        n = len(self.image_list)
        for step in range(1, n + 1):
            idx = (self.image_index + step) % n
            cand = self.image_list[idx]
            if cand.with_suffix(".json").exists():
                self.image_index = idx
                self.load_and_predict(cand)
                return
        messagebox.showinfo("แจ้งเตือน", "ไม่พบรูปภาพอื่นที่มีไฟล์เฉลย .JSON")

    def load_and_predict(self, img_path):
        self.current_img_path = img_path
        json_path = img_path.with_suffix(".json")
        has_gt = json_path.exists()
        gt_tag = "  [🎯 มีเฉลย .JSON]" if has_gt else "  [⚪ AI อย่างเดียว]"
        self.lbl_img_status.config(text=f"ไฟล์: {img_path.name} ({self.image_index + 1}/{len(self.image_list)}){gt_tag}")

        self.cv_image = cv2.imread(str(img_path))
        if self.cv_image is None:
            messagebox.showerror("ผิดพลาด", f"ไม่สามารถโหลดภาพ: {img_path}")
            return

        self.run_prediction()

    def run_prediction(self):
        if self.cv_image is None:
            return

        conf_thres = float(self.conf_slider.get())
        dof_mode = self.combo_dof.get()

        self.annotated_cv_image = self.cv_image.copy()
        self.detected_plates = []

        json_path = self.current_img_path.with_suffix(".json") if self.current_img_path else None
        has_gt_file = json_path is not None and json_path.exists()
        use_gt = has_gt_file and self.use_gt_var.get()

        if use_gt:
            self.lbl_speed.config(text="0.0 ms (GT)")
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    gt_data = json.load(f)
                
                gt_shapes = [s for s in gt_data.get("shapes", []) if len(s.get("points", [])) == 4]

                for i, s in enumerate(gt_shapes):
                    pts_gt = np.array(s["points"], dtype=np.float32)
                    pts_int = np.int32(pts_gt)

                    # วาดกรอบเฉลยสี Cyan / Aqua นีออน
                    cv2.polylines(self.annotated_cv_image, [pts_int], isClosed=True, color=(255, 220, 0), thickness=3)
                    for pt in pts_int:
                        cv2.circle(self.annotated_cv_image, (int(pt[0]), int(pt[1])), 6, (255, 255, 255), -1)
                        cv2.circle(self.annotated_cv_image, (int(pt[0]), int(pt[1])), 4, (255, 200, 0), -1)

                    rectified, raw_crop, H, dof_name, dof_desc, par_err, ortho_err = analyze_dof_and_rectify(
                        self.cv_image, pts_gt, mode=dof_mode, is_gt=True
                    )

                    tag_text = f"เฉลย #{i+1} [ตรง 100%] ({dof_name.split()[0]})"
                    min_y = min(pts_int[:, 1])
                    min_x = min(pts_int[:, 0])
                    cv2.putText(self.annotated_cv_image, tag_text, (min_x, max(25, min_y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 2)

                    if rectified is not None:
                        h, w = rectified.shape[:2]
                        self.detected_plates.append({
                            "index": i + 1,
                            "conf": 1.0,
                            "is_gt": True,
                            "rectified_img": rectified,
                            "raw_crop_img": raw_crop,
                            "width": w,
                            "height": h,
                            "box": pts_gt,
                            "homography_matrix": H,
                            "dof_name": dof_name,
                            "dof_desc": dof_desc,
                            "par_err": par_err,
                            "ortho_err": ortho_err,
                            "current_display_img": rectified.copy()
                        })
            except Exception as e:
                print("Error reading GT JSON:", e)

        # ถ้าไม่ได้ใช้ GT หรือไม่มีป้ายจาก GT ให้ใช้ AI Model YOLO-OBB
        if not self.detected_plates and self.model is not None:
            results = self.model.predict(source=str(self.current_img_path), conf=conf_thres, save=False, verbose=False)
            res = results[0]
            speed = res.speed.get("inference", 0.0)
            self.lbl_speed.config(text=f"{speed:.1f} ms")

            if res.obb is not None and len(res.obb) > 0:
                boxes = res.obb.xyxyxyxy.cpu().numpy()
                confs = res.obb.conf.cpu().numpy()

                for i, (box, conf) in enumerate(zip(boxes, confs)):
                    pts = np.int32(box)

                    # 1. วาดกรอบ OBB นีออนสีเขียว
                    cv2.polylines(self.annotated_cv_image, [pts], isClosed=True, color=(0, 255, 100), thickness=3)

                    # วาดจุดมุม 4 จุด
                    for pt in pts:
                        cv2.circle(self.annotated_cv_image, (int(pt[0]), int(pt[1])), 5, (0, 220, 255), -1)

                    # 2. ปรับตรงด้วย Geometric DOF Analysis & Homography Matrix
                    rectified, raw_crop, H, dof_name, dof_desc, par_err, ortho_err = analyze_dof_and_rectify(
                        self.cv_image, box, mode=dof_mode, is_gt=False
                    )

                    tag_text = f"Plate #{i+1}: {conf*100:.1f}% ({dof_name.split()[0]})"
                    cv2.putText(self.annotated_cv_image, tag_text, (pts[0][0], max(20, pts[0][1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)

                    if rectified is not None:
                        h, w = rectified.shape[:2]
                        self.detected_plates.append({
                            "index": i + 1,
                            "conf": conf,
                            "is_gt": False,
                            "rectified_img": rectified,
                            "raw_crop_img": raw_crop,
                            "width": w,
                            "height": h,
                            "box": box,
                            "homography_matrix": H,
                            "dof_name": dof_name,
                            "dof_desc": dof_desc,
                            "par_err": par_err,
                            "ortho_err": ortho_err,
                            "current_display_img": rectified.copy()
                        })

        gt_label_str = " (🎯 เฉลย JSON)" if use_gt else ""
        self.lbl_plate_count.config(text=f"{len(self.detected_plates)} ป้าย{gt_label_str}")
        if self.detected_plates:
            self.lbl_homography_badge.config(text=self.detected_plates[0]["dof_name"].split()[0])
        else:
            self.lbl_homography_badge.config(text=dof_mode.split()[0])

        self.render_main_image()
        self.render_cropped_plates()

    def render_main_image(self):
        if self.annotated_cv_image is None:
            return

        cw = self.canvas_main.winfo_width()
        ch = self.canvas_main.winfo_height()
        if cw < 10 or ch < 10:
            return

        orig_h, orig_w = self.annotated_cv_image.shape[:2]
        scale = min(cw / orig_w, ch / orig_h)
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))

        offset_x = (cw - new_w) // 2
        offset_y = (ch - new_h) // 2

        rgb_img = cv2.cvtColor(self.annotated_cv_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img).resize((new_w, new_h), Image.Resampling.BILINEAR)
        self.tk_main_img = ImageTk.PhotoImage(pil_img)

        self.canvas_main.delete("all")
        self.canvas_main.create_image(offset_x, offset_y, anchor="nw", image=self.tk_main_img)

    def render_cropped_plates(self):
        for widget in self.crop_scroll_frame.winfo_children():
            widget.destroy()
        self.crop_tk_widgets = []

        if not self.detected_plates:
            no_plate_frame = tk.Frame(self.crop_scroll_frame, bg=BG_PANEL, pady=40)
            no_plate_frame.pack(fill=tk.BOTH, expand=True)

            tk.Label(no_plate_frame, text="🔍 ไม่พบป้ายทะเบียนในภาพนี้", font=("Segoe UI", 12, "bold"), fg=TEXT_MUTED, bg=BG_PANEL).pack(pady=5)
            tk.Label(no_plate_frame, text="ลองปรับลดแถบ Confidence หรือเลือกภาพอื่นดูนะครับ", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_PANEL).pack()
            return

        for idx, plate in enumerate(self.detected_plates):
            card = tk.Frame(self.crop_scroll_frame, bg=BG_CARD, padx=14, pady=12, relief="solid", bd=1)
            card.pack(fill=tk.X, padx=8, pady=8)

            # Header row
            top_bar = tk.Frame(card, bg=BG_CARD)
            top_bar.pack(fill=tk.X, pady=(0, 4))

            if plate.get("is_gt"):
                badge_title = f"🎯 เฉลยจริง Ground Truth #{plate['index']}  (พิกัด 4 จุดจริง - ตรงเป๊ะ 100%)"
                title_color = "#00e5ff"
            else:
                badge_title = f"🏷️ ป้ายทะเบียน #{plate['index']}  (ตรวจจับ AI: {plate['conf']*100:.1f}%)"
                title_color = ACCENT_GREEN
            tk.Label(top_bar, text=badge_title, font=("Segoe UI", 11, "bold"), fg=title_color, bg=BG_CARD).pack(side=tk.LEFT)

            # DOF Theory Badge
            badge_homography = f"📐 {plate['dof_name']}"
            tk.Label(top_bar, text=badge_homography, font=("Segoe UI", 9, "bold"), fg="#111111", bg=ACCENT_PURPLE, padx=6, pady=2).pack(side=tk.RIGHT)

            # Description Subtitle
            lbl_desc = tk.Label(card, text=f"💡 การวิเคราะห์เส้น: {plate['dof_desc']}", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD)
            lbl_desc.pack(anchor="w", pady=(0, 6))

            # Side-by-Side Container
            comp_box = tk.Frame(card, bg=BG_CARD_INNER, padx=10, pady=10, relief="groove", bd=1)
            comp_box.pack(fill=tk.X, pady=4)

            # Left sub-column: Raw Crop
            col_raw = tk.Frame(comp_box, bg=BG_CARD_INNER)
            col_raw.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

            tk.Label(col_raw, text="1. ตัดตามมุมกล้องเดิม (Raw Crop):", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=BG_CARD_INNER).pack(anchor="w")

            raw_cv = plate["raw_crop_img"]
            if raw_cv is not None and raw_cv.size > 0:
                raw_rgb = cv2.cvtColor(raw_cv, cv2.COLOR_BGR2RGB)
                raw_pil = Image.fromarray(raw_rgb)
                rw = 180
                rh = max(20, int(raw_pil.height * (rw / max(1, raw_pil.width))))
                raw_resized = raw_pil.resize((rw, rh), Image.Resampling.BILINEAR)
                tk_raw = ImageTk.PhotoImage(raw_resized)
                self.crop_tk_widgets.append(tk_raw)

                lbl_raw_img = tk.Label(col_raw, image=tk_raw, bg="#000000", relief="solid", bd=1)
                lbl_raw_img.pack(pady=4)
                tk.Label(col_raw, text=f"ขนาด: {raw_cv.shape[1]}x{raw_cv.shape[0]} px (เอียงตามกล้อง)", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD_INNER).pack()
            else:
                tk.Label(col_raw, text="ไม่มีข้อมูลภาพ", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD_INNER).pack()

            # Arrow
            tk.Label(comp_box, text=f"➔\n{plate['dof_name'].split()[0]}", font=("Segoe UI", 9, "bold"), fg=ACCENT_BLUE, bg=BG_CARD_INNER).pack(side=tk.LEFT, padx=6)

            # Right sub-column: Homography Rectified
            col_homo = tk.Frame(comp_box, bg=BG_CARD_INNER)
            col_homo.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

            tk.Label(col_homo, text="2. ดึงกลับตรงตามทฤษฎี Homography (Theory-Rectified):", font=("Segoe UI", 9, "bold"), fg=ACCENT_GREEN, bg=BG_CARD_INNER).pack(anchor="w")

            homo_cv = plate["current_display_img"]
            homo_rgb = cv2.cvtColor(homo_cv, cv2.COLOR_BGR2RGB)
            homo_pil = Image.fromarray(homo_rgb)

            hw = 240
            hh = max(20, int(homo_pil.height * (hw / max(1, homo_pil.width))))
            homo_resized = homo_pil.resize((hw, hh), Image.Resampling.LANCZOS)
            tk_homo = ImageTk.PhotoImage(homo_resized)
            self.crop_tk_widgets.append(tk_homo)

            lbl_homo_img = tk.Label(col_homo, image=tk_homo, bg="#000000", relief="solid", bd=2)
            lbl_homo_img.pack(pady=4)

            tk.Label(col_homo, text=f"ขนาดมาตรฐาน: {homo_cv.shape[1]}x{homo_cv.shape[0]} px (หน้าตรง 100%)", font=("Segoe UI", 8, "bold"), fg=ACCENT_GREEN, bg=BG_CARD_INNER).pack()

            # Actions Bar
            action_bar = tk.Frame(card, bg=BG_CARD)
            action_bar.pack(fill=tk.X, pady=(8, 2))

            btn_rot90 = tk.Button(action_bar, text="🔄 หมุน 90°", bg=BG_PANEL, fg=TEXT_WHITE,
                                  font=("Segoe UI", 8), bd=1, relief="ridge", padx=6, pady=3,
                                  command=lambda p=plate: self.rotate_plate(p, 90))
            btn_rot90.pack(side=tk.LEFT, padx=3)

            btn_rot180 = tk.Button(action_bar, text="🔃 กลับหัว 180°", bg=BG_PANEL, fg=TEXT_WHITE,
                                   font=("Segoe UI", 8), bd=1, relief="ridge", padx=6, pady=3,
                                   command=lambda p=plate: self.rotate_plate(p, 180))
            btn_rot180.pack(side=tk.LEFT, padx=3)

            btn_flip = tk.Button(action_bar, text="🪞 พลิกซ้ายขวา", bg=BG_PANEL, fg=TEXT_WHITE,
                                 font=("Segoe UI", 8), bd=1, relief="ridge", padx=6, pady=3,
                                 command=lambda p=plate: self.flip_plate(p))
            btn_flip.pack(side=tk.LEFT, padx=3)

            btn_matrix = tk.Button(action_bar, text="📐 ดูเมทริกซ์ 3x3 & DOF", bg=BG_PANEL, fg=ACCENT_PURPLE,
                                   font=("Segoe UI", 8, "bold"), bd=1, relief="ridge", padx=6, pady=3,
                                   command=lambda p=plate: self.show_homography_matrix(p))
            btn_matrix.pack(side=tk.LEFT, padx=3)

            btn_save = tk.Button(action_bar, text="💾 บันทึกรูปป้ายนี้", bg=ACCENT_GREEN, fg="#000000",
                                 font=("Segoe UI", 8, "bold"), bd=0, padx=10, pady=3,
                                 command=lambda p=plate: self.save_single_crop(p))
            btn_save.pack(side=tk.RIGHT)

    def rotate_plate(self, plate, angle):
        if angle == 90:
            plate["current_display_img"] = cv2.rotate(plate["current_display_img"], cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            plate["current_display_img"] = cv2.rotate(plate["current_display_img"], cv2.ROTATE_180)
        self.render_cropped_plates()

    def flip_plate(self, plate):
        plate["current_display_img"] = cv2.flip(plate["current_display_img"], 1)
        self.render_cropped_plates()

    def show_homography_matrix(self, plate):
        H = plate.get("homography_matrix")
        if H is None:
            messagebox.showinfo("Homography Matrix", "ไม่มีข้อมูลเมทริกซ์")
            return

        matrix_text = f"📐 Transformation Theory Analysis:\n" + "="*50 + "\n"
        matrix_text += f"• หมวดหมู่การแปลง: {plate.get('dof_name')}\n"
        matrix_text += f"• คำอธิบายเส้น: {plate.get('dof_desc')}\n"
        matrix_text += f"• ความคลาดเคลื่อนเส้นขนาน (Parallelism Error): {plate.get('par_err', 0.0):.2f}°\n"
        matrix_text += f"• ความคลาดเคลื่อนมุมฉาก (Orthogonality Error): {plate.get('ortho_err', 0.0):.2f}°\n\n"
        matrix_text += "📐 3x3 Homography Matrix (H):\n" + "-"*50 + "\n"
        for row in H:
            matrix_text += f"[ {row[0]:12.5f}  {row[1]:12.5f}  {row[2]:12.5f} ]\n"

        matrix_text += "\n" + "="*50 + "\n"
        matrix_text += "คุณสมบัติ: แปลงพิกัด 4 จุดจากมุมมองกล้อง CCTV ไปยังระนาบป้ายมาตรฐานหน้าตรง 340 x 150 px โดยรักษาสภาพทางเรขาคณิตตามทฤษฎี"
        messagebox.showinfo("Homography Theory & Matrix (H)", matrix_text)

    def save_single_crop(self, plate):
        if not self.current_img_path:
            return
        filename = CROPS_SAVE_DIR / f"{self.current_img_path.stem}_plate_{plate['index']}_homography.png"
        cv2.imwrite(str(filename), plate["current_display_img"])
        messagebox.showinfo("บันทึกสำเร็จ", f"บันทึกภาพป้ายทะเบียน Homography เรียบร้อยที่:\n{filename}")

    def save_all_crops(self):
        if not self.detected_plates or not self.current_img_path:
            messagebox.showwarning("แจ้งเตือน", "ไม่พบป้ายทะเบียนที่ต้องการบันทึก")
            return

        count = 0
        for plate in self.detected_plates:
            filename = CROPS_SAVE_DIR / f"{self.current_img_path.stem}_plate_{plate['index']}_homography.png"
            cv2.imwrite(str(filename), plate["current_display_img"])
            count += 1

        messagebox.showinfo("บันทึกสำเร็จ", f"บันทึกภาพป้ายทะเบียน Homography {count} ป้ายเรียบร้อยใน:\npredictions/cropped_plates/")

if __name__ == "__main__":
    app = LicensePlateTestApp()
    app.mainloop()
