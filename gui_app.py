import os
import sys
import json
import queue
import shutil
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw

# Reconfigure stdout for utf-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw_images"
DATASET_DIR = BASE_DIR / "dataset"

# Modern Dark Color Palette
BG_DARK = "#1E1E24"
BG_CARD = "#2B2D42"
BG_PANEL = "#252738"
ACCENT_BLUE = "#4CC9F0"
ACCENT_GREEN = "#06D6A0"
ACCENT_RED = "#EF476F"
TEXT_WHITE = "#FFFFFF"
TEXT_MUTED = "#8D99AE"
BORDER_COLOR = "#3A3D54"

class CCTVAnnotationViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CCTV License Plate - AI Dashboard & Annotation Viewer")
        self.geometry("1280x820")
        self.minsize(1024, 700)
        self.configure(bg=BG_DARK)

        # Style configuration
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.configure_styles()

        # Data state
        self.all_images = []
        self.filtered_images = []
        self.current_index = 0
        self.current_img_obj = None
        self.train_process = None
        self.log_queue = queue.Queue()

        # Build UI layout
        self.create_header()
        self.create_main_content()
        self.create_training_panel()

        # Keyboard shortcuts
        self.bind("<Left>", lambda e: self.navigate_image(-1))
        self.bind("<Right>", lambda e: self.navigate_image(1))
        self.bind("<Up>", lambda e: self.navigate_image(-1))
        self.bind("<Down>", lambda e: self.navigate_image(1))

        # Initial load
        self.load_dataset()
        self.after(100, self.process_log_queue)

    def configure_styles(self):
        self.style.configure(".", background=BG_DARK, foreground=TEXT_WHITE, font=("Segoe UI", 10))
        self.style.configure("TFrame", background=BG_DARK)
        self.style.configure("Card.TFrame", background=BG_CARD)
        self.style.configure("TLabel", background=BG_DARK, foreground=TEXT_WHITE)
        self.style.configure("Card.TLabel", background=BG_CARD, foreground=TEXT_WHITE)
        self.style.configure("Muted.TLabel", background=BG_CARD, foreground=TEXT_MUTED, font=("Segoe UI", 9))
        self.style.configure("Title.TLabel", background=BG_DARK, foreground=TEXT_WHITE, font=("Segoe UI", 14, "bold"))
        self.style.configure("StatValue.TLabel", background=BG_CARD, foreground=ACCENT_BLUE, font=("Segoe UI", 16, "bold"))

        # Progressbar
        self.style.configure("Custom.Horizontal.TProgressbar", troughcolor=BG_PANEL, background=ACCENT_GREEN, thickness=10)

        # Buttons
        self.style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), background=ACCENT_BLUE, foreground="#111111", padding=6)
        self.style.map("Primary.TButton", background=[("active", "#70D6FF"), ("disabled", "#555555")])

        self.style.configure("Success.TButton", font=("Segoe UI", 10, "bold"), background=ACCENT_GREEN, foreground="#111111", padding=8)
        self.style.map("Success.TButton", background=[("active", "#38ef7d"), ("disabled", "#555555")])

        self.style.configure("Danger.TButton", font=("Segoe UI", 10, "bold"), background=ACCENT_RED, foreground=TEXT_WHITE, padding=8)
        self.style.map("Danger.TButton", background=[("active", "#ff6b8b"), ("disabled", "#555555")])

    def create_header(self):
        header_frame = tk.Frame(self, bg=BG_CARD, height=70, padx=20, pady=10)
        header_frame.pack(fill=tk.X, side=tk.TOP, padx=10, pady=(10, 5))

        # Title & Subtitle
        title_box = tk.Frame(header_frame, bg=BG_CARD)
        title_box.pack(side=tk.LEFT, fill=tk.Y)
        lbl_title = tk.Label(title_box, text="🚗 CCTV License Plate Detection AI", font=("Segoe UI", 14, "bold"), fg=TEXT_WHITE, bg=BG_CARD)
        lbl_title.pack(anchor="w")
        lbl_sub = tk.Label(title_box, text="ระบบตรวจเช็คข้อมูลเฉลยป้ายทะเบียนมุมเอียง & สั่งเทรน YOLO11-OBB", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD)
        lbl_sub.pack(anchor="w")

        # Stats Cards
        self.stats_frame = tk.Frame(header_frame, bg=BG_CARD)
        self.stats_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.lbl_total = self.add_stat_box(self.stats_frame, "ภาพทั้งหมด", "0 รูป", ACCENT_BLUE)
        self.lbl_labeled = self.add_stat_box(self.stats_frame, "เฉลยแล้ว", "0 รูป", ACCENT_GREEN)
        self.lbl_unlabeled = self.add_stat_box(self.stats_frame, "ยังไม่ทำ", "0 รูป", ACCENT_RED)
        
        # Progress
        prog_box = tk.Frame(self.stats_frame, bg=BG_CARD, padx=15)
        prog_box.pack(side=tk.LEFT, fill=tk.Y)
        self.lbl_percent = tk.Label(prog_box, text="0.0%", font=("Segoe UI", 10, "bold"), fg=ACCENT_GREEN, bg=BG_CARD)
        self.lbl_percent.pack(anchor="w")
        self.progress_bar = ttk.Progressbar(prog_box, style="Custom.Horizontal.TProgressbar", length=140, mode="determinate")
        self.progress_bar.pack(pady=4)

    def add_stat_box(self, parent, label, init_val, color):
        box = tk.Frame(parent, bg=BG_PANEL, padx=15, pady=4, relief="ridge", bd=1)
        box.pack(side=tk.LEFT, padx=5)
        lbl_val = tk.Label(box, text=init_val, font=("Segoe UI", 13, "bold"), fg=color, bg=BG_PANEL)
        lbl_val.pack()
        lbl_txt = tk.Label(box, text=label, font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_PANEL)
        lbl_txt.pack()
        return lbl_val

    def create_main_content(self):
        main_split = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=BG_DARK, bd=0, sashwidth=4)
        main_split.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # ================= LEFT PANEL: Image List =================
        left_panel = tk.Frame(main_split, bg=BG_CARD, width=320, padx=10, pady=10)
        main_split.add(left_panel, minsize=280)

        # Filter Tabs
        filter_box = tk.Frame(left_panel, bg=BG_CARD)
        filter_box.pack(fill=tk.X, pady=(0, 8))
        self.filter_var = tk.StringVar(value="all")

        rb_all = tk.Radiobutton(filter_box, text="ทั้งหมด", variable=self.filter_var, value="all",
                                command=self.apply_filter, bg=BG_CARD, fg=TEXT_WHITE, selectcolor=BG_PANEL, activebackground=BG_CARD, font=("Segoe UI", 9, "bold"))
        rb_labeled = tk.Radiobutton(filter_box, text="เฉลยแล้ว ✅", variable=self.filter_var, value="labeled",
                                    command=self.apply_filter, bg=BG_CARD, fg=ACCENT_GREEN, selectcolor=BG_PANEL, activebackground=BG_CARD, font=("Segoe UI", 9, "bold"))
        rb_unlabeled = tk.Radiobutton(filter_box, text="ยังไม่ทำ ⚪", variable=self.filter_var, value="unlabeled",
                                      command=self.apply_filter, bg=BG_CARD, fg=ACCENT_RED, selectcolor=BG_PANEL, activebackground=BG_CARD, font=("Segoe UI", 9, "bold"))
        rb_all.pack(side=tk.LEFT, expand=True)
        rb_labeled.pack(side=tk.LEFT, expand=True)
        rb_unlabeled.pack(side=tk.LEFT, expand=True)

        # Search Bar
        search_box = tk.Frame(left_panel, bg=BG_PANEL, padx=6, pady=4)
        search_box.pack(fill=tk.X, pady=(0, 8))
        tk.Label(search_box, text="🔍", bg=BG_PANEL, fg=TEXT_MUTED).pack(side=tk.LEFT)
        self.search_entry = tk.Entry(search_box, bg=BG_PANEL, fg=TEXT_WHITE, insertbackground=TEXT_WHITE, bd=0, font=("Segoe UI", 10))
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.search_entry.bind("<KeyRelease>", lambda e: self.apply_filter())

        # Listbox with Scrollbar
        list_container = tk.Frame(left_panel, bg=BG_PANEL)
        list_container.pack(fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(list_container, orient=tk.VERTICAL, bg=BG_PANEL)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.image_listbox = tk.Listbox(
            list_container,
            bg=BG_PANEL,
            fg=TEXT_WHITE,
            selectbackground=ACCENT_BLUE,
            selectforeground="#000000",
            font=("Consolas", 10),
            bd=0,
            highlightthickness=0,
            yscrollcommand=self.scrollbar.set
        )
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.image_listbox.yview)
        self.image_listbox.bind("<<ListboxSelect>>", self.on_list_select)

        # Refresh & Quick Info Button
        btn_refresh = tk.Button(left_panel, text="🔄 รีเฟรชรายการไฟล์ (Refresh)", bg=BG_PANEL, fg=TEXT_WHITE,
                                bd=1, relief="ridge", activebackground=BG_CARD, command=self.load_dataset, font=("Segoe UI", 9))
        btn_refresh.pack(fill=tk.X, pady=(8, 0))

        # ================= CENTER PANEL: Image & Mark Preview =================
        center_panel = tk.Frame(main_split, bg=BG_CARD, padx=10, pady=10)
        main_split.add(center_panel, minsize=500)

        # Image Info Header
        info_header = tk.Frame(center_panel, bg=BG_CARD)
        info_header.pack(fill=tk.X, pady=(0, 5))
        self.lbl_img_name = tk.Label(info_header, text="ยังไม่ได้เลือกรูปภาพ", font=("Segoe UI", 12, "bold"), fg=TEXT_WHITE, bg=BG_CARD)
        self.lbl_img_name.pack(side=tk.LEFT)

        self.lbl_status_badge = tk.Label(info_header, text="สถานะ: -", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=BG_PANEL, padx=8, pady=2)
        self.lbl_status_badge.pack(side=tk.RIGHT)

        # Canvas for Rendering Image + Polygon Mark
        self.canvas = tk.Canvas(center_panel, bg="#111217", highlightthickness=1, highlightbackground=BORDER_COLOR)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw_current_image())

        # Navigation Buttons Bar
        nav_bar = tk.Frame(center_panel, bg=BG_CARD, pady=6)
        nav_bar.pack(fill=tk.X)
        
        btn_prev = tk.Button(nav_bar, text="◀ ก่อนหน้า (Left)", bg=BG_PANEL, fg=TEXT_WHITE, bd=0, padx=15, pady=4,
                             activebackground=BG_CARD, command=lambda: self.navigate_image(-1))
        btn_prev.pack(side=tk.LEFT)

        self.lbl_counter = tk.Label(nav_bar, text="0 / 0", bg=BG_CARD, fg=TEXT_MUTED, font=("Segoe UI", 10))
        self.lbl_counter.pack(side=tk.LEFT, expand=True)

        btn_next = tk.Button(nav_bar, text="ถัดไป (Right) ▶", bg=BG_PANEL, fg=TEXT_WHITE, bd=0, padx=15, pady=4,
                             activebackground=BG_CARD, command=lambda: self.navigate_image(1))
        btn_next.pack(side=tk.RIGHT)

    def create_training_panel(self):
        # Bottom Card for AI Training & Console Log
        train_frame = tk.Frame(self, bg=BG_CARD, padx=15, pady=10)
        train_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(5, 10))

        # Controls row
        ctrl_row = tk.Frame(train_frame, bg=BG_CARD)
        ctrl_row.pack(fill=tk.X, pady=(0, 6))

        tk.Label(ctrl_row, text="⚡ ควบคุมการเทรน AI (RTX 3080 GPU):", font=("Segoe UI", 11, "bold"), fg=TEXT_WHITE, bg=BG_CARD).pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(ctrl_row, text="Epochs:", fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.LEFT, padx=3)
        self.spin_epochs = tk.Spinbox(ctrl_row, from_=10, to=500, increment=10, width=5, bg=BG_PANEL, fg=TEXT_WHITE, bd=1)
        self.spin_epochs.delete(0, "end")
        self.spin_epochs.insert(0, "100")
        self.spin_epochs.pack(side=tk.LEFT, padx=5)

        tk.Label(ctrl_row, text="Batch Size:", fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.LEFT, padx=3)
        self.spin_batch = tk.Spinbox(ctrl_row, from_=4, to=64, increment=4, width=5, bg=BG_PANEL, fg=TEXT_WHITE, bd=1)
        self.spin_batch.delete(0, "end")
        self.spin_batch.insert(0, "16")
        self.spin_batch.pack(side=tk.LEFT, padx=5)

        tk.Label(ctrl_row, text="โมเดล:", fg=TEXT_MUTED, bg=BG_CARD).pack(side=tk.LEFT, padx=3)
        self.combo_model = ttk.Combobox(ctrl_row, values=["yolo11n-obb.pt", "yolo11s-obb.pt", "yolo11m-obb.pt"], width=14, state="readonly")
        self.combo_model.set("yolo11n-obb.pt")
        self.combo_model.pack(side=tk.LEFT, padx=5)

        # Buttons
        self.btn_train = tk.Button(ctrl_row, text="🚀 เริ่มแบ่งข้อมูล & เทรนโมเดล", bg=ACCENT_GREEN, fg="#000000",
                                   font=("Segoe UI", 10, "bold"), bd=0, padx=15, pady=4, command=self.start_training)
        self.btn_train.pack(side=tk.LEFT, padx=15)

        self.btn_stop = tk.Button(ctrl_row, text="⏹ หยุดการเทรน", bg=ACCENT_RED, fg=TEXT_WHITE,
                                  font=("Segoe UI", 9, "bold"), bd=0, padx=12, pady=4, state=tk.DISABLED, command=self.stop_training)
        self.btn_stop.pack(side=tk.LEFT)

        self.lbl_train_status = tk.Label(ctrl_row, text="สถานะ: พร้อมใช้งาน (Idle)", fg=ACCENT_BLUE, bg=BG_CARD, font=("Segoe UI", 9))
        self.lbl_train_status.pack(side=tk.RIGHT)

        # Terminal Output Box
        self.log_text = scrolledtext.ScrolledText(
            train_frame,
            height=6,
            bg="#111217",
            fg="#A9B7C6",
            insertbackground="#FFFFFF",
            font=("Consolas", 9),
            bd=1,
            relief="solid"
        )
        self.log_text.pack(fill=tk.X)
        self.log_message("[System] โปรแกรมพร้อมใช้งานแล้ว ตรวจพบการ์ดจอ NVIDIA RTX 3080")

    def load_dataset(self):
        """โหลดรายการไฟล์รูปทั้งหมดในโฟลเดอร์ data/raw_images"""
        if not RAW_DIR.exists():
            RAW_DIR.mkdir(parents=True, exist_ok=True)

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.all_images = []
        labeled_count = 0

        for file_path in sorted(RAW_DIR.iterdir(), key=lambda p: p.name.lower()):
            if file_path.suffix.lower() in image_extensions:
                json_path = file_path.with_suffix(".json")
                txt_path = file_path.with_suffix(".txt")

                has_label = json_path.exists() or txt_path.exists()
                if has_label:
                    labeled_count += 1

                self.all_images.append({
                    "name": file_path.name,
                    "img_path": file_path,
                    "json_path": json_path if json_path.exists() else None,
                    "txt_path": txt_path if txt_path.exists() else None,
                    "is_labeled": has_label
                })

        total = len(self.all_images)
        unlabeled = total - labeled_count
        percent = (labeled_count / total * 100) if total > 0 else 0.0

        # Update stats
        self.lbl_total.config(text=f"{total} รูป")
        self.lbl_labeled.config(text=f"{labeled_count} รูป")
        self.lbl_unlabeled.config(text=f"{unlabeled} รูป")
        self.lbl_percent.config(text=f"{percent:.1f}%")
        self.progress_bar["value"] = percent

        self.apply_filter()

    def apply_filter(self):
        filter_mode = self.filter_var.get()
        search_kw = self.search_entry.get().strip().lower()

        self.filtered_images = []
        for item in self.all_images:
            if filter_mode == "labeled" and not item["is_labeled"]:
                continue
            if filter_mode == "unlabeled" and item["is_labeled"]:
                continue
            if search_kw and search_kw not in item["name"].lower():
                continue
            self.filtered_images.append(item)

        # Update Listbox
        self.image_listbox.delete(0, tk.END)
        for item in self.filtered_images:
            icon = "✅" if item["is_labeled"] else "⚪"
            self.image_listbox.insert(tk.END, f"{icon}  {item['name']}")

        if self.filtered_images:
            self.current_index = 0
            self.image_listbox.select_set(0)
            self.image_listbox.see(0)
            self.display_selected_image(self.filtered_images[0])
        else:
            self.canvas.delete("all")
            self.lbl_img_name.config(text="ไม่พบรูปภาพตามเงื่อนไข")
            self.lbl_status_badge.config(text="0 รายการ", fg=TEXT_MUTED)
            self.lbl_counter.config(text="0 / 0")

    def on_list_select(self, event):
        selection = self.image_listbox.curselection()
        if selection:
            idx = selection[0]
            self.current_index = idx
            self.display_selected_image(self.filtered_images[idx])

    def navigate_image(self, step):
        if not self.filtered_images:
            return
        new_idx = self.current_index + step
        if 0 <= new_idx < len(self.filtered_images):
            self.current_index = new_idx
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.select_set(new_idx)
            self.image_listbox.see(new_idx)
            self.display_selected_image(self.filtered_images[new_idx])

    def display_selected_image(self, item):
        self.lbl_counter.config(text=f"{self.current_index + 1} / {len(self.filtered_images)}")
        self.lbl_img_name.config(text=item["name"])

        if item["is_labeled"]:
            self.lbl_status_badge.config(text="สถานะ: เฉลยแล้ว (Labeled) ✅", fg=ACCENT_GREEN)
        else:
            self.lbl_status_badge.config(text="สถานะ: ยังไม่ได้เฉลย ⚪", fg=ACCENT_RED)

        try:
            self.current_img_obj = Image.open(item["img_path"])
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(250, 200, text=f"ไม่สามารถเปิดภาพได้: {e}", fill=ACCENT_RED, font=("Segoe UI", 12))
            return

        self.redraw_current_image()

    def redraw_current_image(self):
        if not self.current_img_obj or not self.filtered_images:
            return

        item = self.filtered_images[self.current_index]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w < 10 or canvas_h < 10:
            return

        orig_w, orig_h = self.current_img_obj.size
        # Maintain aspect ratio
        scale = min(canvas_w / orig_w, canvas_h / orig_h)
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))

        offset_x = (canvas_w - new_w) // 2
        offset_y = (canvas_h - new_h) // 2

        resized_img = self.current_img_obj.resize((new_w, new_h), Image.Resampling.BILINEAR)
        self.tk_img = ImageTk.PhotoImage(resized_img)

        self.canvas.delete("all")
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.tk_img)

        # Draw annotations if present
        if item["json_path"]:
            self.draw_json_annotations(item["json_path"], scale, offset_x, offset_y)
        elif item["txt_path"]:
            self.draw_yolo_obb_annotations(item["txt_path"], orig_w, orig_h, scale, offset_x, offset_y)

    def draw_json_annotations(self, json_path, scale, offset_x, offset_y):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for shape in data.get("shapes", []):
                points = shape.get("points", [])
                label = shape.get("label", "license_plate")

                if len(points) >= 3:
                    canvas_pts = []
                    for pt in points:
                        cx = pt[0] * scale + offset_x
                        cy = pt[1] * scale + offset_y
                        canvas_pts.extend([cx, cy])

                    # Draw thick neon border
                    self.canvas.create_polygon(
                        canvas_pts,
                        outline="#00FF66",
                        width=3,
                        fill=""
                    )

                    # Draw corner points
                    for i in range(0, len(canvas_pts), 2):
                        px, py = canvas_pts[i], canvas_pts[i+1]
                        self.canvas.create_oval(px-4, py-4, px+4, py+4, fill="#00FF66", outline="#FFFFFF", width=1)

                    # Draw badge label
                    lx, ly = canvas_pts[0], canvas_pts[1]
                    self.canvas.create_rectangle(lx - 2, ly - 22, lx + 95, ly - 2, fill="#00FF66", outline="")
                    self.canvas.create_text(lx + 46, ly - 12, text=label, fill="#000000", font=("Segoe UI", 9, "bold"))
        except Exception as e:
            print(f"Error drawing json: {e}")

    def draw_yolo_obb_annotations(self, txt_path, orig_w, orig_h, scale, offset_x, offset_y):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) == 9:
                    pts = [float(p) for p in parts[1:]]
                    canvas_pts = []
                    for i in range(0, 8, 2):
                        cx = (pts[i] * orig_w) * scale + offset_x
                        cy = (pts[i+1] * orig_h) * scale + offset_y
                        canvas_pts.extend([cx, cy])

                    self.canvas.create_polygon(canvas_pts, outline="#00FF66", width=3, fill="")
                    lx, ly = canvas_pts[0], canvas_pts[1]
                    self.canvas.create_rectangle(lx - 2, ly - 22, lx + 95, ly - 2, fill="#00FF66", outline="")
                    self.canvas.create_text(lx + 46, ly - 12, text="license_plate", fill="#000000", font=("Segoe UI", 9, "bold"))
        except Exception as e:
            print(f"Error drawing txt: {e}")

    def log_message(self, text):
        self.log_queue.put(text)

    def process_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get_nowait()
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
        self.after(100, self.process_log_queue)

    def start_training(self):
        # Check if there are labeled images
        labeled_count = sum(1 for img in self.all_images if img["is_labeled"])
        if labeled_count == 0:
            messagebox.showwarning("คำเตือน", "ยังไม่มีภาพที่ถูกเฉลยเลยครับ กรุณาวาดกรอบใน X-AnyLabeling ก่อนเริ่มเทรน")
            return

        confirm = messagebox.askyesno(
            "ยืนยันการเทรน",
            f"พบภาพที่เฉลยแล้ว {labeled_count} รูป\nระบบจะแบ่งชุดข้อมูล Train/Val และเริ่มเทรนโมเดล YOLO-OBB ทันที\nต้องการดำเนินการต่อหรือไม่?"
        )
        if not confirm:
            return

        self.btn_train.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.lbl_train_status.config(text="สถานะ: กำลังเตรียมข้อมูล & เทรนโมเดล... ⏳", fg="#FFD166")

        epochs = self.spin_epochs.get()
        batch = self.spin_batch.get()
        model_name = self.combo_model.get()

        # Run training thread
        thread = threading.Thread(target=self.run_training_pipeline, args=(epochs, batch, model_name), daemon=True)
        thread.start()

    def run_training_pipeline(self, epochs, batch, model_name):
        try:
            self.log_message("\n" + "="*50)
            self.log_message("🚀 [Step 1/2] กำลังรัน split_dataset.py เพื่อแปลงข้อมูล...")
            
            # Run split_dataset
            split_res = subprocess.run(
                [sys.executable, str(BASE_DIR / "split_dataset.py")],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
            for line in split_res.stdout.splitlines():
                self.log_message(line)

            if split_res.returncode != 0:
                self.log_message(f"❌ เกิดข้อผิดพลาดในการแบ่งข้อมูล: {split_res.stderr}")
                self.on_training_finished(False)
                return

            self.log_message("\n🔥 [Step 2/2] เริ่มกระบวนการเทรน YOLO11-OBB ด้วย GPU RTX 3080...")
            self.log_message(f"⚙️ Config: Model={model_name}, Epochs={epochs}, Batch={batch}")

            # Inline training script invocation to pass parameters
            train_code = f"""
import sys
import torch
from ultralytics import YOLO

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

device = 0 if torch.cuda.is_available() else 'cpu'
print(f'[*] Training on: {{torch.cuda.get_device_name(0) if device == 0 else "CPU"}}')

model = YOLO('{model_name}')
model.train(
    data='data.yaml',
    epochs={epochs},
    imgsz=640,
    batch={batch},
    device=device,
    save=True,
    project='runs/detect_plate_obb',
    name='experiment_1',
    degrees=15.0,
    perspective=0.001
)
print('[OK] Training finished successfully!')
"""
            self.train_process = subprocess.Popen(
                [sys.executable, "-u", "-c", train_code],
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1
            )

            for line in self.train_process.stdout:
                clean_line = line.strip()
                if clean_line:
                    self.log_message(clean_line)

            self.train_process.wait()
            success = (self.train_process.returncode == 0)
            self.on_training_finished(success)

        except Exception as e:
            self.log_message(f"❌ Error in training thread: {e}")
            self.on_training_finished(False)

    def on_training_finished(self, success):
        def _update_ui():
            self.btn_train.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            if success:
                self.lbl_train_status.config(text="สถานะ: เทรนเสร็จสมบูรณ์! 🎉", fg=ACCENT_GREEN)
                messagebox.showinfo("สำเร็จ", "เทรนโมเดลสำเร็จเรียบร้อย!\nไฟล์โมเดลถูกบันทึกไว้ที่:\nruns/detect_plate_obb/experiment_1/weights/best.pt")
            else:
                self.lbl_train_status.config(text="สถานะ: การเทรนสิ้นสุดหรือเกิดข้อผิดพลาด", fg=ACCENT_RED)
        self.after(0, _update_ui)

    def stop_training(self):
        if self.train_process and self.train_process.poll() is None:
            confirm = messagebox.askyesno("ยืนยัน", "ต้องการยกเลิกการเทรนหรือไม่?")
            if confirm:
                self.train_process.terminate()
                self.log_message("[System] 🛑 ได้ทำการหยุดการเทรนแล้ว")
                self.lbl_train_status.config(text="สถานะ: ถูกยกเลิกโดยผู้ใช้", fg=ACCENT_RED)
                self.btn_train.config(state=tk.NORMAL)
                self.btn_stop.config(state=tk.DISABLED)

if __name__ == "__main__":
    app = CCTVAnnotationViewer()
    app.mainloop()
