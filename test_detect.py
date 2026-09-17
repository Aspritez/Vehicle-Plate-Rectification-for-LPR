"""
สคริปต์ทดสอบโมเดลป้ายทะเบียน (Inference & Geometric DOF Homography Rectification)
วิเคราะห์เวกเตอร์ของเส้น (3-DOF, 4-DOF, 6-DOF, 8-DOF) เพื่อคำนวณ Homography Matrix H
ดึงภาพป้ายทะเบียนให้กลับมาตั้งตรงตามหลักทฤษฎีเรขาคณิตการแปลงภาพ (Transformation Theory)
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Reconfigure stdout for utf-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "predictions"
OUTPUT_DIR.mkdir(exist_ok=True)
CROP_DIR = OUTPUT_DIR / "cropped_plates"
CROP_DIR.mkdir(exist_ok=True)

def find_latest_model():
    models = list(BASE_DIR.glob("runs/**/best.pt"))
    root_best = BASE_DIR / "best.pt"
    if root_best.exists() and root_best not in models:
        models.append(root_best)
    models.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return models[0] if models else BASE_DIR / "best.pt"

def get_line_angle(p1, p2):
    """คำนวณมุมเอียงของเส้นเมื่อเทียบกับแกนแนวนอน [0, 90 องศา]"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = abs(np.degrees(np.arctan2(dy, dx))) % 180.0
    if angle > 90.0:
        angle = 180.0 - angle
    return angle

def order_plate_corners_physical(pts):
    pts = np.array(pts, dtype="float32")
    angle_A = (get_line_angle(pts[0], pts[1]) + get_line_angle(pts[2], pts[3])) / 2.0
    angle_B = (get_line_angle(pts[1], pts[2]) + get_line_angle(pts[3], pts[0])) / 2.0

    if angle_A <= angle_B:
        edge_1 = (pts[0], pts[1])
        edge_2 = (pts[3], pts[2])
    else:
        edge_1 = (pts[1], pts[2])
        edge_2 = (pts[0], pts[3])

    y_1 = (edge_1[0][1] + edge_1[1][1]) / 2.0
    y_2 = (edge_2[0][1] + edge_2[1][1]) / 2.0

    top_edge = edge_1 if y_1 < y_2 else edge_2
    bottom_edge = edge_2 if y_1 < y_2 else edge_1

    tl = top_edge[0] if top_edge[0][0] < top_edge[1][0] else top_edge[1]
    tr = top_edge[1] if top_edge[0][0] < top_edge[1][0] else top_edge[0]

    bl = bottom_edge[0] if bottom_edge[0][0] < bottom_edge[1][0] else bottom_edge[1]
    br = bottom_edge[1] if bottom_edge[0][0] < bottom_edge[1][0] else bottom_edge[0]

    return np.array([tl, tr, br, bl], dtype="float32")

def rectify_plate_dof(image, pts, target_size=(340, 150), mode="Auto"):
    rect = order_plate_corners_physical(pts)
    (tl, tr, br, bl) = rect

    target_w, target_h = target_size
    dst_pts = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype="float32")

    # Initial Homography
    H = cv2.getPerspectiveTransform(rect, dst_pts)
    warp_init = cv2.warpPerspective(image, H, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # คำนวณ Full 8-DOF Homography Matrix
    H = cv2.getPerspectiveTransform(rect, dst_pts)

    r_tl, r_tr, r_br, r_bl = rect
    v_top = r_tr - r_tl
    v_bot = r_br - r_bl
    v_left = r_bl - r_tl
    v_right = r_br - r_tr

    cos_h = abs(np.dot(v_top, v_bot)) / (np.linalg.norm(v_top) * np.linalg.norm(v_bot) + 1e-6)
    delta_h = np.degrees(np.arccos(np.clip(cos_h, 0.0, 1.0)))
    cos_v = abs(np.dot(v_left, v_right)) / (np.linalg.norm(v_left) * np.linalg.norm(v_right) + 1e-6)
    delta_v = np.degrees(np.arccos(np.clip(cos_v, 0.0, 1.0)))
    max_par_err = max(delta_h, delta_v)

    cos_corner = abs(np.dot(v_top, v_left)) / (np.linalg.norm(v_top) * np.linalg.norm(v_left) + 1e-6)
    ortho_err = abs(np.degrees(np.arccos(np.clip(cos_corner, 0.0, 1.0))) - 90.0)

    if "8-DOF" in mode or (mode == "Auto"):
        dof_name = "8-DOF (Projective)"
        rectified = cv2.warpPerspective(image, H, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    elif "6-DOF" in mode:
        dof_name = "6-DOF (Affine)"
        M_aff = cv2.getAffineTransform(rect[:3], dst_pts[:3])
        H = np.vstack([M_aff, [0.0, 0.0, 1.0]])
        rectified = cv2.warpAffine(image, M_aff, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    elif "3-DOF" in mode:
        dof_name = "3-DOF (Euclidean)"
        M_sim, _ = cv2.estimateAffinePartial2D(rect, dst_pts)
        if M_sim is not None:
            H = np.vstack([M_sim, [0.0, 0.0, 1.0]])
            rectified = cv2.warpAffine(image, M_sim, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            rectified = cv2.warpPerspective(image, H, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        dof_name = "4-DOF (Similarity)"
        M_sim, _ = cv2.estimateAffinePartial2D(rect, dst_pts)
        if M_sim is not None:
            H = np.vstack([M_sim, [0.0, 0.0, 1.0]])
            rectified = cv2.warpAffine(image, M_sim, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            rectified = cv2.warpPerspective(image, H, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # ปรับระนาบตัวหนังสือในกรณีสี่เหลี่ยมผืนผ้า OBB
    if max_par_err < 1.5 and ortho_err < 1.5:
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
        else:
            rectified = cv2.warpPerspective(image, H, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if rectified.shape[0] > rectified.shape[1]:
        rectified = cv2.rotate(rectified, cv2.ROTATE_90_CLOCKWISE)

    return rectified, H, dof_name, max_par_err, ortho_err

def test_model(image_path=None):
    model_path = find_latest_model()
    if not model_path.exists():
        print(f"❌ ไม่พบไฟล์โมเดลที่: {model_path}")
        return

    print(f"[*] โหลดโมเดลล่าสุด: {model_path}")
    model = YOLO(str(model_path))

    if image_path is None:
        raw_images = list((BASE_DIR / "data" / "raw_images").glob("*.png"))
        if not raw_images:
            raw_images = list((BASE_DIR / "data" / "raw_images").glob("*.jpg"))
        test_images = raw_images[35:40] if len(raw_images) > 40 else raw_images[:5]
    else:
        test_images = [Path(image_path)]

    print(f"[*] กำลังทดสอบตรวจจับ {len(test_images)} รูปด้วย Geometric DOF Homography...")

    for img_p in test_images:
        orig_img = cv2.imread(str(img_p))
        if orig_img is None:
            continue

        results = model.predict(source=str(img_p), conf=0.25, save=False, verbose=False)
        annotated_img = orig_img.copy()

        plate_count = 0
        for r in results:
            if r.obb is not None and len(r.obb) > 0:
                xyxyxyxy = r.obb.xyxyxyxy.cpu().numpy()
                confs = r.obb.conf.cpu().numpy()

                for box, conf in zip(xyxyxyxy, confs):
                    pts = np.int32(box)
                    plate_count += 1

                    # วาดกรอบ OBB
                    cv2.polylines(annotated_img, [pts], isClosed=True, color=(0, 255, 100), thickness=3)

                    # ดึงตรงตามทฤษฎีเรขาคณิต DOF
                    rectified_plate, H, dof_name, par_err, ortho_err = rectify_plate_dof(orig_img, box)

                    tag_text = f"Plate: {conf*100:.1f}% ({dof_name})"
                    cv2.putText(annotated_img, tag_text, (pts[0][0], max(20, pts[0][1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

                    if rectified_plate is not None:
                        crop_filename = CROP_DIR / f"{img_p.stem}_dof_{plate_count}.png"
                        cv2.imwrite(str(crop_filename), rectified_plate)
                        print(f"   📐 [{dof_name}] ดึงตรงสำเร็จ: {crop_filename.name} (ParErr={par_err:.2f}°, OrthoErr={ortho_err:.2f}°)")

        out_path = OUTPUT_DIR / f"result_{img_p.name}"
        cv2.imwrite(str(out_path), annotated_img)
        print(f"✅ ตรวจจับ {img_p.name}: พบป้าย {plate_count} ป้าย -> บันทึกไว้ที่ predictions/{out_path.name}")

    print("\n🎉 ทดสอบเสร็จสิ้น! ดูผลลัพธ์ Homography ได้ที่โฟลเดอร์ 'predictions/cropped_plates/'")

if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    test_model(img_arg)
