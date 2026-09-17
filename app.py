"""Streamlit entry point. User images stay in this browser session's memory."""

import hashlib
import json
from time import perf_counter

import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from src.detection import auto_rectify, detect_candidates, web_detector_config
from src.geometry import ordered_quad, rectify_corners
from src.imaging import decode_image, png_bytes, prepare_ocr, resized
from src.matching import match_rectify

st.set_page_config(page_title="PlateLab · ปรับมุมป้ายทะเบียน", page_icon="▱", layout="wide")
st.markdown("""
<style>
 .stApp {background: #f5f7fb;}
 .block-container {max-width: 1260px; padding-top: 2.4rem; padding-bottom: 3rem;}
 h1,h2,h3 {letter-spacing: -.025em; color: #102b3f;}
 .eyebrow {font-size: .78rem; font-weight: 700; letter-spacing: .18em; color: #087f8c; margin-bottom: .4rem;}
 .intro {color: #556779; font-size: 1.05rem; max-width: 760px; line-height: 1.8;}
 div[data-testid="stMetric"] {background: white; padding: 1rem; border: 1px solid #e0e7ed; border-radius: 12px;}
 div[data-testid="stFileUploader"] {background: white; border-radius: 12px;}
 .footer {color: #778695; font-size: .8rem; margin-top: 2rem; border-top: 1px solid #dce4ed; padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)


def forget_result():
    st.session_state.pop("result", None)


def image_upload(label, key):
    upload = st.file_uploader(label, type=["jpg", "jpeg", "png"], key=key,
                              max_upload_size=10, help="JPG/PNG ไม่เกิน 10 MB และ 24 ล้านพิกเซล")
    if upload is None:
        return None, None
    data = upload.getvalue()
    token = hashlib.sha256(data).hexdigest()[:16]
    cache_key = f"decoded_{key}"
    if st.session_state.get(cache_key, {}).get("token") != token:
        try:
            st.session_state[cache_key] = {"token": token, "image": decode_image(data)}
        except ValueError as error:
            st.error(str(error))
            return None, None
    return st.session_state[cache_key]["image"], token


def select_region(image, prefix, title):
    st.markdown(f"**{title}**")
    c1, c2 = st.columns([1, 2])
    with c2:
        with st.expander("ตั้งค่าค้นหาขอบ"):
            low = st.slider("Canny ต่ำ", 5, 150, 40, 5, key=f"{prefix}_low")
            high = st.slider("Canny สูง", 50, 250, 140, 5, key=f"{prefix}_high")
    with c1:
        if st.button("ค้นหาป้าย", key=f"{prefix}_detect", width="stretch"):
            with st.spinner("กำลังค้นหารูปทรงที่อาจเป็นป้าย…"):
                candidates, edges = detect_candidates(image, low, high)
            st.session_state[f"{prefix}_detections"] = candidates
            st.session_state[f"{prefix}_edges"] = edges
            forget_result()
    detections = st.session_state.get(f"{prefix}_detections", [])
    candidate = None
    choice = st.selectbox("บริเวณที่จะใช้", [-1, *range(len(detections))],
                           format_func=lambda i: "เลือกพื้นที่เอง / ใช้ภาพเต็ม" if i == -1 else f"บริเวณ {i + 1} · คะแนนจัดอันดับ {detections[i].score:.2f}",
                           key=f"{prefix}_choice")
    if f"{prefix}_detections" in st.session_state and not detections:
        st.info("ยังไม่พบรูปทรงที่เหมาะสม ใช้ภาพเต็มหรือปรับกรอบด้านล่างได้")
    if choice != -1:
        candidate = detections[choice]
        x, y, w, h = candidate.box
    else:
        h, w = image.shape[:2]
        x, y = 0, 0
    # Per-selection keys avoid retaining coordinates from a different candidate.
    suffix = f"{prefix}_roi_{choice}"
    with st.expander("ปรับกรอบบริเวณป้าย", expanded=False):
        st.caption("พิกัดอ้างอิงภาพต้นฉบับหลังปรับทิศทาง EXIF กรอบควรครอบคลุมป้ายทั้งหมด")
        cols = st.columns(4)
        x0 = int(cols[0].number_input("ซ้าย (x)", 0, image.shape[1] - 16, x, key=f"{suffix}_x"))
        y0 = int(cols[1].number_input("บน (y)", 0, image.shape[0] - 16, y, key=f"{suffix}_y"))
        x1 = int(cols[2].number_input("ขวา", 16, image.shape[1], x + w, key=f"{suffix}_r"))
        y1 = int(cols[3].number_input("ล่าง", 16, image.shape[0], y + h, key=f"{suffix}_b"))
    if x1 - x0 < 16 or y1 - y0 < 16:
        st.error("กรอบต้องกว้างและสูงอย่างน้อย 16 พิกเซล และขอบขวา/ล่างต้องมากกว่าซ้าย/บน")
        return None, None, None
    preview, scale = resized(image, 900)
    preview = preview.copy()
    for i, item in enumerate(detections):
        bx, by, bw, bh = item.box
        a = tuple(np.round([bx * scale[0, 0], by * scale[1, 1]]).astype(int))
        b = tuple(np.round([(bx + bw) * scale[0, 0], (by + bh) * scale[1, 1]]).astype(int))
        cv2.rectangle(preview, a, b, (170, 180, 190), 1)
        cv2.putText(preview, str(i + 1), (a[0], max(18, a[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, .6, (30, 75, 95), 2)
    a = tuple(np.round([x0 * scale[0, 0], y0 * scale[1, 1]]).astype(int))
    b = tuple(np.round([(x1 - 1) * scale[0, 0], (y1 - 1) * scale[1, 1]]).astype(int))
    cv2.rectangle(preview, a, b, (0, 165, 155), 2)
    st.image(preview, caption="กรอบสีเขียวคือบริเวณที่จะประมวลผล · คะแนนใช้จัดอันดับ ไม่ใช่ความแม่นยำ", width="stretch")
    if f"{prefix}_edges" in st.session_state:
        with st.expander("ดูภาพขอบสำหรับตรวจสอบ"):
            st.image(st.session_state[f"{prefix}_edges"], width="stretch")
    return image[y0:y1, x0:x1], (x0, y0, x1, y1), candidate


def corner_editor(roi, box, candidate, key):
    x0, y0, _, _ = box
    point_key = f"{key}_points"
    nonce_key = f"{key}_nonce"
    st.session_state.setdefault(point_key, [])
    st.session_state.setdefault(nonce_key, 0)
    controls = st.columns(3)
    auto = controls[0].button("ใช้มุมที่ตรวจพบ", key=f"{key}_auto", width="stretch",
                              disabled=candidate is None or candidate.corners is None)
    undo = controls[1].button("ย้อนหนึ่งจุด", key=f"{key}_undo", width="stretch")
    clear = controls[2].button("เลือกใหม่", key=f"{key}_clear", width="stretch")
    if auto:
        local = candidate.corners - [x0, y0]
        if (local < 0).any() or (local > [roi.shape[1] - 1, roi.shape[0] - 1]).any():
            st.warning("มุมที่ตรวจพบอยู่นอกกรอบที่แก้ไข กรุณาระบุมุมใหม่")
        else:
            st.session_state[point_key] = local.tolist()
    if undo:
        st.session_state[point_key] = st.session_state[point_key][:-1]
    if clear:
        st.session_state[point_key] = []
    if auto or undo or clear:
        st.session_state[nonce_key] += 1
        forget_result()
    points = st.session_state[point_key]
    preview, scale = resized(roi, 1000)
    preview = preview.copy()
    if len(points) == 4:
        try:
            local_ordered = ordered_quad(np.array(points), roi.shape)
            drawing = np.round(local_ordered * [scale[0, 0], scale[1, 1]]).astype(np.int32)
            cv2.polylines(preview, [drawing], True, (0, 155, 130), 2)
        except ValueError:
            pass
    for i, point in enumerate(points):
        pixel = tuple(np.round(np.array(point) * [scale[0, 0], scale[1, 1]]).astype(int))
        cv2.circle(preview, pixel, 6, (245, 130, 45), -1)
        cv2.putText(preview, str(i + 1), (pixel[0] + 9, pixel[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, .7, (245, 130, 45), 2)
    st.caption(f"คลิกมุมป้ายบนภาพเต็ม • เลือกแล้ว {len(points)}/4 จุด • ระบบเรียงมุมให้ ตรวจแนวภาพก่อนดาวน์โหลด")
    click = streamlit_image_coordinates(preview, width="stretch", cursor="crosshair",
                                        key=f"{key}_click_{st.session_state[nonce_key]}")
    if click:
        click_id = (st.session_state[nonce_key], click.get("unix_time"), click.get("x"), click.get("y"))
        if st.session_state.get(f"{key}_last_click") != click_id:
            st.session_state[f"{key}_last_click"] = click_id
            if len(points) < 4:
                # The component reports rendered dimensions, which change on mobile.
                px = np.clip(click["x"] * roi.shape[1] / click["width"], 0, roi.shape[1] - 1)
                py = np.clip(click["y"] * roi.shape[0] / click["height"], 0, roi.shape[0] - 1)
                if not points or np.min(np.linalg.norm(np.array(points) - [px, py], axis=1)) >= 3:
                    st.session_state[point_key] = points + [[float(px), float(py)]]
                    forget_result()
                    st.rerun()
                else:
                    st.warning("จุดนี้ใกล้กับจุดเดิมเกินไป")
            else:
                st.info("ครบ 4 จุดแล้ว กดย้อนหนึ่งจุดหรือเลือกใหม่เพื่อแก้ไข")
    with st.expander("ระบุพิกัด 4 จุดด้วยตัวเลข"):
        st.caption("ทางเลือกแทนการคลิก ใช้พิกัดภายในบริเวณป้ายที่แสดงด้านบน")
        default = "\n".join(f"{p[0]:.1f}, {p[1]:.1f}" for p in points)
        values = st.text_area("หนึ่งจุดต่อบรรทัด: x, y", default,
                               placeholder="10, 10\n300, 20\n290, 150\n15, 140", key=f"{key}_text_{len(points)}_{st.session_state[nonce_key]}")
        if st.button("ใช้พิกัดนี้", key=f"{key}_apply"):
            try:
                parsed = np.array([[float(v.strip()) for v in row.split(",")] for row in values.strip().splitlines()])
                parsed = ordered_quad(parsed, roi.shape)
                st.session_state[point_key] = parsed.tolist()
                st.session_state[nonce_key] += 1
                forget_result()
                st.rerun()
            except (ValueError, TypeError) as exc:
                st.error(f"พิกัดไม่ถูกต้อง: {exc}")
    return np.asarray(points, dtype=np.float32)


st.markdown('<div class="eyebrow">PLATELAB / COMPUTER VISION</div>', unsafe_allow_html=True)
st.title("ปรับมุมป้ายทะเบียนให้พร้อมอ่าน")
st.markdown('<p class="intro">เลือกป้าย ปรับภาพให้ตรง และเตรียมภาพสำหรับ OCR<br>ตรวจสอบทุกขั้นตอนได้ ตั้งแต่ขอบป้ายจนถึงจุดที่จับคู่</p>', unsafe_allow_html=True)
st.caption("CP461 · ไม่ใช้โมเดลที่ต้องฝึก · ยังไม่ได้ประเมินกับชุดภาพป้ายจริง")

with st.sidebar:
    st.markdown("### วิธีใช้งาน")
    st.markdown("**01** อัปโหลดภาพ\n\n**02** ค้นหาและปรับป้าย หรือระบุมุมบนภาพเต็ม\n\n**03** ตรวจภาพก่อน–หลังและดาวน์โหลด")
    st.divider()
    st.markdown("**ภาพเดียว** หาป้ายและปรับมุมอัตโนมัติ หากเลือกผิดให้ระบุมุมเอง\n\n**มี reference** ใช้ SIFT/ORB + matching + RANSAC")
    st.caption("ภาพถูกส่งไปประมวลผลบนเซิร์ฟเวอร์และเก็บชั่วคราวใน session แอปไม่บันทึกไฟล์อัปโหลดลงดิสก์")
    if st.button("ล้างงานทั้งหมด", width="stretch"):
        next_generation = st.session_state.get("upload_generation", 0) + 1
        st.session_state.clear()
        st.session_state["upload_generation"] = next_generation
        st.rerun()
    with st.expander("ขอบเขตระบบ"):
        st.write("รองรับ JPG/PNG ครั้งละหนึ่งป้าย ไม่มี OCR อ่านตัวอักษรหรือการประมวลผลวิดีโอในเวอร์ชันนี้ คะแนน matching ไม่ยืนยันความถูกต้องของทะเบียน")

mode = st.radio("โหมดการปรับภาพ", ["ภาพเดียว · Auto / เลือกมุม", "มี reference · SIFT / ORB"], horizontal=True)
feature_mode = mode.startswith("มี reference")
if feature_mode:
    st.info("ใช้ภาพเอียงและภาพด้านหน้าของป้ายเดียวกัน เลือกบริเวณป้ายในทั้งสองภาพ ผลจะมีมุมมองและสัดส่วนตาม reference")

st.subheader("01 / ภาพและบริเวณป้าย")
upload_columns = st.columns(2) if feature_mode else [st.container()]
with upload_columns[0]:
    source, source_token = image_upload("ภาพรถหรือป้ายที่ต้องการปรับ", f"source_upload_{st.session_state.get('upload_generation', 0)}")
reference, reference_token = None, None
if feature_mode:
    with upload_columns[1]:
        reference, reference_token = image_upload("ภาพอ้างอิงด้านหน้าของป้ายเดียวกัน", f"reference_upload_{st.session_state.get('upload_generation', 0)}")

active_input = (source_token, reference_token, mode)
if st.session_state.get("active_input") != active_input:
    # Keep only current decoded uploads/widgets; release old image-heavy processing state.
    for old_key in list(st.session_state):
        if old_key.startswith(("work_", "result")):
            del st.session_state[old_key]
    st.session_state["active_input"] = active_input
if source is None:
    st.markdown("#### เริ่มจากภาพของคุณ")
    st.write("อัปโหลด JPG หรือ PNG เพื่อเลือกป้ายและปรับมุม ยังไม่ต้องเตรียมข้อมูล train")
    st.stop()
if feature_mode and reference is None:
    st.image(source, caption="ภาพต้นฉบับ · รอภาพอ้างอิง", width=640)
    st.stop()

source_roi = source
source_box = (0, 0, source.shape[1], source.shape[0])
selected = None
reference_roi, reference_box = None, None
if feature_mode:
    region_columns = st.columns(2)
    with region_columns[0]:
        source_roi, source_box, selected = select_region(source, f"work_src_{source_token}", "ภาพที่ต้องการปรับ")
    with region_columns[1]:
        reference_roi, reference_box, _ = select_region(reference, f"work_ref_{reference_token}", "ภาพอ้างอิง")
if source_roi is None or (feature_mode and reference_roi is None):
    st.stop()

st.subheader("02 / ปรับมุมภาพ")
settings = {}
points = np.empty((0, 2))
auto_payload = None
manual = False
if feature_mode:
    method = st.radio("วิธีหาจุดเด่น", ["SIFT", "ORB"], horizontal=True)
    st.caption("SIFT ใช้ L2 distance · ORB ใช้ Hamming distance · ทั้งสองใช้ KNN, Lowe’s ratio test และ RANSAC")
    with st.expander("ปรับค่าการจับคู่ขั้นสูง"):
        cols = st.columns(3)
        ratio = cols[0].slider("Lowe’s ratio", 0.50, 0.90, 0.75, 0.01)
        ransac_px = cols[1].slider("RANSAC threshold (px)", 1.0, 8.0, 3.0, 0.5)
        min_inliers = cols[2].slider("Inliers ขั้นต่ำ", 8, 40, 8)
        st.caption("Threshold อ้างอิงภาพ reference ที่ย่อสำหรับ matching สูงสุด 1200 px ไม่ใช่ภาพต้นฉบับ ค่าตั้งต้นยังไม่ผ่านการปรับด้วยข้อมูลป้ายจริง")
    settings = dict(method=method, ratio=ratio, ransac_px=ransac_px, min_inliers=min_inliers)
else:
    with st.expander("สัดส่วนผลลัพธ์และค่าค้นหาขั้นสูง"):
        cols = st.columns(3)
        width = cols[0].slider("ความกว้างผลลัพธ์ (px)", 320, 1600, 640, 80)
        aspect = cols[1].number_input("สัดส่วนกว้าง ÷ สูง", 0.80, 5.0, 2.0, 0.05)
        rotate = cols[2].checkbox("สลับแนวมุม 90°", help="ใช้เมื่อระบบตีความด้านกว้างและด้านสูงสลับกัน")
        st.caption("สัดส่วน 2.0 เป็นค่าเริ่มต้น ปรับตามป้ายจริงหากภาพยืดหรือบีบ")
        low = st.slider("Canny ต่ำ", 5, 150, 40, 5, key="work_auto_low")
        high = st.slider("Canny สูง", 50, 250, 140, 5, key="work_auto_high")
        st.caption("ใช้ค่าค้นหาล่าสุดที่ปรับจากชุดภาพป้ายไทย หากไม่ผ่านเกณฑ์ ระบบให้เลือกมุมเอง")
    settings = dict(width=width, aspect=aspect, rotate=rotate)
    config = web_detector_config(low=min(low, high), high=max(low, high))
    search_key = json.dumps([source_token, settings, vars(config)], sort_keys=True)
    if st.session_state.get("work_single_search_key") != search_key:
        st.session_state.pop("work_single_detection", None)
        st.session_state.pop("work_single_auto_payload", None)
        forget_result()
        st.session_state["work_single_search_key"] = search_key
    buttons = st.columns(2)
    search = buttons[0].button("ค้นหาและปรับป้าย", type="primary", width="stretch")
    edit = buttons[1].button("ระบุมุมเองบนภาพเต็ม", width="stretch")
    key = f"work_corners_full_{source_token}"
    if search:
        forget_result()
        st.session_state.pop("work_single_detection", None)
        st.session_state.pop("work_single_auto_payload", None)
        with st.spinner("กำลังหาขอบและกลุ่มตัวอักษร ตรวจมุม และปรับป้าย…"):
            started = perf_counter()
            try:
                detection, warped, homography, quad = auto_rectify(source, **settings, config=config)
                st.session_state["work_single_detection"] = detection
                st.session_state["work_single_manual"] = detection.selected is None
                st.session_state.pop("work_single_auto_payload", None)
                if detection.selected is not None:
                    st.session_state["work_single_auto_payload"] = dict(
                        accepted=True, reason=detection.reason, warped=warped, homography=homography,
                        quad=quad, visualization=None,
                        metrics={**detection.selected.metrics, "candidate_score": detection.selected.score,
                                 "detector_thresholds": vars(config),
                                 "elapsed_ms": round((perf_counter() - started) * 1000, 2)},
                        method="Auto: edges + text groups + border refinement + Homography")
                else:
                    # Never carry old manual points into a new failed auto attempt.
                    st.session_state[f"{key}_points"] = []
                    st.session_state[f"{key}_nonce"] = st.session_state.get(f"{key}_nonce", 0) + 1
            except (ValueError, cv2.error) as exc:
                st.session_state["work_single_manual"] = True
                st.session_state.pop("work_single_auto_payload", None)
                st.error(f"ค้นหาอัตโนมัติไม่ได้ กรุณาระบุมุมเอง: {exc}")
    detection = st.session_state.get("work_single_detection")
    auto_payload = st.session_state.get("work_single_auto_payload")
    if edit:
        st.session_state["work_single_manual"] = True
        forget_result()
    manual = st.session_state.get("work_single_manual", False)
    if detection is not None:
        if detection.selected is None:
            st.warning(detection.reason)
        with st.expander("รายละเอียดการค้นหาอัตโนมัติ"):
            st.caption("คะแนนเป็นเกณฑ์จัดอันดับ ไม่ใช่เปอร์เซ็นต์ความแม่นยำ ค่าล่าสุดปรับจากภาพป้ายจริง 36 ภาพ ยังไม่ยืนยันผลบนภาพใหม่")
            rows = []
            for i, c in enumerate(detection.candidates):
                rows.append({"บริเวณ": i + 1, "เลือก": c is detection.selected,
                             "คะแนน": round(c.score, 3), "ผ่าน": c.accepted, "เหตุผล": c.reason,
                             "ข้อความ": c.metrics.get("text_score", 0), "ชิ้นหลัก": c.metrics.get("components", 0),
                             "ขอบ": c.metrics.get("edge_support", 0), "แหล่งที่พบ": ", ".join(sorted(c.sources))})
            if rows:
                st.dataframe(rows, hide_index=True, width="stretch")
            else:
                st.write("ไม่พบ candidate ที่ผ่านขั้นเสนอพื้นที่")
            st.caption(f"เวลาค้นหาและตรวจมุม {detection.elapsed_ms:,.0f} ms")
            diag_columns = st.columns(2)
            for i, (label, diagnostic) in enumerate(detection.diagnostics.items()):
                diag_columns[i % 2].image(diagnostic, caption=label, width="stretch")
    if manual:
        st.info("เลือกมุมป้ายจริงบนภาพเต็ม หรือใช้มุมที่ตรวจพบเป็นจุดเริ่มต้น กดเลือกใหม่หากระบบเลือกผิดบริเวณ")
        selected = detection.selected if detection else None
        points = corner_editor(source, source_box, selected, key)
    elif auto_payload is None:
        st.image(source, caption="ภาพเต็ม · กดค้นหาและปรับป้าย หรือระบุมุมเอง", width="stretch")

signature = hashlib.sha256(json.dumps([active_input, source_box, reference_box, settings,
                                       points.tolist(), manual,
                                       None if feature_mode else search_key], sort_keys=True).encode()).hexdigest()
if st.session_state.get("result", {}).get("signature") != signature:
    forget_result()
if auto_payload is not None and not manual:
    st.session_state["result"] = dict(auto_payload, signature=signature, source_box=source_box, reference_box=None)
run = False
if feature_mode or manual:
    run = st.button("ปรับป้ายให้ตรง" if feature_mode else "ปรับภาพจากมุมที่เลือก", type="primary", disabled=not feature_mode and len(points) != 4)
if run:
    with st.spinner("กำลังประมวลผล…"):
        start = perf_counter()
        try:
            if feature_mode:
                match = match_rectify(source_roi, reference_roi, **settings)
                result = dict(accepted=match.accepted, reason=match.reason, warped=match.warped,
                              homography=match.homography, visualization=match.visualization,
                              metrics=match.metrics, method=f"{settings['method']} + KNN + ratio test + RANSAC",
                              quad=match.source_quad)
            else:
                global_points = points + [source_box[0], source_box[1]]
                warped, homography, quad = rectify_corners(source, global_points, **settings)
                result = dict(accepted=True, reason="ปรับด้วยมุมป้าย 4 จุดแล้ว โปรดตรวจสัดส่วนและแนวภาพ",
                              warped=warped, homography=homography, visualization=None,
                              metrics={"elapsed_ms": round((perf_counter() - start) * 1000, 2)},
                              method="4 corners + Homography (ไม่มี descriptor matching / RANSAC)", quad=quad)
            # Export all homographies in the same full-source-image coordinate frame.
            if feature_mode and result["homography"] is not None:
                translation = np.array([[1, 0, -source_box[0]], [0, 1, -source_box[1]], [0, 0, 1]])
                result["homography"] = result["homography"] @ translation
                result["quad"] = result["quad"] + [source_box[0], source_box[1]]
            result.update(signature=signature, source_box=source_box, reference_box=reference_box)
            st.session_state["result"] = result
        except (ValueError, cv2.error) as exc:
            st.error(f"ประมวลผลไม่ได้ กรุณาตรวจบริเวณป้ายและมุม: {exc}")

result = st.session_state.get("result")
if result:
    st.subheader("03 / ผลลัพธ์")
    (st.success if result["accepted"] else st.warning)(result["reason"])
    st.caption(f"วิธีที่ใช้จริง: {result['method']}")
    if not result["accepted"]:
        st.write("แก้บริเวณป้าย เปลี่ยนภาพอ้างอิง หรือเลือกโหมดภาพเดียวเพื่อระบุมุม ระบบจะไม่เปลี่ยนวิธีให้เอง")
    metrics = result["metrics"]
    cols = st.columns(4 if feature_mode else 1)
    cols[0].metric("เวลาประมวลผล", f"{metrics.get('elapsed_ms', 0):,.0f} ms")
    if feature_mode:
        cols[1].metric("คู่หลังกรอง", metrics.get("good_matches", 0))
        cols[2].metric("Inliers", f"{metrics.get('inliers', 0)} / {metrics.get('good_matches', 0)}")
        cols[3].metric("Median error", f"{metrics['median_reprojection_px']:.2f} px" if "median_reprojection_px" in metrics else "—")
    if result["visualization"] is not None:
        with st.expander("ดูจุดที่จับคู่และรายละเอียด", expanded=not result["accepted"]):
            st.image(result["visualization"], caption="ซ้าย: ภาพที่ต้องการปรับ · ขวา: reference · เมื่อคำนวณได้ เส้นสีเขียวแสดง inliers สูงสุด 150 คู่", width="stretch")
            st.json(metrics)
    if result["accepted"]:
        if not feature_mode and not manual:
            if st.button("เลือกผิดป้าย / แก้มุมด้วยตนเอง"):
                st.session_state["work_single_manual"] = True
                st.session_state[f"{key}_points"] = result["quad"].tolist()
                st.session_state[f"{key}_nonce"] = st.session_state.get(f"{key}_nonce", 0) + 1
                forget_result()
                st.rerun()
        st.markdown("**เตรียมภาพสำหรับ OCR**")
        opts = st.columns([2, 1, 1])
        output_mode = opts[0].selectbox("รูปแบบภาพ", ["Grayscale", "สี", "ขาวดำ (Adaptive threshold)"])
        contrast = opts[1].checkbox("เพิ่ม Contrast (CLAHE)", value=True)
        denoise = opts[2].checkbox("ลด Noise", value=False)
        prepared = prepare_ocr(result["warped"], output_mode, contrast, denoise)
        before = source_roi.copy()
        if result["quad"] is not None:
            local_quad = result["quad"] - [source_box[0], source_box[1]]
            cv2.polylines(before, [np.round(local_quad).astype(np.int32)], True, (0, 165, 130), 2)
        cols = st.columns(3)
        cols[0].image(before, caption="01 / ป้ายต้นฉบับและขอบที่จะปรับ", width="stretch")
        cols[1].image(result["warped"], caption="02 / ป้ายที่ปรับตรง", width="stretch")
        cols[2].image(prepared, caption="03 / ภาพเตรียม OCR", width="stretch")
        st.caption("ยังไม่มีการอ่านตัวอักษรหรือประเมินความแม่นยำ OCR ภาพที่เบลอหรือรายละเอียดหายอาจไม่ดีขึ้นหลังปรับมุม")
        report = {"schema_version": 1, "method": result["method"], "accepted": True,
                  "source_size": [source.shape[1], source.shape[0]],
                  "source_roi_xyxy": result["source_box"], "reference_roi_xyxy": result["reference_box"],
                  "output_size": [result["warped"].shape[1], result["warped"].shape[0]],
                  "homography_source_to_output": result["homography"].tolist(),
                  "coordinate_frame": "full source image after EXIF orientation; pixels",
                  "source_corners": result["quad"].tolist(), "settings": settings,
                  "ocr_preparation": {"mode": output_mode, "clahe": contrast, "denoise": denoise},
                  "metrics": metrics,
                  "dataset_evaluation": {"scope": "auto localization tuning only",
                                         "tuning_images": 36, "independent_test_completed": False,
                                         "ocr_evaluated": False}}
        buttons = st.columns(3)
        buttons[0].download_button("ดาวน์โหลดป้าย PNG", png_bytes(result["warped"]), "plate_rectified.png", "image/png", width="stretch")
        buttons[1].download_button("ดาวน์โหลดภาพ OCR", png_bytes(prepared), "plate_ocr_ready.png", "image/png", width="stretch")
        buttons[2].download_button("ดาวน์โหลดรายละเอียด JSON", json.dumps(report, ensure_ascii=False, indent=2), "rectification_report.json", "application/json", width="stretch")
        with st.expander("Homography และข้อมูลการแปลง"):
            st.json(report)

st.markdown('<div class="footer">PlateLab · SIFT / ORB / Homography · ระบบต้นแบบสำหรับ CP461</div>', unsafe_allow_html=True)
