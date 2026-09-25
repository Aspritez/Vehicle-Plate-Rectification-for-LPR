"""Thai license plate recognition: Streamlit app.

Run:  streamlit run streamlit_app.py
Flow: upload an image (or pick a sample) -> the plate is located automatically -> if it is wrong, draw a
rough box around the plate (or type the 4 corners) and the plate is found inside it.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

sys.path.insert(0, str(Path(__file__).parent))
from backend.app.pipeline import PlateReader  # noqa: E402

SAMPLES_DIR = Path(__file__).parent / "samples"
MAX_UPLOAD_MB = 10
MAX_PIXELS = 40_000_000
DISPLAY_WIDTH = 640          # width of the drawing image on screen (pixels)

st.set_page_config(page_title="Thai Plate Recognition", page_icon="🚗", layout="wide")


@st.cache_resource(show_spinner="Loading models (the first start takes a minute or two)...")
def get_reader() -> PlateReader:
    return PlateReader()


def decode(data: bytes):
    """Decode an uploaded image; returns (image, error message)."""
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return None, f"Image is larger than {MAX_UPLOAD_MB} MB."
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None, "That file is not a readable image."
    if image.shape[0] * image.shape[1] > MAX_PIXELS:
        return None, f"Image has more than {MAX_PIXELS // 1_000_000} megapixels."
    return image, None


def rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def overlay(image: np.ndarray, corners, scale: float) -> Image.Image:
    """The image at display size with the plate corners drawn on it (numbered 1-4, clockwise)."""
    small = cv2.resize(rgb(image), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if corners:
        pts = (np.array(corners, np.float32) * scale).astype(np.int32)
        cv2.polylines(small, [pts], True, (43, 130, 255), 2)
        for i, (x, y) in enumerate(pts, start=1):
            cv2.circle(small, (int(x), int(y)), 9, (43, 130, 255), -1)
            cv2.putText(small, str(i), (int(x) - 5, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return Image.fromarray(small)


def set_result(result: dict) -> None:
    st.session_state.result = result
    corners = result.get("corners") or result.get("suggested_corners")
    st.session_state.corners = [list(map(float, c)) for c in corners] if corners else None
    st.session_state.corner_version += 1     # resets the numeric corner inputs


def load_image(name: str, data: bytes) -> None:
    """A new image was chosen: reset the state and read it automatically."""
    image, error = decode(data)
    st.session_state.update(image_id=name, image=image, load_error=error, last_box=None, result=None, corners=None)
    if image is not None:
        with st.spinner("Locating and reading the plate..."):
            set_result(get_reader().recognize(image))


def main() -> None:
    for key, default in (("image_id", None), ("image", None), ("load_error", None), ("last_box", None),
                         ("result", None), ("corners", None), ("corner_version", 0)):
        st.session_state.setdefault(key, default)

    st.title("🚗 Thai License Plate Recognition")
    st.caption("SIFT locator · corner snapping · text straightening · OCR · optional super-resolution")
    reader = get_reader()

    # ---------- input ----------
    top_left, top_right = st.columns([2, 1])
    with top_left:
        upload = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png", "webp"])
    with top_right:
        samples = sorted(p.name for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in (".jpg", ".png"))
        sample = st.selectbox("...or try a sample", ["-"] + samples) if samples else "-"

    if upload is not None:
        chosen = (f"upload:{upload.name}:{upload.size}", upload.getvalue)
    elif sample != "-":
        chosen = (f"sample:{sample}", (SAMPLES_DIR / sample).read_bytes)
    else:
        chosen = None

    if chosen is None:
        st.info("Upload an image or pick a sample to begin.")
        return
    if chosen[0] != st.session_state.image_id:
        load_image(chosen[0], chosen[1]())

    if st.session_state.load_error:
        st.error(st.session_state.load_error)
        return

    image = st.session_state.image
    result = st.session_state.result or {}
    height, width = image.shape[:2]
    scale = min(1.0, DISPLAY_WIDTH / width)

    col_draw, col_pipeline, col_result = st.columns([1.15, 1.0, 0.85], gap="large")

    # ---------- column 1: draw a box / adjust corners ----------
    with col_draw:
        st.subheader("1 · Plate box")
        st.caption("Drag on the image to draw a rough box around the plate. The plate is found inside it.")
        shown = overlay(image, st.session_state.corners, scale)
        drag = streamlit_image_coordinates(
            shown, key=f"draw-{st.session_state.image_id}", click_and_drag=True, width="stretch")

        if drag and drag.get("x1") is not None:
            # Coordinates are in the pixels the browser displayed, which may be smaller than the image we
            # sent (the column can be narrower): convert back to original-image pixels and stay inside it.
            per_shown = shown.width / (drag.get("width") or shown.width)
            to_original = per_shown / scale
            x0, x1 = sorted((drag["x1"], drag["x2"]))
            y0, y1 = sorted((drag["y1"], drag["y2"]))
            box = [round(min(max(v * to_original, 0.0), limit), 1)
                   for v, limit in ((x0, width - 1), (y0, height - 1), (x1, width - 1), (y1, height - 1))]
            if (box[2] - box[0]) >= 14 and (box[3] - box[1]) >= 14 and box != st.session_state.last_box:
                st.session_state.last_box = box
                with st.spinner("Finding the plate inside your box..."):
                    set_result(reader.recognize_box(image, box))
                st.rerun()

        with st.expander("Or type the 4 corners (pixels)"):
            st.caption("1 = top-left, then clockwise. Numbers are in original image pixels.")
            corners = st.session_state.corners or [[0.0, 0.0]] * 4
            version = st.session_state.corner_version
            entered = []
            for i, (cx, cy) in enumerate(corners, start=1):
                a, b = st.columns(2)
                entered.append([a.number_input(f"{i} · x", 0.0, float(width), float(cx), key=f"cx{i}-{version}"),
                                b.number_input(f"{i} · y", 0.0, float(height), float(cy), key=f"cy{i}-{version}")])
            snap = st.checkbox("Snap to the plate edges", help="Use when the corners are only roughly placed.")
            if st.button("Apply corners", type="primary"):
                with st.spinner("Reading the plate..."):
                    set_result(reader.recognize_corners(image, entered, snap))
                st.rerun()

    # ---------- column 2: pipeline ----------
    with col_pipeline:
        st.subheader("2 · Pipeline")
        images = result.get("images") or {}
        tabs = st.tabs(["Original", "Detected", "Deskewed", "Enhanced"])
        with tabs[0]:
            st.image(rgb(image), width="stretch")
        for tab, key in zip(tabs[1:], ("visualization", "deskewed", "enhanced")):
            with tab:
                if key in images:
                    st.image(rgb(images[key]), width="stretch")
                else:
                    st.caption("Not available yet.")

    # ---------- column 3: result ----------
    with col_result:
        st.subheader("3 · Result")
        ocr = result.get("ocr_result")
        if ocr:
            st.markdown(f"<div style='font-size:2.6rem;font-weight:600;line-height:1.1'>{ocr['text'] or '-'}</div>",
                        unsafe_allow_html=True)
            st.markdown(f"**{ocr['province'] or '-'}**")
            st.progress(min(1.0, max(0.0, float(ocr["confidence"]))), text=f"Confidence {ocr['confidence']:.0%}")
            (st.success if ocr["is_valid"] else st.error)("Valid Thai plate format" if ocr["is_valid"] else "Invalid format")
            if result.get("super_resolution_used"):
                st.warning("Read from a super-resolved copy of the plate. Very blurry plates can be reconstructed "
                           "with the wrong characters, so please check the text.")
            st.caption(f"Processing time: {result['processing_time_ms']} ms")
        else:
            st.warning(result.get("message") or "No result yet.")
            st.caption("Draw a box around the plate on the left, or type the corners.")


main()
