"""Thai license plate recognition: Streamlit app.

Run:  streamlit run streamlit_app.py

The page is the same single-screen GUI as the FastAPI version (the frontend/ folder), embedded as a Streamlit
custom component. Its JavaScript talks to this Python script through Streamlit's component protocol instead of
HTTP: it sends {recognize | box | corners} requests and receives the same JSON the API would return.
"""
import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent))
from backend.app.pipeline import MODEL_PATH, PlateReader  # noqa: E402
from backend.app.core.super_resolution import SuperResolver  # noqa: E402

ROOT = Path(__file__).parent
MAX_UPLOAD_MB = 10
MAX_PIXELS = 40_000_000

st.set_page_config(page_title="Thai Plate Recognition", page_icon="🚗", layout="wide",
                   initial_sidebar_state="collapsed")

# The GUI: frontend/index.html + css + js, served by Streamlit as a component.
plate_app = components.declare_component("plate_app", path=str(ROOT / "frontend"))

# Remove Streamlit's own page chrome so the GUI fills the window (same frame colour as the GUI's page).
PAGE_CSS = """
<style>
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stStatusWidget"],
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
.stApp { background: #9298a8; }
.block-container { padding: 0 !important; max-width: 100% !important; }
iframe { border: 0 !important; display: block; }
</style>
"""


@st.cache_resource(show_spinner=False)
def get_reader() -> PlateReader:
    return PlateReader()


def decode_upload(data_url: str):
    """A data URL from the page -> (image, error message)."""
    encoded = data_url.split(",", 1)[-1]
    if len(encoded) > MAX_UPLOAD_MB * 1024 * 1024 * 4 // 3 + 16:
        return None, f"Image is larger than {MAX_UPLOAD_MB} MB."
    try:
        data = base64.b64decode(encoded)
    except Exception:
        return None, "The image data could not be read."
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None, "That file is not a readable image."
    if image.shape[0] * image.shape[1] > MAX_PIXELS:
        return None, f"Image has more than {MAX_PIXELS // 1_000_000} megapixels."
    return image, None


def data_url(image: np.ndarray, fmt: str) -> str:
    """PNG for the small plate crops (lossless, good for saving), JPEG for the full-size picture."""
    if fmt == "png":
        buffer, mime = cv2.imencode(".png", image)[1], "image/png"
    else:
        buffer, mime = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])[1], "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(buffer).decode()}"


def to_reply(result: dict) -> dict:
    """The pipeline result (numpy images) -> the JSON the page expects (data-URL images)."""
    images = result.pop("images", None)
    if images:
        result["deskewed_plate_base64"] = data_url(images["deskewed"], "png")
        result["enhanced_plate_base64"] = data_url(images["enhanced"], "png")
        result["visualization_base64"] = data_url(images["visualization"], "jpg")
    return result


def handle(event: dict, reader: PlateReader) -> dict:
    """Answer one request from the page: {"result": ...} or {"error": message}."""
    kind = event.get("kind")
    try:
        if kind == "recognize":
            image, error = decode_upload(event.get("image", ""))
            if error:
                return {"error": error}
            st.session_state.image = image
            return {"result": to_reply(reader.recognize(image))}

        image = st.session_state.get("image")
        if image is None:
            return {"error": "The image is no longer on the server. Please upload it again."}
        if kind == "box" and len(event.get("box", [])) == 4:
            return {"result": to_reply(reader.recognize_box(image, event["box"]))}
        if kind == "corners" and len(event.get("corners", [])) == 4:
            return {"result": to_reply(reader.recognize_corners(image, event["corners"], bool(event.get("snap"))))}
        return {"error": f"Unknown or incomplete request: {kind}"}
    except Exception as exc:  # shown in the page instead of a stack trace
        return {"error": str(exc)}


def main() -> None:
    for key, default in (("image", None), ("reply", None), ("last_event", None)):
        st.session_state.setdefault(key, default)
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    # What the status pill shows. Checked from the files so the page can appear before the models finish loading.
    health = {"status": "healthy", "locator_loaded": MODEL_PATH.exists(),
              "super_resolution_available": SuperResolver().available}

    # The page is drawn first; loading the models afterwards does not delay it.
    event = plate_app(reply=st.session_state.reply, health=health, key="plate_app", default=None)
    reader = get_reader()

    if event and event.get("id") != st.session_state.last_event:
        st.session_state.last_event = event["id"]
        reply = handle(event, reader)
        reply["id"] = event["id"]
        st.session_state.reply = reply
        st.rerun()


main()
