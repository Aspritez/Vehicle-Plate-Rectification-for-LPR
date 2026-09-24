"""
Vehicle Plate Rectification for LPR – Streamlit Application
"""

from __future__ import annotations

import base64
import io
import os
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from typing import List

from cv_pipeline import (
    detect_plate_corners,
    rectify_plate,
    validate_quadrilateral,
    order_points,
    enhance_plate_for_ocr
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Plate Rectification",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
#MainMenu, footer, header { visibility: hidden; }
.app-header { text-align: center; padding-bottom: 1rem; }
.upload-zone { border: 2px dashed #555; border-radius: 12px; padding: 3rem 2rem; text-align: center; }
div.stButton > button { border-radius: 8px; font-weight: 600; padding: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Bi-directional Draggable Canvas Component (With Pan & Zoom) ───────────────
_HTML_CONTENT = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body { margin: 0; padding: 0; display: flex; justify-content: center; background: transparent; overflow: hidden; }
    
    .viewport { position: relative; width: 100%; max-width: 100%; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.15); border: 1px solid rgba(255,255,255,0.12); touch-action: none; background: #111; }
    .zoom-wrap { position: relative; width: 100%; transform-origin: 0 0; }
    img { display: block; width: 100%; height: auto; pointer-events: none; }
    canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: crosshair; }
    
    .overlay-btns { position: absolute; top: 12px; right: 12px; display: flex; gap: 8px; z-index: 100; }
    .icon-btn { width: 34px; height: 34px; border-radius: 50%; border: none; background-color: rgba(0,0,0,0.6); color: white; font-size: 16px; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: background 0.2s; }
    .icon-btn:hover { background-color: rgba(100,100,100,0.9); }
    #close-btn:hover { background-color: rgba(255,50,50,0.9); }
    
    .hint { position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.5); color: #fff; padding: 4px 12px; border-radius: 20px; font-size: 12px; pointer-events: none; opacity: 0.8; z-index: 10; white-space: nowrap; }
  </style>
</head>
<body>
  <div class="viewport" id="viewport">
    <div class="zoom-wrap" id="zoom-wrap">
      <img id="bg" draggable="false"/>
      <canvas id="cv"></canvas>
    </div>
    <div class="overlay-btns">
        <button class="icon-btn" id="reset-zoom-btn" title="Reset View" style="display:none; font-size:20px;">⟲</button>
        <button class="icon-btn" id="close-btn" title="Remove Image">&#x2715;</button>
    </div>
    <div class="hint">Scroll to zoom &nbsp;&bull;&nbsp; Drag background to pan</div>
  </div>

  <script>
    function sendMessageToStreamlitClient(type, data) {
      window.parent.postMessage({ isStreamlitMessage: true, type: type, ...data }, "*");
    }
    function init() { sendMessageToStreamlitClient("streamlit:componentReady", {apiVersion: 1}); }
    function setFrameHeight(height) { sendMessageToStreamlitClient("streamlit:setFrameHeight", {height: height}); }
    function sendDataToPython(data) { sendMessageToStreamlitClient("streamlit:setComponentValue", data); }

    const bg = document.getElementById("bg");
    const canvas = document.getElementById("cv");
    const ctx = canvas.getContext("2d");
    
    const viewport = document.getElementById("viewport");
    const zoomWrap = document.getElementById("zoom-wrap");
    const resetZoomBtn = document.getElementById("reset-zoom-btn");
    
    const COLORS = ["#FF4444", "#44CC44", "#4488FF", "#FFCC00"];
    const LABELS = ["TL", "TR", "BR", "BL"];

    let pts = [];
    let scale = 1.0;
    
    // Zoom/Pan State
    let zoomScale = 1.0;
    let panX = 0;
    let panY = 0;
    
    // Interaction State
    let dragging = -1;
    let isPanning = false;
    let lastPanX = 0;
    let lastPanY = 0;
    const R = 12;

    function updateTransform() {
      zoomWrap.style.transform = `translate(${panX}px, ${panY}px) scale(${zoomScale})`;
      resetZoomBtn.style.display = (zoomScale > 1.0 || panX !== 0 || panY !== 0) ? "flex" : "none";
      draw(); // Redraw to maintain perfectly sized dots regardless of CSS zoom scale
    }

    resetZoomBtn.addEventListener("click", () => {
      zoomScale = 1.0; panX = 0; panY = 0; updateTransform();
    });

    viewport.addEventListener("wheel", (e) => {
      e.preventDefault();
      const zoomSensitivity = 0.002;
      const delta = -e.deltaY * zoomSensitivity;
      let newScale = zoomScale * Math.exp(delta);
      newScale = Math.max(1.0, Math.min(newScale, 15.0)); // Allow massive zoom for tiny plates
      
      const rect = viewport.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      
      panX = mx - (mx - panX) * (newScale / zoomScale);
      panY = my - (my - panY) * (newScale / zoomScale);
      zoomScale = newScale;
      
      if (zoomScale === 1.0) { panX = 0; panY = 0; }
      updateTransform();
    }, {passive: false});

    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (pts.length !== 4) return;
      
      const rect = canvas.getBoundingClientRect();
      const currentScaleX = rect.width / canvas.width; 
      
      // Dynamic rendering sizes so dots don't blow up on the screen when zoomed in heavily
      const dynamicR = R / currentScaleX;
      const dynamicLine = 2 / currentScaleX;
      const dynamicFont = Math.max(8, 11 / currentScaleX) + "px sans-serif";

      // Polygon
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < 4; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.closePath();
      ctx.fillStyle = "rgba(255,255,255,0.15)";
      ctx.strokeStyle = "rgba(255,255,255,0.8)";
      ctx.lineWidth = dynamicLine;
      ctx.fill();
      ctx.stroke();

      // Corners
      for (let i = 0; i < 4; i++) {
        const [x, y] = pts[i];
        ctx.beginPath();
        ctx.arc(x, y, dynamicR, 0, Math.PI * 2);
        ctx.fillStyle = COLORS[i];
        ctx.fill();
        ctx.lineWidth = dynamicLine;
        ctx.strokeStyle = "#fff";
        ctx.stroke();
        ctx.fillStyle = "#fff";
        ctx.font = "bold " + dynamicFont;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(LABELS[i], x, y);
      }
    }

    function getPos(e) {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      return [(e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY];
    }

    canvas.addEventListener("pointerdown", (e) => {
      if(pts.length !== 4) return;
      const [mx, my] = getPos(e);
      let bestDist = Infinity;
      let bestIdx = -1;
      
      const rect = canvas.getBoundingClientRect();
      const currentScaleX = canvas.width / rect.width;
      const hitRadius = (R + 15) * currentScaleX; 

      for (let i = 0; i < 4; i++) {
        const dx = mx - pts[i][0], dy = my - pts[i][1];
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist <= hitRadius && dist < bestDist) {
           bestDist = dist;
           bestIdx = i;
        }
      }
      
      if (bestIdx >= 0) {
          dragging = bestIdx;
          canvas.setPointerCapture(e.pointerId);
          e.preventDefault();
      } else {
          isPanning = true;
          lastPanX = e.clientX;
          lastPanY = e.clientY;
          canvas.setPointerCapture(e.pointerId);
          canvas.style.cursor = "grabbing";
          e.preventDefault();
      }
    });

    canvas.addEventListener("pointermove", (e) => {
      if (dragging >= 0) {
          const [mx, my] = getPos(e);
          pts[dragging] = [Math.max(0,Math.min(mx,canvas.width)), Math.max(0,Math.min(my,canvas.height))];
          draw();
          e.preventDefault();
      } else if (isPanning) {
          panX += e.clientX - lastPanX;
          panY += e.clientY - lastPanY;
          lastPanX = e.clientX;
          lastPanY = e.clientY;
          updateTransform();
          e.preventDefault();
      }
    });

    canvas.addEventListener("pointerup", (e) => {
      canvas.style.cursor = "crosshair";
      if (isPanning) {
          isPanning = false;
          canvas.releasePointerCapture(e.pointerId);
          return;
      }
      if (dragging >= 0) {
          dragging = -1;
          canvas.releasePointerCapture(e.pointerId);
          const origPts = pts.map(p => [Math.round(p[0]/scale), Math.round(p[1]/scale)]);
          sendDataToPython({value: {action: "update", corners: origPts}});
      }
    });

    document.getElementById("close-btn").addEventListener("click", () => {
      sendDataToPython({value: {action: "clear"}});
    });

    window.addEventListener("message", function(event) {
      if (event.data.type === "streamlit:render") {
        const args = event.data.args;
        bg.src = args.image_uri;
        canvas.width = args.disp_w;
        canvas.height = args.disp_h;
        scale = args.scale;
        pts = args.corners.map(p => [p[0] * scale, p[1] * scale]);
        draw();
        setTimeout(() => { setFrameHeight(viewport.offsetHeight + 15); }, 50);
      }
    });
    
    window.addEventListener("resize", () => {
      setFrameHeight(viewport.offsetHeight + 15);
    });

    init();
  </script>
</body>
</html>
"""

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend_draggable")
os.makedirs(_FRONTEND_DIR, exist_ok=True)
with open(os.path.join(_FRONTEND_DIR, "index.html"), "w", encoding="utf-8") as f:
    f.write(_HTML_CONTENT)

_draggable_canvas_component = components.declare_component("draggable_canvas", path=_FRONTEND_DIR)

def _draggable_canvas(pil_img: Image.Image, corners: List[List[int]], key="canvas"):
    orig_w, orig_h = pil_img.size
    # Compress for efficient websocket transfer, but keep it big enough for zoom detail
    disp_w = min(orig_w, 1500)
    scale = disp_w / orig_w
    disp_h = int(orig_h * scale)
    
    buf = io.BytesIO()
    pil_img.copy().resize((disp_w, disp_h), Image.LANCZOS).save(buf, format="JPEG", quality=85)
    data_uri = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    return _draggable_canvas_component(
        image_uri=data_uri,
        corners=corners,
        scale=scale,
        disp_w=disp_w,
        disp_h=disp_h,
        key=key,
        default={"action": "none"}
    )


# ── Session state ───────────────────────────────────────────────────────────
if "pil_image" not in st.session_state:
    st.session_state.pil_image = None
if "cv_image" not in st.session_state:
    st.session_state.cv_image = None
if "corners" not in st.session_state:
    st.session_state.corners = None
if "rectified" not in st.session_state:
    st.session_state.rectified = None
if "enhanced" not in st.session_state:
    st.session_state.enhanced = None

def _clear():
    st.session_state.pil_image = None
    st.session_state.cv_image = None
    st.session_state.corners = None
    st.session_state.rectified = None
    st.session_state.enhanced = None

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv_to_pil(img: np.ndarray) -> Image.Image:
    if len(img.shape) == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ── Main App ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <h1>Vehicle Plate Rectification</h1>
    <p>Simple deskew tool for license plates</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.pil_image is None:
    # Upload view
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload an image containing a license plate", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is not None:
        pil = Image.open(uploaded).convert("RGB")
        cv_img = pil_to_cv(pil)
        st.session_state.pil_image = pil
        st.session_state.cv_image = cv_img

        corners = detect_plate_corners(cv_img)
        if not corners or len(corners) != 4:
            h, w = cv_img.shape[:2]
            corners = [[w*0.2, h*0.2], [w*0.8, h*0.2], [w*0.8, h*0.8], [w*0.2, h*0.8]]
            st.toast("Auto-detection fallback used.")
        else:
            st.toast("Corners detected!")

        # Order them consistently
        st.session_state.corners = order_points(np.array(corners, dtype="float32")).astype(int).tolist()
        st.rerun()

else:
    # Editor view
    st.info("**Drag any dot** to precisely match the plate corners.")
    
    # Display the interactive component
    result = _draggable_canvas(st.session_state.pil_image, st.session_state.corners)
    
    # Handle component events
    if result:
        if result.get("action") == "clear":
            _clear()
            st.rerun()
        elif result.get("action") == "update":
            new_corners = result.get("corners")
            if new_corners and new_corners != st.session_state.corners:
                st.session_state.corners = new_corners
                st.rerun()

    st.markdown("---")
    
    col_btn, col_out = st.columns(2)
    
    with col_btn:
        st.markdown("### 1. Deskew & Enhance")
        is_valid, val_msg = validate_quadrilateral(st.session_state.corners, st.session_state.cv_image.shape)
        
        if is_valid:
            if st.button("**Rectify Plate**", use_container_width=True, type="primary"):
                # 1. Rectify (Deskew)
                out = rectify_plate(
                    st.session_state.cv_image, 
                    st.session_state.corners, 
                    output_width=400, 
                    output_height=130
                )
                st.session_state.rectified = out
                
                # 2. Enhance for OCR
                st.session_state.enhanced = enhance_plate_for_ocr(out)
        else:
            st.error(val_msg)

    with col_out:
        st.markdown("### 2. Output")
        if st.session_state.rectified is not None:
            show_enhanced = st.toggle("Apply OCR Sharpness & Contrast", value=True)
        
            display_img = (
                st.session_state.enhanced 
                if show_enhanced and st.session_state.enhanced is not None 
                else st.session_state.rectified
            )

            st.image(cv_to_pil(display_img), use_container_width=True)
            
            # Download button
            buf = io.BytesIO()
            cv_to_pil(display_img).save(buf, format="PNG")
            st.download_button(
                "Download Result",
                data=buf.getvalue(),
                file_name="enhanced_plate.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.write("Click 'Rectify Plate' to see the result here.")
