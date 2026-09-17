"""UI smoke checks using procedural images, independent of dataset evaluation."""

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from src.imaging import png_bytes
from src.geometry import project_points

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_empty_app_and_mode_switch_are_usable():
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    assert not app.exception
    assert app.title[0].value == "ปรับมุมป้ายทะเบียนให้พร้อมอ่าน"
    app.radio[0].set_value("มี reference · SIFT / ORB").run()
    assert not app.exception
    assert any("ภาพด้านหน้า" in x.value for x in app.info)
    next(b for b in app.button if b.label == "ล้างงานทั้งหมด").click().run()
    assert not app.exception
    assert app.session_state["upload_generation"] == 1


def test_single_image_manual_coordinates_to_downloads():
    data = png_bytes(np.full((160, 360, 3), 230, np.uint8))
    with patch("streamlit.file_uploader", return_value=BytesIO(data)), \
         patch("streamlit_image_coordinates.streamlit_image_coordinates", return_value=None):
        app = AppTest.from_file(APP_PATH, default_timeout=20).run()
        assert not app.exception
        next(b for b in app.button if b.label == "ระบุมุมเองบนภาพเต็ม").click().run()
        app.text_area[0].set_value("15, 15\n340, 20\n335, 145\n20, 140")
        next(b for b in app.button if b.label == "ใช้พิกัดนี้").click().run()
        assert not app.exception
        next(b for b in app.button if b.label == "ปรับภาพจากมุมที่เลือก").click().run()
        assert not app.exception
        assert len(app.success) == 1
        assert app.session_state["result"]["accepted"]
        assert "4 corners" in app.session_state["result"]["method"]
        assert len(app.get("download_button")) == 3
        # Updating geometry must remove stale output before processing again.
        next(s for s in app.slider if s.label == "ความกว้างผลลัพธ์ (px)").set_value(800).run()
        assert not app.exception
        assert len(app.get("download_button")) == 0


def test_feature_failure_shows_reason_without_downloads():
    data = png_bytes(np.full((160, 360, 3), 230, np.uint8))
    with patch("streamlit.file_uploader", return_value=BytesIO(data)):
        app = AppTest.from_file(APP_PATH, default_timeout=20).run()
        app.radio[0].set_value("มี reference · SIFT / ORB").run()
        assert not app.exception
        next(b for b in app.button if b.label == "ปรับป้ายให้ตรง").click().run()
        assert not app.exception
        assert not app.session_state["result"]["accepted"]
        assert len(app.warning) >= 1
        assert len(app.get("download_button")) == 0


def test_feature_success_exports_full_image_coordinates_after_cropping():
    rng = np.random.default_rng(71)
    data = png_bytes(rng.integers(0, 256, (260, 600, 3), dtype=np.uint8))
    with patch("streamlit.file_uploader", return_value=BytesIO(data)), \
         patch("streamlit_image_coordinates.streamlit_image_coordinates", return_value=None):
        app = AppTest.from_file(APP_PATH, default_timeout=20).run()
        app.radio[0].set_value("มี reference · SIFT / ORB").run()
        left_inputs = [n for n in app.number_input if n.label == "ซ้าย (x)"]
        left_inputs[0].set_value(20)
        left_inputs[1].set_value(40)
        app.run()
        next(b for b in app.button if b.label == "ปรับป้ายให้ตรง").click().run()
        assert not app.exception
        result = app.session_state["result"]
        assert result["accepted"], result["reason"]
        assert result["warped"].shape == (260, 560, 3)
        assert np.allclose(project_points(np.array([[100, 100]]), result["homography"]), [[60, 100]], atol=1)
        assert len(app.get("download_button")) == 3


def test_auto_single_click_then_manual_correction_uses_full_image():
    from test_detection import perspective_scene
    image, _ = perspective_scene()
    with patch("streamlit.file_uploader", return_value=BytesIO(png_bytes(image))), \
         patch("streamlit_image_coordinates.streamlit_image_coordinates", return_value=None):
        app = AppTest.from_file(APP_PATH, default_timeout=20).run()
        next(b for b in app.button if b.label == "ค้นหาและปรับป้าย").click().run()
        assert not app.exception
        result = app.session_state["result"]
        assert result["accepted"] and result["method"].startswith("Auto:")
        import json
        from src.detection import DetectorConfig
        frozen = json.loads((APP_PATH.parent / "evaluation/tuned_config.json").read_text())
        assert result["metrics"]["detector_thresholds"] == vars(DetectorConfig(**frozen))
        assert len(app.get("download_button")) == 3
        next(b for b in app.button if b.label == "เลือกผิดป้าย / แก้มุมด้วยตนเอง").click().run()
        assert not app.exception
        assert len(app.get("download_button")) == 0
        assert app.session_state["work_single_manual"]
        # Full-image coordinates extend beyond the plate ROI, yet are valid.
        app.text_area[0].set_value("100, 80\n900, 90\n890, 530\n110, 520")
        next(b for b in app.button if b.label == "ใช้พิกัดนี้").click().run()
        next(b for b in app.button if b.label == "ปรับภาพจากมุมที่เลือก").click().run()
        assert not app.exception
        corrected = app.session_state["result"]
        assert corrected["source_box"] == (0, 0, 1000, 600)
        assert "4 corners" in corrected["method"]
        assert np.allclose(project_points(np.array([[100, 80]]), corrected["homography"]), [[0, 0]], atol=.01)


def test_auto_rejection_opens_manual_without_old_downloads():
    data = png_bytes(np.full((200, 400, 3), 240, np.uint8))
    with patch("streamlit.file_uploader", return_value=BytesIO(data)), \
         patch("streamlit_image_coordinates.streamlit_image_coordinates", return_value=None):
        app = AppTest.from_file(APP_PATH, default_timeout=20).run()
        next(b for b in app.button if b.label == "ค้นหาและปรับป้าย").click().run()
        assert not app.exception
        assert len(app.warning) >= 1
        assert app.session_state["work_single_manual"]
        assert len(app.text_area) == 1
        assert len(app.get("download_button")) == 0


def test_auto_settings_change_invalidates_output_and_diagnostics():
    from test_detection import perspective_scene
    image, _ = perspective_scene()
    with patch("streamlit.file_uploader", return_value=BytesIO(png_bytes(image))):
        app = AppTest.from_file(APP_PATH, default_timeout=20).run()
        next(b for b in app.button if b.label == "ค้นหาและปรับป้าย").click().run()
        assert len(app.get("download_button")) == 3
        next(s for s in app.slider if s.label == "Canny ต่ำ").set_value(60).run()
        assert not app.exception
        assert len(app.get("download_button")) == 0
        assert "work_single_detection" not in app.session_state
