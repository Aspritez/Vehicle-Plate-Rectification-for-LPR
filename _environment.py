"""Runtime environment detection: local vs Streamlit Community Cloud.

Streamlit Community Cloud sets the ``STREAMLIT_SHARING_MODE`` environment
variable to ``"true"`` before the app process starts.  Any other runtime
(local venv, Docker, custom server) leaves it unset, so the flag reliably
distinguishes the two cases without fragile path or hostname heuristics.
"""

from __future__ import annotations

import os
import platform
import sys


IS_STREAMLIT_CLOUD: bool = os.getenv("STREAMLIT_SHARING_MODE") == "true"


def runtime_info() -> dict[str, str]:
    """Return a snapshot of the current runtime for display or logging."""
    return {
        "environment": "Streamlit Community Cloud" if IS_STREAMLIT_CLOUD else "local",
        "python": sys.version.split()[0],
        "platform": platform.platform(terse=True),
        "opencv": _opencv_version(),
    }


def _opencv_version() -> str:
    try:
        import cv2  # noqa: PLC0415
        return cv2.__version__
    except ImportError:
        return "not installed"
