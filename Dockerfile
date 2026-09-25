FROM python:3.13-slim

# runtime libraries used by OpenCV / PyTorch / scikit-learn
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run containers as user id 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only PyTorch (much smaller than the CUDA build)
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user backend/requirements-docker.txt backend/requirements-docker.txt
RUN pip install --no-cache-dir -r backend/requirements-docker.txt

# EasyOCR downloads its Thai/English models on first use: do it at build time instead
RUN python -c "import easyocr; easyocr.Reader(['th', 'en'], gpu=False)"

COPY --chown=user backend backend
COPY --chown=user frontend frontend

EXPOSE 7860
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
