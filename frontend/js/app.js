class LicensePlateApp {
    constructor() {
        this.currentImage = null;
        this.currentImageBase64 = null;
        this.currentResult = null;
        this.corners = null;
        this.canvasViewer = null;

        this.setupEventListeners();
        this.checkBackendHealth();
    }

    setupEventListeners() {
        // Drag & Drop
        const dragDropZone = document.getElementById('dragDropZone');
        const fileInput = document.getElementById('fileInput');

        dragDropZone.addEventListener('click', () => fileInput.click());
        document.getElementById('navUploadBtn').addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e.target.files[0]));

        dragDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dragDropZone.classList.add('dragover');
        });

        dragDropZone.addEventListener('dragleave', () => dragDropZone.classList.remove('dragover'));
        dragDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dragDropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                this.handleFileSelect(e.dataTransfer.files[0]);
            }
        });

        // Tab Switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
                e.target.classList.add('active');
                const tabId = `${e.target.dataset.tab}-tab`;
                const tabContent = document.getElementById(tabId);
                if (tabContent) tabContent.style.display = 'block';
            });
        });

        // Buttons
        document.getElementById('copyBtn').addEventListener('click', () => this.copyToClipboard());
        document.getElementById('editBtn').addEventListener('click', () => this.openEditModal());
        document.getElementById('resetCornersBtn').addEventListener('click', () => this.resetCorners());
        document.getElementById('applyCornersBtn').addEventListener('click', () => this.applyCornersChange());

        // Webcam
        document.getElementById('captureBtn').addEventListener('click', () => this.captureWebcam());

        // Modal
        const editModal = document.getElementById('editModal');
        document.querySelector('.modal-close').addEventListener('click', () => {
            editModal.classList.remove('active');
        });
        document.getElementById('cancelEditBtn').addEventListener('click', () => {
            editModal.classList.remove('active');
        });
        document.getElementById('saveEditBtn').addEventListener('click', () => this.saveEditText());

        window.addEventListener('click', (e) => {
            if (e.target === editModal) {
                editModal.classList.remove('active');
            }
        });
    }

    async handleFileSelect(file) {
        if (!file) return;

        const reader = new FileReader();
        reader.onload = async (e) => {
            this.currentImageBase64 = e.target.result;
            pipelineViewer.setImage('original', this.currentImageBase64);
            pipelineViewer.display();

            await this.recognizePlate(file);
        };
        reader.readAsDataURL(file);
    }

    async recognizePlate(file) {
        try {
            document.getElementById('dragDropZone').innerHTML = '<div class="loading"></div><p>Processing...</p>';

            const result = await apiClient.recognize(file);

            if (result.status === 'success' && result.plate_found) {
                this.acceptResult(result);
                this.setCornerStatus('Plate located automatically. Drag the corners if the box is off, then press Apply.');
            } else {
                // Not found: let the user place the 4 corners by hand instead of giving up.
                this.currentResult = null;
                this.detectedCorners = result.suggested_corners || result.corners;
                document.getElementById('dragDropZone').innerHTML =
                    '<p style="color: #d9822b;">Plate not recognized automatically</p>';
                this.showCornerAdjustment(this.detectedCorners);
                const reason = (result.message || 'No plate detected').replace(/[.\s]+$/, '');
                this.setCornerStatus(`${reason}. Drag the 4 corners onto the plate, then press Apply.`);
            }
        } catch (error) {
            console.error('Error:', error);
            this.showMessage(`Error processing image: ${error.message}`);
            this.resetUI();
        }
    }

    acceptResult(result) {
        this.currentResult = result;
        this.detectedCorners = result.corners;
        this.displayResults(result);
        this.showCornerAdjustment(result.corners);
    }

    setCornerStatus(text) {
        document.getElementById('cornerStatus').textContent = text;
    }

    displayResults(result) {
        const processingCard = document.getElementById('processingCard');

        document.getElementById('plateText').textContent = result.ocr_result.text || '-';
        document.getElementById('provinceText').textContent = result.ocr_result.province || '-';
        document.getElementById('confidenceValue').textContent =
            `${Math.round(result.ocr_result.confidence * 100)}%`;
        document.getElementById('confidenceFill').style.width =
            `${result.ocr_result.confidence * 100}%`;

        const badge = document.getElementById('statusBadge');
        if (result.ocr_result.is_valid) {
            badge.textContent = 'Valid';
            badge.classList.remove('invalid');
        } else {
            badge.textContent = 'Invalid Format';
            badge.classList.add('invalid');
        }

        pipelineViewer.setAllImages(result);
        pipelineViewer.display();

        document.getElementById('srNote').style.display = result.super_resolution_used ? 'block' : 'none';

        document.getElementById('resultsEmpty').style.display = 'none';
        document.querySelector('.steps').style.display = 'none';
        document.getElementById('resultsBody').style.display = 'flex';
        processingCard.style.display = 'inline';
        document.getElementById('processingTime').textContent = `${result.processing_time_ms}ms`;

        document.getElementById('dragDropZone').innerHTML =
            '<p style="color: #28a745;">✓ Image loaded successfully</p>';
    }

    showCornerAdjustment(corners) {
        document.getElementById('cornerAdjustSection').style.display = 'flex';
        document.getElementById('cornerEmpty').style.display = 'none';

        // Reuse one viewer: creating a new one per image would stack event listeners on the same canvas.
        if (!this.canvasViewer) {
            this.canvasViewer = new CanvasViewer('cornerCanvas');
            this.canvasViewer.onBox = (box) => this.recognizeBox(box);
        }
        this.canvasViewer.setImage(this.currentImageBase64);
        this.canvasViewer.setCorners(corners);
    }

    async recognizeBox(box) {
        if (!this.currentImageBase64) return;

        const applyBtn = document.getElementById('applyCornersBtn');
        try {
            applyBtn.disabled = true;
            this.setCornerStatus('Finding the plate inside your box...');
            this.canvasViewer.setBoxCorners(box);   // show the box while working

            const result = await apiClient.recognizeBox(this.currentImageBase64, box);

            if (result.status === 'success') {
                this.detectedCorners = result.corners;
                this.canvasViewer.setCorners(result.corners);
                this.displayResults(result);
                this.setCornerStatus('Plate found inside your box. Drag the corners to fine-tune, then press Apply.');
            } else {
                const corners = result.corners || result.suggested_corners;
                if (corners) this.canvasViewer.setCorners(corners);
                const reason = (result.message || 'Could not read the plate').replace(/[.\s]+$/, '');
                this.setCornerStatus(`${reason}.`);
            }
        } catch (error) {
            console.error('Error:', error);
            this.setCornerStatus(`Error: ${error.message}`);
        } finally {
            applyBtn.disabled = false;
        }
    }

    resetCorners() {
        if (this.canvasViewer && this.detectedCorners) {
            this.canvasViewer.setCorners(this.detectedCorners);
        }
    }

    async applyCornersChange() {
        if (!this.canvasViewer || !this.currentImageBase64) return;

        try {
            document.getElementById('applyCornersBtn').disabled = true;
            this.setCornerStatus('Recognizing...');

            const snap = document.getElementById('snapCheckbox').checked;
            const result = await apiClient.deskewInteractive(
                this.currentImageBase64, this.canvasViewer.getCorners(), snap);

            if (snap && result.corners) {
                this.canvasViewer.setCorners(result.corners);  // show where the corners were snapped to
            }

            if (result.status === 'success') {
                this.currentResult = result;
                this.displayResults(result);
                        this.setCornerStatus('Done. Adjust the corners and press Apply again to re-run.');
            } else {
                this.setCornerStatus(result.message || 'Could not recognize the plate from these corners.');
            }
        } catch (error) {
            console.error('Error:', error);
            this.setCornerStatus(`Error applying corner adjustment: ${error.message}`);
        } finally {
            document.getElementById('applyCornersBtn').disabled = false;
        }
    }

    copyToClipboard() {
        const text = document.getElementById('plateText').textContent;
        navigator.clipboard.writeText(text).then(() => {
            this.showMessage('Copied to clipboard!');
        });
    }

    openEditModal() {
        const text = document.getElementById('plateText').textContent;
        document.getElementById('editInput').value = text;
        document.getElementById('editModal').classList.add('active');
    }

    saveEditText() {
        const newText = document.getElementById('editInput').value.trim();
        if (newText) {
            document.getElementById('plateText').textContent = newText;
            document.getElementById('editModal').classList.remove('active');
            this.showMessage('Text updated!');
        }
    }

    async captureWebcam() {
        const video = document.getElementById('webcamVideo');
        if (!video.srcObject) {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            return;
        }

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);

        canvas.toBlob((blob) => {
            const file = new File([blob], 'webcam.jpg', { type: 'image/jpeg' });
            this.handleFileSelect(file);
        });
    }

    showMessage(message) {
        alert(message);
    }

    resetUI() {
        document.getElementById('dragDropZone').innerHTML = `
            <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
            <p class="drop-text">Drag & Drop Image or <span class="browse-link">Browse</span></p>
            <p class="drop-subtext">Supported: JPG, PNG, WEBP</p>
        `;
    }

    async checkBackendHealth() {
        const pill = document.getElementById('healthPill');
        const text = document.getElementById('healthText');
        try {
            const health = await apiClient.healthCheck();
            if (health.status !== 'healthy') throw new Error('unhealthy');
            const extras = [];
            if (!health.locator_loaded) extras.push('no locator model');
            if (health.super_resolution_available) extras.push('super-res');
            pill.className = 'health online';
            text.textContent = 'Backend online' + (extras.length ? ` · ${extras.join(' · ')}` : '');
        } catch (error) {
            pill.className = 'health offline';
            text.textContent = 'Backend offline';
            console.warn('Backend not responding at', apiClient.baseUrl);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new LicensePlateApp();
});
