class CanvasViewer {
    constructor(canvasId, imageData) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.image = new Image();
        this.corners = [];
        this.draggingCorner = null;
        this.drawStart = null;      // where a new box is being drawn from (empty space was pressed)
        this.drawRect = null;       // the box being drawn, for the overlay
        this.onBox = null;          // callback(box) when a box has been drawn: [x0, y0, x1, y1] in image pixels
        this.handleCssRadius = 8;  // corner handle size on screen, independent of the image resolution

        this.image.onload = () => {
            this.resizeCanvasToImage();
            // No corners supplied (e.g. nothing detected): start from a centred plate-shaped box.
            if (this.corners.length !== 4) this.setDefaultCorners();
            this.draw();
        };
        if (imageData) this.setImage(imageData);

        // Pointer events keep the drag alive outside the canvas and also work on touch screens.
        this.canvas.style.touchAction = 'none';
        this.canvas.addEventListener('pointermove', (e) => this.onPointerMove(e));
        this.canvas.addEventListener('pointerdown', (e) => this.onPointerDown(e));
        this.canvas.addEventListener('pointerup', (e) => this.onPointerUp(e));
        this.canvas.addEventListener('pointercancel', (e) => this.onPointerUp(e));
    }

    setImage(imageData) {
        this.image.src = imageData;
    }

    // Canvas pixels per on-screen pixel (the canvas is scaled by CSS).
    displayScale() {
        const rect = this.canvas.getBoundingClientRect();
        return rect.width > 0 ? this.canvas.width / rect.width : 1;
    }

    handleRadius() {
        return this.handleCssRadius * this.displayScale();
    }

    // Pointer position in canvas (image) pixels.
    getCanvasPosition(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left) * (this.canvas.width / rect.width),
            y: (e.clientY - rect.top) * (this.canvas.height / rect.height),
        };
    }

    resizeCanvasToImage() {
        this.canvas.width = this.image.width;
        this.canvas.height = this.image.height;
    }

    setDefaultCorners() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        const boxH = Math.min(w * 0.4 / 2.26, h * 0.4);
        const boxW = boxH * 2.26;
        const x0 = (w - boxW) / 2;
        const y0 = (h - boxH) / 2;
        this.corners = [[x0, y0], [x0 + boxW, y0], [x0 + boxW, y0 + boxH], [x0, y0 + boxH]]
            .map(c => ({ x: c[0], y: c[1] }));
    }

    setCorners(corners) {
        this.corners = Array.isArray(corners) && corners.length === 4
            ? corners.map(c => ({ x: c[0], y: c[1] }))
            : [];
        this.draw();
    }

    getCorners() {
        return this.corners.map(c => [c.x, c.y]);
    }

    onPointerMove(e) {
        const { x, y } = this.getCanvasPosition(e);

        if (this.drawStart) {
            this.drawRect = this.rectBetween(this.drawStart, { x, y });
            this.draw();
            return;
        }

        if (!this.draggingCorner) {
            this.canvas.style.cursor = this.getCornerAtPosition(x, y) ? 'grab' : 'crosshair';
            return;
        }

        this.draggingCorner.x = Math.min(Math.max(x, 0), this.canvas.width);
        this.draggingCorner.y = Math.min(Math.max(y, 0), this.canvas.height);
        this.draw();
    }

    onPointerDown(e) {
        const { x, y } = this.getCanvasPosition(e);

        this.draggingCorner = this.getCornerAtPosition(x, y);
        if (this.draggingCorner) {
            this.canvas.setPointerCapture(e.pointerId);
            this.canvas.style.cursor = 'grabbing';
            e.preventDefault();
        } else {
            // Pressed on empty space: start drawing a rough box around the plate.
            this.drawStart = { x, y };
            this.drawRect = null;
            this.canvas.setPointerCapture(e.pointerId);
            e.preventDefault();
        }
    }

    onPointerUp(e) {
        if (this.canvas.hasPointerCapture(e.pointerId)) {
            this.canvas.releasePointerCapture(e.pointerId);
        }

        if (this.drawStart) {
            const rect = this.drawRect;
            this.drawStart = null;
            this.drawRect = null;
            this.draw();
            const minSide = 14 * this.displayScale();   // ignore clicks and tiny accidental drags
            if (rect && rect.x1 - rect.x0 >= minSide && rect.y1 - rect.y0 >= minSide && this.onBox) {
                this.onBox([rect.x0, rect.y0, rect.x1, rect.y1]);
            }
        }

        this.draggingCorner = null;
        this.canvas.style.cursor = 'crosshair';
    }

    rectBetween(a, b) {
        const clamp = (v, max) => Math.min(Math.max(v, 0), max);
        return {
            x0: clamp(Math.min(a.x, b.x), this.canvas.width),
            y0: clamp(Math.min(a.y, b.y), this.canvas.height),
            x1: clamp(Math.max(a.x, b.x), this.canvas.width),
            y1: clamp(Math.max(a.y, b.y), this.canvas.height),
        };
    }

    setBoxCorners(box) {
        const [x0, y0, x1, y1] = box;
        this.setCorners([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
    }

    getCornerAtPosition(x, y) {
        const reach = this.handleRadius() * 1.6;
        let nearest = null;
        let best = reach;
        for (const corner of this.corners) {
            const dist = Math.hypot(corner.x - x, corner.y - y);
            if (dist < best) {
                best = dist;
                nearest = corner;
            }
        }
        return nearest;
    }

    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.image, 0, 0);

        if (this.corners.length === 4) {
            const radius = this.handleRadius();

            this.ctx.strokeStyle = '#4a9eff';
            this.ctx.lineWidth = 2 * this.displayScale();
            this.ctx.beginPath();
            this.ctx.moveTo(this.corners[0].x, this.corners[0].y);
            for (let i = 1; i < 4; i++) this.ctx.lineTo(this.corners[i].x, this.corners[i].y);
            this.ctx.closePath();
            this.ctx.stroke();

            this.corners.forEach((corner, idx) => {
                this.ctx.fillStyle = '#4a9eff';
                this.ctx.beginPath();
                this.ctx.arc(corner.x, corner.y, radius, 0, 2 * Math.PI);
                this.ctx.fill();

                this.ctx.fillStyle = '#1a1a2e';
                this.ctx.font = `bold ${Math.round(radius * 1.3)}px Arial`;
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                this.ctx.fillText((idx + 1).toString(), corner.x, corner.y);
            });
        }

        if (this.drawRect) {
            const r = this.drawRect;
            const unit = this.displayScale();
            this.ctx.fillStyle = 'rgba(255, 138, 61, 0.18)';
            this.ctx.fillRect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0);
            this.ctx.strokeStyle = '#ff8a3d';
            this.ctx.lineWidth = 2 * unit;
            this.ctx.setLineDash([6 * unit, 4 * unit]);
            this.ctx.strokeRect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0);
            this.ctx.setLineDash([]);
        }
    }
}

class PipelineViewer {
    constructor() {
        this.currentStep = 'original';
        this.images = {};
        this.initTabListeners();
    }

    initTabListeners() {
        document.querySelectorAll('.pipeline-tab').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.pipeline-tab').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.currentStep = e.target.dataset.step;
                this.display();
            });
        });
    }

    setImage(step, imageBase64) {
        this.images[step] = imageBase64;
    }

    setAllImages(data) {
        if (data.visualization_base64) this.setImage('original', data.visualization_base64);
        if (data.visualization_base64) this.setImage('detected', data.visualization_base64);
        if (data.deskewed_plate_base64) this.setImage('deskewed', data.deskewed_plate_base64);
        if (data.enhanced_plate_base64) this.setImage('enhanced', data.enhanced_plate_base64);
    }

    display() {
        const img = document.getElementById('pipelineImage');
        const info = document.getElementById('pipelineInfo');

        if (this.images[this.currentStep]) {
            img.src = this.images[this.currentStep];
            img.style.display = 'block';
            info.style.display = 'none';
        } else {
            img.style.display = 'none';
            info.textContent = `No ${this.currentStep} image available`;
            info.style.display = 'block';
        }
    }
}

const pipelineViewer = new PipelineViewer();
