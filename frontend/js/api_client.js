class APIClient {
    constructor(baseUrl = '') {  // same origin as the page (works locally and when deployed)
        this.baseUrl = baseUrl;
    }

    async _errorFromResponse(response) {
        try {
            const body = await response.json();
            if (body && body.message) return new Error(body.message);
        } catch (e) { /* body was not JSON */ }
        return new Error(`HTTP error! status: ${response.status}`);
    }

    async recognize(imageFile) {
        const formData = new FormData();
        formData.append('file', imageFile);

        try {
            const response = await fetch(`${this.baseUrl}/api/v1/recognize`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw await this._errorFromResponse(response);
            }

            return await response.json();
        } catch (error) {
            console.error('Recognition API error:', error);
            throw error;
        }
    }

    async deskewInteractive(imageBase64, corners, snap = false) {
        try {
            const response = await fetch(`${this.baseUrl}/api/v1/interactive-deskew`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_base64: imageBase64,
                    corners: corners,
                    snap: snap,
                }),
            });

            if (!response.ok) {
                throw await this._errorFromResponse(response);
            }

            return await response.json();
        } catch (error) {
            console.error('Deskew API error:', error);
            throw error;
        }
    }

    async recognizeBox(imageBase64, box) {
        try {
            const response = await fetch(`${this.baseUrl}/api/v1/recognize-box`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_base64: imageBase64, box: box }),
            });

            if (!response.ok) {
                throw await this._errorFromResponse(response);
            }

            return await response.json();
        } catch (error) {
            console.error('Recognize-box API error:', error);
            throw error;
        }
    }

    async healthCheck() {
        try {
            const response = await fetch(`${this.baseUrl}/api/v1/health`);
            return await response.json();
        } catch (error) {
            console.error('Health check error:', error);
            return { status: 'error' };
        }
    }
}

// When this page runs inside the Streamlit app (as a custom component in an iframe) there is no HTTP API:
// requests travel to Python through Streamlit's component protocol and the answers come back as render args.
class StreamlitBridgeClient {
    constructor() {
        this.baseUrl = 'Streamlit';
        this.pending = new Map();
        this.seq = 0;
        this.session = Math.random().toString(36).slice(2, 8);
        this.health = null;
        this.healthWaiters = [];

        window.addEventListener('message', (event) => this.onMessage(event));
        this.post('streamlit:componentReady', { apiVersion: 1 });
        this.sizeFrame();
        try {
            window.parent.addEventListener('resize', () => this.sizeFrame());
        } catch (error) { /* parent not reachable: keep the initial size */ }
    }

    post(type, payload) {
        window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type: type }, payload || {}), '*');
    }

    sizeFrame() {
        let height;
        try {
            height = window.parent.innerHeight - 16;
        } catch (error) {
            height = Math.round(window.screen.availHeight * 0.85);
        }
        this.post('streamlit:setFrameHeight', { height: Math.max(640, height) });
    }

    onMessage(event) {
        const data = event.data;
        if (!data || data.type !== 'streamlit:render') return;
        const args = data.args || {};

        if (args.health && !this.health) {
            this.health = args.health;
            this.healthWaiters.splice(0).forEach((resolve) => resolve(this.health));
        }

        const reply = args.reply;
        if (reply && this.pending.has(reply.id)) {
            const waiter = this.pending.get(reply.id);
            this.pending.delete(reply.id);
            if (reply.error) waiter.reject(new Error(reply.error));
            else waiter.resolve(reply.result);
        }
        this.sizeFrame();
    }

    request(payload) {
        return new Promise((resolve, reject) => {
            const id = `${this.session}-${++this.seq}`;
            this.pending.set(id, { resolve, reject });
            this.post('streamlit:setComponentValue', { value: Object.assign({ id: id }, payload), dataType: 'json' });
            setTimeout(() => {
                if (this.pending.delete(id)) reject(new Error('The server took too long to answer.'));
            }, 300000);
        });
    }

    recognize(imageFile) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => this.request({ kind: 'recognize', name: imageFile.name || 'image', image: reader.result })
                .then(resolve, reject);
            reader.onerror = () => reject(new Error('Could not read the file.'));
            reader.readAsDataURL(imageFile);
        });
    }

    // The Python side keeps the uploaded image, so only the small parameters are sent.
    deskewInteractive(imageBase64, corners, snap = false) {
        return this.request({ kind: 'corners', corners: corners, snap: snap });
    }

    recognizeBox(imageBase64, box) {
        return this.request({ kind: 'box', box: box });
    }

    healthCheck() {
        if (this.health) return Promise.resolve(this.health);
        return new Promise((resolve) => {
            this.healthWaiters.push(resolve);
            setTimeout(() => resolve({ status: 'error' }), 30000);
        });
    }
}

const inStreamlit = window.parent !== window && /\/component\//.test(window.location.pathname);
const apiClient = inStreamlit ? new StreamlitBridgeClient() : new APIClient();
