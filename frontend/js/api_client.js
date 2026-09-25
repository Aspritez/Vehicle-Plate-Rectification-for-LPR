class APIClient {
    constructor(baseUrl = 'http://localhost:8000') {
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

const apiClient = new APIClient();
