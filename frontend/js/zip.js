// Minimal ZIP writer (no compression, "store" method) so the pipeline images can be downloaded as one file
// without any library. Usage: ZipTools.makeZip([{name: 'a.png', data: Uint8Array}, ...]) -> Blob
const ZipTools = (() => {
    const table = (() => {
        const t = new Uint32Array(256);
        for (let n = 0; n < 256; n++) {
            let c = n;
            for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
            t[n] = c >>> 0;
        }
        return t;
    })();

    function crc32(bytes) {
        let c = 0xFFFFFFFF;
        for (let i = 0; i < bytes.length; i++) c = table[(c ^ bytes[i]) & 0xFF] ^ (c >>> 8);
        return (c ^ 0xFFFFFFFF) >>> 0;
    }

    function makeZip(files) {
        const encoder = new TextEncoder();
        const now = new Date();
        const dosTime = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1);
        const dosDate = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
        const parts = [];
        const central = [];
        let offset = 0;
        let centralSize = 0;

        files.forEach((file) => {
            const name = encoder.encode(file.name);
            const crc = crc32(file.data);

            const local = new DataView(new ArrayBuffer(30));
            local.setUint32(0, 0x04034b50, true);
            local.setUint16(4, 20, true);
            local.setUint16(6, 0x0800, true);          // UTF-8 file names
            local.setUint16(8, 0, true);               // stored, not compressed
            local.setUint16(10, dosTime, true);
            local.setUint16(12, dosDate, true);
            local.setUint32(14, crc, true);
            local.setUint32(18, file.data.length, true);
            local.setUint32(22, file.data.length, true);
            local.setUint16(26, name.length, true);
            local.setUint16(28, 0, true);
            parts.push(local.buffer, name, file.data);

            const entry = new DataView(new ArrayBuffer(46));
            entry.setUint32(0, 0x02014b50, true);
            entry.setUint16(4, 20, true);
            entry.setUint16(6, 20, true);
            entry.setUint16(8, 0x0800, true);
            entry.setUint16(10, 0, true);
            entry.setUint16(12, dosTime, true);
            entry.setUint16(14, dosDate, true);
            entry.setUint32(16, crc, true);
            entry.setUint32(20, file.data.length, true);
            entry.setUint32(24, file.data.length, true);
            entry.setUint16(28, name.length, true);
            entry.setUint32(42, offset, true);
            central.push(entry.buffer, name);

            offset += 30 + name.length + file.data.length;
            centralSize += 46 + name.length;
        });

        const end = new DataView(new ArrayBuffer(22));
        end.setUint32(0, 0x06054b50, true);
        end.setUint16(8, files.length, true);
        end.setUint16(10, files.length, true);
        end.setUint32(12, centralSize, true);
        end.setUint32(16, offset, true);
        return new Blob(parts.concat(central, [end.buffer]), { type: 'application/zip' });
    }

    // "data:image/png;base64,...." -> {bytes: Uint8Array, mime, ext}
    function decodeDataUrl(url) {
        const match = /^data:([^;,]+)(;base64)?,([\s\S]*)$/.exec(url);
        if (!match) return null;
        const mime = match[1];
        const binary = match[2] ? atob(match[3]) : decodeURIComponent(match[3]);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        const ext = { 'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp' }[mime] || 'bin';
        return { bytes, mime, ext };
    }

    function save(blob, filename) {
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 5000);
    }

    return { makeZip, decodeDataUrl, save };
})();
