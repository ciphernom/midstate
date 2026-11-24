// qr_scan.js: reads QR codes for the send form's "scan" button, from the
// camera or from a photo. The browser's own BarcodeDetector reads them where
// it has one (Chrome on Android and macOS); everywhere else jsQR does, on a
// worker (qr_reader.js, bundled into qr_reader.bundle.js). The page loads
// this module only when someone scans.

// ── decoding ────────────────────────────────────────────────────────────────

let reader = null; // the jsQR worker, started on first use
let seq = 0;
const waiting = new Map();

function readerWorker() {
    if (!reader) {
        reader = new Worker(new URL('./qr_reader.bundle.js', import.meta.url), { type: 'module' });
        reader.onmessage = (e) => {
            const { id, text } = e.data || {};
            const w = waiting.get(id);
            if (w) {
                waiting.delete(id);
                w.resolve(text ?? null);
            }
        };
        reader.onerror = (e) => {
            e.preventDefault();
            for (const w of waiting.values()) w.reject(new Error('the QR reader did not load'));
            waiting.clear();
            reader = null;
        };
    }
    return reader;
}

/** jsQR on the worker; `invert` is its inversionAttempts. */
function readOnWorker(image, invert) {
    return new Promise((resolve, reject) => {
        const id = ++seq;
        waiting.set(id, { resolve, reject });
        const { width, height, data } = image;
        readerWorker().postMessage({ id, width, height, data: data.buffer, invert }, [data.buffer]);
    });
}

let detector; // undefined until looked for; then a BarcodeDetector, or null
async function nativeReader() {
    if (detector === undefined) {
        detector = null;
        try {
            if (typeof BarcodeDetector === 'function'
                && (await BarcodeDetector.getSupportedFormats()).includes('qr_code')) {
                detector = new BarcodeDetector({ formats: ['qr_code'] });
            }
        } catch (_) {
            // jsQR, then.
        }
    }
    return detector;
}

/** The text of the first QR code drawn on `canvas`, or null. */
async function readCanvas(canvas, ctx, invert) {
    const native = await nativeReader();
    if (native) {
        try {
            const found = await native.detect(canvas);
            return found.length ? found[0].rawValue : null;
        } catch (_) {
            detector = null; // one that cannot read a canvas: jsQR from now on
        }
    }
    return readOnWorker(ctx.getImageData(0, 0, canvas.width, canvas.height), invert);
}

function surface(width, height) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return { canvas, ctx: canvas.getContext('2d', { willReadFrequently: true }) };
}

// ── the camera ──────────────────────────────────────────────────────────────

/** Whether this page may ask for a camera at all (HTTPS or localhost). */
export const cameraSupported = () =>
    !!(window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

/** The cameras the browser lists. */
export async function cameraList() {
    try {
        return (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
    } catch (_) {
        return [];
    }
}

const stopTracks = stream => { for (const t of stream.getTracks()) t.stop(); };

/** A camera shown in `video` and read one frame at a time. */
export class Camera {
    constructor(video) {
        this.video = video;
        this.stream = null;
        this.closed = false;
        this.frames = 0;
        this.surface = null;
        /** Called when the browser or the system ends the camera (not stop()). */
        this.onended = null;
    }

    /** Starts the back camera, or `deviceId`; rejects as getUserMedia does. */
    async start(deviceId) {
        this.stop();
        const which = deviceId ? { deviceId: { exact: deviceId } } : { facingMode: { ideal: 'environment' } };
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: { ...which, width: { ideal: 1280 }, height: { ideal: 720 } },
        });
        if (this.closed) {
            // Closed while the browser was asking for permission.
            stopTracks(stream);
            return;
        }
        this.stop(); // a start that overlapped this one
        this.stream = stream;
        for (const track of stream.getVideoTracks()) {
            // Fires only when something else ends it (stop() does not).
            track.addEventListener('ended', () => {
                if (this.stream !== stream) return;
                this.stop();
                if (this.onended) this.onended();
            });
        }
        this.video.srcObject = stream;
        try {
            await this.video.play();
        } catch (_) {
            // Stopped before it played, or already playing (autoplay).
        }
    }

    get live() {
        return !!this.stream;
    }

    deviceId() {
        const track = this.stream && this.stream.getVideoTracks()[0];
        return (track && track.getSettings().deviceId) || null;
    }

    /** Stops the camera, so the browser's camera light goes out. */
    stop() {
        if (this.stream) stopTracks(this.stream);
        this.stream = null;
        this.video.srcObject = null;
    }

    /** Stops it for good, including a start still waiting for permission. */
    close() {
        this.closed = true;
        this.stop();
    }

    /** The text of a QR code in the current frame, or null. Reads the
     *  frame's centred square: the part the page shows. */
    async read() {
        const v = this.video;
        if (!this.stream || v.readyState < 2 || !v.videoWidth) return null;
        const side = Math.min(v.videoWidth, v.videoHeight);
        const size = Math.min(side, 640);
        if (!this.surface || this.surface.canvas.width !== size) this.surface = surface(size, size);
        const { canvas, ctx } = this.surface;
        ctx.drawImage(v, (v.videoWidth - side) / 2, (v.videoHeight - side) / 2, side, side, 0, 0, size, size);
        // Alternate frames between dark-on-light codes and light-on-dark ones.
        return readCanvas(canvas, ctx, this.frames++ % 2 ? 'onlyInvert' : 'dontInvert');
    }
}

// ── photos ──────────────────────────────────────────────────────────────────

/** The text of a QR code in an image file (a photo or a screenshot), or null. */
export async function readImageFile(file) {
    const image = await createImageBitmap(file);
    try {
        // Small first: quick, and enough for a screenshot. Then larger, for a
        // code that is small in a big photo.
        for (const max of [1024, 2048]) {
            const scale = Math.min(1, max / Math.max(image.width, image.height));
            const width = Math.max(1, Math.round(image.width * scale));
            const height = Math.max(1, Math.round(image.height * scale));
            const { canvas, ctx } = surface(width, height);
            ctx.drawImage(image, 0, 0, width, height);
            const text = await readCanvas(canvas, ctx, 'attemptBoth');
            if (text != null) return text;
            if (scale === 1) break;
        }
        return null;
    } finally {
        if (image.close) image.close();
    }
}
