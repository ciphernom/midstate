// qr_reader.js: decodes QR codes for qr_scan.js off the page's thread, with
// jsQR. Bundled into qr_reader.bundle.js (`npm run bundle`).
//
// In:  { id, width, height, data: ArrayBuffer of RGBA pixels, invert }
//      (`invert` is jsQR's inversionAttempts)
// Out: { id, text }, text null when there is no readable code.

import jsQR from 'jsqr';

self.onmessage = (e) => {
    const { id, width, height, data, invert } = e.data || {};
    let text = null;
    try {
        const code = jsQR(new Uint8ClampedArray(data), width, height, {
            inversionAttempts: invert || 'attemptBoth',
        });
        if (code) text = code.data;
    } catch (_) {
        // An unreadable frame is a frame without a code.
    }
    self.postMessage({ id, text });
};
