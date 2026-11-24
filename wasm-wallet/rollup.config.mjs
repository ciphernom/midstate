// Builds both browser libp2p bundles, and the QR reader's.
//
// Rollup accepts an array of configs, so `npx rollup -c` produces
// light_client.bundle.js, pool_client.bundle.js and qr_reader.bundle.js in one
// pass. Keeping them in
// a single file matters more than it looks: both entry points depend on the SAME
// libp2p and @libp2p/webrtc packages, so if their resolver settings ever drift
// apart the app ends up shipping two subtly different builds of one library.
// The shared `browserResolve` factory below makes that drift impossible.
//
//   npm install
//   npx rollup -c
//
// Both outputs are ES modules with named exports (`LightClient`, `PoolClient`),
// which is what net.js's dynamic `import()` calls expect.

import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';

// Single source of truth for module resolution. libp2p ships browser-specific
// entry points; without `browser: true` and these conditions the resolver can
// select Node builds that reference 'dgram'/'net' and fail at runtime rather
// than at build time.
const browserResolve = () =>
    resolve({
        preferBuiltins: false,
        browser: true,
        exportConditions: ['browser', 'import', 'module', 'default'],
    });

// libp2p emits a large number of benign circular-dependency warnings. Silencing
// only that class keeps genuine warnings visible.
const quietCircular = (warning, warn) => {
    if (warning.code === 'CIRCULAR_DEPENDENCY') return;
    warn(warning);
};

export default [
    // ── Light client: wallet ↔ full node ──
    {
        input: 'light_client.js',
        output: {
            file: 'light_client.bundle.js',
            format: 'es',
        },
        plugins: [browserResolve(), commonjs()],
        onwarn: quietCircular,
    },

    // ── Pool client: wallet ↔ mining pool ──
    {
        input: 'pool_client.js',
        output: {
            file: 'pool_client.bundle.js',
            format: 'es',
            // Forces a single output file even if a dependency uses a dynamic
            // import. `output.file` (as opposed to `output.dir`) cannot emit
            // multiple chunks, and net.js loads this by a fixed filename.
            inlineDynamicImports: true,
        },
        plugins: [browserResolve(), commonjs()],
        onwarn: quietCircular,
    },

    // ── QR reader: jsQR on a worker, for the send form's "Scan" ──
    // qr_scan.js starts it only when someone scans (the browser's own
    // BarcodeDetector is used instead where there is one).
    {
        input: 'qr_reader.js',
        output: {
            file: 'qr_reader.bundle.js',
            format: 'es',
            banner: '/*! jsQR 1.4.0, https://github.com/cozmo/jsQR, Apache License 2.0 */',
        },
        plugins: [browserResolve(), commonjs()],
    },
];
