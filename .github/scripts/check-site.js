// check-site.js: run on the staged site before it is published. Fails if a
// page or script there names a file beside it ('./mining.js', '../x.wasm',
// …) that the site doesn't have, or if a page's src/href or a web app
// manifest's icons do. The deploy copies a list of files, and a file left
// off it is a 404 in production: one missing import blanks the wallet, or
// stops every mining thread. wasm-bindgen's pkg/ and rollup's bundles are
// skipped; their own references are internal.
//
//   node .github/scripts/check-site.js <staged directory>

const fs = require('fs');
const path = require('path');

const root = process.argv[2];
const walk = (dir) => fs.readdirSync(dir, { withFileTypes: true }).flatMap((d) =>
    d.isDirectory() ? walk(path.join(dir, d.name)) : [path.join(dir, d.name)]);
const pages = walk(root).filter((f) => /\.(m?js|html)$/.test(f)
    && !/\.bundle\.js$/.test(f) && !f.split(path.sep).includes('pkg'));

const missing = new Set();
const need = (file, ref) => {
    if (!fs.existsSync(path.join(path.dirname(file), ref))) missing.add(`${path.relative(root, file)}: ${ref}`);
};
for (const file of pages) {
    const src = fs.readFileSync(file, 'utf8');
    for (const m of src.matchAll(/['"`](\.\.?\/[\w./-]+\.(?:m?js|wasm|wgsl|json|css|webmanifest))['"`]/g)) need(file, m[1]);
    // A page's own files: <script src="app.js">, <link href="wallet.css">.
    if (file.endsWith('.html')) {
        for (const m of src.matchAll(/\b(?:src|href)=["']([\w][\w./-]*\.(?:m?js|css|json|svg|png|webmanifest|wasm))["']/g)) need(file, m[1]);
    }
}
// What an installed app's icons and start page are.
for (const file of walk(root).filter((f) => /(^|[\\/])manifest\.json$|\.webmanifest$/.test(f))) {
    let manifest;
    try { manifest = JSON.parse(fs.readFileSync(file, 'utf8')); } catch (_) { missing.add(`${path.relative(root, file)}: not JSON`); continue; }
    for (const icon of manifest.icons || []) if (icon && icon.src && !/^[a-z]+:/i.test(icon.src)) need(file, icon.src);
    if (manifest.start_url && !/^[a-z]+:/i.test(manifest.start_url)) need(file, manifest.start_url.split(/[?#]/)[0]);
}
if (missing.size) {
    console.error(`The site names files it doesn't have. Add them to the deploy's file list:\n  ${[...missing].join('\n  ')}`);
    process.exit(1);
}
console.log(`Checked ${pages.length} pages and scripts: every file they name is there.`);
