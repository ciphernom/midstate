// check-site.js: run on the staged site before it is published. Fails if a
// page or script there names a file beside it ('./mining.js', '../x.wasm',
// …) that the site doesn't have. The deploy copies a list of files, and a
// file left off it is a 404 in production: one missing import blanks the
// wallet, or stops every mining thread. wasm-bindgen's pkg/ and rollup's
// bundles are skipped; their own references are internal.
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
for (const file of pages) {
    const src = fs.readFileSync(file, 'utf8');
    for (const m of src.matchAll(/['"`](\.\.?\/[\w./-]+\.(?:m?js|wasm|wgsl|json|css|webmanifest))['"`]/g)) {
        if (!fs.existsSync(path.join(path.dirname(file), m[1]))) missing.add(`${path.relative(root, file)}: ${m[1]}`);
    }
}
if (missing.size) {
    console.error(`The site names files it doesn't have. Add them to the deploy's file list:\n  ${[...missing].join('\n  ')}`);
    process.exit(1);
}
console.log(`Checked ${pages.length} pages and scripts: every file they name is there.`);
