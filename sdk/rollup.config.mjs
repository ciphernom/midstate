import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';

export default {
    input: 'src/index.js', // <--- FIXED THIS LINE
    output: {
        file: 'dist/midstate-sdk.bundle.js',
        format: 'es',
        inlineDynamicImports: true
    },
    plugins: [
        resolve({ 
            browser: true,
            preferBuiltins: false 
        }),
        commonjs(),
        {
            name: 'ignore-node-builtins',
            resolveId(id) {
                if (['fs', 'fs/promises', 'path', 'os', 'crypto'].includes(id)) {
                    return id;
                }
                return null;
            },
            load(id) {
                if (['fs', 'fs/promises', 'path', 'os', 'crypto'].includes(id)) {
                    return 'export default {};';
                }
                return null;
            }
        }
    ]
};
