import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';

export default {
    input: 'light_client.js',
    output: {
        file: 'light_client.bundle.js',
        format: 'es'
    },
    plugins: [
        resolve({ 
            preferBuiltins: false,
            browser: true,
            exportConditions: ['browser', 'import', 'module', 'default']
        }),
        commonjs()
    ]
};
