import { compute_coin_id_hex, blake3_hash_hex } from '../pkg/wasm_wallet.js';

const MDS_KILO = 1024;
const MDS_MEGA = 1048576;
const MDS_GIGA = 1073741824;

export const MidstateUtils = {
    formatMDS(n) {
        n = Number(n) || 0;
        if (n === 0) return { value: '0', prefix: 'MDS' };
        if (n >= MDS_GIGA) return { value: parseFloat((n / MDS_GIGA).toFixed(4)).toString(), prefix: 'gMDS' };
        if (n >= MDS_MEGA) return { value: parseFloat((n / MDS_MEGA).toFixed(4)).toString(), prefix: 'mMDS' };
        if (n >= MDS_KILO) return { value: parseFloat((n / MDS_KILO).toFixed(4)).toString(), prefix: 'kMDS' };
        return { value: n.toLocaleString('en'), prefix: 'MDS' };
    },
    computeCoinId(addressHex, value, saltHex) {
        return compute_coin_id_hex(addressHex, BigInt(value), saltHex);
    },
    hash(hexData) {
        return blake3_hash_hex(hexData);
    }
};
