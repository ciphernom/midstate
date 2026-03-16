/* tslint:disable */
/* eslint-disable */

export class WebWallet {
    free(): void;
    [Symbol.dispose](): void;
    build_reveal(spend_context_json: string, server_commitment_hex: string, server_salt_hex: string): string;
    check_filter(filter_hex: string, block_hash_hex: string, n: number): boolean;
    static from_seed_hex(seed_hex: string): WebWallet;
    get_mss_address(index: number, height: number, progress_cb?: Function | null): string;
    get_wots_address(index: number): string;
    constructor(phrase: string);
    prepare_spend(available_utxos_json: string, to_address_hex: string, send_amount: bigint, next_wots_index: number): string;
    set_mss_leaf_index(address_hex: string, leaf_index: number): void;
    set_watchlist(addrs_json: string): void;
}

export function compute_coin_id_hex(address_hex: string, value: bigint, salt_hex: string): string;

export function decompose_amount(amount: bigint): BigUint64Array;

export function decrypt_cli_wallet(data: Uint8Array, password: string): string;

export function generate_phrase(): string;

export function mine_commitment_pow(commitment_hex: string, required_pow: number): bigint;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_webwallet_free: (a: number, b: number) => void;
    readonly compute_coin_id_hex: (a: number, b: number, c: number, d: bigint, e: number, f: number) => void;
    readonly decompose_amount: (a: bigint) => number;
    readonly decrypt_cli_wallet: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly generate_phrase: (a: number) => void;
    readonly mine_commitment_pow: (a: number, b: number, c: number) => bigint;
    readonly webwallet_build_reveal: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly webwallet_check_filter: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly webwallet_from_seed_hex: (a: number, b: number, c: number) => void;
    readonly webwallet_get_mss_address: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly webwallet_get_wots_address: (a: number, b: number, c: number) => void;
    readonly webwallet_new: (a: number, b: number, c: number) => void;
    readonly webwallet_prepare_spend: (a: number, b: number, c: number, d: number, e: number, f: number, g: bigint, h: number) => void;
    readonly webwallet_set_mss_leaf_index: (a: number, b: number, c: number, d: number) => void;
    readonly webwallet_set_watchlist: (a: number, b: number, c: number) => void;
    readonly __wbindgen_export: (a: number) => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export2: (a: number, b: number) => number;
    readonly __wbindgen_export3: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export4: (a: number, b: number, c: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
