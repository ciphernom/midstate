/* tslint:disable */
/* eslint-disable */

export class WebWallet {
    free(): void;
    [Symbol.dispose](): void;
    build_reveal(spend_context_json: string, server_commitment_hex: string, server_salt_hex: string): string;
    check_filter(filter_hex: string, block_hash_hex: string, n: number, addrs_json: string): boolean;
    /**
     * Derives a reusable MSS address for receiving funds (Height 5 recommended)
     */
    get_mss_address(index: number, height: number): string;
    /**
     * Derives a single-use WOTS address (used internally for change outputs)
     */
    get_wots_address(index: number): string;
    constructor(phrase: string);
    prepare_spend(available_utxos_json: string, to_address_hex: string, send_amount: bigint, next_wots_index: number): string;
}

export function compute_coin_id_hex(address_hex: string, value: bigint, salt_hex: string): string;

export function decompose_amount(amount: bigint): BigUint64Array;

export function generate_phrase(): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_webwallet_free: (a: number, b: number) => void;
    readonly compute_coin_id_hex: (a: number, b: number, c: bigint, d: number, e: number) => [number, number];
    readonly decompose_amount: (a: bigint) => any;
    readonly generate_phrase: () => [number, number];
    readonly webwallet_build_reveal: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number, number];
    readonly webwallet_check_filter: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => number;
    readonly webwallet_get_mss_address: (a: number, b: number, c: number) => [number, number, number, number];
    readonly webwallet_get_wots_address: (a: number, b: number) => [number, number];
    readonly webwallet_new: (a: number, b: number) => [number, number, number];
    readonly webwallet_prepare_spend: (a: number, b: number, c: number, d: number, e: number, f: bigint, g: number) => [number, number, number, number];
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
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
