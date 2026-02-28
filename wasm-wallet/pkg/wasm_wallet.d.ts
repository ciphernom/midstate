/* tslint:disable */
/* eslint-disable */

export class WebWallet {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Phase 1: Build the Commit payload
     */
    build_commit(inputs_json: string, to_address: string, send_amount: bigint, fee: bigint): string;
    /**
     * Phase 2: Sign the transaction and build the Reveal payload
     */
    build_reveal(commitment_hex: string, server_salt_hex: string): string;
    /**
     * Checks a block's compact filter to see if it contains our address
     */
    check_filter(filter_hex: string, block_hash_hex: string, n: number): boolean;
    get_primary_address(): string;
    constructor(phrase: string);
}

/**
 * Helper to generate a brand new seed phrase in the browser
 */
export function generate_new_phrase(): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_webwallet_free: (a: number, b: number) => void;
    readonly generate_new_phrase: () => [number, number];
    readonly webwallet_build_commit: (a: number, b: number, c: number, d: number, e: number, f: bigint, g: bigint) => [number, number, number, number];
    readonly webwallet_build_reveal: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly webwallet_check_filter: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly webwallet_get_primary_address: (a: number) => [number, number];
    readonly webwallet_new: (a: number, b: number) => [number, number, number];
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
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
