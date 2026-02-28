// Wasm-safe modules (Pure Math & Cryptography)
pub mod core;
pub mod wallet;

//  Native-only modules (Storage, Networking, Sockets)
#[cfg(not(target_arch = "wasm32"))]
pub mod storage;

#[cfg(not(target_arch = "wasm32"))]
pub mod mempool;

#[cfg(not(target_arch = "wasm32"))]
pub mod network;

#[cfg(not(target_arch = "wasm32"))]
pub mod node;

#[cfg(not(target_arch = "wasm32"))]
pub mod rpc;

#[cfg(not(target_arch = "wasm32"))]
pub mod metrics;

#[cfg(not(target_arch = "wasm32"))]
pub mod sync;

#[cfg(not(target_arch = "wasm32"))]
pub mod mix;

// Export the core types everywhere
pub use core::types::*;
