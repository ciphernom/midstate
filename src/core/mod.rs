pub mod finality;
pub mod types;
pub mod wots;
pub mod transaction;
pub mod extension;
pub mod walk;
pub mod state;
pub mod mmr;  
pub mod mss;
pub mod script;
pub mod filter;
pub mod simd_mining;
pub mod wots_simd;
#[cfg(not(target_arch = "wasm32"))]
pub mod gpu_mining;  
#[cfg(not(target_arch = "wasm32"))]
pub mod cuda_walk;
#[cfg(not(target_arch = "wasm32"))]
pub mod wgpu_walk;

pub use finality::*;
pub use types::*;
pub use state::adjust_difficulty;

// Q-Bolt v2 payment channels (shared by native + wasm wallets).
pub mod channel;

// Cross-chain DEX order announcements (shared by native + wasm wallets).
pub mod dex;

// Midwimble mining bonds for rent: offers, announcements, takes.
pub mod rental;

// Inheritance plans and payment streams: covenants on existing opcodes.
pub mod plans;
