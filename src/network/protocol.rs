use crate::core::{Batch, Transaction};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Message {
    /// Announce a new transaction
    Transaction(Transaction),
    
    /// Announce a new batch
    Batch(Batch),
    
    /// Request current state info
    GetState,
    
    /// Response with state info
    StateInfo {
        height: u64,
        depth: u64,
        midstate: [u8; 32],
    },
    
    /// Request peer addresses
    GetAddr,
    
    /// Share peer addresses
    Addr(Vec<SocketAddr>),
    
    /// Ping (heartbeat)
    Ping { nonce: u64 },
    
    /// Pong (heartbeat response)
    Pong { nonce: u64 },
    
    /// Version handshake
    Version {
        version: u32,
        services: u64,
        timestamp: u64,
        addr_recv: SocketAddr,
        addr_from: SocketAddr,
    },
    
    /// Version acknowledgment
    Verack,
}

impl Message {
    pub fn serialize(&self) -> Vec<u8> {
        bincode::serialize(self).expect("Serialization failed")
    }
    
    pub fn deserialize(bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(bincode::deserialize(bytes)?)
    }
}

pub const PROTOCOL_VERSION: u32 = 1;
