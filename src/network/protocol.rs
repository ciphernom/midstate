use crate::core::{Batch, Transaction};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Message {
    Transaction(Transaction),
    Batch(Batch),
    GetState,
    StateInfo {
        height: u64,
        depth: u64,
        midstate: [u8; 32],
    },
    GetAddr,
    Addr(Vec<SocketAddr>),
    Ping { nonce: u64 },
    Pong { nonce: u64 },
    Version {
        version: u32,
        services: u64,
        timestamp: u64,
        addr_recv: SocketAddr,
        addr_from: SocketAddr,
    },
    Verack,
    GetBatches {
        start_height: u64,
        count: u64,
    },
    Batches(Vec<Batch>),
}

impl Message {
    pub fn serialize(&self) -> Vec<u8> {
        bincode::serialize(self).expect("Serialization failed")
    }

    pub fn deserialize(bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(bincode::deserialize(bytes)?)
    }
}

pub const PROTOCOL_VERSION: u32 = 2;
