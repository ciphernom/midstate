use crate::core::{Batch, Transaction};
use futures::{AsyncRead, AsyncWrite, AsyncReadExt, AsyncWriteExt};
use libp2p::StreamProtocol;
use serde::{Deserialize, Serialize};
use std::io;
use std::net::SocketAddr;
use async_trait::async_trait; // <--- Import this

pub const MAX_GETBATCHES_COUNT: u64 = 100;

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
    GetBatches {
        start_height: u64,
        count: u64,
    },
    Batches {
        start_height: u64,
        batches: Vec<Batch>,
    },
}

impl Message {
    pub fn serialize_bin(&self) -> Vec<u8> {
        bincode::serialize(self).expect("Serialization failed")
    }

    pub fn deserialize_bin(bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(bincode::deserialize(bytes)?)
    }
}

// ── libp2p request-response codec ───────────────────────────────────────────

pub const MIDSTATE_PROTOCOL: StreamProtocol = StreamProtocol::new("/midstate/1.0.0");
const MAX_MSG_SIZE: usize = 10_000_000;

#[derive(Debug, Clone, Default)]
pub struct MidstateCodec;

#[async_trait] // <--- Add this attribute
impl libp2p::request_response::Codec for MidstateCodec {
    type Protocol = StreamProtocol;
    type Request = Message;
    type Response = Message;

    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        read_message(io).await
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        read_message(io).await
    }

    async fn write_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        write_message(io, &req).await
    }

    async fn write_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        res: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        write_message(io, &res).await
    }
}

async fn read_message<T: AsyncRead + Unpin + Send>(io: &mut T) -> io::Result<Message> {
    let mut len_bytes = [0u8; 4];
    io.read_exact(&mut len_bytes).await?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    if len > MAX_MSG_SIZE {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "message too large"));
    }
    let mut buf = vec![0u8; len];
    io.read_exact(&mut buf).await?;
    Message::deserialize_bin(&buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

async fn write_message<T: AsyncWrite + Unpin + Send>(io: &mut T, msg: &Message) -> io::Result<()> {
    let bytes = msg.serialize_bin();
    let len = (bytes.len() as u32).to_le_bytes();
    io.write_all(&len).await?;
    io.write_all(&bytes).await?;
    io.close().await?;
    Ok(())
}

// ============================================================
// ADD THIS ENTIRE BLOCK at the bottom of src/network/protocol.rs
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_serialize_deserialize_transaction() {
        let tx = Transaction::Commit { commitment: [0xAA; 32], spam_nonce: 42 };
        let msg = Message::Transaction(tx);
        let bytes = msg.serialize_bin();
        let msg2 = Message::deserialize_bin(&bytes).unwrap();
        match msg2 {
            Message::Transaction(Transaction::Commit { commitment, spam_nonce }) => {
                assert_eq!(commitment, [0xAA; 32]);
                assert_eq!(spam_nonce, 42);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn message_serialize_deserialize_state_info() {
        let msg = Message::StateInfo {
            height: 100,
            depth: 5000,
            midstate: [0xBB; 32],
        };
        let bytes = msg.serialize_bin();
        let msg2 = Message::deserialize_bin(&bytes).unwrap();
        match msg2 {
            Message::StateInfo { height, depth, midstate } => {
                assert_eq!(height, 100);
                assert_eq!(depth, 5000);
                assert_eq!(midstate, [0xBB; 32]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn message_serialize_deserialize_get_batches() {
        let msg = Message::GetBatches { start_height: 50, count: 20 };
        let bytes = msg.serialize_bin();
        let msg2 = Message::deserialize_bin(&bytes).unwrap();
        match msg2 {
            Message::GetBatches { start_height, count } => {
                assert_eq!(start_height, 50);
                assert_eq!(count, 20);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn message_deserialize_garbage_fails() {
        let garbage = vec![0xFF, 0xFE, 0xFD, 0xFC];
        assert!(Message::deserialize_bin(&garbage).is_err());
    }

    #[test]
    fn message_all_variants_round_trip() {
        use std::net::{IpAddr, Ipv4Addr, SocketAddr};

        let messages = vec![
            Message::GetState,
            Message::GetAddr,
            Message::Ping { nonce: 12345 },
            Message::Pong { nonce: 54321 },
            Message::Addr(vec![SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8080)]),
            Message::GetBatches { start_height: 0, count: 100 },
            Message::Batches { start_height: 0, batches: vec![] },
        ];

        for msg in messages {
            let bytes = msg.serialize_bin();
            assert!(Message::deserialize_bin(&bytes).is_ok());
        }
    }
}
