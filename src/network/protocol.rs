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
