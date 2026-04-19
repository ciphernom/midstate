// src/network/light_protocol.rs
//
// Light-client protocol for browser wallets connecting over WebRTC.
//
// Uses JSON framing (4-byte LE length + JSON body) so browsers don't need
// a bincode implementation. Runs as a separate request_response::Behaviour
// on the same swarm alongside the full /midstate/2.0.0 binary protocol.
//
// The protocol ID includes "light" so full nodes can rate-limit/prioritize
// light client traffic independently from peer-to-peer sync traffic.

use async_trait::async_trait;
use futures::{AsyncRead, AsyncWrite, AsyncReadExt, AsyncWriteExt};
use libp2p::StreamProtocol;
use serde::{Deserialize, Serialize};
use std::io;

pub const LIGHT_PROTOCOL: StreamProtocol = StreamProtocol::new("/midstate/light/2.0.0");
pub const LIGHT_PUSH_PROTOCOL: StreamProtocol = StreamProtocol::new("/midstate/light-push/2.0.0");

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum LightNotification {
    NewBlockTip {
        height: u64,
        target: String,
        filter_hex: String,
        block_hash: String,
        element_count: u64,
    }
}

/// Maximum message size for light client requests/responses.
/// Light clients don't transfer full batch payloads, so 2 MB is generous.
const MAX_LIGHT_MSG_SIZE: usize = 2_000_000;

// ── Request / Response Types ────────────────────────────────────────────────

/// Every request from a browser light client.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", content = "params")]
pub enum LightRequest {
    /// GET /state equivalent
    #[serde(rename = "get_state")]
    GetState,

    /// GET /block/{height} equivalent
    #[serde(rename = "get_block")]
    GetBlock { height: u64 },

    /// POST /filters equivalent
    #[serde(rename = "get_filters")]
    GetFilters { start_height: u64, end_height: u64 },

    /// GET /mempool equivalent
    #[serde(rename = "get_mempool")]
    GetMempool,

    /// POST /block_template equivalent
    #[serde(rename = "block_template")]
    BlockTemplate { coinbase: serde_json::Value },

    /// POST /api/internal/submit_batch equivalent
    #[serde(rename = "submit_batch")]
    SubmitBatch { batch: serde_json::Value },

    /// POST /commit equivalent
    #[serde(rename = "commit")]
    Commit { commitment: String, spam_nonce: u64 },

    /// POST /send equivalent (reveal transaction)
    #[serde(rename = "send")]
    Send { reveal: serde_json::Value },

    /// POST /check equivalent
    #[serde(rename = "check")]
    CheckCoin { coin: String },

    /// GET /check_commitment equivalent
    #[serde(rename = "check_commitment")]
    CheckCommitment { commitment: String },

    /// POST /mss_state equivalent
    #[serde(rename = "mss_state")]
    MssState { master_pk: String },
}

/// Every response sent to a browser light client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl LightResponse {
    pub fn success(data: serde_json::Value) -> Self {
        LightResponse { ok: true, data: Some(data), error: None }
    }

    pub fn error(msg: impl Into<String>) -> Self {
        LightResponse { ok: false, data: None, error: Some(msg.into()) }
    }
}

// ── Raw-stream helpers ───────────────────────────────────────────────────────
//
// Called directly by the network layer on the raw stream that
// libp2p_stream::Behaviour hands us.  No request_response envelope involved.

/// Read a length-prefixed JSON request from a raw stream.
pub async fn read_request_raw<T: AsyncRead + Unpin + Send>(io: &mut T) -> io::Result<LightRequest> {
    let bytes = read_length_prefixed(io).await?;
    serde_json::from_slice(&bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Write a length-prefixed JSON response to a raw stream and close the write side.
pub async fn write_response_raw<T: AsyncWrite + Unpin + Send>(
    io: &mut T,
    res: LightResponse,
) -> io::Result<()> {
    let bytes = serde_json::to_vec(&res)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    write_length_prefixed(io, &bytes).await
}

// ── JSON Codec ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct LightCodec;

#[async_trait]
impl libp2p::request_response::Codec for LightCodec {
    type Protocol = StreamProtocol;
    type Request = LightRequest;
    type Response = LightResponse;

    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        let bytes = read_length_prefixed(io).await?;
        serde_json::from_slice(&bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        let bytes = read_length_prefixed(io).await?;
        serde_json::from_slice(&bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
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
        let bytes = serde_json::to_vec(&req)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        write_length_prefixed(io, &bytes).await
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
        let bytes = serde_json::to_vec(&res)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        write_length_prefixed(io, &bytes).await
    }
}

// ── Wire helpers (4-byte LE length prefix, same framing as the binary protocol) ─

async fn read_length_prefixed<T: AsyncRead + Unpin + Send>(io: &mut T) -> io::Result<Vec<u8>> {
    let mut len_bytes = [0u8; 4];
    io.read_exact(&mut len_bytes).await?;
    let len = u32::from_le_bytes(len_bytes) as usize;

    if len > MAX_LIGHT_MSG_SIZE {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "light message too large"));
    }

    let initial_alloc = std::cmp::min(len, 65_536);
    let mut buf = Vec::with_capacity(initial_alloc);
    let mut handle = io.take(len as u64);
    handle.read_to_end(&mut buf).await?;

    if buf.len() != len {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "incomplete light message"));
    }

    Ok(buf)
}

async fn write_length_prefixed<T: AsyncWrite + Unpin + Send>(io: &mut T, data: &[u8]) -> io::Result<()> {
    let len = (data.len() as u32).to_le_bytes();
    io.write_all(&len).await?;
    io.write_all(data).await?;
    io.close().await?;
    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_request_get_state_serializes() {
        let req = LightRequest::GetState;
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("get_state"));
        let _: LightRequest = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn light_request_get_block_round_trip() {
        let req = LightRequest::GetBlock { height: 42 };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: LightRequest = serde_json::from_str(&json).unwrap();
        match parsed {
            LightRequest::GetBlock { height } => assert_eq!(height, 42),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn light_request_commit_round_trip() {
        let req = LightRequest::Commit {
            commitment: "ab".repeat(32),
            spam_nonce: 999,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: LightRequest = serde_json::from_str(&json).unwrap();
        match parsed {
            LightRequest::Commit { commitment, spam_nonce } => {
                assert_eq!(spam_nonce, 999);
                assert_eq!(commitment.len(), 64);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn light_response_success_serializes() {
        let resp = LightResponse::success(serde_json::json!({ "height": 100 }));
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"ok\":true"));
        assert!(!json.contains("error"));
    }

    #[test]
    fn light_response_error_serializes() {
        let resp = LightResponse::error("bad request");
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"ok\":false"));
        assert!(json.contains("bad request"));
        assert!(!json.contains("\"data\""));
    }

    #[test]
    fn all_request_variants_parse_from_browser_json() {
        // Simulate what the browser would actually send
        let cases = vec![
            r#"{"method":"get_state"}"#,
            r#"{"method":"get_block","params":{"height":10}}"#,
            r#"{"method":"get_filters","params":{"start_height":0,"end_height":100}}"#,
            r#"{"method":"get_mempool"}"#,
            r#"{"method":"check","params":{"coin":"aabbccdd"}}"#,
            r#"{"method":"mss_state","params":{"master_pk":"0011223344"}}"#,
            r#"{"method":"commit","params":{"commitment":"abcd","spam_nonce":42}}"#,
            r#"{"method":"check_commitment","params":{"commitment":"aabbccdd"}}"#,
        ];
        for json in cases {
            let parsed: Result<LightRequest, _> = serde_json::from_str(json);
            assert!(parsed.is_ok(), "Failed to parse: {}", json);
        }
    }
}
