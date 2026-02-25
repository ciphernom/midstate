use crate::core::{Batch, BatchHeader, Transaction};
use futures::{AsyncRead, AsyncWrite, AsyncReadExt, AsyncWriteExt};
use libp2p::StreamProtocol;
use serde::{Deserialize, Serialize};
use std::io;
use async_trait::async_trait;

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
    /// Peer exchange: list of multiaddr strings peers can dial
    Addr(Vec<String>),
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
    /// Request headers-only to verify PoW quickly
    GetHeaders {
        start_height: u64,
        count: u64,
    },
    /// Response with headers
    Headers {
        start_height: u64,
        headers: Vec<BatchHeader>,
    },
    // ── Dark Pool CoinJoin ──────────────────────────────────────────
    /// Announce intent to mix a specific denomination.
    MixAnnounce {
        mix_id: [u8; 32],
        denomination: u64,
    },
    /// Join an announced mix session with input and desired output.
    MixJoin {
        mix_id: [u8; 32],
        input: crate::core::InputReveal,
        output: crate::core::OutputData,
    },
    MixFee {
        mix_id: [u8; 32],
        input: crate::core::InputReveal,
    },
    /// Broadcast the canonical proposal so all participants can sign.
    MixProposal {
        mix_id: [u8; 32],
        inputs: Vec<crate::core::InputReveal>,
        outputs: Vec<crate::core::OutputData>,
        salt: [u8; 32],
        commitment: [u8; 32],
    },
    /// A participant's signature for their input in the mix.
    MixSign {
        mix_id: [u8; 32],
        /// Index into the proposal's canonical input list.
        input_index: usize,
        signature: Vec<u8>,
    },
}

impl Message {
    pub fn serialize_bin(&self) -> Vec<u8> {
        use bincode::Options;
        bincode::DefaultOptions::new()
            .with_limit(MAX_MSG_SIZE as u64)
            .serialize(self)
            .expect("Serialization failed")
    }

    pub fn deserialize_bin(bytes: &[u8]) -> anyhow::Result<Self> {
        use bincode::Options;
        Ok(bincode::DefaultOptions::new()
            .with_limit(MAX_MSG_SIZE as u64)
            .deserialize(bytes)?)
    }
}

// ── libp2p request-response codec ───────────────────────────────────────────
pub const MIDSTATE_PROTOCOL: StreamProtocol = StreamProtocol::new("/midstate/1.0.0");
const MAX_MSG_SIZE: usize = 10_000_000;

#[derive(Debug, Clone, Default)]
pub struct MidstateCodec;

#[async_trait]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_serialize_deserialize_transaction() {
        let tx = Transaction::Commit {
            commitment: [0xAA; 32],
            spam_nonce: 42,
        };
        let msg = Message::Transaction(tx);
        let bytes = msg.serialize_bin();
        let msg2 = Message::deserialize_bin(&bytes).unwrap();
        match msg2 {
            Message::Transaction(Transaction::Commit {
                commitment,
                spam_nonce,
            }) => {
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
            Message::StateInfo {
                height,
                depth,
                midstate,
            } => {
                assert_eq!(height, 100);
                assert_eq!(depth, 5000);
                assert_eq!(midstate, [0xBB; 32]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn message_serialize_deserialize_get_batches() {
        let msg = Message::GetBatches {
            start_height: 50,
            count: 20,
        };
        let bytes = msg.serialize_bin();
        let msg2 = Message::deserialize_bin(&bytes).unwrap();
        match msg2 {
            Message::GetBatches {
                start_height,
                count,
            } => {
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
        use crate::core::types::{InputReveal, OutputData};
        use crate::core::types::Predicate;

        let messages = vec![
            Message::GetState,
            Message::GetAddr,
            Message::Ping { nonce: 12345 },
            Message::Pong { nonce: 54321 },
            Message::Addr(vec![
                "/ip4/203.0.113.10/tcp/9333/p2p/12D3KooWTest".to_string(),
                "/ip4/10.0.0.1/udp/9333/quic-v1/p2p/12D3KooWOther".to_string(),
            ]),
            Message::GetBatches {
                start_height: 0,
                count: 100,
            },
            Message::Batches {
                start_height: 0,
                batches: vec![],
            },
            Message::GetHeaders {
                start_height: 0,
                count: 50,
            },
            Message::Headers {
                start_height: 0,
                headers: vec![],
            },
            Message::MixAnnounce {
                mix_id: [0; 32],
                denomination: 8,
            },
            Message::MixJoin {
                mix_id: [0; 32],
                input: InputReveal {
                    predicate: Predicate::p2pk(&[1; 32]),
                    value: 8,
                    salt: [2; 32],
                },
                output: OutputData {
                    address: [3; 32],
                    value: 8,
                    salt: [4; 32],
                },
            },
            Message::MixProposal {
                mix_id: [0; 32],
                inputs: vec![],
                outputs: vec![],
                salt: [0; 32],
                commitment: [0; 32],
            },
            Message::MixSign {
                mix_id: [0; 32],
                input_index: 0,
                signature: vec![],
            },
        ];

        for msg in messages {
            let bytes = msg.serialize_bin();
            assert!(Message::deserialize_bin(&bytes).is_ok());
        }
    }

    // ── PEX message tests ───────────────────────────────────────────
    #[test]
    fn addr_message_preserves_multiaddr_strings() {
        let addrs = vec![
            "/ip4/1.2.3.4/tcp/9333/p2p/12D3KooWAbCdEf".to_string(),
            "/ip4/5.6.7.8/udp/9333/quic-v1/p2p/12D3KooWGhIjKl".to_string(),
            "/ip4/10.0.0.1/tcp/9333/p2p/12D3KooWRelay/p2p-circuit/p2p/12D3KooWNatted".to_string(),
        ];
        let msg = Message::Addr(addrs.clone());
        let bytes = msg.serialize_bin();
        let msg2 = Message::deserialize_bin(&bytes).unwrap();
        match msg2 {
            Message::Addr(got) => assert_eq!(got, addrs),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn addr_message_empty_vec() {
        let msg = Message::Addr(vec![]);
        let bytes = msg.serialize_bin();
        match Message::deserialize_bin(&bytes).unwrap() {
            Message::Addr(addrs) => assert!(addrs.is_empty()),
            _ => panic!("wrong variant"),
        }
    }

    // ── CoinJoin dark pool messages ─────────────────────────────────
    #[test]
    fn mix_announce_round_trip() {
        let msg = Message::MixAnnounce {
            mix_id: [0xAA; 32],
            denomination: 16,
        };
        let bytes = msg.serialize_bin();
        match Message::deserialize_bin(&bytes).unwrap() {
            Message::MixAnnounce { mix_id, denomination } => {
                assert_eq!(mix_id, [0xAA; 32]);
                assert_eq!(denomination, 16);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn mix_join_round_trip() {
        use crate::core::types::{InputReveal, OutputData};
        use crate::core::types::Predicate;

        let msg = Message::MixJoin {
            mix_id: [0xBB; 32],
            input: InputReveal {
                predicate: Predicate::p2pk(&[1; 32]),
                value: 8,
                salt: [2; 32],
            },
            output: OutputData {
                address: [3; 32],
                value: 8,
                salt: [4; 32],
            },
        };

        let bytes = msg.serialize_bin();
        match Message::deserialize_bin(&bytes).unwrap() {
            Message::MixJoin { mix_id, input, output } => {
                assert_eq!(mix_id, [0xBB; 32]);
                assert_eq!(input.value, 8);
                assert_eq!(output.value, 8);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn mix_proposal_round_trip() {
        use crate::core::types::{InputReveal, OutputData};
        use crate::core::types::Predicate;

        let msg = Message::MixProposal {
            mix_id: [0xCC; 32],
            inputs: vec![
                InputReveal {
                    predicate: Predicate::p2pk(&[1; 32]),
                    value: 8,
                    salt: [2; 32],
                },
                InputReveal {
                    predicate: Predicate::p2pk(&[3; 32]),
                    value: 1,
                    salt: [4; 32],
                },
            ],
            outputs: vec![OutputData {
                address: [5; 32],
                value: 8,
                salt: [6; 32],
            }],
            salt: [0xDD; 32],
            commitment: [0xEE; 32],
        };

        let bytes = msg.serialize_bin();
        match Message::deserialize_bin(&bytes).unwrap() {
            Message::MixProposal {
                mix_id,
                inputs,
                outputs,
                salt,
                commitment,
            } => {
                assert_eq!(mix_id, [0xCC; 32]);
                assert_eq!(inputs.len(), 2);
                assert_eq!(outputs.len(), 1);
                assert_eq!(salt, [0xDD; 32]);
                assert_eq!(commitment, [0xEE; 32]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn mix_sign_round_trip() {
        let msg = Message::MixSign {
            mix_id: [0xFF; 32],
            input_index: 2,
            signature: vec![0xAB; 576],
        };
        let bytes = msg.serialize_bin();
        match Message::deserialize_bin(&bytes).unwrap() {
            Message::MixSign {
                mix_id,
                input_index,
                signature,
            } => {
                assert_eq!(mix_id, [0xFF; 32]);
                assert_eq!(input_index, 2);
                assert_eq!(signature.len(), 576);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn addr_message_large_peer_list() {
        // PEX should handle up to ~1000 addrs without hitting MAX_MSG_SIZE
        let addrs: Vec<String> = (0..1000)
            .map(|i| {
                format!(
                    "/ip4/10.{}.{}.{}/tcp/9333/p2p/12D3KooWTest{}",
                    i / 65536,
                    (i / 256) % 256,
                    i % 256,
                    i
                )
            })
            .collect();
        let msg = Message::Addr(addrs.clone());
        let bytes = msg.serialize_bin();
        assert!(bytes.len() < MAX_MSG_SIZE);
        match Message::deserialize_bin(&bytes).unwrap() {
            Message::Addr(got) => assert_eq!(got.len(), 1000),
            _ => panic!("wrong variant"),
        }
    }
}
