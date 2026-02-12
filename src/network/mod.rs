pub mod protocol;

pub use protocol::{Message, MidstateCodec, MIDSTATE_PROTOCOL, MAX_GETBATCHES_COUNT};

use anyhow::Result;
use futures::StreamExt;
use libp2p::{
    identify, kad,
    request_response::{self, OutboundRequestId, ProtocolSupport, ResponseChannel, Config as RequestResponseConfig},
    swarm::{NetworkBehaviour, SwarmEvent},
    Multiaddr, PeerId, Swarm,
    noise, tcp, yamux,
    identity::Keypair,
};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::time::Duration;

// ── Behaviour ───────────────────────────────────────────────────────────────

#[derive(NetworkBehaviour)]
pub struct MidstateBehaviour {
    pub rr: request_response::Behaviour<MidstateCodec>,
    pub kademlia: kad::Behaviour<kad::store::MemoryStore>,
    pub identify: identify::Behaviour,
}

// ── Events ──────────────────────────────────────────────────────────────────

pub enum NetworkEvent {
    MessageReceived {
        peer: PeerId,
        message: Message,
        channel: Option<ResponseChannel<Message>>,
    },
    PeerConnected(PeerId),
    PeerDisconnected(PeerId),
}

// ── Network API ─────────────────────────────────────────────────────────────

pub struct MidstateNetwork {
    swarm: Swarm<MidstateBehaviour>,
    connected: HashSet<PeerId>,
    pending_requests: HashMap<OutboundRequestId, PeerId>,
}

impl MidstateNetwork {
    pub async fn new(
        keypair: Keypair,
        listen_addr: Multiaddr,
        bootstrap_peers: Vec<Multiaddr>,
    ) -> Result<Self> {
        let peer_id = keypair.public().to_peer_id();
        tracing::info!("Local peer id: {}", peer_id);

        let swarm = libp2p::SwarmBuilder::with_existing_identity(keypair.clone())
            .with_tokio()
            .with_tcp(
                tcp::Config::default(),
                noise::Config::new,
                yamux::Config::default,
            )?
            .with_behaviour(|key| {
                let rr = request_response::Behaviour::new(
                    [(MIDSTATE_PROTOCOL, ProtocolSupport::Full)],
                    RequestResponseConfig::default(),
                );

                let kad_store = kad::store::MemoryStore::new(key.public().to_peer_id());
                let mut kademlia = kad::Behaviour::new(
                    key.public().to_peer_id(),
                    kad_store,
                );
                kademlia.set_mode(Some(kad::Mode::Server));

                let identify = identify::Behaviour::new(
                    identify::Config::new(
                        "/midstate/id/1.0.0".to_string(),
                        key.public(),
                    )
                    .with_push_listen_addr_updates(true),
                );

                MidstateBehaviour { rr, kademlia, identify }
            })?
            .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(120)))
            .build();

        let mut net = Self {
            swarm,
            connected: HashSet::new(),
            pending_requests: HashMap::new(),
        };

        net.swarm.listen_on(listen_addr)?;

        for addr in bootstrap_peers {
            if let Err(e) = net.swarm.dial(addr.clone()) {
                tracing::warn!("Failed to dial {}: {}", addr, e);
            }
        }

        Ok(net)
    }

    pub fn local_peer_id(&self) -> PeerId {
        *self.swarm.local_peer_id()
    }

    pub fn send(&mut self, peer: PeerId, msg: Message) {
        let req_id = self.swarm.behaviour_mut().rr.send_request(&peer, msg);
        self.pending_requests.insert(req_id, peer);
    }

    pub fn broadcast(&mut self, msg: Message) {
        let peers: Vec<PeerId> = self.connected.iter().copied().collect();
        for peer in peers {
            self.send(peer, msg.clone());
        }
    }

    pub fn broadcast_except(&mut self, exclude: Option<PeerId>, msg: Message) {
        let peers: Vec<PeerId> = self.connected.iter()
            .filter(|p| Some(**p) != exclude)
            .copied()
            .collect();
        for peer in peers {
            self.send(peer, msg.clone());
        }
    }

    pub fn respond(&mut self, channel: ResponseChannel<Message>, msg: Message) {
        if let Err(_) = self.swarm.behaviour_mut().rr.send_response(channel, msg) {
            tracing::warn!("Failed to send response (channel closed)");
        }
    }

    pub fn peer_count(&self) -> usize {
        self.connected.len()
    }

    pub fn connected_peers(&self) -> Vec<PeerId> {
        self.connected.iter().copied().collect()
    }

    pub fn peer_addrs(&self) -> Vec<String> {
        self.connected.iter().map(|p| p.to_string()).collect()
    }

    pub fn add_kad_address(&mut self, peer: PeerId, addr: Multiaddr) {
        self.swarm.behaviour_mut().kademlia.add_address(&peer, addr);
    }

    pub fn random_peer(&self) -> Option<PeerId> {
        use rand::seq::IteratorRandom;
        self.connected.iter().copied().choose(&mut rand::thread_rng())
    }

    pub async fn next_event(&mut self) -> NetworkEvent {
        loop {
            match self.swarm.select_next_some().await {
                SwarmEvent::Behaviour(MidstateBehaviourEvent::Rr(
                    request_response::Event::Message { peer, message },
                )) => match message {
                    request_response::Message::Request {
                        request, channel, ..
                    } => {
                        return NetworkEvent::MessageReceived {
                            peer,
                            message: request,
                            channel: Some(channel),
                        };
                    }
                    request_response::Message::Response {
                        request_id,
                        response,
                    } => {
                        self.pending_requests.remove(&request_id);
                        return NetworkEvent::MessageReceived {
                            peer,
                            message: response,
                            channel: None,
                        };
                    }
                },
                SwarmEvent::Behaviour(MidstateBehaviourEvent::Rr(
                    request_response::Event::OutboundFailure {
                        peer, request_id, error,
                    },
                )) => {
                    self.pending_requests.remove(&request_id);
                    tracing::warn!("Outbound request to {} failed: {}", peer, error);
                }
                SwarmEvent::Behaviour(MidstateBehaviourEvent::Rr(
                    request_response::Event::InboundFailure { peer, error, .. },
                )) => {
                    tracing::warn!("Inbound request from {} failed: {}", peer, error);
                }
                SwarmEvent::Behaviour(MidstateBehaviourEvent::Rr(
                    request_response::Event::ResponseSent { .. },
                )) => {}
                SwarmEvent::Behaviour(MidstateBehaviourEvent::Identify(
                    identify::Event::Received { peer_id, info, .. },
                )) => {
                    for addr in info.listen_addrs {
                        self.swarm
                            .behaviour_mut()
                            .kademlia
                            .add_address(&peer_id, addr);
                    }
                }
                SwarmEvent::Behaviour(MidstateBehaviourEvent::Kademlia(_)) => {}
                SwarmEvent::Behaviour(MidstateBehaviourEvent::Identify(_)) => {}
                SwarmEvent::ConnectionEstablished {
                    peer_id,
                    ..
                } => {
                    self.connected.insert(peer_id);
                    tracing::info!("Peer connected: {} (total: {})", peer_id, self.connected.len());
                    return NetworkEvent::PeerConnected(peer_id);
                }
                SwarmEvent::ConnectionClosed {
                    peer_id,
                    num_established,
                    ..
                } => {
                    if num_established == 0 {
                        self.connected.remove(&peer_id);
                        tracing::info!("Peer disconnected: {} (total: {})", peer_id, self.connected.len());
                        return NetworkEvent::PeerDisconnected(peer_id);
                    }
                }
                SwarmEvent::NewListenAddr { address, .. } => {
                    tracing::info!("Listening on {}", address);
                }
                _ => {}
            }
        }
    }
}

pub fn socket_to_multiaddr(addr: SocketAddr) -> Multiaddr {
    use std::net::IpAddr;
    let mut ma = Multiaddr::empty();
    match addr.ip() {
        IpAddr::V4(ip) => ma.push(libp2p::multiaddr::Protocol::Ip4(ip)),
        IpAddr::V6(ip) => ma.push(libp2p::multiaddr::Protocol::Ip6(ip)),
    }
    ma.push(libp2p::multiaddr::Protocol::Tcp(addr.port()));
    ma
}
