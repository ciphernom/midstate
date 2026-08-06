//! Amino (public IPFS/libp2p) DHT rendezvous — borrowed-commons peer discovery.
//!
//! # Reasoning
//!
//! Midstate's cold-start path depends entirely on operator-run infrastructure:
//! two hardcoded bootstrap VPS nodes and a Cloudflare Worker on a domain that
//! must be renewed forever. If that infrastructure lapses, existing nodes carry
//! on (they restore from the on-disk address book) but **no new node can ever
//! join**, and the network dies by attrition.
//!
//! This module removes that dependency by publishing a provider record on the
//! public IPFS DHT — infrastructure operated by several independent parties who
//! have never heard of this project, and which has run for years without anyone
//! associated with Midstate lifting a finger.
//!
//! ## Why a separate Swarm
//!
//! The obvious implementation adds a second `kad::Behaviour` to
//! `MidstateBehaviour`. That does not work: `kad::Config::default()` uses the
//! protocol name `/ipfs/kad/1.0.0`, which is exactly what the existing Midstate
//! Kademlia already claims, and two sub-behaviours in one `NetworkBehaviour`
//! cannot share a protocol name. Resolving that would mean renaming the
//! Midstate DHT, which partitions Kademlia across the version boundary.
//!
//! Running Amino as an isolated Swarm avoids the collision entirely:
//! - Zero changes to `MidstateBehaviour`; no protocol rename; full wire
//!   compatibility with existing nodes.
//! - Foreign IPFS peers can never enter the Midstate routing table, and
//!   Midstate queries can never leak into a foreign network.
//! - The whole subsystem can be disabled or deleted without touching
//!   consensus-adjacent networking code.
//!
//! Per `docs/standards.md` §3 this is **Tier 3** — it touches neither money nor
//! consensus state. Reasoning and pre/post conditions are mandatory; Z schemas
//! are given only for the two operations with real external state impact.
//!
//! # Safety / Invariants
//!
//! - **Discovery is not trust.** Every address produced here is fed through the
//!   normal `dial_addr` path and must still pass the full Midstate handshake:
//!   protocol id, genesis match, and a chain consistent with known checkpoints.
//!   A wholly poisoned provider set costs dial attempts and nothing else.
//! - **Never load-bearing.** Any failure — no bootstrap peers, DNS blocked, DHT
//!   unreachable — degrades to the other seed sources and never blocks startup.

use anyhow::Result;
use futures::StreamExt;
use libp2p::{
    identify, identity::Keypair, kad, multiaddr::Protocol, noise,
    swarm::{StreamProtocol, SwarmEvent},
    tcp, yamux, Multiaddr, PeerId, Swarm,
};
use std::collections::HashSet;
use std::time::Duration;
use tokio::sync::mpsc;

// ── Constants ───────────────────────────────────────────────────────────────

/// Protocol name of the public IPFS DHT. This is `kad::Config::default()`'s
/// value, spelled out so the coupling is explicit rather than incidental.
pub const AMINO_KAD_PROTOCOL: &str = "/ipfs/kad/1.0.0";

/// Domain-separation label for the rendezvous key. Changing this partitions old
/// and new nodes onto unrelated rendezvous points.
const RENDEZVOUS_LABEL: &[u8] = b"midstate-amino-rendezvous-v1";

/// Multicodec identifier for BLAKE3 in the multiformats table.
const MULTIHASH_CODE_BLAKE3: u8 = 0x1e;

/// Upper bound on addresses returned from one discovery round.
///
/// Bounds the damage from a poisoned keyspace region: an attacker who dominates
/// the neighbourhood of our key can waste at most this many dial slots per
/// round, and the caller's subnet caps constrain it further.
pub const MAX_ADDRS_PER_ROUND: usize = 32;

/// Grace period after startup before the first DHT query, allowing the routing
/// table to fill from the bootstrap fleet.
const WARMUP: Duration = Duration::from_secs(20);

/// Bootstrap entry points into the Amino DHT.
///
/// # Reasoning
/// These are deliberately *not* Midstate infrastructure. They are the public
/// IPFS bootstrap fleet, run by multiple independent operators with no
/// knowledge of or interest in this project — which is precisely why depending
/// on them is safer than depending on a domain someone here must keep renewing.
///
/// This is the bootstrap-of-the-bootstrap and it is the one remaining external
/// dependency. It is not zero-dependency; it is a dependency on a large, old,
/// diversely-operated commons rather than on one person's credit card.
///
/// The `/dnsaddr/` entries need DNS TXT resolution (hence `.with_dns()` on this
/// swarm's transport). The literal `/ip4/` entry needs no DNS and is retained
/// as a floor for censored or DNS-less environments.
///
/// # Safety / Invariants
/// - **No trust conferred.** These peers route DHT queries and never supply
///   Midstate consensus data. A malicious bootstrap node can degrade discovery
///   but cannot inject an invalid chain.
pub const AMINO_BOOTSTRAP: &[&str] = &[
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN",
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa",
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb",
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt",
    "/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ",
];

// ── Rendezvous key ──────────────────────────────────────────────────────────

/// Derives the DHT rendezvous key from the network's own identity anchor.
///
/// # Reasoning
/// The key must be computable by a node with *no chain state at all* — empty
/// database, zero peers, nothing but its binary and a clock. Deriving it from
/// the chain tip would invert the dependency: you would need peers in order to
/// compute the place where peers are found. The network anchor is compiled in
/// and never changes, so it has no such circularity.
///
/// Free property worth noting: testnet, mainnet, and any hard fork that changes
/// the anchor land on completely unrelated DHT keys with zero coordination.
///
/// The digest is wrapped as a well-formed BLAKE3 multihash rather than emitted
/// raw, because some DHT implementations validate that provider keys decode as
/// multihashes and will otherwise reject `ADD_PROVIDER` silently.
///
/// # Formal Specification
///
/// ```text
/// Pre:
///   - anchor is the 32-byte compiled-in network identity constant
///
/// Post:
///   key! = ⟨0x1e⟩ ⌢ ⟨0x20⟩ ⌢ BLAKE3(anchor ⌢ RENDEZVOUS_LABEL)
///   #key! = 34
///   key! decodes as a well-formed multihash
///   key! is a pure function of anchor (no clock, no chain state, no config)
/// ```
///
/// # Safety / Invariants
/// - **Determinism:** every node on a network derives the identical key. Any
///   divergence silently partitions discovery, so this must never depend on
///   time, configuration, or local state.
/// - **Not secret.** The derivation is public; an adversary computes the same
///   key. Nothing in this design assumes otherwise.
pub fn rendezvous_key(anchor: &[u8; 32]) -> kad::RecordKey {
    let mut preimage = Vec::with_capacity(32 + RENDEZVOUS_LABEL.len());
    preimage.extend_from_slice(anchor);
    preimage.extend_from_slice(RENDEZVOUS_LABEL);

    let digest = crate::core::types::hash(&preimage);

    // Multihash framing: <code varint> <length varint> <digest>.
    // 0x1e and 32 are both single-byte varints.
    let mut mh = Vec::with_capacity(34);
    mh.push(MULTIHASH_CODE_BLAKE3);
    mh.push(32u8);
    mh.extend_from_slice(&digest);

    kad::RecordKey::new(&mh)
}

// ── Behaviour ───────────────────────────────────────────────────────────────

#[derive(libp2p::swarm::NetworkBehaviour)]
struct AminoBehaviour {
    kad: kad::Behaviour<kad::store::MemoryStore>,
    identify: identify::Behaviour,
}

// ── Public handle ───────────────────────────────────────────────────────────

/// Commands accepted by the background Amino task.
#[derive(Debug)]
enum AminoCommand {
    /// Publish ourselves as a provider, advertising the given directly-dialable
    /// addresses of the *main* Midstate swarm.
    Announce(Vec<Multiaddr>),
    /// Query the rendezvous key for other Midstate nodes.
    Discover,
}

/// Handle to the background Amino rendezvous task.
///
/// # Safety / Invariants
/// - **Non-blocking:** every method uses `try_send`. The node event loop must
///   never stall on DHT work, so a full command queue silently drops the
///   request; the next interval tick retries.
pub struct AminoHandle {
    tx: mpsc::Sender<AminoCommand>,
}

impl AminoHandle {
    /// Publishes this node as a provider of the rendezvous key.
    ///
    /// # Reasoning
    /// This is how a node becomes findable by strangers with no Midstate-run
    /// infrastructure involved. `start_providing` republishes automatically on
    /// libp2p's `provider_publication_interval` (12h by default) for as long as
    /// the record stays in the local store — unlike the seed registry, which
    /// must be re-POSTed before its 3600s TTL, and unlike Cloudflare KV, there
    /// is no daily write quota capping network size.
    ///
    /// Callers must pass only directly routable addresses. `/p2p-circuit` paths
    /// describe a route through someone else's node and would advertise a relay
    /// operator's IP as ours; webrtc-direct is a browser transport that
    /// `dial_addr` refuses server-to-server.
    ///
    /// # Formal Specification
    ///
    /// ```text
    /// Pre:
    ///   - nat_status ≠ Private          (enforced by the caller)
    ///   - #addrs > 0
    ///   - ∀ a ∈ addrs • is_routable(a)  (no circuit, no loopback, no CGNAT)
    ///
    /// Post:
    ///   command queued ⇒ eventually providers' = providers ⊕ {key ↦ (self, addrs, now + TTL)}
    ///   queue full     ⇒ providers' = providers   (dropped; retried next tick)
    /// ```
    ///
    /// ```zed
    ///     AminoAnnounce
    ///     ----------------
    ///     ΔProviderStore
    ///     key?   : Multihash
    ///     addrs? : seq Multiaddr
    ///     self   : PeerId
    ///
    ///     pre  nat_status ≠ Private
    ///     pre  #addrs? > 0
    ///
    ///     post providers' = providers ⊕ {key? ↦ (self, addrs?, now + TTL)}
    ///     post ∀ k ∈ dom providers • k ≠ key? ⇒ providers'(k) = providers(k)
    /// ```
    pub fn announce(&self, addrs: Vec<Multiaddr>) {
        if addrs.is_empty() {
            tracing::debug!("Amino announce skipped: no routable addresses");
            return;
        }
        let _ = self.tx.try_send(AminoCommand::Announce(addrs));
    }

    /// Queries the rendezvous key for other Midstate nodes.
    ///
    /// # Reasoning
    /// Callers must invoke this only when peer-starved. Polling unconditionally
    /// would put avoidable load on a commons lending us capacity for free, and
    /// a node with a healthy outbound set gains nothing from the answer.
    ///
    /// # Formal Specification
    ///
    /// ```text
    /// Pre:  outbound_peer_count < TARGET_OUTBOUND_PEERS
    /// Post:
    ///   a GET_PROVIDERS query is eventually issued for the rendezvous key
    ///   results arrive asynchronously on the discovery channel
    ///   provider store unchanged (read-only)
    /// ```
    ///
    /// ```zed
    ///     AminoDiscover
    ///     ----------------
    ///     ΞProviderStore
    ///     key?   : Multihash
    ///     addrs! : seq Multiaddr
    ///
    ///     pre  outbound_peer_count < TARGET_OUTBOUND_PEERS
    ///
    ///     post addrs! = { a | ∃ p • (key? ↦ p) ∈ providers ∧ a ∈ addr_of(p) ∧ p ≠ self }
    ///     post #addrs! ≤ MAX_ADDRS_PER_ROUND
    /// ```
    pub fn discover(&self) {
        let _ = self.tx.try_send(AminoCommand::Discover);
    }
}

// ── Spawning ────────────────────────────────────────────────────────────────

/// Starts the Amino rendezvous task and returns a handle plus a discovery
/// channel.
///
/// # Reasoning
/// Uses the node's existing libp2p keypair so the PeerId advertised in the
/// provider record is the *same* identity the main swarm listens under. A
/// discoverer therefore dials our real Midstate listener and speaks the real
/// Midstate protocol; this swarm exists only to publish and to look up.
///
/// This swarm deliberately does not listen. It needs outbound connectivity to
/// route DHT queries and nothing more, so it opens no port and adds no inbound
/// attack surface. Its "external addresses" are injected from the main swarm on
/// each announce, which is what makes provider records point somewhere useful.
///
/// # Formal Specification
///
/// ```text
/// Pre:  keypair is the node's persistent libp2p identity
/// Post:
///   result = Ok((handle!, rx!)) ⇒
///     a background task owns an isolated Swarm speaking only /ipfs/kad/1.0.0
///     handle!.peer_id = main_swarm.peer_id
///     no listener is opened
///   result = Err(_) ⇒ caller proceeds without DHT discovery (non-fatal)
/// ```
///
/// # Safety / Invariants
/// - **Isolation:** this Swarm shares no behaviour, routing table, or
///   connection pool with `MidstateBehaviour`. Foreign peers cannot cross over.
/// - **Bounded channels:** both directions use bounded channels so neither the
///   node loop nor the DHT task can grow memory under backpressure.
pub fn spawn(
    keypair: Keypair,
    anchor: [u8; 32],
) -> Result<(AminoHandle, mpsc::Receiver<Vec<String>>)> {
    let (cmd_tx, cmd_rx) = mpsc::channel::<AminoCommand>(8);
    let (found_tx, found_rx) = mpsc::channel::<Vec<String>>(4);

    let local_peer = keypair.public().to_peer_id();

    let swarm = libp2p::SwarmBuilder::with_existing_identity(keypair.clone())
        .with_tokio()
        .with_tcp(
            tcp::Config::default().nodelay(true),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_quic()
        // Required for the /dnsaddr/ bootstrap entries. Note this is scoped to
        // this swarm only — the main Midstate transport is untouched.
        .with_dns()?
        .with_behaviour(|key: &Keypair| {
            let store = kad::store::MemoryStore::new(local_peer);

            // Config::new(protocol) is the libp2p 0.56 constructor; the older
            // Config::default() + set_protocol_names pairing is gone.
            let cfg = kad::Config::new(StreamProtocol::new(AMINO_KAD_PROTOCOL));

            let mut kad_behaviour = kad::Behaviour::with_config(local_peer, store, cfg);
            // Client mode is deliberate: a server must answer routing queries
            // for arbitrary IPFS content, which costs real bandwidth and makes
            // the node look like an IPFS gateway to hosting abuse desks.
            // Publishing and retrieving provider records works fine as a client.
            kad_behaviour.set_mode(Some(kad::Mode::Client));

            let identify = identify::Behaviour::new(identify::Config::new(
                "/ipfs/id/1.0.0".to_string(),
                key.public(),
            ));

            AminoBehaviour {
                kad: kad_behaviour,
                identify,
            }
        })?
        .build();

    tokio::spawn(async move {
        run_task(swarm, local_peer, anchor, cmd_rx, found_tx).await;
    });

    Ok((AminoHandle { tx: cmd_tx }, found_rx))
}

// ── Task ────────────────────────────────────────────────────────────────────

/// Seeds the routing table with the public bootstrap fleet.
///
/// # Formal Specification
///
/// ```text
/// Pre:  true
/// Post:
///   ∀ a ∈ AMINO_BOOTSTRAP • parses(a) ∧ has_peer_id(a) ⇒ (peer_id(a) ↦ a) ∈ routing_table'
///   count! = number of entries successfully added
///   count! = 0 ⇒ DHT discovery inoperable this session (logged, non-fatal)
/// ```
///
/// # Safety / Invariants
/// - **Per-entry isolation:** a malformed or stale entry must never prevent the
///   remaining entries from being tried.
fn seed_bootstrap(swarm: &mut Swarm<AminoBehaviour>) -> usize {
    let mut added = 0usize;

    for entry in AMINO_BOOTSTRAP {
        let addr: Multiaddr = match entry.parse() {
            Ok(a) => a,
            Err(e) => {
                tracing::debug!("Skipping malformed Amino bootstrap {}: {}", entry, e);
                continue;
            }
        };

        let peer = addr.iter().find_map(|p| match p {
            Protocol::P2p(id) => Some(id),
            _ => None,
        });

        match peer {
            Some(id) => {
                swarm.behaviour_mut().kad.add_address(&id, addr.clone());
                let _ = swarm.dial(addr);
                added += 1;
            }
            None => tracing::debug!("Amino bootstrap entry has no peer id: {}", entry),
        }
    }

    if added == 0 {
        tracing::warn!("No Amino bootstrap peers usable; DHT discovery inoperable this session");
    } else {
        tracing::info!("Seeded Amino DHT with {} bootstrap peers", added);
    }
    added
}

/// Resolves provider PeerIds to dialable multiaddrs via the local routing table.
///
/// # Reasoning
/// `GetProvidersOk::FoundProviders` yields only PeerIds. The addresses arrive
/// separately: Kademlia inserts provider peers into the routing table as it
/// learns them, so the kbuckets are the authoritative local source. This mirrors
/// the pattern already used by `MidstateNetwork::connected_peer_addrs`.
///
/// Every emitted address carries a `/p2p/<id>` suffix so the main swarm can
/// verify identity during the handshake rather than trusting the DHT's word.
///
/// # Formal Specification
///
/// ```text
/// Pre:  providers ⊆ PeerId
/// Post:
///   addrs! = { a ⌢ /p2p/p | p ∈ providers ∧ p ≠ local ∧ a ∈ routing_table(p) }
///   #addrs! ≤ MAX_ADDRS_PER_ROUND
///   ∀ a ∈ addrs! • a carries an explicit peer id
/// ```
fn resolve_addrs(
    swarm: &mut Swarm<AminoBehaviour>,
    providers: &HashSet<PeerId>,
    local: &PeerId,
) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for bucket in swarm.behaviour_mut().kad.kbuckets() {
        for entry in bucket.iter() {
            let peer = entry.node.key.preimage();
            if peer == local || !providers.contains(peer) {
                continue;
            }
            for addr in entry.node.value.iter() {
                let s = if addr.to_string().contains("/p2p/") {
                    addr.to_string()
                } else {
                    addr.clone().with(Protocol::P2p(*peer)).to_string()
                };
                if seen.insert(s.clone()) {
                    out.push(s);
                }
            }
        }
    }

    out.truncate(MAX_ADDRS_PER_ROUND);
    out
}

/// The background task: owns the isolated Swarm and services commands.
///
/// # Safety / Invariants
/// - **Terminates cleanly:** returns when the command channel closes, i.e. when
///   the node drops its `AminoHandle`.
/// - **Never panics into the node:** all failures are logged and swallowed;
///   this task must not be able to take down block processing.
async fn run_task(
    mut swarm: Swarm<AminoBehaviour>,
    local_peer: PeerId,
    anchor: [u8; 32],
    mut cmd_rx: mpsc::Receiver<AminoCommand>,
    found_tx: mpsc::Sender<Vec<String>>,
) {
    let key = rendezvous_key(&anchor);
    seed_bootstrap(&mut swarm);

    // Let the routing table fill before the first iterative query; a bootstrap
    // walk against an almost-empty table wastes a round trip.
    let mut warmup = tokio::time::interval_at(
        tokio::time::Instant::now() + WARMUP,
        Duration::from_secs(30 * 60),
    );
    warmup.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            _ = warmup.tick() => {
                if let Err(e) = swarm.behaviour_mut().kad.bootstrap() {
                    tracing::debug!("Amino bootstrap walk failed: {:?}", e);
                }
            }

            cmd = cmd_rx.recv() => {
                match cmd {
                    Some(AminoCommand::Announce(addrs)) => {
                        // Provider records carry the addresses libp2p considers
                        // external for *this* swarm. Since it never listens, we
                        // inject the main swarm's addresses so discoverers dial
                        // the real Midstate listener.
                        for a in addrs {
                            let clean = strip_peer_id(a);
                            swarm.add_external_address(clean);
                        }
                        match swarm.behaviour_mut().kad.start_providing(key.clone()) {
                            Ok(_) => tracing::info!("Announced on Amino DHT rendezvous key"),
                            Err(e) => tracing::warn!("Amino announce failed: {:?}", e),
                        }
                    }
                    Some(AminoCommand::Discover) => {
                        tracing::debug!("Querying Amino DHT for Midstate peers");
                        swarm.behaviour_mut().kad.get_providers(key.clone());
                    }
                    // Node dropped the handle; shut down.
                    None => {
                        tracing::debug!("Amino task stopping");
                        return;
                    }
                }
            }

            event = swarm.select_next_some() => {
                if let SwarmEvent::Behaviour(AminoBehaviourEvent::Kad(
                    kad::Event::OutboundQueryProgressed {
                        result: kad::QueryResult::GetProviders(Ok(
                            kad::GetProvidersOk::FoundProviders { providers, .. }
                        )),
                        ..
                    },
                )) = event
                {
                    if providers.is_empty() {
                        continue;
                    }
                    let addrs = resolve_addrs(&mut swarm, &providers, &local_peer);
                    if addrs.is_empty() {
                        tracing::debug!(
                            "Amino: {} providers found but no addresses known yet",
                            providers.len()
                        );
                        continue;
                    }
                    tracing::info!("Amino rendezvous: {} candidate addresses", addrs.len());
                    // try_send: if the node is busy, dropping is correct —
                    // the next discovery tick will repeat the query.
                    let _ = found_tx.try_send(addrs);
                }
            }
        }
    }
}

/// Removes a trailing `/p2p/<id>` component.
///
/// `Swarm::add_external_address` expects a transport address; libp2p appends the
/// peer id itself when building provider records, and a doubled suffix would
/// produce an address no peer can parse back into a dial.
fn strip_peer_id(addr: Multiaddr) -> Multiaddr {
    addr.into_iter()
        .filter(|p| !matches!(p, Protocol::P2p(_)))
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rendezvous_key_is_deterministic() {
        let anchor = [7u8; 32];
        assert_eq!(rendezvous_key(&anchor), rendezvous_key(&anchor));
    }

    #[test]
    fn rendezvous_key_separates_networks() {
        // Testnet, or any fork changing the anchor, must not collide.
        assert_ne!(rendezvous_key(&[1u8; 32]), rendezvous_key(&[2u8; 32]));
    }

    #[test]
    fn rendezvous_key_is_well_formed_multihash() {
        let bytes = rendezvous_key(&[0u8; 32]).to_vec();
        assert_eq!(bytes.len(), 34, "code + length + 32-byte digest");
        assert_eq!(bytes[0], MULTIHASH_CODE_BLAKE3);
        assert_eq!(bytes[1], 32);
    }

    #[test]
    fn bootstrap_entries_parse_and_carry_peer_ids() {
        for entry in AMINO_BOOTSTRAP {
            let addr: Multiaddr = entry.parse().expect("bootstrap entry must parse");
            assert!(
                addr.iter().any(|p| matches!(p, Protocol::P2p(_))),
                "bootstrap entry needs a peer id: {}",
                entry
            );
        }
    }

    #[test]
    fn strip_peer_id_removes_only_the_p2p_component() {
        let with: Multiaddr =
            "/ip4/1.2.3.4/tcp/9333/p2p/12D3KooWPbR63SQg1UBLpAMiNngqrRHGM4LaMP8ieAJUxhfw7dxv"
                .parse()
                .unwrap();
        let stripped = strip_peer_id(with);
        assert_eq!(stripped.to_string(), "/ip4/1.2.3.4/tcp/9333");
        // Idempotent on addresses that never had one.
        let plain: Multiaddr = "/ip4/1.2.3.4/udp/9333/quic-v1".parse().unwrap();
        assert_eq!(strip_peer_id(plain.clone()), plain);
    }
}
