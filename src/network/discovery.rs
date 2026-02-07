use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::path::Path;
use std::time::{Duration, SystemTime};
use anyhow::Result;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerInfo {
    pub addr: SocketAddr,
    pub last_seen: SystemTime,
    pub misbehavior_score: u32,
    pub services: u64,
}

#[derive(Serialize, Deserialize)]
pub struct AddressBook {
    tried: HashMap<SocketAddr, PeerInfo>,
    new: HashMap<SocketAddr, PeerInfo>,
    #[serde(skip)]
    connected: HashSet<SocketAddr>,
    banned: HashSet<SocketAddr>,
}

impl AddressBook {
    pub fn new() -> Self {
        Self {
            tried: HashMap::new(),
            new: HashMap::new(),
            connected: HashSet::new(),
            banned: HashSet::new(),
        }
    }

    /// Load from disk, falling back to empty if missing or corrupt.
    pub fn load(path: &Path) -> Self {
        match std::fs::read(path) {
            Ok(bytes) => {
                match serde_json::from_slice::<AddressBook>(&bytes) {
                    Ok(book) => {
                        tracing::info!(
                            "Loaded address book: {} tried, {} new, {} banned",
                            book.tried.len(),
                            book.new.len(),
                            book.banned.len()
                        );
                        book
                    }
                    Err(e) => {
                        tracing::warn!("Corrupt address book, starting fresh: {}", e);
                        Self::new()
                    }
                }
            }
            Err(_) => Self::new(),
        }
    }

    /// Persist to disk.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = serde_json::to_vec(self)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    pub fn add_address(&mut self, addr: SocketAddr) {
        if self.banned.contains(&addr) {
            return;
        }
        
        if !self.tried.contains_key(&addr) && !self.connected.contains(&addr) {
            self.new.insert(addr, PeerInfo {
                addr,
                last_seen: SystemTime::now(),
                misbehavior_score: 0,
                services: 0,
            });
        }
    }
    
    pub fn mark_tried(&mut self, addr: SocketAddr) {
        if let Some(info) = self.new.remove(&addr) {
            self.tried.insert(addr, info);
        }
    }
    
    pub fn mark_connected(&mut self, addr: SocketAddr) {
        self.connected.insert(addr);
    }
    
    pub fn mark_disconnected(&mut self, addr: SocketAddr) {
        self.connected.remove(&addr);
    }
    
    pub fn mark_misbehavior(&mut self, addr: SocketAddr, score: u32) {
        if let Some(info) = self.tried.get_mut(&addr) {
            info.misbehavior_score += score;
            
            if info.misbehavior_score >= 100 {
                self.ban_peer(addr);
            }
        }
    }
    
    pub fn ban_peer(&mut self, addr: SocketAddr) {
        tracing::warn!("Banning peer: {}", addr);
        self.banned.insert(addr);
        self.tried.remove(&addr);
        self.new.remove(&addr);
        self.connected.remove(&addr);
    }
    
    pub fn is_banned(&self, addr: SocketAddr) -> bool {
        self.banned.contains(&addr)
    }
    
    pub fn get_peers_to_try(&self, count: usize) -> Vec<SocketAddr> {
        use rand::seq::SliceRandom;
        
        let mut rng = rand::thread_rng();
        
        let from_new = count / 2;
        let from_tried = count - from_new;
        
        let mut result = Vec::new();
        
        let new_addrs: Vec<_> = self.new.keys()
            .filter(|addr| !self.connected.contains(addr))
            .copied()
            .collect();
        
        let mut new_sample: Vec<_> = new_addrs.choose_multiple(&mut rng, from_new).copied().collect();
        result.append(&mut new_sample);
        
        let tried_addrs: Vec<_> = self.tried.keys()
            .filter(|addr| !self.connected.contains(addr))
            .copied()
            .collect();
        
        let mut tried_sample: Vec<_> = tried_addrs.choose_multiple(&mut rng, from_tried).copied().collect();
        result.append(&mut tried_sample);
        
        result
    }
    
    pub fn get_random_peers(&self, count: usize) -> Vec<SocketAddr> {
        use rand::seq::SliceRandom;
        
        let all: Vec<_> = self.tried.keys()
            .chain(self.new.keys())
            .filter(|addr| !self.banned.contains(addr))
            .copied()
            .collect();
        
        all.choose_multiple(&mut rand::thread_rng(), count)
            .copied()
            .collect()
    }
    
    pub fn prune_old(&mut self, max_age: Duration) {
        let cutoff = SystemTime::now() - max_age;
        
        self.new.retain(|_, info| info.last_seen > cutoff);
        self.tried.retain(|_, info| info.last_seen > cutoff);
    }
}

pub const SEED_NODES: &[&str] = &[
    // For local testing - add real DNS names for production
];

pub async fn discover_peers(address_book: &mut AddressBook) -> Result<()> {
    for seed in SEED_NODES {
        match seed.parse::<SocketAddr>() {
            Ok(addr) => {
                address_book.add_address(addr);
            }
            Err(_) => {
                match tokio::net::lookup_host(seed).await {
                    Ok(addrs) => {
                        for addr in addrs {
                            address_book.add_address(addr);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to resolve seed {}: {}", seed, e);
                    }
                }
            }
        }
    }
    
    Ok(())
}
