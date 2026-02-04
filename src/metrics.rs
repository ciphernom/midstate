use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone)]
pub struct Metrics {
    transactions_processed: Arc<AtomicU64>,
    batches_processed: Arc<AtomicU64>,
    batches_mined: Arc<AtomicU64>,
    invalid_batches: Arc<AtomicU64>,
    invalid_transactions: Arc<AtomicU64>,
    reorgs: Arc<AtomicU64>,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            transactions_processed: Arc::new(AtomicU64::new(0)),
            batches_processed: Arc::new(AtomicU64::new(0)),
            batches_mined: Arc::new(AtomicU64::new(0)),
            invalid_batches: Arc::new(AtomicU64::new(0)),
            invalid_transactions: Arc::new(AtomicU64::new(0)),
            reorgs: Arc::new(AtomicU64::new(0)),
        }
    }
    
    pub fn inc_transactions_processed(&self) {
        self.transactions_processed.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn inc_batches_processed(&self) {
        self.batches_processed.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn inc_batches_mined(&self) {
        self.batches_mined.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn inc_invalid_batches(&self) {
        self.invalid_batches.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn inc_invalid_transactions(&self) {
        self.invalid_transactions.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn inc_reorgs(&self) {
        self.reorgs.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn report(&self) {
        tracing::info!(
            "Metrics: txs={} batches={} mined={} invalid_batches={} invalid_txs={} reorgs={}",
            self.transactions_processed.load(Ordering::Relaxed),
            self.batches_processed.load(Ordering::Relaxed),
            self.batches_mined.load(Ordering::Relaxed),
            self.invalid_batches.load(Ordering::Relaxed),
            self.invalid_transactions.load(Ordering::Relaxed),
            self.reorgs.load(Ordering::Relaxed),
        );
    }
}
