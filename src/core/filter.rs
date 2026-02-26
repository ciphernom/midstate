use crate::core::types::{Batch, Transaction, hash_concat};

/// False Positive Rate = 1 / FPR_INVERSE (1 in 1,000,000)
const FPR_INVERSE: u64 = 1_000_000;
/// Golomb-Rice parameter: 2^P ≈ FPR_INVERSE. For 1,000,000, P = 20.
const P: u8 = 20; 

pub struct CompactFilter {
    pub data: Vec<u8>,
}

impl CompactFilter {
    /// Build a Golomb-Coded Set filter for a given Batch
    pub fn build(batch: &Batch) -> Self {
        let mut items = Vec::new();

        // 1. Extract all identifiable elements from the batch
        for tx in &batch.transactions {
            match tx {
                Transaction::Commit { commitment, .. } => {
                    items.push(*commitment);
                }
                Transaction::Reveal { inputs, outputs, .. } => {
                    for input in inputs {
                        items.push(input.coin_id());
                        items.push(input.predicate.address());
                    }
                    for output in outputs {
                        if let Some(cid) = output.coin_id() {
                            items.push(cid);
                        }
                        items.push(output.address());
                    }
                }
            }
        }
        for cb in &batch.coinbase {
            items.push(cb.coin_id());
            items.push(cb.address);
        }

        // Deduplicate
        items.sort();
        items.dedup();

        let n = items.len() as u64;
        if n == 0 {
            return Self { data: vec![] };
        }

        // 2. Hash items into a uniform distribution [0, N * FPR]
        let modulus = n * FPR_INVERSE;
        let mut hashes: Vec<u64> = items.into_iter().map(|item| {
            // Key the hash with the block's final_hash to prevent precomputation attacks
            let h = hash_concat(&batch.extension.final_hash, &item);
            let raw = u64::from_le_bytes(h[..8].try_into().unwrap());
            raw % modulus
        }).collect();

        // 3. Sort hashes to encode deltas
        hashes.sort();

        // 4. Golomb-Rice encoding of the deltas
        let mut writer = BitWriter::new();
        let mut last = 0u64;
        for h in hashes {
            let diff = h - last;
            encode_golomb(&mut writer, diff);
            last = h;
        }

        Self { data: writer.into_bytes() }
    }
}

// ── Bit Fiddling Helpers ────────────────────────────────────────────────────

struct BitWriter {
    buffer: Vec<u8>,
    current_byte: u8,
    bits_in_byte: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self { buffer: Vec::new(), current_byte: 0, bits_in_byte: 0 }
    }

    fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current_byte |= 1 << (7 - self.bits_in_byte);
        }
        self.bits_in_byte += 1;
        if self.bits_in_byte == 8 {
            self.buffer.push(self.current_byte);
            self.current_byte = 0;
            self.bits_in_byte = 0;
        }
    }

    fn write_bits(&mut self, value: u64, count: u8) {
        for i in (0..count).rev() {
            self.write_bit((value >> i) & 1 == 1);
        }
    }

    fn into_bytes(mut self) -> Vec<u8> {
        if self.bits_in_byte > 0 {
            self.buffer.push(self.current_byte);
        }
        self.buffer
    }
}

/// Golomb-Rice encoding: Quotient as unary, remainder as binary
fn encode_golomb(writer: &mut BitWriter, value: u64) {
    let quotient = value >> P;
    let remainder = value & ((1 << P) - 1);

    // Unary encode quotient (Q '1's followed by a '0')
    for _ in 0..quotient {
        writer.write_bit(true);
    }
    writer.write_bit(false);

    // Binary encode remainder
    writer.write_bits(remainder, P);
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{CoinbaseOutput, InputReveal, OutputData, Predicate, Extension};

    // Helper to generate a minimal valid-looking Batch
    fn dummy_batch() -> Batch {
        Batch {
            prev_midstate: [0; 32],
            transactions: vec![],
            extension: Extension { nonce: 0, final_hash: [1; 32], checkpoints: vec![] },
            coinbase: vec![],
            timestamp: 0,
            target: [0xff; 32],
            state_root: [0; 32],
        }
    }

    // ── BitWriter Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_bit_writer_basic() {
        let mut writer = BitWriter::new();
        writer.write_bit(true);  // 1
        writer.write_bit(false); // 0
        writer.write_bits(0b1011, 4); // 1011
        writer.write_bit(false); // 0
        writer.write_bit(true);  // 1
        
        // Expected bits: 10101101 -> 0xAD
        let bytes = writer.into_bytes();
        assert_eq!(bytes, vec![0xAD]);
    }

    #[test]
    fn test_bit_writer_cross_byte_boundary() {
        let mut writer = BitWriter::new();
        writer.write_bits(0xFF, 8); // 11111111
        writer.write_bit(true);     // 10000000 (padded to byte)
        
        let bytes = writer.into_bytes();
        assert_eq!(bytes, vec![0xFF, 0x80]); // Second byte is padded with trailing zeros
    }

    // ── CompactFilter Builder Tests ─────────────────────────────────────────

    #[test]
    fn test_empty_batch_yields_empty_filter() {
        let batch = dummy_batch();
        let filter = CompactFilter::build(&batch);
        assert!(filter.data.is_empty(), "Empty batch should have an empty filter");
    }

    #[test]
    fn test_filter_with_coinbase() {
        let mut batch = dummy_batch();
        batch.coinbase.push(CoinbaseOutput {
            address: [2; 32],
            value: 50,
            salt: [3; 32],
        });

        let filter = CompactFilter::build(&batch);
        assert!(!filter.data.is_empty(), "Filter should contain encoded data");
    }

    #[test]
    fn test_filter_deduplication() {
        // Create a batch with duplicate items (e.g., a tx sending to the same address twice)
        let mut batch_dup = dummy_batch();
        let address = [2; 32]; // <--- Change 'addr' to 'address'
        batch_dup.coinbase.push(CoinbaseOutput { address, value: 50, salt: [3; 32] });
        batch_dup.coinbase.push(CoinbaseOutput { address, value: 25, salt: [4; 32] });

        // Create a batch with only the unique identifiable elements
        let mut batch_single = dummy_batch();
        batch_single.coinbase.push(CoinbaseOutput { address, value: 50, salt: [3; 32] });

        let filter_dup = CompactFilter::build(&batch_dup);
        let filter_single = CompactFilter::build(&batch_single);

        assert!(!filter_dup.data.is_empty());
        
        // Let's force an exact duplicate item insertion
        let mut batch_exact_dup = dummy_batch();
        let cb = CoinbaseOutput { address, value: 50, salt: [3; 32] };
        batch_exact_dup.coinbase.push(cb.clone());
        batch_exact_dup.coinbase.push(cb); // Exact duplicate
        
        let filter_exact_dup = CompactFilter::build(&batch_exact_dup);
        assert_eq!(filter_exact_dup.data, filter_single.data, "Exact duplicate elements should be deduped");
    }

    #[test]
    fn test_filter_determinism() {
        let mut batch = dummy_batch();
        batch.transactions.push(Transaction::Commit {
            commitment: [5; 32],
            spam_nonce: 123,
        });

        let filter1 = CompactFilter::build(&batch);
        let filter2 = CompactFilter::build(&batch);
        
        assert_eq!(filter1.data, filter2.data, "Building the same filter twice must yield identical bytes");
    }

    #[test]
    fn test_filter_extracts_all_tx_elements() {
        let mut batch = dummy_batch();
        
        // Add a commit
        batch.transactions.push(Transaction::Commit {
            commitment: [7; 32],
            spam_nonce: 0,
        });

        // Add a reveal
        batch.transactions.push(Transaction::Reveal {
            inputs: vec![InputReveal {
                predicate: Predicate::p2pk(&[8; 32]),
                value: 10,
                salt: [9; 32],
            }],
            witnesses: vec![],
            outputs: vec![OutputData::Standard {
                address: [10; 32],
                value: 9,
                salt: [11; 32],
            }],
            salt: [12; 32],
        });

        let filter = CompactFilter::build(&batch);
        assert!(!filter.data.is_empty());
        // Since P=20, each element adds at least ~20 bits (2.5 bytes). 
        // We have 1 commit + (1 input coin_id + 1 input addr) + (1 output coin_id + 1 output addr) = 5 items.
        // 5 items * 2.5 bytes = ~12.5 bytes minimum.
        assert!(filter.data.len() >= 12, "Filter size should reflect the number of elements");
    }
}
