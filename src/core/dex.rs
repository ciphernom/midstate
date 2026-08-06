//! On-chain DEX order announcements.
//!
//! Ported byte-for-byte from `worker.js` so the native and web wallets read
//! and write the same order book. Every integer here is **big-endian**, unlike
//! the Q-Bolt channel wire which is little-endian — mixing them up produces
//! announcements that decode to plausible garbage, so the two live in separate
//! modules deliberately.
//!
//! Why announcements exist at all: a swap's coin salt is the one piece of
//! state a wallet cannot re-derive from its seed. Lose it and the funds are
//! stranded even though the key is fine. Publishing the salt (never the
//! preimage) in a zero-value `DataBurn` makes every order recoverable from
//! seed alone.
//!
//! Why fragmentation exists: consensus caps a burn payload at
//! [`MAX_BURN_DATA_SIZE`] = 80 bytes, but a self-contained MDXA is 72 bytes of
//! header plus 81 per unit — it can never fit. So an announcement is split
//! into MDXF fragments and all of them ride as separate burns inside the *same*
//! funding transaction, landing in one block and reassembling trivially.


use super::script;
use super::types::{hash, InputReveal, Predicate, Witness, MAX_BURN_DATA_SIZE};
use anyhow::{bail, Result};

pub const ANN_MAGIC: [u8; 4] = *b"MDXA";
pub const FRAG_MAGIC: [u8; 4] = *b"MDXF";
pub const TAKER_MAGIC: [u8; 4] = *b"MDXT";
pub const ANN_VER: u8 = 1;
pub const TAKER_VER: u8 = 1;

/// magic 4 + groupId 6 + idx 1 + total 1
pub const FRAG_HEADER_BYTES: usize = 12;
pub const FRAG_PAYLOAD_BYTES: usize = MAX_BURN_DATA_SIZE - FRAG_HEADER_BYTES; // 68

/// One sellable unit of a maker's order: a coin of `value` offered for
/// `wei_amount`, hash-locked to `secret_hash`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnnUnit {
    pub secret_hash: [u8; 32],
    /// The coin salt — the value that cannot be re-derived, and the whole
    /// reason these announcements are published.
    pub salt: [u8; 32],
    /// Coin value. Must be a power of two; stored on the wire as its exponent.
    pub value: u64,
    pub wei_amount: u128,
}

/// A maker's limit order, covering one or more coins.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MakerAnnouncement {
    pub maker_evm_addr: [u8; 20],
    pub maker_mds_pk: [u8; 32],
    pub timeout_height: u64,
    pub group_id: [u8; 6],
    pub units: Vec<AnnUnit>,
}

/// The taker side of a swap: someone who locked MDS to fill a resting bid.
/// Same recovery role as [`MakerAnnouncement`], different shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TakerAnnouncement {
    /// Refund key — how the taker recognises their own lock on a rescan.
    pub taker_mds_pk: [u8; 32],
    pub secret_hash: [u8; 32],
    pub salt: [u8; 32],
    /// Buyer's receiving address, needed to rebuild the covenant address.
    pub receiver_addr: [u8; 32],
    pub timeout_height: u64,
    pub value: u64,
    pub wei_amount: u128,
}

fn log2_exact(v: u64) -> Result<u8> {
    if v == 0 || (v & (v - 1)) != 0 {
        bail!("dex: unit value {v} is not a power of two");
    }
    Ok(v.trailing_zeros() as u8)
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_be_bytes());
}
fn put_u128(out: &mut Vec<u8>, v: u128) {
    out.extend_from_slice(&v.to_be_bytes());
}
fn take<'a>(b: &'a [u8], o: &mut usize, n: usize) -> Option<&'a [u8]> {
    let s = b.get(*o..*o + n)?;
    *o += n;
    Some(s)
}
fn take32(b: &[u8], o: &mut usize) -> Option<[u8; 32]> {
    take(b, o, 32)?.try_into().ok()
}

// ── MDXA: maker orders ──────────────────────────────────────────────────

impl MakerAnnouncement {
    /// 72-byte header + 81 bytes per unit.
    pub fn encode(&self) -> Result<Vec<u8>> {
        if self.units.is_empty() || self.units.len() > 255 {
            bail!("dex: unit count out of range (1..=255)");
        }
        let mut out = Vec::with_capacity(72 + self.units.len() * 81);
        out.extend_from_slice(&ANN_MAGIC);
        out.push(ANN_VER);
        out.extend_from_slice(&self.maker_evm_addr);
        out.extend_from_slice(&self.maker_mds_pk);
        put_u64(&mut out, self.timeout_height);
        out.extend_from_slice(&self.group_id);
        out.push(self.units.len() as u8);
        for u in &self.units {
            out.extend_from_slice(&u.secret_hash);
            out.extend_from_slice(&u.salt);
            out.push(log2_exact(u.value)?);
            put_u128(&mut out, u.wei_amount);
        }
        Ok(out)
    }

    pub fn decode(b: &[u8]) -> Option<Self> {
        let mut o = 0usize;
        if take(b, &mut o, 4)? != ANN_MAGIC {
            return None;
        }
        if *take(b, &mut o, 1)?.first()? != ANN_VER {
            return None;
        }
        let maker_evm_addr: [u8; 20] = take(b, &mut o, 20)?.try_into().ok()?;
        let maker_mds_pk = take32(b, &mut o)?;
        let timeout_height = u64::from_be_bytes(take(b, &mut o, 8)?.try_into().ok()?);
        let group_id: [u8; 6] = take(b, &mut o, 6)?.try_into().ok()?;
        let n = *take(b, &mut o, 1)?.first()? as usize;
        if n == 0 {
            return None;
        }
        let mut units = Vec::with_capacity(n);
        for _ in 0..n {
            let secret_hash = take32(b, &mut o)?;
            let salt = take32(b, &mut o)?;
            let exp = *take(b, &mut o, 1)?.first()?;
            if exp >= 64 {
                return None;
            }
            let wei_amount = u128::from_be_bytes(take(b, &mut o, 16)?.try_into().ok()?);
            units.push(AnnUnit { secret_hash, salt, value: 1u64 << exp, wei_amount });
        }
        Some(Self { maker_evm_addr, maker_mds_pk, timeout_height, group_id, units })
    }

    pub fn total_wei(&self) -> u128 {
        self.units.iter().map(|u| u.wei_amount).sum()
    }
    pub fn total_value(&self) -> u64 {
        self.units.iter().map(|u| u.value).sum()
    }
}

// ── MDXT: taker locks ───────────────────────────────────────────────────

impl TakerAnnouncement {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(158);
        out.extend_from_slice(&TAKER_MAGIC);
        out.push(TAKER_VER);
        out.extend_from_slice(&self.taker_mds_pk);
        out.extend_from_slice(&self.secret_hash);
        out.extend_from_slice(&self.salt);
        out.extend_from_slice(&self.receiver_addr);
        put_u64(&mut out, self.timeout_height);
        out.push(log2_exact(self.value)?);
        put_u128(&mut out, self.wei_amount);
        Ok(out)
    }

    pub fn decode(b: &[u8]) -> Option<Self> {
        let mut o = 0usize;
        if take(b, &mut o, 4)? != TAKER_MAGIC {
            return None;
        }
        if *take(b, &mut o, 1)?.first()? != TAKER_VER {
            return None;
        }
        let taker_mds_pk = take32(b, &mut o)?;
        let secret_hash = take32(b, &mut o)?;
        let salt = take32(b, &mut o)?;
        let receiver_addr = take32(b, &mut o)?;
        let timeout_height = u64::from_be_bytes(take(b, &mut o, 8)?.try_into().ok()?);
        let exp = *take(b, &mut o, 1)?.first()?;
        if exp >= 64 {
            return None;
        }
        let wei_amount = u128::from_be_bytes(take(b, &mut o, 16)?.try_into().ok()?);
        Some(Self {
            taker_mds_pk,
            secret_hash,
            salt,
            receiver_addr,
            timeout_height,
            value: 1u64 << exp,
            wei_amount,
        })
    }
}

// ── MDXF: fragmentation ─────────────────────────────────────────────────

/// Split an encoded announcement into burn-sized fragments. All of them must
/// be published in the SAME transaction so they land in one block.
pub fn fragment(body: &[u8], group_id: &[u8; 6]) -> Result<Vec<Vec<u8>>> {
    let total = body.len().div_ceil(FRAG_PAYLOAD_BYTES).max(1);
    if total > 255 {
        bail!("dex: announcement needs {total} fragments (max 255)");
    }
    let mut out = Vec::with_capacity(total);
    for i in 0..total {
        let lo = i * FRAG_PAYLOAD_BYTES;
        let hi = ((i + 1) * FRAG_PAYLOAD_BYTES).min(body.len());
        let mut f = Vec::with_capacity(MAX_BURN_DATA_SIZE);
        f.extend_from_slice(&FRAG_MAGIC);
        f.extend_from_slice(group_id);
        f.push(i as u8);
        f.push(total as u8);
        f.extend_from_slice(&body[lo..hi]);
        debug_assert!(f.len() <= MAX_BURN_DATA_SIZE);
        out.push(f);
    }
    Ok(out)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fragment {
    pub group_id: [u8; 6],
    pub idx: u8,
    pub total: u8,
    pub chunk: Vec<u8>,
}

pub fn parse_fragment(b: &[u8]) -> Option<Fragment> {
    if b.len() <= FRAG_HEADER_BYTES || b[..4] != FRAG_MAGIC {
        return None;
    }
    let group_id: [u8; 6] = b[4..10].try_into().ok()?;
    let (idx, total) = (b[10], b[11]);
    if total == 0 || idx >= total {
        return None;
    }
    Some(Fragment { group_id, idx, total, chunk: b[FRAG_HEADER_BYTES..].to_vec() })
}

/// Accumulates fragments across blocks until a group is complete. Fragments
/// from one announcement normally arrive together, but a reorg or a partial
/// scan can split them, so the pool is deliberately tolerant.
#[derive(Default)]
pub struct FragmentPool {
    groups: std::collections::HashMap<([u8; 6], u8), Vec<Option<Vec<u8>>>>,
}

impl FragmentPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a fragment; returns the reassembled body once every piece is in.
    pub fn add(&mut self, f: Fragment) -> Option<Vec<u8>> {
        let slots = self
            .groups
            .entry((f.group_id, f.total))
            .or_insert_with(|| vec![None; f.total as usize]);
        slots[f.idx as usize] = Some(f.chunk);
        if slots.iter().any(|s| s.is_none()) {
            return None;
        }
        let body: Vec<u8> = slots.iter().flat_map(|s| s.clone().unwrap()).collect();
        self.groups.remove(&(f.group_id, f.total));
        Some(body)
    }

    pub fn pending(&self) -> usize {
        self.groups.len()
    }

    /// Drop partially-received groups so a long-running scan cannot grow
    /// without bound on fragments whose siblings never arrive.
    pub fn clear(&mut self) {
        self.groups.clear();
    }
}

/// Anything a burn payload can turn out to be.
#[derive(Clone, Debug)]
pub enum Announcement {
    Maker(MakerAnnouncement),
    Taker(TakerAnnouncement),
}

/// Classify one burn payload. Fragments go to the pool and only yield an
/// announcement once the group completes.
pub fn ingest(payload: &[u8], pool: &mut FragmentPool) -> Option<Announcement> {
    if let Some(t) = TakerAnnouncement::decode(payload) {
        return Some(Announcement::Taker(t));
    }
    if let Some(m) = MakerAnnouncement::decode(payload) {
        return Some(Announcement::Maker(m));
    }
    let f = parse_fragment(payload)?;
    let body = pool.add(f)?;
    if let Some(m) = MakerAnnouncement::decode(&body) {
        return Some(Announcement::Maker(m));
    }
    TakerAnnouncement::decode(&body).map(Announcement::Taker)
}

// ── Spending a limit order ──────────────────────────────────────────────
//
// The covenant behind a maker ask is not a plain HTLC, and the difference
// matters for both witnesses:
//
// ```text
// IF   HASH <secret_hash> EQUALVERIFY
//      THIS_ADDRESS SUM_TO_ADDR <max_claim> ADD
//      INPUT_VALUE GREATER_OR_EQUAL VERIFY
// ELSE DROP <timeout> CHECKTIMEVERIFY <refund_pk> CHECKSIGVERIFY
// ENDIF 1
// ```
//
// The claim branch contains **no signature check**. That is deliberate: a
// buyer arriving from the public order book has no key relationship with the
// maker, so holding the preimage *is* the authorisation. Copying the shape of
// a channel HTLC — which does check a signature — produces a witness that can
// never satisfy this script.
//
// The claim branch also enforces a remainder continuation: whatever is not
// claimed must be paid back into the same covenant. With one coin per unit and
// `max_claim` equal to that unit's value, a unit is taken whole and no
// continuation output is needed.

/// Address a maker ask's coins sit at.
pub fn limit_order_address(
    secret_hash: &[u8; 32],
    max_claim: u64,
    timeout_height: u64,
    refund_pk: &[u8; 32],
) -> [u8; 32] {
    hash(&script::compile_limit_order_covenant(secret_hash, max_claim, timeout_height, refund_pk))
}

/// Take a unit by revealing the preimage. Witness: `[preimage, 0x01]`.
pub fn limit_claim_input(
    secret_hash: &[u8; 32],
    max_claim: u64,
    timeout_height: u64,
    refund_pk: &[u8; 32],
    value: u64,
    salt: [u8; 32],
    preimage: &[u8; 32],
) -> Result<(InputReveal, Witness)> {
    if hash(preimage) != *secret_hash {
        bail!("dex: that preimage does not open this order");
    }
    if value > max_claim {
        bail!(
            "dex: claiming {value} exceeds the order's per-unit maximum of {max_claim}; the \
             remainder would have to be paid back into the covenant"
        );
    }
    let bytecode =
        script::compile_limit_order_covenant(secret_hash, max_claim, timeout_height, refund_pk);
    Ok((
        InputReveal { predicate: Predicate::Script { bytecode }, value, salt, commitment: None },
        Witness::ScriptInputs(vec![preimage.to_vec(), vec![0x01]]),
    ))
}

/// Maker reclaims an unsold unit after `timeout_height`.
/// Witness: `[refund_sig, <32-byte filler>, 0x00]` — the ELSE branch drops the
/// filler before checking the timelock, so the slot must still be occupied.
pub fn limit_reclaim_input(
    secret_hash: &[u8; 32],
    max_claim: u64,
    timeout_height: u64,
    refund_pk: &[u8; 32],
    value: u64,
    salt: [u8; 32],
    refund_sig: &[u8],
) -> Result<(InputReveal, Witness)> {
    if refund_sig.is_empty() {
        bail!("dex: reclaiming an order needs the maker's signature");
    }
    let bytecode =
        script::compile_limit_order_covenant(secret_hash, max_claim, timeout_height, refund_pk);
    Ok((
        InputReveal { predicate: Predicate::Script { bytecode }, value, salt, commitment: None },
        Witness::ScriptInputs(vec![refund_sig.to_vec(), vec![0u8; 32], vec![0x00]]),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(b: u8, value: u64, wei: u128) -> AnnUnit {
        AnnUnit { secret_hash: [b; 32], salt: [b ^ 0xff; 32], value, wei_amount: wei }
    }

    fn maker(n: usize) -> MakerAnnouncement {
        MakerAnnouncement {
            maker_evm_addr: [0xab; 20],
            maker_mds_pk: [0x11; 32],
            timeout_height: 250_000,
            group_id: [1, 2, 3, 4, 5, 6],
            units: (0..n).map(|i| unit(i as u8, 1 << (10 + i), 10_u128.pow(15) * (i as u128 + 1))).collect(),
        }
    }

    #[test]
    fn maker_roundtrip_and_header_size() {
        let m = maker(3);
        let enc = m.encode().unwrap();
        // 72-byte header + 81 per unit, exactly as worker.js computes it.
        assert_eq!(enc.len(), 72 + 3 * 81);
        assert_eq!(MakerAnnouncement::decode(&enc).unwrap(), m);
    }

    #[test]
    fn taker_roundtrip() {
        let t = TakerAnnouncement {
            taker_mds_pk: [9; 32],
            secret_hash: [8; 32],
            salt: [7; 32],
            receiver_addr: [6; 32],
            timeout_height: 987_654,
            value: 4096,
            wei_amount: 123_456_789_000_000_000,
        };
        let enc = t.encode().unwrap();
        assert_eq!(enc.len(), 158);
        assert_eq!(TakerAnnouncement::decode(&enc).unwrap(), t);
    }

    #[test]
    fn integers_are_big_endian() {
        // Guards against copying the little-endian Q-Bolt convention: the
        // high byte of the timeout must appear first.
        let mut m = maker(1);
        m.timeout_height = 0x0102_0304_0506_0708;
        let enc = m.encode().unwrap();
        assert_eq!(&enc[57..65], &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn every_fragment_fits_a_burn() {
        let m = maker(20); // 72 + 1620 = 1692 bytes, far past one burn
        let body = m.encode().unwrap();
        let frags = fragment(&body, &m.group_id).unwrap();
        assert!(frags.len() > 1);
        assert!(frags.iter().all(|f| f.len() <= MAX_BURN_DATA_SIZE));
    }

    #[test]
    fn fragments_reassemble_in_any_order() {
        let m = maker(12);
        let body = m.encode().unwrap();
        let mut frags = fragment(&body, &m.group_id).unwrap();
        frags.reverse(); // arrival order must not matter

        let mut pool = FragmentPool::new();
        let mut got = None;
        for (i, f) in frags.iter().enumerate() {
            let parsed = parse_fragment(f).unwrap();
            let out = pool.add(parsed);
            if i + 1 < frags.len() {
                assert!(out.is_none(), "completed early at fragment {i}");
            } else {
                got = out;
            }
        }
        assert_eq!(got.unwrap(), body);
        assert_eq!(pool.pending(), 0);
    }

    #[test]
    fn ingest_classifies_all_three_shapes() {
        let mut pool = FragmentPool::new();

        let t = TakerAnnouncement {
            taker_mds_pk: [1; 32],
            secret_hash: [2; 32],
            salt: [3; 32],
            receiver_addr: [4; 32],
            timeout_height: 1,
            value: 2,
            wei_amount: 3,
        };
        assert!(matches!(
            ingest(&t.encode().unwrap(), &mut pool),
            Some(Announcement::Taker(_))
        ));

        let m = maker(1);
        assert!(matches!(
            ingest(&m.encode().unwrap(), &mut pool),
            Some(Announcement::Maker(_))
        ));

        let big = maker(10);
        let body = big.encode().unwrap();
        let frags = fragment(&body, &big.group_id).unwrap();
        let mut last = None;
        for f in &frags {
            last = ingest(f, &mut pool);
        }
        match last {
            Some(Announcement::Maker(d)) => assert_eq!(d, big),
            other => panic!("expected a reassembled maker order, got {other:?}"),
        }

        assert!(ingest(b"not an announcement", &mut pool).is_none());
    }

    #[test]
    fn claim_witness_has_no_signature_slot() {
        let secret = [42u8; 32];
        let h = hash(&secret);
        let (input, witness) =
            limit_claim_input(&h, 1024, 500, &[1; 32], 1024, [2; 32], &secret).unwrap();
        let Witness::ScriptInputs(items) = witness;
        // Exactly two: preimage and the branch selector. A third item would
        // mean this was modelled on the channel HTLC, whose claim path does
        // check a signature.
        assert_eq!(items.len(), 2, "claim takes no signature");
        assert_eq!(items[0], secret.to_vec());
        assert_eq!(items[1], vec![0x01]);
        assert_eq!(input.value, 1024);
    }

    #[test]
    fn reclaim_witness_keeps_the_dropped_slot_occupied() {
        let (_, witness) =
            limit_reclaim_input(&[7; 32], 1024, 500, &[1; 32], 1024, [2; 32], &[9u8; 64]).unwrap();
        let Witness::ScriptInputs(items) = witness;
        assert_eq!(items.len(), 3);
        // The ELSE branch opens with DROP, so something 32 bytes wide must sit
        // there or the timelock check reads the signature.
        assert_eq!(items[1].len(), 32);
        assert_eq!(items[2], vec![0x00]);
        assert!(limit_reclaim_input(&[7; 32], 1024, 500, &[1; 32], 1024, [2; 32], &[]).is_err());
    }

    #[test]
    fn a_wrong_preimage_is_caught_before_broadcast() {
        let h = hash(&[42u8; 32]);
        assert!(limit_claim_input(&h, 1024, 500, &[1; 32], 1024, [2; 32], &[43u8; 32]).is_err());
    }

    #[test]
    fn claiming_more_than_the_unit_allows_is_refused() {
        let secret = [42u8; 32];
        let h = hash(&secret);
        // The script would demand the excess be paid back into the covenant;
        // refuse here rather than build a transaction that cannot validate.
        assert!(limit_claim_input(&h, 1024, 500, &[1; 32], 2048, [2; 32], &secret).is_err());
    }

    #[test]
    fn address_matches_the_compiled_covenant() {
        let h = [3u8; 32];
        assert_eq!(
            limit_order_address(&h, 512, 900, &[4; 32]),
            hash(&script::compile_limit_order_covenant(&h, 512, 900, &[4; 32]))
        );
    }

    #[test]
    fn non_power_of_two_value_is_refused() {
        let mut m = maker(1);
        m.units[0].value = 3000;
        assert!(m.encode().is_err());
    }
}
