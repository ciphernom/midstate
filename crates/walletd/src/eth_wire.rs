//! MDS -> ETH sell orders on midstate's chat bus.
//!
//! Four messages, each one chat message, all in the clear:
//!
//! ```text
//! words   = [WORD_DEX, WORD_MARK, kind]
//! advert  = VERSION | order_id[16] | seller_evm[20] | mds[8] | min[8] | wei_per_gmds[16] | expires[8]
//! take    = VERSION | order_id[16] | take_id[16] | mds[8] | wei[16] | claim_pk[32] | buyer_evm[20]
//! lock    = VERSION | take_id[16] | 1 | secret_hash[32] | timeout[8] | refund_pk[32]
//! decline = VERSION | take_id[16] | 0 | reason[1]
//! ```
//!
//! Nothing here needs to be secret or authentic, because nobody moves money
//! on a message alone. The buyer locks ETH only once the lock a `lock`
//! message describes is confirmed on midstate and claimable by the buyer's
//! own key; the seller claims ETH only once the buyer's lock on Base pays
//! the seller what the take offered. A forged or altered message costs its
//! target an attempt, never funds.
//!
//! Integers are little-endian.

use midstate::chat::{ChatAttachment, ChatMessage};

/// "dex": the first word of every message here, so the MDW market's parser
/// (whose messages lead with "swap") never sees them.
pub const WORD_DEX: u8 = 37;
/// "⚡", as qbolt's and the MDW market's frames lead with.
pub const WORD_MARK: u8 = 255;
/// "sell": an order advert.
pub const KIND_SELL: u8 = 31;
/// "take": a buyer's request.
pub const KIND_TAKE: u8 = 43;
/// "trade": the seller's answer, a lock or a decline.
pub const KIND_TRADE: u8 = 32;
/// "who": a name claim for an order or a take. Its own message, so wallets
/// that predate it, which read only the kinds above, see the orders as before.
pub const KIND_WHO: u8 = 91;

pub const VERSION: u8 = 1;

/// A sell order, as advertised.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Advert {
    pub order_id: [u8; 16],
    /// Where the buyer's ETH goes.
    pub seller_evm: [u8; 20],
    /// MDS (units) for sale now.
    pub mds: u64,
    /// The smallest take the seller accepts.
    pub min: u64,
    /// Price: wei per 2^30 units.
    pub wei_per_gmds: u128,
    /// Midstate height after which the order no longer stands.
    pub expires: u64,
}

impl Advert {
    pub const LEN: usize = 1 + 16 + 20 + 8 + 8 + 16 + 8;

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::LEN);
        out.push(VERSION);
        out.extend_from_slice(&self.order_id);
        out.extend_from_slice(&self.seller_evm);
        out.extend_from_slice(&self.mds.to_le_bytes());
        out.extend_from_slice(&self.min.to_le_bytes());
        out.extend_from_slice(&self.wei_per_gmds.to_le_bytes());
        out.extend_from_slice(&self.expires.to_le_bytes());
        out
    }

    pub fn decode(b: &[u8]) -> Option<Self> {
        if b.len() != Self::LEN || b[0] != VERSION {
            return None;
        }
        let mut r = Reader { b, o: 1 };
        Some(Self {
            order_id: r.arr()?,
            seller_evm: r.arr()?,
            mds: r.u64()?,
            min: r.u64()?,
            wei_per_gmds: r.u128()?,
            expires: r.u64()?,
        })
    }
}

/// A buyer's request to take `mds` of an order for `wei`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Take {
    pub order_id: [u8; 16],
    /// Names the swap from here on; random, the buyer's.
    pub take_id: [u8; 16],
    pub mds: u64,
    pub wei: u128,
    /// The buyer's one-time key: the only key that can claim the lock.
    pub claim_pk: [u8; 32],
    /// The account the buyer's ETH lock comes from and refunds to.
    pub buyer_evm: [u8; 20],
}

impl Take {
    pub const LEN: usize = 1 + 16 + 16 + 8 + 16 + 32 + 20;

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::LEN);
        out.push(VERSION);
        out.extend_from_slice(&self.order_id);
        out.extend_from_slice(&self.take_id);
        out.extend_from_slice(&self.mds.to_le_bytes());
        out.extend_from_slice(&self.wei.to_le_bytes());
        out.extend_from_slice(&self.claim_pk);
        out.extend_from_slice(&self.buyer_evm);
        out
    }

    pub fn decode(b: &[u8]) -> Option<Self> {
        if b.len() != Self::LEN || b[0] != VERSION {
            return None;
        }
        let mut r = Reader { b, o: 1 };
        Some(Self {
            order_id: r.arr()?,
            take_id: r.arr()?,
            mds: r.u64()?,
            wei: r.u128()?,
            claim_pk: r.arr()?,
            buyer_evm: r.arr()?,
        })
    }
}

/// Why a seller turned a take down.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decline {
    /// No such order, or it has ended.
    Gone,
    /// More than is for sale, or less than the order's minimum.
    Amount,
    /// Less ETH than the order's price.
    Price,
    /// The seller already has as many takes waiting on ETH as it allows.
    Busy,
    /// The seller's wallet cannot lock right now.
    Unable,
}

impl Decline {
    fn byte(self) -> u8 {
        match self {
            Decline::Gone => 0,
            Decline::Amount => 1,
            Decline::Price => 2,
            Decline::Busy => 3,
            Decline::Unable => 4,
        }
    }
    fn of(b: u8) -> Option<Self> {
        Some(match b {
            0 => Decline::Gone,
            1 => Decline::Amount,
            2 => Decline::Price,
            3 => Decline::Busy,
            4 => Decline::Unable,
            _ => return None,
        })
    }
    pub fn says(self) -> &'static str {
        match self {
            Decline::Gone => "the order has ended",
            Decline::Amount => "that amount is not on offer",
            Decline::Price => "the price has changed",
            Decline::Busy => "the seller is busy with other buyers; try again shortly",
            Decline::Unable => "the seller's wallet cannot lock right now",
        }
    }
}

/// The seller's answer to a take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Answer {
    /// The seller has locked (or is locking) the MDS under these terms; the
    /// coins follow from them (`eth_trade::lock_coins`).
    Lock { take_id: [u8; 16], secret_hash: [u8; 32], timeout: u64, refund_pk: [u8; 32] },
    Decline { take_id: [u8; 16], reason: Decline },
}

impl Answer {
    pub fn take_id(&self) -> [u8; 16] {
        match self {
            Answer::Lock { take_id, .. } | Answer::Decline { take_id, .. } => *take_id,
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = vec![VERSION];
        match self {
            Answer::Lock { take_id, secret_hash, timeout, refund_pk } => {
                out.extend_from_slice(take_id);
                out.push(1);
                out.extend_from_slice(secret_hash);
                out.extend_from_slice(&timeout.to_le_bytes());
                out.extend_from_slice(refund_pk);
            }
            Answer::Decline { take_id, reason } => {
                out.extend_from_slice(take_id);
                out.push(0);
                out.push(reason.byte());
            }
        }
        out
    }

    pub fn decode(b: &[u8]) -> Option<Self> {
        if b.first() != Some(&VERSION) {
            return None;
        }
        let mut r = Reader { b, o: 1 };
        let take_id = r.arr()?;
        match r.u8()? {
            1 if b.len() == 1 + 16 + 1 + 32 + 8 + 32 => {
                Some(Answer::Lock { take_id, secret_hash: r.arr()?, timeout: r.u64()?, refund_pk: r.arr()? })
            }
            0 if b.len() == 1 + 16 + 1 + 1 => Some(Answer::Decline { take_id, reason: Decline::of(r.u8()?)? }),
            _ => None,
        }
    }
}

/// What a name claim is about.
pub const CLAIM_ORDER: u8 = 0;
pub const CLAIM_TAKE: u8 = 1;

/// "This order (or take) is `name`'s", signed by the Base account the order
/// pays (or the take locks from). Display only: the account stays the source
/// of truth, and a wallet shows the name only when the name's `evm` record is
/// that same account, so both sides say it.
///
/// ```text
/// claim = VERSION | subject | id[16] | name_len | name | sig[65]
/// sig   = EIP-191 signature over claim_message(subject, id, name)
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NameClaim {
    pub subject: u8,
    pub id: [u8; 16],
    /// Canonical, without `.mds`.
    pub name: String,
    pub sig: [u8; 65],
}

/// What a claim's account signs.
pub fn claim_message(subject: u8, id: &[u8; 16], name: &str) -> Vec<u8> {
    let what = if subject == CLAIM_TAKE { "take" } else { "order" };
    format!("midstate name claim v1\n{what} {}\n{name}.mds", hex::encode(id)).into_bytes()
}

impl NameClaim {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = vec![VERSION, self.subject];
        out.extend_from_slice(&self.id);
        out.push(self.name.len() as u8);
        out.extend_from_slice(self.name.as_bytes());
        out.extend_from_slice(&self.sig);
        out
    }

    pub fn decode(b: &[u8]) -> Option<Self> {
        if b.len() < 19 || b[0] != VERSION || b[1] > CLAIM_TAKE {
            return None;
        }
        let n = b[18] as usize;
        if b.len() != 19 + n + 65 || n == 0 || n > midstate::core::names::MAX_NAME_LEN {
            return None;
        }
        let name = std::str::from_utf8(&b[19..19 + n]).ok()?.to_string();
        // Only canonical names: the one form a name is shown and signed in.
        if midstate::core::names::normalize(&name).ok()? != name {
            return None;
        }
        Some(Self { subject: b[1], id: b[2..18].try_into().ok()?, name, sig: b[19 + n..].try_into().ok()? })
    }

    /// The account that signed it.
    pub fn signer(&self) -> Option<[u8; 20]> {
        crate::evm::recover_message(&claim_message(self.subject, &self.id, &self.name), &self.sig)
    }
}

/// Any message of this protocol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Frame {
    Advert(Advert),
    Take(Take),
    Answer(Answer),
    Claim(NameClaim),
}

/// One chat message to send: `(words, attachments)`.
pub type Outgoing = (Vec<u8>, Vec<ChatAttachment>);

fn outgoing(kind: u8, body: Vec<u8>) -> Outgoing {
    (vec![WORD_DEX, WORD_MARK, kind], vec![ChatAttachment::Signature(body)])
}

impl Frame {
    pub fn chat(&self) -> Outgoing {
        match self {
            Frame::Advert(a) => outgoing(KIND_SELL, a.encode()),
            Frame::Take(t) => outgoing(KIND_TAKE, t.encode()),
            Frame::Answer(a) => outgoing(KIND_TRADE, a.encode()),
            Frame::Claim(c) => outgoing(KIND_WHO, c.encode()),
        }
    }
}

/// Parses a chat message, or `None` if it is not one of ours.
pub fn parse(m: &ChatMessage) -> Option<Frame> {
    if m.words.len() != 3 || m.words[0] != WORD_DEX || m.words[1] != WORD_MARK || m.attachments.len() != 1 {
        return None;
    }
    let ChatAttachment::Signature(b) = &m.attachments[0] else { return None };
    match m.words[2] {
        KIND_SELL => Advert::decode(b).map(Frame::Advert),
        KIND_TAKE => Take::decode(b).map(Frame::Take),
        KIND_TRADE => Answer::decode(b).map(Frame::Answer),
        KIND_WHO => NameClaim::decode(b).map(Frame::Claim),
        _ => None,
    }
}

struct Reader<'a> {
    b: &'a [u8],
    o: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let s = self.b.get(self.o..self.o + n)?;
        self.o += n;
        Some(s)
    }
    fn arr<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.take(N)?.try_into().ok()
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.arr()?))
    }
    fn u128(&mut self) -> Option<u128> {
        Some(u128::from_le_bytes(self.arr()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat(o: &Outgoing) -> ChatMessage {
        ChatMessage { sender: "p".into(), timestamp: 0, nonce: 0, reply_to: None, words: o.0.clone(), attachments: o.1.clone() }
    }

    #[test]
    fn every_message_round_trips_through_the_bus() {
        let frames = [
            Frame::Advert(Advert {
                order_id: [1; 16],
                seller_evm: [2; 20],
                mds: 5 << 30,
                min: 1 << 30,
                wei_per_gmds: 4_000_000_000_000_000,
                expires: 400_000,
            }),
            Frame::Take(Take { order_id: [1; 16], take_id: [3; 16], mds: 1 << 30, wei: 4_000_000_000_000_000, claim_pk: [4; 32], buyer_evm: [5; 20] }),
            Frame::Answer(Answer::Lock { take_id: [3; 16], secret_hash: [6; 32], timeout: 400_187, refund_pk: [7; 32] }),
            Frame::Answer(Answer::Decline { take_id: [3; 16], reason: Decline::Busy }),
        ];
        for f in frames {
            assert_eq!(parse(&chat(&f.chat())), Some(f));
        }
    }

    #[test]
    fn a_name_claim_round_trips_and_names_its_signer() {
        let key = crate::evm::EvmKey::from_secret(&[5; 32]).unwrap();
        let msg = claim_message(CLAIM_ORDER, &[1; 16], "nick");
        let c = NameClaim { subject: CLAIM_ORDER, id: [1; 16], name: "nick".into(), sig: key.sign_message(&msg).unwrap() };
        let f = Frame::Claim(c.clone());
        assert_eq!(parse(&chat(&f.chat())), Some(f));
        assert_eq!(c.signer(), Some(key.address));
        // Moved to another order, another name or a take: another signer.
        assert_ne!(NameClaim { id: [2; 16], ..c.clone() }.signer(), Some(key.address));
        assert_ne!(NameClaim { name: "nicky".into(), ..c.clone() }.signer(), Some(key.address));
        assert_ne!(NameClaim { subject: CLAIM_TAKE, ..c.clone() }.signer(), Some(key.address));
        // An old wallet reads the claim as nothing it knows (its own kind).
        let (w, _) = Frame::Claim(c).chat();
        assert_eq!(w[2], KIND_WHO);
    }

    #[test]
    fn eip191_signatures_match_ethereum_tools() {
        // Recovering what we sign, and refusing a mangled signature.
        let key = crate::evm::EvmKey::from_secret(&[9; 32]).unwrap();
        let sig = key.sign_message(b"hello").unwrap();
        assert_eq!(crate::evm::recover_message(b"hello", &sig), Some(key.address));
        let mut bad = sig;
        bad[64] = 30;
        assert_eq!(crate::evm::recover_message(b"hello", &bad), None);
        // The digest is personal_sign's: keccak256("\x19Ethereum Signed Message:\n5hello").
        assert_eq!(
            hex::encode(crate::evm::eip191_digest(b"hello")),
            "50b2c43fd39106bafbba0da34fc430e1f91e3c96ea2acee2bc34119f92b37750"
        );
    }

    #[test]
    fn other_protocols_and_truncations_are_not_ours() {
        let a = Frame::Advert(Advert { order_id: [1; 16], seller_evm: [2; 20], mds: 1, min: 1, wei_per_gmds: 1, expires: 1 });
        let mut o = a.chat();
        o.0[0] = 33; // the MDW market's "swap"
        assert_eq!(parse(&chat(&o)), None);
        let mut o = a.chat();
        if let ChatAttachment::Signature(b) = &mut o.1[0] {
            b.pop();
        }
        assert_eq!(parse(&chat(&o)), None);
    }
}
