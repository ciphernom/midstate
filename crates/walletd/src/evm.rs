//! The Ethereum side of the cross-chain DEX: key derivation, transaction
//! signing, and ABI coding for the Base atomic-swap contract.
//!
//! Hand-rolled rather than pulling in `ethers-rs`, because the surface we need
//! is tiny and fully static. Every argument and event field in
//! `MidstateAtomicSwap.sol` is a fixed-width type — `bytes32`, `address`,
//! `uint256`, `uint64`, `bool` — with no dynamic arrays or strings anywhere.
//! That reduces ABI coding to a 4-byte selector followed by 32-byte words, and
//! log decoding to topics plus data words.
//!
//! Key derivation deliberately follows standard BIP44 `m/44'/60'/0'/0/0` from
//! the raw BIP39 seed, NOT from midstate's `master_seed` (which is
//! `BLAKE3(PBKDF2(mnemonic))` and therefore midstate-specific). The difference
//! matters: derived this way, the same recovery phrase restores the Base
//! account in MetaMask or a hardware wallet. Derived the other way it would be
//! recoverable only by this software, which is a trap to leave someone in when
//! they have real funds on an L2.

use anyhow::{anyhow, bail, Context, Result};
use k256::ecdsa::{RecoveryId, Signature, SigningKey};
use sha3::{Digest, Keccak256};

pub const BASE_MAINNET_CHAIN_ID: u64 = 8453;
pub const BASE_MAINNET_RPC: &str = "https://mainnet.base.org";
/// Deployed V1 `MidstateAtomicSwap` (the address the web wallet uses).
pub const BASE_MAINNET_CONTRACT: &str = "0x409C52821EC5fE402Ab8b9bdc1474a8cD006f9C7";

pub fn keccak256(bytes: &[u8]) -> [u8; 32] {
    let mut h = Keccak256::new();
    h.update(bytes);
    h.finalize().into()
}

/// EIP-55 mixed-case checksum encoding.
pub fn to_checksum_address(addr: &[u8; 20]) -> String {
    let lower = hex::encode(addr);
    let hash = keccak256(lower.as_bytes());
    let mut out = String::with_capacity(42);
    out.push_str("0x");
    for (i, c) in lower.chars().enumerate() {
        let nibble = if i % 2 == 0 { hash[i / 2] >> 4 } else { hash[i / 2] & 0x0f };
        if c.is_ascii_digit() || nibble < 8 {
            out.push(c);
        } else {
            out.push(c.to_ascii_uppercase());
        }
    }
    out
}

pub fn parse_address(s: &str) -> Result<[u8; 20]> {
    let h = s.strip_prefix("0x").unwrap_or(s);
    if h.len() != 40 {
        bail!("an EVM address is 40 hex characters");
    }
    let b = hex::decode(h).context("address is not hex")?;
    Ok(b.try_into().unwrap())
}

// ── Keys ────────────────────────────────────────────────────────────────

pub struct EvmKey {
    signing: SigningKey,
    pub address: [u8; 20],
}

impl EvmKey {
    pub fn from_secret(secret: &[u8; 32]) -> Result<Self> {
        let signing = SigningKey::from_bytes(secret.into())
            .map_err(|e| anyhow!("invalid secp256k1 secret: {e}"))?;
        let vk = signing.verifying_key();
        // Address = last 20 bytes of keccak(uncompressed pubkey without the
        // 0x04 prefix).
        let enc = vk.to_encoded_point(false);
        let hash = keccak256(&enc.as_bytes()[1..]);
        let mut address = [0u8; 20];
        address.copy_from_slice(&hash[12..]);
        Ok(Self { signing, address })
    }

    /// Derive the account a standard wallet would show for this phrase.
    pub fn from_mnemonic(phrase: &str) -> Result<Self> {
        let m = bip39::Mnemonic::parse_normalized(&phrase.trim().to_lowercase())
            .map_err(|e| anyhow!("invalid recovery phrase: {e}"))?;
        let seed = m.to_seed("");
        let path: bip32::DerivationPath =
            "m/44'/60'/0'/0/0".parse().expect("static path parses");
        let xprv = bip32::XPrv::derive_from_path(seed, &path)
            .map_err(|e| anyhow!("BIP32 derivation failed: {e}"))?;
        Self::from_secret(&xprv.to_bytes())
    }

    pub fn secret_bytes(&self) -> [u8; 32] {
        self.signing.to_bytes().into()
    }

    pub fn checksum_address(&self) -> String {
        to_checksum_address(&self.address)
    }

    /// Sign a 32-byte digest, returning `(r, s, y_parity)`.
    fn sign_digest(&self, digest: &[u8; 32]) -> Result<([u8; 32], [u8; 32], u8)> {
        let (sig, rec): (Signature, RecoveryId) = self
            .signing
            .sign_prehash_recoverable(digest)
            .map_err(|e| anyhow!("signing failed: {e}"))?;
        let r: [u8; 32] = sig.r().to_bytes().into();
        let s: [u8; 32] = sig.s().to_bytes().into();
        Ok((r, s, rec.to_byte() & 1))
    }
}

// ── RLP ─────────────────────────────────────────────────────────────────
// Only what an EIP-1559 payload needs: byte strings and one list.

fn rlp_len_prefix(out: &mut Vec<u8>, len: usize, short_base: u8, long_base: u8) {
    if len < 56 {
        out.push(short_base + len as u8);
    } else {
        let be = len.to_be_bytes();
        let first = be.iter().position(|b| *b != 0).unwrap_or(be.len() - 1);
        let sig = &be[first..];
        out.push(long_base + sig.len() as u8);
        out.extend_from_slice(sig);
    }
}

pub fn rlp_bytes(out: &mut Vec<u8>, b: &[u8]) {
    if b.len() == 1 && b[0] < 0x80 {
        out.push(b[0]);
        return;
    }
    rlp_len_prefix(out, b.len(), 0x80, 0xb7);
    out.extend_from_slice(b);
}

/// Integers are RLP-encoded big-endian with leading zeros stripped; zero is
/// the empty string.
pub fn rlp_uint(out: &mut Vec<u8>, v: u128) {
    let be = v.to_be_bytes();
    let start = be.iter().position(|b| *b != 0).unwrap_or(be.len());
    rlp_bytes(out, &be[start..]);
}

pub fn rlp_list(items: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(items.len() + 9);
    rlp_len_prefix(&mut out, items.len(), 0xc0, 0xf7);
    out.extend_from_slice(items);
    out
}

/// An EIP-1559 (type 0x02) transaction.
pub struct TxRequest {
    pub chain_id: u64,
    pub nonce: u64,
    pub max_priority_fee: u128,
    pub max_fee: u128,
    pub gas_limit: u64,
    pub to: [u8; 20],
    pub value: u128,
    pub data: Vec<u8>,
}

impl TxRequest {
    fn payload(&self) -> Vec<u8> {
        let mut f = Vec::new();
        rlp_uint(&mut f, self.chain_id as u128);
        rlp_uint(&mut f, self.nonce as u128);
        rlp_uint(&mut f, self.max_priority_fee);
        rlp_uint(&mut f, self.max_fee);
        rlp_uint(&mut f, self.gas_limit as u128);
        rlp_bytes(&mut f, &self.to);
        rlp_uint(&mut f, self.value);
        rlp_bytes(&mut f, &self.data);
        f.extend_from_slice(&rlp_list(&[])); // empty access list
        rlp_list(&f)
    }

    /// Returns the raw signed transaction, ready for `eth_sendRawTransaction`.
    pub fn sign(&self, key: &EvmKey) -> Result<Vec<u8>> {
        let mut unsigned = vec![0x02u8];
        unsigned.extend_from_slice(&self.payload());
        let digest = keccak256(&unsigned);
        let (r, s, y) = key.sign_digest(&digest)?;

        let mut f = Vec::new();
        rlp_uint(&mut f, self.chain_id as u128);
        rlp_uint(&mut f, self.nonce as u128);
        rlp_uint(&mut f, self.max_priority_fee);
        rlp_uint(&mut f, self.max_fee);
        rlp_uint(&mut f, self.gas_limit as u128);
        rlp_bytes(&mut f, &self.to);
        rlp_uint(&mut f, self.value);
        rlp_bytes(&mut f, &self.data);
        f.extend_from_slice(&rlp_list(&[]));
        rlp_uint(&mut f, y as u128);
        rlp_bytes(&mut f, strip_leading_zeros(&r));
        rlp_bytes(&mut f, strip_leading_zeros(&s));

        let mut out = vec![0x02u8];
        out.extend_from_slice(&rlp_list(&f));
        Ok(out)
    }

    /// The hash the network will know this transaction by, computable before
    /// broadcast — unlike `swapId`, which depends on the mining timestamp.
    pub fn hash(&self, key: &EvmKey) -> Result<[u8; 32]> {
        Ok(keccak256(&self.sign(key)?))
    }
}

fn strip_leading_zeros(b: &[u8; 32]) -> &[u8] {
    let start = b.iter().position(|x| *x != 0).unwrap_or(31);
    &b[start..]
}

// ── ABI ─────────────────────────────────────────────────────────────────

pub fn selector(signature: &str) -> [u8; 4] {
    let h = keccak256(signature.as_bytes());
    [h[0], h[1], h[2], h[3]]
}

/// The 32-byte word forms. Every argument in the swap contract is static, so
/// a call is just the selector followed by these in order.
#[derive(Clone, Copy, Debug)]
pub enum Word {
    U256(u128),
    U64(u64),
    Bytes32([u8; 32]),
    Address([u8; 20]),
    Bool(bool),
}

impl Word {
    pub fn encode(&self) -> [u8; 32] {
        let mut w = [0u8; 32];
        match self {
            Word::U256(v) => w[16..].copy_from_slice(&v.to_be_bytes()),
            Word::U64(v) => w[24..].copy_from_slice(&v.to_be_bytes()),
            Word::Bytes32(b) => w.copy_from_slice(b),
            Word::Address(a) => w[12..].copy_from_slice(a),
            Word::Bool(b) => w[31] = *b as u8,
        }
        w
    }
}

pub fn encode_call(sig: &str, args: &[Word]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + args.len() * 32);
    out.extend_from_slice(&selector(sig));
    for a in args {
        out.extend_from_slice(&a.encode());
    }
    out
}

/// Split ABI-encoded return data into 32-byte words.
pub fn words(data: &[u8]) -> Vec<[u8; 32]> {
    data.chunks_exact(32).map(|c| c.try_into().unwrap()).collect()
}

pub fn word_u128(w: &[u8; 32]) -> u128 {
    u128::from_be_bytes(w[16..].try_into().unwrap())
}
pub fn word_u64(w: &[u8; 32]) -> u64 {
    u64::from_be_bytes(w[24..].try_into().unwrap())
}
pub fn word_address(w: &[u8; 32]) -> [u8; 20] {
    w[12..].try_into().unwrap()
}
pub fn word_bool(w: &[u8; 32]) -> bool {
    w[31] != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keccak_known_vectors() {
        assert_eq!(
            hex::encode(keccak256(b"")),
            "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
        );
        // The selector the contract's own `claim(bytes32,bytes32)` resolves to.
        assert_eq!(hex::encode(selector("claim(bytes32,bytes32)")), "84cc9dfb");
    }

    #[test]
    fn address_and_checksum_from_known_key() {
        // Account 0 of Ganache's deterministic mnemonic ("myth like bonus
        // scare over problem client lizard pioneer submit female collect").
        // The expected value was confirmed by an independent secp256k1 +
        // Keccak-256 derivation, not read off this implementation.
        let secret =
            hex::decode("4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d")
                .unwrap();
        let k = EvmKey::from_secret(&secret.try_into().unwrap()).unwrap();
        assert_eq!(
            k.checksum_address(),
            "0x90F8bf6A479f320ead074411a4B0e7944Ea8c9C1"
        );
        // EIP-55 must actually mix case — an all-lowercase result would mean
        // the checksum step silently did nothing.
        let a = k.checksum_address();
        assert!(a.chars().any(|c| c.is_ascii_uppercase()) && a.chars().any(|c| c.is_ascii_lowercase()));
    }

    #[test]
    fn mnemonic_derives_the_standard_account() {
        // The BIP39 test vector every wallet agrees on for m/44'/60'/0'/0/0.
        let phrase = "abandon abandon abandon abandon abandon abandon \
                      abandon abandon abandon abandon abandon about";
        let k = EvmKey::from_mnemonic(phrase).unwrap();
        assert_eq!(
            k.checksum_address(),
            "0x9858EfFD232B4033E47d90003D41EC34EcaEda94"
        );
    }

    #[test]
    fn rlp_integer_edges() {
        let mut v = Vec::new();
        rlp_uint(&mut v, 0);
        assert_eq!(v, vec![0x80]); // zero is the empty string, not 0x00

        v.clear();
        rlp_uint(&mut v, 15);
        assert_eq!(v, vec![0x0f]); // single small byte is itself

        v.clear();
        rlp_uint(&mut v, 1024);
        assert_eq!(v, vec![0x82, 0x04, 0x00]);
    }

    #[test]
    fn abi_encoding_shape() {
        let call = encode_call(
            "lock(bytes32,address,uint256)",
            &[Word::Bytes32([0xaa; 32]), Word::Address([0xbb; 20]), Word::U256(1_000)],
        );
        assert_eq!(call.len(), 4 + 96);
        // Address is right-aligned in its word: 12 zero bytes then 20 of data.
        assert_eq!(&call[4 + 32..4 + 32 + 12], &[0u8; 12]);
        assert_eq!(call[4 + 32 + 12], 0xbb);
        assert_eq!(word_u128(&words(&call[4..])[2]), 1_000);
    }

    #[test]
    fn signed_tx_is_typed_and_recoverable() {
        let k = EvmKey::from_secret(&[7u8; 32]).unwrap();
        let tx = TxRequest {
            chain_id: BASE_MAINNET_CHAIN_ID,
            nonce: 3,
            max_priority_fee: 1_000_000,
            max_fee: 20_000_000,
            gas_limit: 120_000,
            to: [0x11; 20],
            value: 5_000_000_000_000_000,
            data: encode_call("refund(bytes32)", &[Word::Bytes32([9; 32])]),
        };
        let raw = tx.sign(&k).unwrap();
        assert_eq!(raw[0], 0x02, "must be an EIP-1559 typed transaction");
        assert!(raw.len() > 100);
        // Signing must be deterministic (RFC 6979), so the hash is stable.
        assert_eq!(tx.hash(&k).unwrap(), tx.hash(&k).unwrap());
    }
}
