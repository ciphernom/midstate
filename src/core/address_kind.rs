//! What kind of address a script makes: a key, a multisig, a vault, a plan.
//!
//! An address is the hash of its script, so the address alone says nothing.
//! Given the script (a name's `script` records publish it, and a wallet knows
//! its own), [`classify`] reads its shape. Recognised shapes are the ones this
//! codebase compiles; anything else is a covenant or a custom script.
//! Classification is a label for people, never a check of what a script
//! allows: the script itself is what the chain runs.

use super::script::{
    OP_ADD, OP_CHECKSIG, OP_CHECKSIGVERIFY, OP_CHECKTIMEVERIFY, OP_DIV, OP_DROP, OP_ELSE, OP_GREATER_OR_EQUAL, OP_HASH,
    OP_IF, OP_OUTPUT_ADDRESS, OP_PUSH_DATA, OP_READ_INPUT_STATE, OP_READ_OUTPUT_STATE, OP_SUM_INPUT_VALUE, OP_SUM_TO_ADDR,
    OP_SWAP,
};
use serde::{Deserialize, Serialize};

/// A script's shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AddressKind {
    /// One key signs (`PUSH pk CHECKSIGVERIFY PUSH 1`).
    Key,
    /// One key, at an address tagged apart from others (a plan's payout).
    TaggedKey,
    /// `m` of `n` keys sign.
    Multisig { m: u8, n: u8 },
    /// Keys sign only from a height on: a vault.
    Vault,
    /// A midwimble mining bond: its owner's key, from a height on.
    MiningBond,
    /// Paid out against a hash's preimage, or refunded after a height.
    HashLock,
    /// An inheritance plan: its owner's key any time, its heirs after a deadline.
    Plan { heirs: u8 },
    /// A tranche of a payment stream.
    Stream,
    /// The MSN-1 name registry's thread.
    NameRegistry,
    /// Another script that restricts where its coins go.
    Covenant,
    /// Anything else.
    Script,
}

impl AddressKind {
    /// For people: "2-of-3 multisig", "vault", …
    pub fn label(&self) -> String {
        match self {
            AddressKind::Key => "single key".into(),
            AddressKind::TaggedKey => "single key (tagged)".into(),
            AddressKind::Multisig { m, n } => format!("{m}-of-{n} multisig"),
            AddressKind::Vault => "vault (time-locked)".into(),
            AddressKind::MiningBond => "mining bond".into(),
            AddressKind::HashLock => "hash lock (HTLC)".into(),
            AddressKind::Plan { heirs } => format!("inheritance plan ({heirs} heir{})", if *heirs == 1 { "" } else { "s" }),
            AddressKind::Stream => "payment stream".into(),
            AddressKind::NameRegistry => "name registry".into(),
            AddressKind::Covenant => "covenant".into(),
            AddressKind::Script => "custom script".into(),
        }
    }

    /// A shared address: more than one key decides, or a covenant does.
    pub fn is_shared(&self) -> bool {
        !matches!(self, AddressKind::Key | AddressKind::TaggedKey)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Tok<'a> {
    Push(&'a [u8]),
    Op(u8),
}

/// The script's instructions; `None` if a push runs off its end.
fn tokens(bc: &[u8]) -> Option<Vec<Tok<'_>>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < bc.len() {
        let op = bc[i];
        i += 1;
        if op == OP_PUSH_DATA {
            let len = u16::from_le_bytes([*bc.get(i)?, *bc.get(i + 1)?]) as usize;
            i += 2;
            out.push(Tok::Push(bc.get(i..i + len)?));
            i += len;
        } else {
            out.push(Tok::Op(op));
        }
    }
    Some(out)
}

fn is_key(t: &Tok) -> bool {
    matches!(t, Tok::Push(p) if p.len() == 32)
}

/// A small integer as `push_int` writes it (minimal little-endian).
fn small(t: &Tok) -> Option<u64> {
    match t {
        Tok::Push(p) if !p.is_empty() && p.len() <= 8 => {
            let mut b = [0u8; 8];
            b[..p.len()].copy_from_slice(p);
            Some(u64::from_le_bytes(b))
        }
        _ => None,
    }
}

/// `PUSH pk CHECKSIG (SWAP PUSH pk CHECKSIG ADD)* PUSH m GREATER_OR_EQUAL`,
/// as `compile_multisig_2of2` and `compile_multisig_2of3` write it.
fn multisig(t: &[Tok]) -> Option<AddressKind> {
    if t.len() < 4 || !is_key(&t[0]) || t[1] != Tok::Op(OP_CHECKSIG) {
        return None;
    }
    let mut n = 1u64;
    let mut i = 2;
    while i + 4 <= t.len() - 2 && t[i] == Tok::Op(OP_SWAP) {
        if !is_key(&t[i + 1]) || t[i + 2] != Tok::Op(OP_CHECKSIG) || t[i + 3] != Tok::Op(OP_ADD) {
            return None;
        }
        n += 1;
        i += 4;
    }
    let m = small(t.get(i)?)?;
    if i + 2 != t.len() || t[i + 1] != Tok::Op(OP_GREATER_OR_EQUAL) || n < 2 || m == 0 || m > n || n > 255 {
        return None;
    }
    Some(AddressKind::Multisig { m: m as u8, n: n as u8 })
}

/// The shape of `bytecode`.
pub fn classify(bytecode: &[u8]) -> AddressKind {
    if bytecode == super::names::covenant() {
        return AddressKind::NameRegistry;
    }
    let Some(t) = tokens(bytecode) else { return AddressKind::Script };
    let op = |o: u8| Tok::Op(o);
    match t.as_slice() {
        [k, c, one] if is_key(k) && *c == op(OP_CHECKSIGVERIFY) && small(one) == Some(1) => return AddressKind::Key,
        [tag, d, k, c] if is_key(tag) && *d == op(OP_DROP) && is_key(k) && *c == op(OP_CHECKSIG) => return AddressKind::TaggedKey,
        [m, d, h, ctv, k, c, one]
            if is_key(m)
                && *d == op(OP_DROP)
                && small(h).is_some()
                && *ctv == op(OP_CHECKTIMEVERIFY)
                && is_key(k)
                && *c == op(OP_CHECKSIGVERIFY)
                && small(one) == Some(1) =>
        {
            return AddressKind::MiningBond
        }
        _ => {}
    }
    if let Some(k) = multisig(&t) {
        return k;
    }
    let has = |o: u8| t.contains(&op(o));
    let count = |o: u8| t.iter().filter(|x| **x == op(o)).count();
    let signs = has(OP_CHECKSIG) || has(OP_CHECKSIGVERIFY);
    if has(OP_SUM_INPUT_VALUE) && has(OP_DIV) && has(OP_SUM_TO_ADDR) && has(OP_CHECKTIMEVERIFY) {
        // A plan opens with its owner's key; so does a cancellable stream, but
        // a stream pays one address and a plan one per heir, each checked.
        let owner_first = t.len() > 3 && t[0] == op(OP_IF) && is_key(&t[1]) && t[2] == op(OP_CHECKSIG) && t[3] == op(OP_ELSE);
        let payouts = count(OP_SUM_TO_ADDR);
        if owner_first && t.iter().any(|x| *x == op(super::script::OP_VERIFY)) {
            return AddressKind::Plan { heirs: payouts.min(255) as u8 };
        }
        return AddressKind::Stream;
    }
    if has(OP_SUM_TO_ADDR) || has(OP_OUTPUT_ADDRESS) || has(OP_READ_INPUT_STATE) || has(OP_READ_OUTPUT_STATE) {
        return AddressKind::Covenant;
    }
    if signs && has(OP_HASH) && has(OP_IF) {
        return AddressKind::HashLock;
    }
    if signs && has(OP_CHECKTIMEVERIFY) && !has(OP_HASH) {
        return AddressKind::Vault;
    }
    AddressKind::Script
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::script::{self, push_data, push_int};

    #[test]
    fn the_scripts_this_codebase_compiles_are_recognised() {
        let (a, b, c) = ([1u8; 32], [2u8; 32], [3u8; 32]);
        assert_eq!(classify(&script::compile_p2pk(&a)), AddressKind::Key);
        assert_eq!(classify(&script::compile_multisig_2of2(&a, &b)), AddressKind::Multisig { m: 2, n: 2 });
        assert_eq!(classify(&script::compile_multisig_2of3(&a, &b, &c)), AddressKind::Multisig { m: 2, n: 3 });
        assert_eq!(classify(&script::compile_mining_bond(&a, 500_000, &b)), AddressKind::MiningBond);
        assert_eq!(classify(&crate::core::plans::payout_script(&a, &b)), AddressKind::TaggedKey);
        assert_eq!(classify(crate::core::names::covenant()), AddressKind::NameRegistry);
        assert_eq!(classify(&script::compile_htlc(&[9; 32], &a, 1000, &b)), AddressKind::HashLock);

        let heirs = vec![
            crate::core::plans::Heir { pk: b, share: 40_000 },
            crate::core::plans::Heir { pk: c, share: (crate::core::plans::SHARE_DENOM - 1 - 40_000) as u32 },
        ];
        let plan = crate::core::plans::InheritancePlan { owner_pk: a, heirs, start: 100, period: 52_560, secret: [7; 32] };
        assert_eq!(classify(&plan.script(1)), AddressKind::Plan { heirs: 2 });
    }

    #[test]
    fn a_time_locked_key_is_a_vault_and_garbage_is_a_script() {
        // Hot key after a delay, or the cold key at once.
        let mut v = vec![OP_IF];
        push_int(&mut v, 400_000);
        v.push(OP_CHECKTIMEVERIFY);
        push_data(&mut v, &[1; 32]);
        v.push(OP_CHECKSIG);
        v.push(OP_ELSE);
        push_data(&mut v, &[2; 32]);
        v.push(OP_CHECKSIG);
        v.push(crate::core::script::OP_ENDIF);
        assert_eq!(classify(&v), AddressKind::Vault);
        assert_eq!(classify(&[OP_PUSH_DATA, 9]), AddressKind::Script, "a push off the end");
        assert_eq!(classify(&[]), AddressKind::Script);
        // Three keys but one signature's worth: not the multisig shape.
        let mut odd = script::compile_multisig_2of3(&[1; 32], &[2; 32], &[3; 32]);
        odd.pop();
        assert_ne!(classify(&odd), AddressKind::Multisig { m: 2, n: 3 });
    }

    #[test]
    fn labels_read_naturally() {
        assert_eq!(AddressKind::Multisig { m: 2, n: 3 }.label(), "2-of-3 multisig");
        assert_eq!(AddressKind::Plan { heirs: 1 }.label(), "inheritance plan (1 heir)");
        assert!(AddressKind::Vault.is_shared() && !AddressKind::Key.is_shared());
        assert_eq!(serde_json::to_value(AddressKind::Multisig { m: 2, n: 3 }).unwrap(), serde_json::json!({ "kind": "multisig", "m": 2, "n": 3 }));
    }
}
