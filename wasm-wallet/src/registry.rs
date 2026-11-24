//! Names (MSN-1) for the web wallet: registry spends planned from what a node
//! proves, and the control keys that hold names.
//!
//! A node answers `GET /name/:name/proof` with one leaf of the registry and
//! its Merkle path to the current root: the name's own leaf, or (for a name
//! never registered) the leaf just before it. That path is all a change of
//! the name needs (`names::change`). A registration also needs an empty
//! slot's path. It is taken from the same proof: the largest subtree beside
//! the path that the proof shows to be empty holds nothing, so every sibling
//! below it is the empty subtree's hash, and the ones above are on the path
//! already (`names::register` reads nothing else). The covenant takes any
//! empty slot; nodes' followers fill the lowest free ones for their own
//! wallets, half the registry away from these. Nothing more is asked of a
//! node than it already serves.
//!
//! The plan is checked here before anything is signed: the proof must reach
//! the root, and the root must be the commitment of the thread coin the node
//! names. The page builds and signs the transaction with `prepare_script_spend`
//! and `build_script_reveal`, as the IDE's contract spends are built.
//!
//! A name's control key (its leaf's `owner`) signs changes and nothing else.
//! The page holds names as the desktop app does, with the wallet's own HD MSS
//! keys (`get_mss_address`), so the same seed manages the same names in
//! either app; the owner's signature goes in with `sign_mss_hex`.

use midstate::core::names::{self, By, Leaf, Mem, Set};
use midstate::core::types::{decompose_value, hash_concat, OutputData};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// The largest records a name carries (MSN-1).
const MAX_RECORDS: usize = 1_536;

#[derive(Deserialize)]
struct LeafJson {
    key: String,
    next: String,
    expires: u64,
    tier: u8,
    owner: String,
    data: String,
    deal: String,
}

#[derive(Deserialize)]
struct ThreadJson {
    coin: String,
    salt: String,
}

/// What the page asks for: the node's proof as it came (its slot, leaf,
/// path, root and thread), how the spend is allowed, and what it sets.
#[derive(Deserialize)]
struct PlanReq {
    name: String,
    /// "register" (a name never registered), "owner" (its control key signs)
    /// or "lapse" (it lapsed, or was never claimed: a premium name).
    by: String,
    slot: u64,
    leaf: LeafJson,
    proof: String,
    root: String,
    /// The leaf's terms as they are (hex), for a change.
    #[serde(default)]
    terms: String,
    thread: ThreadJson,
    owner: String,
    records: String,
    #[serde(default = "no_deal")]
    new_terms: String,
    years: u64,
    expires: u64,
    /// The control key's height, for an owner's change: its signature's size
    /// is set aside in the witness, so the fee covers it.
    #[serde(default)]
    signer_height: u32,
}

fn no_deal() -> String {
    hex::encode(names::Deal::None.terms())
}

/// The spend, for `prepare_script_spend`: the registry input's witness
/// (`sign_at`: the item the control key's signature goes in), output 0 (the
/// thread's next coin) and the burns.
#[derive(Serialize, Debug)]
struct PlanOut {
    items: Vec<String>,
    sign_at: Option<usize>,
    covenant: String,
    registry: String,
    thread_coin: String,
    thread_root: String,
    thread_salt: String,
    root: String,
    tip_salt: String,
    slot: u64,
    tier: u8,
    burn: u64,
    burn_payload: String,
    burns: Vec<u64>,
}

fn unhex(s: &str, what: &str) -> Result<Vec<u8>, String> {
    hex::decode(s.trim()).map_err(|_| format!("{what} is not hex"))
}

fn fixed<const N: usize>(s: &str, what: &str) -> Result<[u8; N], String> {
    unhex(s, what)?.try_into().map_err(|_| format!("{what} is not {N} bytes"))
}

fn leaf_of(l: &LeafJson) -> Result<Leaf, String> {
    Ok(Leaf {
        key: fixed(&l.key, "the leaf's key")?,
        next: fixed(&l.next, "the leaf's next")?,
        expires: l.expires,
        tier: l.tier,
        owner: fixed(&l.owner, "the leaf's owner")?,
        data: fixed(&l.data, "the leaf's data")?,
        deal: fixed(&l.deal, "the leaf's deal")?,
    })
}

/// The bytes of an MSS signature from a tree of `height`
/// (`MssSignature::to_bytes`).
pub(crate) fn signature_len(height: u32) -> usize {
    8 + 32 + midstate::core::wots::SIG_SIZE + 4 + height as usize * 32
}

/// Puts what a proof of slot `slot` (leaf hash `leaf`, `None` for empty)
/// says into `mem`: every node on the path and every sibling. Returns the
/// root it reaches. A node two proofs disagree on is refused.
fn absorb(mem: &mut Mem, slot: u64, leaf: Option<[u8; 32]>, proof: &[u8]) -> Result<[u8; 32], String> {
    if proof.len() != names::PROOF_LEN {
        return Err(format!("a proof is {} bytes, not {}", names::PROOF_LEN, proof.len()));
    }
    if names::position(proof) != slot {
        return Err("the proof is for another slot".into());
    }
    let mut put = |level: usize, index: u64, h: [u8; 32]| -> Result<(), String> {
        match mem.get(&(level, index)) {
            Some(old) if *old != h => Err("two proofs disagree about the registry".into()),
            _ => {
                if h != names::empty(level) {
                    mem.insert((level, index), h);
                }
                Ok(())
            }
        }
    };
    let mut acc = leaf.unwrap_or([0; 32]);
    let mut i = slot;
    for level in 0..names::DEPTH {
        let sib: [u8; 32] = proof[level * 33..level * 33 + 32].try_into().expect("32 bytes");
        put(level, i, acc)?;
        put(level, i ^ 1, sib)?;
        acc = if i & 1 == 0 { hash_concat(&acc, &sib) } else { hash_concat(&sib, &acc) };
        i >>= 1;
    }
    put(names::DEPTH, 0, acc)?;
    Ok(acc)
}

/// The first slot of the largest subtree beside `slot`'s path that `proof`
/// shows to be empty, or `None` if every sibling on it holds something.
fn empty_slot(slot: u64, proof: &[u8]) -> Option<u64> {
    (0..names::DEPTH).rev().find(|&l| proof[l * 33..l * 33 + 32] == names::empty(l)).map(|l| ((slot >> l) ^ 1) << l)
}

fn plan(req: &PlanReq) -> Result<PlanOut, String> {
    let name = names::normalize(&req.name).map_err(|e| e.to_string())?;
    let leaf = leaf_of(&req.leaf)?;
    let proof = unhex(&req.proof, "the proof")?;
    let root = fixed::<32>(&req.root, "the root")?;
    let mut mem = Mem::new();
    if absorb(&mut mem, req.slot, Some(leaf.hash()), &proof)? != root {
        return Err("the proof does not reach the registry's root".into());
    }
    // The thread is the coin at the registry's address committing to that root.
    let thread_salt = fixed::<32>(&req.thread.salt, "the thread's salt")?;
    let thread_coin = fixed::<32>(&req.thread.coin, "the thread's coin")?;
    let thread = OutputData::Confidential { address: names::address(), commitment: root, salt: thread_salt };
    if thread.hash_for_commitment() != thread_coin {
        return Err("the thread coin does not commit to that root".into());
    }
    let records = unhex(&req.records, "the records")?;
    if records.len() > MAX_RECORDS || names::parse(&records).is_none() {
        return Err(format!("records are tag, length and value, {MAX_RECORDS} bytes at most"));
    }
    if req.years > names::MAX_YEARS {
        return Err(format!("a spend buys at most {} years", names::MAX_YEARS));
    }
    let set = Set { owner: fixed(&req.owner, "the owner")?, records, terms: unhex(&req.new_terms, "the terms")?, years: req.years, expires: req.expires };
    let k = names::key(name.as_bytes());
    let (mut spend, tier, slot, by) = match req.by.as_str() {
        "register" => {
            if req.years < 1 {
                return Err("a name is registered for a year at least".into());
            }
            let free = empty_slot(req.slot, &proof).ok_or("the registry shows no empty slot beside this proof")?;
            let s = names::register(&mem, &name, &set, req.slot, leaf, free).map_err(|e| format!("{e:#}"))?;
            (s, names::tier_for_len(name.len()), free, None)
        }
        how @ ("owner" | "lapse") => {
            if leaf.key != k {
                return Err("the proof is another name's".into());
            }
            let by = if how == "owner" { By::Owner } else { By::Lapse };
            if by == By::Lapse && req.years < 1 {
                return Err("a lapsed name is claimed for a year at least".into());
            }
            let oterms = unhex(&req.terms, "the terms")?;
            let s = names::change(&mem, by, &name, req.slot, leaf, &oterms, &set).map_err(|e| format!("{e:#}"))?;
            (s, leaf.tier, req.slot, Some(by))
        }
        other => return Err(format!("not a way to spend the registry: {other}")),
    };
    // Item 1 of a change: the owner's signature, its size set aside until the
    // page signs; a reclaim's is never read. Never empty: a node's reveal
    // drops empty witness items.
    let sign_at = match by {
        Some(By::Owner) => {
            if !(1..=20).contains(&req.signer_height) {
                return Err("the control key's height is needed to sign".into());
            }
            spend.items[1] = vec![0; signature_len(req.signer_height)];
            Some(1)
        }
        Some(_) => {
            spend.items[1] = vec![0];
            None
        }
        None => None,
    };
    if let Some(i) = spend.items.iter().position(|it| it.is_empty()) {
        return Err(format!("witness item {i} is empty, which a reveal cannot carry"));
    }
    let burn = names::cost(tier, req.years);
    let OutputData::Confidential { salt: tip_salt, .. } = names::next_tip(spend.root, &thread_coin) else { unreachable!("a tip is confidential") };
    Ok(PlanOut {
        items: spend.items.iter().map(hex::encode).collect(),
        sign_at,
        covenant: hex::encode(names::covenant()),
        registry: hex::encode(names::address()),
        thread_coin: hex::encode(thread_coin),
        thread_root: hex::encode(root),
        thread_salt: hex::encode(thread_salt),
        root: hex::encode(spend.root),
        tip_salt: hex::encode(tip_salt),
        slot,
        tier,
        burn,
        burn_payload: hex::encode([&names::TAG[..], name.as_bytes()].concat()),
        burns: decompose_value(burn),
    })
}

/// The registry spend for a name, planned from a node's proof (JSON as
/// `GET /name/:name/proof` gives it, plus how and what to set). Errors say
/// why the proof or the request does not hold.
#[wasm_bindgen]
pub fn name_plan(req_json: &str) -> Result<String, JsValue> {
    let req: PlanReq = serde_json::from_str(req_json).map_err(|e| JsValue::from_str(&format!("Bad name request: {e}")))?;
    let out = plan(&req).map_err(|e| JsValue::from_str(&e))?;
    serde_json::to_string(&out).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WebWallet;
    use midstate::core::mss;
    use midstate::core::names::Deal;
    use midstate::core::types::compute_address;
    use std::collections::HashMap;

    /// A name's key's height: the desktop app's (and the page's) MSS keys.
    const KEY_HEIGHT: u32 = 10;

    const H: u64 = names::START_HEIGHT + 5_000;

    /// A registry in memory, as a node's follower keeps it: the tree, and the
    /// leaves (with terms) by slot.
    struct Reg {
        tree: Mem,
        leaves: HashMap<u64, (Leaf, Vec<u8>)>,
    }

    impl Reg {
        fn genesis() -> Self {
            let mut r = Reg { tree: Mem::new(), leaves: HashMap::new() };
            for (pos, leaf) in names::genesis_leaves() {
                r.put(pos, leaf, Deal::None.terms());
            }
            r
        }
        fn put(&mut self, pos: u64, leaf: Leaf, terms: Vec<u8>) {
            let p = names::path(&self.tree, pos, Some(leaf.hash())).unwrap();
            names::write(&mut self.tree, &p);
            self.leaves.insert(pos, (leaf, terms));
        }
        fn root(&self) -> [u8; 32] {
            names::root(&self.tree).unwrap()
        }
        fn pred(&self, name: &str) -> (u64, Leaf) {
            let k = names::ord(&names::key(name.as_bytes()));
            let (p, (l, _)) = self.leaves.iter().filter(|(_, (l, _))| names::ord(&l.key) < k).max_by_key(|(_, (l, _))| names::ord(&l.key)).unwrap();
            (*p, *l)
        }
        fn find(&self, name: &str) -> (u64, Leaf, Vec<u8>) {
            let k = names::key(name.as_bytes());
            let (p, (l, t)) = self.leaves.iter().find(|(_, (l, _))| l.key == k).unwrap();
            (*p, *l, t.clone())
        }
        /// The lowest free slot, as a follower fills them.
        fn free(&self) -> u64 {
            (1..).find(|p| !self.leaves.contains_key(p)).unwrap()
        }
        /// Applies a spend the way a follower does.
        fn apply(&mut self, writes: Vec<(u64, Leaf)>, terms: Vec<u8>) {
            for (pos, leaf) in writes {
                let t = self.leaves.get(&pos).map(|(_, t)| t.clone()).unwrap_or_else(|| terms.clone());
                self.put(pos, leaf, t);
            }
        }
        /// What a node answers for `name`.
        fn answer(&self, name: &str) -> serde_json::Value {
            let k = names::key(name.as_bytes());
            let (slot, leaf, terms) = match self.leaves.iter().find(|(_, (l, _))| l.key == k) {
                Some((p, (l, t))) => (*p, *l, t.clone()),
                None => {
                    let (p, l) = self.pred(name);
                    (p, l, self.leaves[&p].1.clone())
                }
            };
            let salt = [9u8; 32];
            let coin = OutputData::Confidential { address: names::address(), commitment: self.root(), salt }.hash_for_commitment();
            serde_json::json!({
                "slot": slot,
                "leaf": {
                    "key": hex::encode(leaf.key), "next": hex::encode(leaf.next), "expires": leaf.expires, "tier": leaf.tier,
                    "owner": hex::encode(leaf.owner), "data": hex::encode(leaf.data), "deal": hex::encode(leaf.deal),
                },
                "terms": hex::encode(terms),
                "proof": hex::encode(names::proof(&self.tree, slot).unwrap()),
                "root": hex::encode(self.root()),
                "thread": { "coin": hex::encode(coin), "salt": hex::encode(salt) },
            })
        }
    }

    fn records(tag_text: &str) -> Vec<u8> {
        names::encode(&[names::Rec::Text(tag_text.into()), names::Rec::Kem([4; 32])]).unwrap()
    }

    /// The page's request: the node's answer, and what to set.
    fn request(answer: &serde_json::Value, name: &str, by: &str, owner: [u8; 32], recs: &[u8], years: u64, expires: u64) -> PlanReq {
        let mut v = answer.clone();
        v["name"] = name.into();
        v["by"] = by.into();
        v["owner"] = hex::encode(owner).into();
        v["records"] = hex::encode(recs).into();
        v["years"] = years.into();
        v["expires"] = expires.into();
        v["signer_height"] = KEY_HEIGHT.into();
        serde_json::from_value(v).unwrap()
    }

    fn items(out: &PlanOut) -> Vec<Vec<u8>> {
        out.items.iter().map(|h| hex::decode(h).unwrap()).collect()
    }

    /// Several registrations in a row, each planned from a proof alone, are
    /// exactly what `names::register` makes with the whole tree, and the
    /// registry they leave is the one a follower computes.
    #[test]
    fn registrations_from_a_proof_are_the_full_trees() {
        let mut reg = Reg::genesis();
        // The desktop's way first: a name in the lowest free slot.
        let (pp, pred) = reg.pred("zebra");
        let set = Set { owner: [1; 32], records: records("z"), terms: Deal::None.terms(), years: 1, expires: H + names::YEAR };
        let free = reg.free();
        let s = names::register(&reg.tree, "zebra", &set, pp, pred, free).unwrap();
        let k = names::key(b"zebra");
        let new = Leaf { key: k, next: pred.next, expires: set.expires, tier: 3, owner: set.owner, data: midstate::core::types::hash(&set.records), deal: Deal::None.hash() };
        reg.apply(vec![(pp, Leaf { next: k, ..pred }), (free, new)], Deal::None.terms());
        assert_eq!(reg.root(), s.root);

        let mut slots = Vec::new();
        for (i, name) in ["alice", "bob1", "nickname", "carolyn", "zz-top", "a1b2c3"].iter().enumerate() {
            let recs = records(name);
            let req = request(&reg.answer(name), name, "register", [i as u8 + 2; 32], &recs, 2, H + 2 * names::YEAR);
            let out = plan(&req).unwrap();
            let (pp, pred) = reg.pred(name);
            assert!(out.slot >= 1 << 16, "well away from the slots followers fill: {}", out.slot);
            let set = Set { owner: [i as u8 + 2; 32], records: recs.clone(), terms: Deal::None.terms(), years: 2, expires: H + 2 * names::YEAR };
            let full = names::register(&reg.tree, name, &set, pp, pred, out.slot).unwrap();
            assert_eq!(items(&out), full.items, "{name}");
            assert_eq!(out.root, hex::encode(full.root));
            assert_eq!(out.tier, names::tier_for_len(name.len()));
            assert_eq!(out.burn, names::cost(out.tier, 2));
            assert_eq!(out.burns, decompose_value(out.burn));
            assert_eq!(out.sign_at, None);
            assert_eq!(out.burn_payload, hex::encode([&b"MSN1"[..], name.as_bytes()].concat()));
            // The next thread coin's salt follows from the coin spent.
            let tip = names::next_tip(full.root, &fixed::<32>(&out.thread_coin, "").unwrap());
            assert_eq!(tip, OutputData::Confidential { address: names::address(), commitment: full.root, salt: fixed(&out.tip_salt, "").unwrap() });
            let k = names::key(name.as_bytes());
            let new = Leaf { key: k, next: pred.next, expires: set.expires, tier: out.tier, owner: set.owner, data: midstate::core::types::hash(&recs), deal: Deal::None.hash() };
            reg.apply(vec![(pp, Leaf { next: k, ..pred }), (out.slot, new)], Deal::None.terms());
            assert_eq!(hex::encode(reg.root()), out.root);
            slots.push(out.slot);
        }
        slots.sort();
        slots.dedup();
        assert_eq!(slots.len(), 6, "each in a slot of its own");
        // The desktop's next name still goes in the lowest free slot, untouched.
        assert_eq!(reg.free(), free + 1);
    }

    #[test]
    fn a_change_from_its_own_proof_is_the_full_trees() {
        let mut reg = Reg::genesis();
        let recs = records("x");
        let out = plan(&request(&reg.answer("nickname"), "nickname", "register", [7; 32], &recs, 1, H + names::YEAR)).unwrap();
        let (pp, pred) = reg.pred("nickname");
        let k = names::key(b"nickname");
        reg.apply(
            vec![(pp, Leaf { next: k, ..pred }), (out.slot, Leaf { key: k, next: pred.next, expires: H + names::YEAR, tier: 3, owner: [7; 32], data: midstate::core::types::hash(&recs), deal: Deal::None.hash() })],
            Deal::None.terms(),
        );

        // Renewed by its owner, with new records: the signature's room set aside.
        let more = records("renewed");
        let req = request(&reg.answer("nickname"), "nickname", "owner", [8; 32], &more, 1, H + 2 * names::YEAR);
        let out = plan(&req).unwrap();
        let (pos, leaf, terms) = reg.find("nickname");
        let set = Set { owner: [8; 32], records: more, terms: Deal::None.terms(), years: 1, expires: H + 2 * names::YEAR };
        let full = names::change(&reg.tree, By::Owner, "nickname", pos, leaf, &terms, &set).unwrap();
        let mut got = items(&out);
        assert_eq!(out.sign_at, Some(1));
        assert_eq!(got[1].len(), signature_len(KEY_HEIGHT));
        got[1] = Vec::new();
        assert_eq!(got, full.items);
        assert_eq!(out.root, hex::encode(full.root));
        assert_eq!((out.slot, out.tier, out.burn), (pos, 3, names::RATES[3]));

        // An edit alone burns nothing.
        let req = request(&reg.answer("nickname"), "nickname", "owner", [7; 32], &records("y"), 0, H + names::YEAR);
        assert_eq!(plan(&req).unwrap().burn, 0);
        assert!(plan(&request(&reg.answer("nickname"), "othername", "owner", [7; 32], &records("y"), 0, 0)).is_err(), "another name's proof");
    }

    #[test]
    fn a_premium_name_is_claimed_as_a_lapsed_one() {
        let reg = Reg::genesis();
        let (pos, leaf, terms) = reg.find("bitcoin");
        assert_eq!((leaf.owner, leaf.expires, leaf.tier), ([0; 32], 0, 0));
        let recs = records("b");
        let out = plan(&request(&reg.answer("bitcoin"), "bitcoin", "lapse", [3; 32], &recs, 1, H + names::YEAR)).unwrap();
        let set = Set { owner: [3; 32], records: recs, terms: Deal::None.terms(), years: 1, expires: H + names::YEAR };
        let full = names::change(&reg.tree, By::Lapse, "bitcoin", pos, leaf, &terms, &set).unwrap();
        let mut got = items(&out);
        assert_eq!(got[1], vec![0], "a reclaim's signature item is never read, but must not be empty");
        got[1] = Vec::new();
        assert_eq!(got, full.items);
        assert_eq!(out.burn, names::RATES[0], "platinum: 64 gMDS a year");
        assert!(plan(&request(&reg.answer("bitcoin"), "bitcoin", "lapse", [3; 32], &records("b"), 0, H)).is_err(), "a year at least");
    }

    #[test]
    fn a_proof_that_does_not_hold_is_refused() {
        let reg = Reg::genesis();
        let good = reg.answer("someone");
        let req = |f: &dyn Fn(&mut serde_json::Value)| {
            let mut v = good.clone();
            f(&mut v);
            plan(&request(&v, "someone", "register", [1; 32], &records("s"), 1, H + names::YEAR))
        };
        assert!(req(&|_| {}).is_ok());
        assert!(req(&|v| v["root"] = hex::encode([5u8; 32]).into()).unwrap_err().contains("does not reach"));
        assert!(req(&|v| v["leaf"]["expires"] = 77.into()).unwrap_err().contains("does not reach"));
        assert!(req(&|v| v["thread"]["salt"] = hex::encode([1u8; 32]).into()).unwrap_err().contains("thread coin"));
        let other = good["slot"].as_u64().unwrap() ^ 1;
        assert!(req(&|v| v["slot"] = other.into()).unwrap_err().contains("another slot"));
        assert!(req(&|v| v["proof"] = "00".into()).is_err());
        // A name the leaf is not the neighbour of.
        let (_, pred) = reg.pred("someone");
        let elsewhere = ["aaaa", "bbbb", "cccc", "dddd", "eeee"]
            .into_iter()
            .find(|n| {
                let k = names::ord(&names::key(n.as_bytes()));
                !(names::ord(&pred.key) < k && k < names::ord(&pred.next))
            })
            .unwrap();
        assert!(plan(&request(&good, elsewhere, "register", [1; 32], &records("s"), 1, H + names::YEAR)).unwrap_err().contains("taken"));
        assert!(plan(&request(&good, "someone", "register", [1; 32], &[1, 5, 0], 1, H + names::YEAR)).is_err(), "records cut short");
        assert!(plan(&request(&good, "someone", "register", [1; 32], &records("s"), 11, H)).is_err(), "ten years at most");
        assert!(plan(&request(&good, "someone", "steal", [1; 32], &records("s"), 1, H)).is_err());
    }

    #[test]
    fn absorbing_two_proofs_of_one_tree_agrees_and_a_forgery_does_not() {
        let reg = Reg::genesis();
        let (a, la, _) = reg.find("bitcoin");
        let (b, lb, _) = reg.find("ethereum");
        let mut mem = Mem::new();
        let ra = absorb(&mut mem, a, Some(la.hash()), &names::proof(&reg.tree, a).unwrap()).unwrap();
        let rb = absorb(&mut mem, b, Some(lb.hash()), &names::proof(&reg.tree, b).unwrap()).unwrap();
        assert_eq!((ra, rb), (reg.root(), reg.root()));
        let mut bad = names::proof(&reg.tree, b).unwrap();
        bad[names::PROOF_LEN - 2] ^= 1;
        assert!(absorb(&mut mem, b, Some(lb.hash()), &bad).is_err(), "disagrees with what is known");
    }

    /// A reveal, read as a node reads `/send`: witness stacks split on commas
    /// (empty items dropped), outputs by type.
    fn as_node_reads(reveal: &serde_json::Value) -> (Vec<Vec<Vec<u8>>>, Vec<OutputData>) {
        let witnesses = reveal["signatures"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().split(',').filter(|t| !t.is_empty()).map(|t| hex::decode(t).unwrap()).collect())
            .collect();
        let outputs = reveal["outputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| match o["type"].as_str().unwrap() {
                "data_burn" => OutputData::DataBurn { payload: hex::decode(o["payload"].as_str().unwrap()).unwrap(), value_burned: o["value_burned"].as_u64().unwrap() },
                "confidential" => OutputData::Confidential {
                    address: fixed(o["address"].as_str().unwrap(), "").unwrap(),
                    commitment: fixed(o["commitment"].as_str().unwrap(), "").unwrap(),
                    salt: fixed(o["salt"].as_str().unwrap(), "").unwrap(),
                },
                _ => OutputData::Standard {
                    address: fixed(o["address"].as_str().unwrap(), "").unwrap(),
                    value: o["value"].as_u64().unwrap(),
                    salt: fixed(o["salt"].as_str().unwrap(), "").unwrap(),
                },
            })
            .collect();
        (witnesses, outputs)
    }

    /// Builds the spend `out` plans as the page does: `prepare_script_spend`
    /// with the registry's input and outputs, the control key's signature
    /// (if any) put in its place, then `build_script_reveal`. Runs the
    /// registry's covenant on it at `height`, as a node would, and returns
    /// the outputs.
    fn build_and_run(w: &mut WebWallet, out: &PlanOut, coin: (u32, u64, [u8; 32]), signer: Option<&str>, height: u64) -> Vec<OutputData> {
        use midstate::core::wots;
        let (index, value, salt) = coin;
        let pk = wots::keygen(&midstate::wallet::hd::derive_wots_seed(&w.master_seed, index as u64));
        let addr = compute_address(&pk);
        let utxos = serde_json::json!([{
            "index": index, "is_mss": false, "mss_height": 0, "mss_leaf": 0, "address": hex::encode(addr), "value": value,
            "salt": hex::encode(salt), "coin_id": hex::encode(midstate::core::types::compute_coin_id(&addr, value, &salt)),
        }]);
        let inputs = serde_json::json!([{ "coin_id": out.thread_coin, "witness": out.items.join(","), "value": 0, "salt": out.thread_salt, "state": out.thread_root }]);
        let mut outs = vec![serde_json::json!({ "out_type": "confidential", "address": out.registry, "state": out.root, "salt": out.tip_salt })];
        for v in &out.burns {
            outs.push(serde_json::json!({ "out_type": "data_burn", "payload": out.burn_payload, "value": v }));
        }
        let ctx_str = w.prepare_script_spend(&utxos.to_string(), &out.covenant, &inputs.to_string(), &serde_json::Value::Array(outs).to_string(), 100).unwrap();
        let mut ctx: serde_json::Value = serde_json::from_str(&ctx_str).unwrap();
        let commitment: [u8; 32] = fixed(ctx["commitment"].as_str().unwrap(), "").unwrap();
        if let (Some(at), Some(pk)) = (out.sign_at, signer) {
            let sig = w.sign_mss_hex(pk, &hex::encode(commitment)).unwrap();
            let mut items: Vec<String> = ctx["contract_inputs"][0]["witness"].as_str().unwrap().split(',').map(String::from).collect();
            assert_eq!(sig.len(), items[at].len(), "the room set aside fits the signature");
            items[at] = sig;
            ctx["contract_inputs"][0]["witness"] = items.join(",").into();
        }
        let reveal: serde_json::Value =
            serde_json::from_str(&w.build_script_reveal(&ctx.to_string(), ctx["commitment"].as_str().unwrap(), ctx["tx_salt"].as_str().unwrap()).unwrap()).unwrap();
        let (witnesses, outputs) = as_node_reads(&reveal);
        assert_eq!(reveal["inputs"][0]["bytecode"], out.covenant);
        assert_eq!(reveal["inputs"][0]["commitment"], out.thread_root);
        assert_eq!(outputs[0], OutputData::Confidential { address: names::address(), commitment: fixed(&out.root, "").unwrap(), salt: fixed(&out.tip_salt, "").unwrap() });
        let burned: u64 = outputs.iter().map(|o| if let OutputData::DataBurn { value_burned, .. } = o { *value_burned } else { 0 }).sum();
        assert_eq!(burned, out.burn);
        let ectx = midstate::core::script::ExecContext {
            commitment: &commitment,
            height,
            outputs: &outputs,
            input_value: 0,
            input_state: Some(fixed(&out.thread_root, "").unwrap()),
            this_address: names::address(),
            sum_input_value: 0,
        };
        midstate::core::script::execute_script(names::covenant(), &witnesses[0], &ectx).expect("the registry's covenant takes it");
        // The fee coin is signed over the same commitment.
        let sig = wots::sig_from_bytes(&witnesses[1][0]).unwrap();
        assert!(wots::verify(&sig, &commitment, &pk));
        outputs
    }

    /// A name's key as the page makes one (the seed's MSS key at `index`, as
    /// `get_mss_address` derives it), in a smaller tree: WOTS keys are slow
    /// to make unoptimised.
    fn small_control(w: &mut WebWallet, index: u64, height: u32) -> String {
        let kp = mss::keygen(&midstate::wallet::hd::derive_mss_seed(&w.master_seed, index), height).unwrap();
        let pk = hex::encode(kp.master_pk);
        w.mss_cache.insert(hex::encode(compute_address(&kp.master_pk)), kp);
        pk
    }

    /// Register, then renew and edit as the owner, each built as the page
    /// builds it and taken by the registry's own covenant.
    #[test]
    fn a_name_built_as_the_page_builds_it_passes_the_covenant() {
        let mut reg = Reg::genesis();
        let mut w = WebWallet { master_seed: [5; 32], mss_cache: HashMap::new(), watchlist: Vec::new() };
        let control = small_control(&mut w, 0, 1);
        let owner: [u8; 32] = fixed(&control, "").unwrap();
        let recs = records("me");
        let out = plan(&request(&reg.answer("nickname"), "nickname", "register", owner, &recs, 1, H + names::YEAR)).unwrap();
        // Coins just over what is burned: each change coin is a one-time key,
        // slow to make unoptimised.
        build_and_run(&mut w, &out, (0, out.burn + (1 << 12), [3; 32]), None, H);
        let (pp, pred) = reg.pred("nickname");
        let k = names::key(b"nickname");
        let held = Leaf { key: k, next: pred.next, expires: H + names::YEAR, tier: 3, owner, data: midstate::core::types::hash(&recs), deal: Deal::None.hash() };
        reg.apply(vec![(pp, Leaf { next: k, ..pred }), (out.slot, held)], Deal::None.terms());

        // A year more, and new records: signed by the control key.
        let more = records("me, a year on");
        let mut req = request(&reg.answer("nickname"), "nickname", "owner", owner, &more, 1, H + 2 * names::YEAR);
        req.signer_height = 1;
        let renew = plan(&req).unwrap();
        build_and_run(&mut w, &renew, (1, renew.burn + (1 << 12), [4; 32]), Some(&control), H + 10);
        // Signed by another key, it is refused.
        let mut other = WebWallet { master_seed: [6; 32], mss_cache: HashMap::new(), watchlist: Vec::new() };
        let theirs = small_control(&mut other, 0, 1);
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| build_and_run(&mut other, &renew, (0, renew.burn + (1 << 12), [5; 32]), Some(&theirs), H + 10)));
        assert!(caught.is_err(), "another key's signature");
    }

    #[test]
    fn signatures_are_the_size_set_aside() {
        let mut kp = mss::keygen(&midstate::wallet::hd::derive_mss_seed(&[6; 32], 0), 1).unwrap();
        assert_eq!(kp.sign(&[1; 32]).unwrap().to_bytes().len(), signature_len(1));
    }
}
