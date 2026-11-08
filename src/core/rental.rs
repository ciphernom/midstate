//! Bond rentals: renting out a midwimble mining bond, with no change to
//! either chain's consensus.
//!
//! An owner parks one coin at an *offer*, the covenant
//! [`script::compile_bond_rental_offer`]. Anyone takes the offer by naming a
//! midwimble mining key, in a transaction that must lock the coin as a mining
//! bond for that key and pay the rent to the owner. The owner keeps the
//! bond's owner key, so the coin comes back at the lock height whoever takes
//! it, and the taker can do nothing with it but mine. Until someone takes it,
//! the owner can cancel with that key.
//!
//! Only one side signs any transaction: the owner funds and cancels, the
//! taker takes. Nobody is ever asked to sign first and wait on the other, so
//! nobody can leave a counterparty's one-time keys spent on a transaction
//! that never happens.
//!
//! The offer is announced in the transaction that funds it, as a zero-value
//! `DataBurn` ([`Offer::announcement`]), as the DEX's asks are. Anyone
//! reading the chain finds it, checks that the coin beside it is unspent, and
//! can take it without the owner being online.
//!
//! Every offer has its own owner key, and the rent goes to that key's own
//! address ([`Offer::pay_to`]). Two offers sharing a key would let one bond,
//! or one payment, answer both (the covenant's documentation says how).

use super::script;
use super::types::{
    compute_address, compute_coin_id, hash, InputReveal, OutputData, Predicate, Transaction, Witness,
};
use serde::{Deserialize, Serialize};

/// Announcement tag: "midstate bond rental".
pub const MAGIC: [u8; 4] = *b"MDBR";
pub const VERSION: u8 = 1;
/// tag, version, log2(bond), lock height, rent, owner key.
pub const ANNOUNCEMENT_LEN: usize = 4 + 1 + 1 + 8 + 8 + 32;

/// An offer's terms: everything its covenant commits to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Offer {
    /// The bond's owner key, and the key that cancels. One per offer.
    #[serde(with = "hex32")]
    pub owner_pk: [u8; 32],
    /// Midstate height the bond stays locked until.
    pub bonded_until: u64,
    /// What a taker pays, per bond.
    pub rent: u64,
    /// The bond's value: one coin, a power of two.
    pub bond: u64,
}

impl Offer {
    /// Where the rent goes: the owner key's own address.
    pub fn pay_to(&self) -> [u8; 32] {
        compute_address(&self.owner_pk)
    }

    pub fn script(&self) -> Vec<u8> {
        script::compile_bond_rental_offer(&self.owner_pk, self.bonded_until, self.rent, self.bond, &self.pay_to())
    }

    pub fn address(&self) -> [u8; 32] {
        hash(&self.script())
    }

    /// The offer's coin, funded with `salt`.
    pub fn coin_id(&self, salt: &[u8; 32]) -> [u8; 32] {
        compute_coin_id(&self.address(), self.bond, salt)
    }

    /// The bond a take makes for `mining_key`.
    pub fn bond_script(&self, mining_key: &[u8; 32]) -> Vec<u8> {
        script::compile_mining_bond(mining_key, self.bonded_until, &self.owner_pk)
    }

    pub fn bond_address(&self, mining_key: &[u8; 32]) -> [u8; 32] {
        hash(&self.bond_script(mining_key))
    }

    /// The offer coin as a reveal input.
    pub fn input(&self, salt: [u8; 32]) -> InputReveal {
        InputReveal {
            predicate: Predicate::Script { bytecode: self.script() },
            value: self.bond,
            salt,
            commitment: None,
        }
    }

    /// The rent a take owes for spending `offer_value` of this offer's coins.
    pub fn rent_due(&self, offer_value: u64) -> u64 {
        (offer_value / self.bond.max(1)).saturating_mul(self.rent)
    }

    /// The payload of the `DataBurn` that announces this offer.
    pub fn announcement(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(ANNOUNCEMENT_LEN);
        out.extend_from_slice(&MAGIC);
        out.push(VERSION);
        out.push(self.bond.trailing_zeros() as u8);
        out.extend_from_slice(&self.bonded_until.to_le_bytes());
        out.extend_from_slice(&self.rent.to_le_bytes());
        out.extend_from_slice(&self.owner_pk);
        out
    }

    /// Reads an announcement; `None` for anything that is not one.
    pub fn from_announcement(payload: &[u8]) -> Option<Offer> {
        if payload.len() != ANNOUNCEMENT_LEN || payload[..4] != MAGIC || payload[4] != VERSION {
            return None;
        }
        let log2 = payload[5];
        if log2 >= 64 {
            return None;
        }
        let u64_at = |at: usize| u64::from_le_bytes(payload[at..at + 8].try_into().unwrap());
        Some(Offer {
            bond: 1u64 << log2,
            bonded_until: u64_at(6),
            rent: u64_at(14),
            owner_pk: payload[22..54].try_into().unwrap(),
        })
    }
}

/// What a taker's witness on the offer coin is: the key, then the take
/// branch.
pub fn take_witness(mining_key: &[u8; 32]) -> Witness {
    Witness::ScriptInputs(vec![mining_key.to_vec(), vec![1]])
}

/// What the owner's witness on the offer coin is to cancel it.
pub fn cancel_witness(owner_sig: Vec<u8>) -> Witness {
    Witness::ScriptInputs(vec![owner_sig, vec![0]])
}

/// An offer announced in a transaction, with the coin it funds there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Announced {
    pub offer: Offer,
    pub salt: [u8; 32],
    pub coin_id: [u8; 32],
}

/// The offers a transaction announces: each announcement that sits beside
/// the coin it describes. An announcement whose coin is not in the same
/// transaction is not an offer.
pub fn announced(tx: &Transaction) -> Vec<Announced> {
    let Transaction::Reveal { outputs, .. } = tx else { return Vec::new() };
    let mut found = Vec::new();
    for out in outputs {
        let OutputData::DataBurn { payload, .. } = out else { continue };
        let Some(offer) = Offer::from_announcement(payload) else { continue };
        let address = offer.address();
        let funded = outputs.iter().find_map(|o| match o {
            OutputData::Standard { address: a, value, salt } if *a == address && *value == offer.bond => Some(*salt),
            _ => None,
        });
        if let Some(salt) = funded {
            found.push(Announced { offer, salt, coin_id: offer.coin_id(&salt) });
        }
    }
    found
}

/// How a transaction spent an offer's coin.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OfferSpend {
    /// Taken for `mining_key`. `bond_salt` is the bond coin's, when the
    /// transaction made one of the offer's full value; `rent_paid` is what
    /// it sent to the owner.
    Taken { mining_key: [u8; 32], bond_salt: Option<[u8; 32]>, rent_paid: u64 },
    /// Spent by the owner's key.
    Cancelled,
}

/// Whether `tx` spends the coin `coin_id` of `offer`, and how.
pub fn spent_by(tx: &Transaction, offer: &Offer, coin_id: &[u8; 32]) -> Option<OfferSpend> {
    let Transaction::Reveal { inputs, witnesses, outputs, .. } = tx else { return None };
    let at = inputs.iter().position(|i| &i.coin_id() == coin_id)?;
    let Witness::ScriptInputs(items) = witnesses.get(at)?;
    match items.as_slice() {
        [key, sel] if sel.as_slice() == [1] && key.len() == 32 => {
            let mining_key: [u8; 32] = key.as_slice().try_into().ok()?;
            let bond = offer.bond_address(&mining_key);
            let pay_to = offer.pay_to();
            let bond_salt = outputs.iter().find_map(|o| match o {
                OutputData::Standard { address, value, salt } if *address == bond && *value == offer.bond => Some(*salt),
                _ => None,
            });
            let rent_paid = outputs
                .iter()
                .filter(|o| matches!(o, OutputData::Standard { address, .. } if *address == pay_to))
                .map(|o| o.value())
                .sum();
            Some(OfferSpend::Taken { mining_key, bond_salt, rent_paid })
        }
        [_, sel] if sel.as_slice() == [0] => Some(OfferSpend::Cancelled),
        _ => None,
    }
}

mod hex32 {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex::encode(v))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        let s = String::deserialize(d)?;
        let v = hex::decode(&s).map_err(serde::de::Error::custom)?;
        v.try_into().map_err(|_| serde::de::Error::custom("expected 32 bytes"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::mmr::UtxoAccumulator;
    use crate::core::script::{execute_script, ExecContext, ScriptError};
    use crate::core::types::State;
    use crate::core::transaction::apply_transaction;
    use crate::core::types::{compute_commitment, is_v2_at};
    use crate::core::{mss, wots};

    /// Midstate's height when this was written; the covenant needs nothing
    /// newer than `COVENANT_SUM_ACTIVATION_HEIGHT`.
    const HEIGHT: u64 = 354_000;
    const BOND: u64 = 1 << 37;
    const RENT: u64 = 1 << 33;
    const UNTIL: u64 = 600_000;

    fn offer(owner_pk: [u8; 32]) -> Offer {
        Offer { owner_pk, bonded_until: UNTIL, rent: RENT, bond: BOND }
    }

    fn std_out(address: [u8; 32], value: u64, salt: u8) -> OutputData {
        OutputData::Standard { address, value, salt: [salt; 32] }
    }

    fn run(o: &Offer, witness: Witness, outputs: &[OutputData], spent: u64, c: &[u8; 32]) -> Result<(), ScriptError> {
        let Witness::ScriptInputs(items) = witness;
        let script = o.script();
        let ctx = ExecContext {
            commitment: c,
            height: HEIGHT,
            outputs,
            input_value: o.bond,
            input_state: None,
            this_address: hash(&script),
            sum_input_value: spent,
        };
        execute_script(&script, &items, &ctx)
    }

    /// The covenant is the IDE language's compilation of
    /// `docs/contracts/bond-rental-offer.mds` (by the SDK's compiler), byte
    /// for byte, at that file's placeholder values.
    #[test]
    fn the_covenant_is_the_contracts_compilation() {
        let compiled = "110101000120401011291210010100202021010300c0270901030001200001010002171601020010011613291210\
01010000161613160104003301200016012000ababababababababababababababababababababababababababab\
ababababab0105003201010001161611301150562421012000cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd\
cdcdcdcdcdcdcdcdcdcdcd5056010500000000002027010500000000000226242110101010411101010000204010\
012000abababababababababababababababababababababababababababababababab3241100101000021424201\
010001";
        let ours = script::compile_bond_rental_offer(&[0xab; 32], 600_000, 1 << 33, 1 << 37, &[0xcd; 32]);
        assert_eq!(hex::encode(&ours), compiled);
        assert_eq!(ours.len(), 233);
        script::validate_structure(&ours, HEIGHT).expect("valid at today's height");
    }

    #[test]
    fn announcements_round_trip_and_reject_the_rest() {
        let o = offer([7; 32]);
        let a = o.announcement();
        assert_eq!(a.len(), ANNOUNCEMENT_LEN);
        assert!(a.len() <= crate::core::types::MAX_BURN_DATA_SIZE);
        assert_eq!(Offer::from_announcement(&a), Some(o));
        let mut bad = a.clone();
        bad[0] ^= 1;
        assert_eq!(Offer::from_announcement(&bad), None, "tag");
        let mut bad = a.clone();
        bad[4] = 2;
        assert_eq!(Offer::from_announcement(&bad), None, "version");
        let mut bad = a.clone();
        bad[5] = 64;
        assert_eq!(Offer::from_announcement(&bad), None, "bond out of range");
        assert_eq!(Offer::from_announcement(&a[..53]), None, "short");
    }

    #[test]
    fn a_take_locks_the_bond_for_the_key_and_pays_the_rent() {
        let o = offer([0xab; 32]);
        let key = [0x11; 32];
        let bond = o.bond_address(&key);
        let pay = o.pay_to();
        let c = [7u8; 32];
        let take = |k: &[u8]| Witness::ScriptInputs(vec![k.to_vec(), vec![1]]);
        let paid = |a: [u8; 32]| vec![std_out(a, BOND, 1), std_out(pay, RENT, 2)];

        assert_eq!(run(&o, take(&key), &paid(bond), BOND, &c), Ok(()));
        // Rent in several coins counts.
        let split = [std_out(bond, BOND, 1), std_out(pay, RENT / 2, 2), std_out(pay, RENT / 2, 3)];
        assert_eq!(run(&o, take(&key), &split, BOND, &c), Ok(()));
        // Rent short by one unit; bond short.
        assert!(run(&o, take(&key), &[std_out(bond, BOND, 1), std_out(pay, RENT - 1, 2)], BOND, &c).is_err());
        assert!(run(&o, take(&key), &[std_out(bond, BOND / 2, 1), std_out(pay, RENT, 2)], BOND, &c).is_err());
        // A bond for another key, lock height or owner.
        for wrong in [
            script::compile_mining_bond(&[0x22; 32], UNTIL, &o.owner_pk),
            script::compile_mining_bond(&key, UNTIL - 1, &o.owner_pk),
            script::compile_mining_bond(&key, UNTIL + 1, &o.owner_pk),
            script::compile_mining_bond(&key, UNTIL, &[0x33; 32]),
        ] {
            assert!(run(&o, take(&key), &paid(hash(&wrong)), BOND, &c).is_err());
        }
        // Keys of any other length, among them one that would run as code
        // inside the bond script (32 zero bytes, then VERIFY: a coin nobody
        // could ever spend).
        let mut brick = vec![0u8; 32];
        brick.push(script::OP_VERIFY);
        let mut bricked = vec![script::OP_PUSH_DATA, 32, 0];
        bricked.extend_from_slice(&brick);
        bricked.extend_from_slice(&o.bond_script(&key)[35..]);
        assert!(run(&o, take(&brick), &paid(hash(&bricked)), BOND, &c).is_err());
        assert!(run(&o, take(&key[..31]), &paid(bond), BOND, &c).is_err());
        // Two coins of the offer: both become bond, both are paid for.
        let other = [0x99; 32];
        assert!(run(&o, take(&key), &[std_out(bond, BOND, 1), std_out(pay, 2 * RENT, 2), std_out(other, BOND, 3)], 2 * BOND, &c).is_err());
        assert!(run(&o, take(&key), &[std_out(bond, BOND, 1), std_out(bond, BOND, 4), std_out(pay, RENT, 2)], 2 * BOND, &c).is_err());
        assert_eq!(run(&o, take(&key), &[std_out(bond, BOND, 1), std_out(bond, BOND, 4), std_out(pay, 2 * RENT, 2)], 2 * BOND, &c), Ok(()));
        // No third branch.
        assert!(run(&o, Witness::ScriptInputs(vec![key.to_vec(), vec![2]]), &paid(bond), BOND, &c).is_err());
    }

    #[test]
    fn only_the_owner_cancels() {
        let seed = hash(b"owner");
        let o = offer(wots::keygen(&seed));
        let c = [9u8; 32];
        let sig = wots::sig_to_bytes(&wots::sign(&seed, &c));
        assert_eq!(run(&o, cancel_witness(sig), &[std_out([1; 32], 64, 1)], BOND, &c), Ok(()));
        let stranger = wots::sig_to_bytes(&wots::sign(&hash(b"stranger"), &c));
        assert!(run(&o, cancel_witness(stranger), &[std_out([1; 32], 64, 1)], BOND, &c).is_err());
    }

    // ── Through consensus ───────────────────────────────────────────────

    fn state() -> State {
        State {
            midstate: [0u8; 32],
            coins: UtxoAccumulator::new(),
            commitments: UtxoAccumulator::new(),
            depth: 0,
            target: [0xff; 32],
            height: HEIGHT,
            timestamp: 1_790_000_000,
            commitment_heights: im::HashMap::new(),
            expirations: im::OrdMap::new(),
            header_hash: [0u8; 32],
            chain_mmr: crate::core::mmr::MerkleMountainRange::new(),
            burned_wots: UtxoAccumulator::new(),
        }
    }

    /// Records the reveal's commitment, as a mined commit would.
    fn commit(state: &mut State, inputs: &[InputReveal], outputs: &[OutputData], salt: &[u8; 32]) -> [u8; 32] {
        let ins: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
        let outs: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();
        let c = compute_commitment(&ins, &outs, salt);
        assert!(state.commitments.insert(c, is_v2_at(state.height)));
        state.commitment_heights.insert(c, state.height - 1);
        c
    }

    fn wots_coin(state: &mut State, seed: &[u8; 32], value: u64) -> InputReveal {
        let input = InputReveal {
            predicate: Predicate::p2pk(&wots::keygen(seed)),
            value,
            salt: hash(seed),
            commitment: None,
        };
        assert!(state.coins.insert(input.coin_id(), is_v2_at(state.height)));
        input
    }

    /// The offer funded and announced in one reveal, by the owner alone.
    fn fund(state: &mut State, o: &Offer) -> (Transaction, [u8; 32]) {
        let seed = hash(b"owner funds");
        let input = wots_coin(state, &seed, 2 * BOND);
        let salt = [0x5a; 32];
        let outputs = vec![
            OutputData::Standard { address: o.address(), value: BOND, salt },
            OutputData::DataBurn { payload: o.announcement(), value_burned: 0 },
            std_out(compute_address(&wots::keygen(&hash(b"change"))), BOND / 2, 3),
        ];
        let tx_salt = [1u8; 32];
        let c = commit(state, std::slice::from_ref(&input), &outputs, &tx_salt);
        let tx = Transaction::Reveal {
            inputs: vec![input],
            witnesses: vec![Witness::sig(wots::sig_to_bytes(&wots::sign(&seed, &c)))],
            outputs,
            salt: tx_salt,
        };
        apply_transaction(state, &tx).expect("funding is an ordinary reveal");
        (tx, salt)
    }

    #[test]
    fn an_offer_is_funded_announced_and_taken_on_chain() {
        let mut s = state();
        let o = offer(wots::keygen(&hash(b"owner")));
        let (funding, salt) = fund(&mut s, &o);
        let found = announced(&funding);
        assert_eq!(found, vec![Announced { offer: o, salt, coin_id: o.coin_id(&salt) }]);
        assert!(s.coins.contains(&o.coin_id(&salt)));

        // The taker alone: its coin for the rent and fee, the offer's coin.
        let seed = hash(b"taker");
        let rent_coin = wots_coin(&mut s, &seed, 2 * RENT);
        let key = [0x11; 32];
        let outputs = vec![
            OutputData::Standard { address: o.bond_address(&key), value: BOND, salt: [0x77; 32] },
            OutputData::Standard { address: o.pay_to(), value: RENT, salt: [0x78; 32] },
            std_out(compute_address(&wots::keygen(&hash(b"taker change"))), RENT / 2, 4),
        ];
        let inputs = vec![rent_coin, o.input(salt)];
        let tx_salt = [2u8; 32];
        let c = commit(&mut s, &inputs, &outputs, &tx_salt);
        let take = Transaction::Reveal {
            inputs: inputs.clone(),
            witnesses: vec![Witness::sig(wots::sig_to_bytes(&wots::sign(&seed, &c))), take_witness(&key)],
            outputs: outputs.clone(),
            salt: tx_salt,
        };
        apply_transaction(&mut s.clone(), &take).expect("a take is valid at consensus");
        assert_eq!(
            spent_by(&take, &o, &o.coin_id(&salt)),
            Some(OfferSpend::Taken { mining_key: key, bond_salt: Some([0x77; 32]), rent_paid: RENT })
        );

        // Short of rent, consensus refuses it.
        let mut short = outputs.clone();
        short[1] = OutputData::Standard { address: o.pay_to(), value: RENT / 2, salt: [0x78; 32] };
        let c2 = commit(&mut s, &inputs, &short, &tx_salt);
        let underpaid = Transaction::Reveal {
            inputs: inputs.clone(),
            witnesses: vec![Witness::sig(wots::sig_to_bytes(&wots::sign(&seed, &c2))), take_witness(&key)],
            outputs: short,
            salt: tx_salt,
        };
        assert!(apply_transaction(&mut s.clone(), &underpaid).is_err());

        let mut after = s.clone();
        apply_transaction(&mut after, &take).unwrap();
        assert!(!after.coins.contains(&o.coin_id(&salt)), "the offer is gone");
        let bond_coin = compute_coin_id(&o.bond_address(&key), BOND, &[0x77; 32]);
        assert!(after.coins.contains(&bond_coin), "the bond is on chain");
    }

    #[test]
    fn the_owner_cancels_on_chain() {
        let mut s = state();
        let mut owner = mss::keygen(&hash(b"owner"), 4).unwrap();
        let o = offer(owner.master_pk);
        let (_, salt) = fund(&mut s, &o);
        let outputs = vec![std_out(compute_address(&wots::keygen(&hash(b"back"))), BOND / 2, 5)];
        let inputs = vec![o.input(salt)];
        let tx_salt = [3u8; 32];
        let c = commit(&mut s, &inputs, &outputs, &tx_salt);
        let cancel = Transaction::Reveal {
            inputs,
            witnesses: vec![cancel_witness(owner.sign(&c).unwrap().to_bytes())],
            outputs,
            salt: tx_salt,
        };
        assert_eq!(spent_by(&cancel, &o, &o.coin_id(&salt)), Some(OfferSpend::Cancelled));
        apply_transaction(&mut s, &cancel).expect("the owner's cancel is valid");
        assert!(!s.coins.contains(&o.coin_id(&salt)));
    }
}
