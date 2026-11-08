//! Bond rentals end to end, through midstate's consensus: transactions the
//! wallet code builds and signs (an owner's offer, a renter's take, an
//! owner's cancel) applied with `apply_transaction` to stand-in chains, one
//! before the v6 opcodes and one after, side by side.

use anyhow::{bail, Result};
use midstate::core::mmr::{MerkleMountainRange, UtxoAccumulator};
use midstate::core::rental::{announced, cancel_witness, spent_by, take_witness, Offer, OfferSpend};
use midstate::core::transaction::apply_transaction;
use midstate::core::types::{compute_address, compute_coin_id, compute_commitment, is_v2_at, OutputData, State, Transaction};
use midstate::wallet::Wallet;
use midstate_walletd::rental::{self, plan_take, resolve_cancel_fee, sign_wallet_inputs, OFFER_KEY_HEIGHT};
use midstate_walletd::sendplan;
use tempfile::TempDir;

const BOND: u64 = 1 << 37;
const RENT: u64 = 1 << 31;

struct Chain {
    state: State,
}

fn chain_at(height: u64) -> Chain {
    Chain {
        state: State {
            midstate: [0; 32],
            coins: UtxoAccumulator::new(),
            commitments: UtxoAccumulator::new(),
            depth: 0,
            target: [0xff; 32],
            height,
            timestamp: 0,
            commitment_heights: Default::default(),
            expirations: Default::default(),
            chain_mmr: MerkleMountainRange::new(),
            header_hash: [0; 32],
            burned_wots: UtxoAccumulator::new(),
        },
    }
}

/// The same transactions on every chain.
struct Chains(Vec<Chain>);

impl Chains {
    /// A coin of `w`'s, at a fresh one-time key.
    fn give(&mut self, w: &mut Wallet, value: u64) -> [u8; 32] {
        let seed: [u8; 32] = rand::random();
        let salt: [u8; 32] = rand::random();
        let id = w.import_coin(seed, value, salt, None).unwrap();
        for c in &mut self.0 {
            assert!(c.state.coins.insert(id, is_v2_at(c.state.height)));
        }
        id
    }

    /// Phase 1, as if mined a block ago.
    fn commit(&mut self, inputs: &[[u8; 32]], outputs: &[OutputData], salt: &[u8; 32]) -> [u8; 32] {
        let outs: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();
        let commitment = compute_commitment(inputs, &outs, salt);
        for c in &mut self.0 {
            c.state.commitments.insert(commitment, is_v2_at(c.state.height));
            c.state.commitment_heights.insert(commitment, c.state.height - 1);
        }
        commitment
    }

    /// Phase 2: the relay's fee floor, then consensus, on every chain.
    fn reveal(&mut self, tx: &Transaction) -> Result<()> {
        let bytes = bincode::serialized_size(tx)? as u128;
        if (tx.fee() as u128) * 1_024 < 10 * bytes {
            bail!("fee {} below the relay floor for {} bytes", tx.fee(), bytes);
        }
        let mut next: Vec<State> = Vec::new();
        for c in &self.0 {
            let mut state = c.state.clone();
            apply_transaction(&mut state, tx).map_err(|e| e.context(format!("at height {}", c.state.height)))?;
            next.push(state);
        }
        for (c, state) in self.0.iter_mut().zip(next) {
            c.state = state;
        }
        Ok(())
    }

    fn has(&self, coin: &[u8; 32]) -> bool {
        let here: Vec<bool> = self.0.iter().map(|c| c.state.coins.contains(coin)).collect();
        assert!(here.windows(2).all(|w| w[0] == w[1]), "the chains disagree");
        here[0]
    }
}

fn wallet(dir: &TempDir, name: &str) -> Wallet {
    Wallet::create(&dir.path().join(name), b"pass").unwrap()
}

/// The owner's side: a fresh tree for the offer, then one send that funds
/// and announces it, signed by the wallet's own reveal path.
fn post(chain: &mut Chains, owner: &mut Wallet, until: u64) -> (Offer, Transaction) {
    let address = owner.generate_mss(OFFER_KEY_HEIGHT, None).unwrap();
    let owner_pk = owner.mss_keys().iter().find(|k| compute_address(&k.master_pk) == address).unwrap().master_pk;
    let offer = Offer { owner_pk, bonded_until: until, rent: RENT, bond: BOND };
    let live: Vec<[u8; 32]> = owner.coins().iter().filter(|c| !c.wots_signed).map(|c| c.coin_id).collect();
    let plan = sendplan::plan_fixed_outputs(
        owner,
        &live,
        vec![
            OutputData::Standard { address: offer.address(), value: BOND, salt: rand::random() },
            OutputData::DataBurn { payload: offer.announcement(), value_burned: 0 },
        ],
    )
    .unwrap();
    let (commitment, _) = owner.prepare_commit(&plan.input_coin_ids, &plan.outputs, plan.change_seeds, false, false).unwrap();
    let pending = owner.find_pending(&commitment).unwrap().clone();
    assert_eq!(chain.commit(&pending.input_coin_ids, &pending.outputs, &pending.salt), commitment);
    let (inputs, witnesses) = owner.sign_reveal(&pending).unwrap();
    let tx = Transaction::Reveal { inputs, witnesses, outputs: pending.outputs.clone(), salt: pending.salt };
    chain.reveal(&tx).expect("the offer's funding send is valid");
    owner.complete_reveal(&commitment).unwrap();
    (offer, tx)
}

/// The renter's side: one send of its own.
fn take(chain: &mut Chains, renter: &mut Wallet, offer: &Offer, salt: [u8; 32], key: &[u8; 32]) -> Result<(Transaction, [u8; 32])> {
    let coin = offer.coin_id(&salt);
    let live: Vec<[u8; 32]> = renter.coins().iter().filter(|c| !c.wots_signed).map(|c| c.coin_id).collect();
    let plan = plan_take(renter, &live, &[], offer, coin, key)?;
    let tx_salt: [u8; 32] = rand::random();
    let c = chain.commit(&plan.input_coin_ids, &plan.outputs, &tx_salt);
    let ours = &plan.input_coin_ids[..plan.input_coin_ids.len() - 1];
    let (mut inputs, mut witnesses): (Vec<_>, Vec<_>) = sign_wallet_inputs(renter, ours, &c)?.into_iter().unzip();
    inputs.push(offer.input(salt));
    witnesses.push(take_witness(key));
    let tx = Transaction::Reveal { inputs, witnesses, outputs: plan.outputs, salt: tx_salt };
    chain.reveal(&tx)?;
    Ok((tx, plan.bond_salt))
}

#[test]
fn rent_out_and_rent_before_and_after_the_v6_opcodes() {
    let v6 = midstate::core::types::V6_ACTIVATION_HEIGHT;
    let height = v6 - 10_000;
    let dir = TempDir::new().unwrap();
    let mut chain = Chains(vec![chain_at(v6 - 10_000), chain_at(v6 + 10_000)]);
    let mut owner = wallet(&dir, "owner.dat");
    let mut renter = wallet(&dir, "renter.dat");
    let mut late = wallet(&dir, "late.dat");
    chain.give(&mut owner, 1 << 39);
    chain.give(&mut renter, 1 << 33);
    chain.give(&mut late, 1 << 33);

    let (offer, funding) = post(&mut chain, &mut owner, height + 90 * 1440);
    let listed = announced(&funding);
    assert_eq!(listed.len(), 1, "the funding send announces the offer beside its coin");
    let salt = listed[0].salt;
    assert_eq!(listed[0].offer, offer);
    assert!(chain.has(&offer.coin_id(&salt)));
    assert!(owner.watched_addresses().contains(&offer.pay_to()), "the rent lands where the owner's wallet looks");

    let key = [0x42; 32];
    let (tx, bond_salt) = take(&mut chain, &mut renter, &offer, salt, &key).expect("the take is valid");
    assert!(!chain.has(&offer.coin_id(&salt)));
    let bond = compute_coin_id(&offer.bond_address(&key), BOND, &bond_salt);
    assert!(chain.has(&bond), "the bond is on chain, named for the renter's key");
    assert_eq!(spent_by(&tx, &offer, &offer.coin_id(&salt)), Some(OfferSpend::Taken { mining_key: key, bond_salt: Some(bond_salt), rent_paid: RENT }));

    // Someone else, a moment too late.
    assert!(take(&mut chain, &mut late, &offer, salt, &[0x43; 32]).is_err(), "an offer is taken once");

    // A second offer, cancelled by its own key.
    let (offer2, funding2) = post(&mut chain, &mut owner, height + 30 * 1440);
    let salt2 = announced(&funding2)[0].salt;
    let coin2 = offer2.coin_id(&salt2);
    let (fee, denoms) = resolve_cancel_fee(&offer2, midstate_walletd::bond::mss_sig_len(OFFER_KEY_HEIGHT)).unwrap();
    assert_eq!(denoms.iter().sum::<u64>() + fee, BOND);
    let outputs: Vec<OutputData> = denoms
        .iter()
        .map(|v| OutputData::Standard { address: [9; 32], value: *v, salt: rand::random() })
        .collect();
    let tx_salt: [u8; 32] = rand::random();
    let c = chain.commit(&[coin2], &outputs, &tx_salt);
    let sig = owner.auto_sign(&offer2.owner_pk, &c).unwrap();
    let cancel = Transaction::Reveal {
        inputs: vec![offer2.input(salt2)],
        witnesses: vec![cancel_witness(sig)],
        outputs,
        salt: tx_salt,
    };
    assert_eq!(spent_by(&cancel, &offer2, &coin2), Some(OfferSpend::Cancelled));
    chain.reveal(&cancel).expect("the owner's cancel is valid");
    assert!(!chain.has(&coin2));
}

#[test]
fn an_offer_announced_without_its_coin_is_no_offer() {
    let mut m = rental::Market::default();
    let offer = Offer { owner_pk: [1; 32], bonded_until: 500_000, rent: RENT, bond: BOND };
    let bare = Transaction::Reveal {
        inputs: vec![],
        witnesses: vec![],
        outputs: vec![OutputData::DataBurn { payload: offer.announcement(), value_burned: 0 }],
        salt: [0; 32],
    };
    m.ingest(1, &bare);
    assert!(m.listings.is_empty());
}
