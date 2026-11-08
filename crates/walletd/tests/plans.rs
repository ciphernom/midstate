//! Inheritance plans and payment streams end to end, through midstate's
//! consensus: transactions built by the wallet's planning code and signed by
//! real wallets, applied with `apply_transaction` (after the relay's fee
//! floor) to a stand-in chain.

use anyhow::{bail, Result};
use midstate::core::mmr::{MerkleMountainRange, UtxoAccumulator};
use midstate::core::plans::{Heir, InheritancePlan, PaymentStream, ACTIVATION_HEIGHT, DAY};
use midstate::core::transaction::apply_transaction;
use midstate::core::types::{compute_address, compute_commitment, decompose_value, is_v2_at, OutputData, State, Transaction};
use midstate::wallet::Wallet;
use midstate_walletd::plans::{
    claim_inputs, fee_for, free_indices, live_plan_coins, live_tranches, owner_inputs, payout_inputs, payouts_in,
    plan_denominations, reveal_size, shares_from_bps, tranche_inputs, witnesses, CovInput,
};
use midstate_walletd::sendplan;
use tempfile::TempDir;

const HEIGHT: u64 = ACTIVATION_HEIGHT + 10_000;

struct Chain(State);

impl Chain {
    fn at(height: u64) -> Self {
        Chain(State {
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
        })
    }

    fn give(&mut self, w: &mut Wallet, value: u64) {
        let id = w.import_coin(rand::random(), value, rand::random(), None).unwrap();
        assert!(self.0.coins.insert(id, is_v2_at(self.0.height)));
    }

    fn commit(&mut self, inputs: &[[u8; 32]], outputs: &[OutputData], salt: &[u8; 32]) -> [u8; 32] {
        let outs: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();
        let c = compute_commitment(inputs, &outs, salt);
        self.0.commitments.insert(c, is_v2_at(self.0.height));
        self.0.commitment_heights.insert(c, self.0.height - 1);
        c
    }

    fn reveal(&mut self, tx: &Transaction) -> Result<()> {
        let bytes = bincode::serialized_size(tx)? as u128;
        if (tx.fee() as u128) * 1_024 < 10 * bytes {
            bail!("fee {} below the relay floor for {} bytes", tx.fee(), bytes);
        }
        let mut next = self.0.clone();
        apply_transaction(&mut next, tx)?;
        self.0 = next;
        Ok(())
    }

    fn live(&self) -> impl Fn(&[u8; 32]) -> bool + '_ {
        |id| self.0.coins.contains(id)
    }

    /// A send of `w`'s own coins paying `outputs`, signed by its reveal path.
    fn fund(&mut self, w: &mut Wallet, outputs: Vec<OutputData>) {
        let live: Vec<[u8; 32]> = w.coins().iter().filter(|c| !c.wots_signed).map(|c| c.coin_id).collect();
        let plan = sendplan::plan_fixed_outputs(w, &live, outputs).unwrap();
        let (c, _) = w.prepare_commit(&plan.input_coin_ids, &plan.outputs, plan.change_seeds, false, false).unwrap();
        let pending = w.find_pending(&c).unwrap().clone();
        assert_eq!(self.commit(&pending.input_coin_ids, &pending.outputs, &pending.salt), c);
        let (inputs, witnesses) = w.sign_reveal(&pending).unwrap();
        let tx = Transaction::Reveal { inputs, witnesses, outputs: pending.outputs.clone(), salt: pending.salt };
        self.reveal(&tx).expect("a funding is an ordinary send");
        w.complete_reveal(&c).unwrap();
    }

    /// A covenant send: `inputs` opened as recorded, signed by `signer`.
    fn spend(&mut self, inputs: &[CovInput], outputs: Vec<OutputData>, signer: Option<(&mut Wallet, [u8; 32])>) -> Result<Transaction> {
        let ids: Vec<[u8; 32]> = inputs.iter().map(|i| i.reveal.coin_id()).collect();
        let salt: [u8; 32] = rand::random();
        let c = self.commit(&ids, &outputs, &salt);
        let sig = match signer {
            Some((w, pk)) => Some(w.auto_sign(&pk, &c)?),
            None => None,
        };
        let tx = Transaction::Reveal {
            inputs: inputs.iter().map(|i| i.reveal.clone()).collect(),
            witnesses: witnesses(inputs, sig.as_deref())?,
            outputs,
            salt,
        };
        self.reveal(&tx)?;
        Ok(tx)
    }
}

fn wallet(dir: &TempDir, name: &str) -> Wallet {
    Wallet::create(&dir.path().join(name), b"pass").unwrap()
}

/// A fresh MSS tree of `w`'s (short: a test signs a handful of times).
fn tree(w: &mut Wallet) -> [u8; 32] {
    let address = w.generate_mss(2, None).unwrap();
    w.mss_keys().iter().find(|k| compute_address(&k.master_pk) == address).unwrap().master_pk
}

fn to_plain(value: u64) -> Vec<OutputData> {
    decompose_value(value)
        .into_iter()
        .map(|v| OutputData::Standard { address: [0x5a; 32], value: v, salt: rand::random() })
        .collect()
}

#[test]
fn a_plan_is_funded_checked_in_claimed_after_its_deadline_and_swept() {
    let dir = TempDir::new().unwrap();
    let mut owner = wallet(&dir, "owner.dat");
    let (mut ann, mut ben) = (wallet(&dir, "ann.dat"), wallet(&dir, "ben.dat"));
    let mut chain = Chain::at(HEIGHT);
    chain.give(&mut owner, 1 << 40);

    let shares = shares_from_bps(&[6_000, 4_000]).unwrap();
    let (ann_pk, ben_pk) = (tree(&mut ann), tree(&mut ben));
    let owner_pk = tree(&mut owner);
    let plan = InheritancePlan {
        owner_pk,
        heirs: vec![Heir { pk: ann_pk, share: shares[0] }, Heir { pk: ben_pk, share: shares[1] }],
        start: HEIGHT,
        period: 365 * DAY,
        secret: rand::random(),
    };
    plan.check().unwrap();
    let amount = (1u64 << 39) + (1 << 37) + 12_345;
    let denoms = plan_denominations(amount);
    let k = plan.epoch_after(HEIGHT);
    chain.fund(&mut owner, denoms.iter().enumerate().map(|(n, v)| plan.output(k, n as u32, *v)).collect());
    let coins = live_plan_coins(&plan, HEIGHT, chain.live());
    assert_eq!(coins.iter().map(|c| c.value).sum::<u64>(), denoms.iter().sum::<u64>());

    // Nobody but the owner, before the deadline.
    chain.0.height = plan.deadline(k) - 1;
    let total: u64 = coins.iter().map(|c| c.value).sum();
    let early = chain.spend(&claim_inputs(&plan, &coins), plan.claim_outputs(k, total, rand::random::<[u8; 32]>), None);
    assert!(early.is_err(), "claimed before the deadline");

    // The owner checks in half a year later: a full period from then.
    chain.0.height = HEIGHT + 180 * DAY;
    let inputs = owner_inputs(&plan, &coins);
    let fee = fee_for(reveal_size(&inputs, midstate_walletd::bond::mss_sig_len(2), 64));
    let to = plan.epoch_after(chain.0.height);
    assert!(plan.deadline(to) >= chain.0.height + plan.period);
    let keep = plan_denominations(total - fee);
    let back = total - fee - keep.iter().sum::<u64>();
    let mut outputs: Vec<OutputData> =
        keep.iter().zip(free_indices(&coins, to, keep.len())).map(|(v, n)| plan.output(to, n, *v)).collect();
    outputs.extend(to_plain(back));
    chain.spend(&inputs, outputs, Some((&mut owner, owner_pk))).expect("a check-in");
    let moved = live_plan_coins(&plan, chain.0.height, chain.live());
    assert!(moved.iter().all(|c| c.k == to));
    let total: u64 = moved.iter().map(|c| c.value).sum();

    // The owner goes quiet. From the new deadline anyone pays the heirs.
    chain.0.height = plan.deadline(to);
    let tx = chain
        .spend(&claim_inputs(&plan, &moved), plan.claim_outputs(to, total, rand::random::<[u8; 32]>), None)
        .expect("a claim from the deadline, by anyone");
    assert!(live_plan_coins(&plan, chain.0.height, chain.live()).is_empty());
    let ann_payouts = payouts_in(&tx, &[(plan.payout_tag(to, 0), ann_pk)], "test");
    let ben_payouts = payouts_in(&tx, &[(plan.payout_tag(to, 1), ben_pk)], "test");
    let unit = total / midstate::core::plans::SHARE_DENOM;
    assert_eq!(ann_payouts.iter().map(|p| p.value).sum::<u64>(), unit * shares[0] as u64);
    assert_eq!(ben_payouts.iter().map(|p| p.value).sum::<u64>(), unit * shares[1] as u64);

    // Each heir sweeps its own with its own key, and only its own.
    let got: u64 = ann_payouts.iter().map(|p| p.value).sum();
    let sweep = payout_inputs(&ann_payouts);
    let fee = fee_for(reveal_size(&sweep, midstate_walletd::bond::mss_sig_len(2), 8));
    assert!(chain.spend(&sweep, to_plain(got - fee), Some((&mut ben, ben_pk))).is_err(), "Ben swept Ann's");
    chain.spend(&sweep, to_plain(got - fee), Some((&mut ann, ann_pk))).expect("Ann's sweep");
    let got: u64 = ben_payouts.iter().map(|p| p.value).sum();
    let sweep = payout_inputs(&ben_payouts);
    chain.spend(&sweep, to_plain(got - fee), Some((&mut ben, ben_pk))).expect("Ben's sweep");
}

#[test]
fn a_stream_pays_what_is_due_and_its_payer_takes_back_the_rest() {
    let dir = TempDir::new().unwrap();
    let mut payer = wallet(&dir, "payer.dat");
    let mut payee = wallet(&dir, "payee.dat");
    let mut chain = Chain::at(HEIGHT);
    chain.give(&mut payer, 1 << 40);
    let (payer_pk, payee_pk) = (tree(&mut payer), tree(&mut payee));
    let s = PaymentStream {
        payer_pk: Some(payer_pk),
        recipient_pk: payee_pk,
        first: HEIGHT,
        period: 7 * DAY,
        amount: (1 << 34) + (1 << 32),
        count: 4,
        secret: rand::random(),
    };
    s.check().unwrap();
    chain.fund(&mut payer, s.funding_outputs());
    assert_eq!(live_tranches(&s, chain.live()).len(), 8);

    // A week in, two payments are due: anyone collects them for the payee.
    chain.0.height = s.due(1);
    let due: Vec<(u32, u32, u64)> = live_tranches(&s, chain.live()).into_iter().filter(|(j, _, _)| s.due(*j) <= chain.0.height).collect();
    assert_eq!(due.len(), 4);
    let mut outputs = Vec::new();
    for j in [0, 1] {
        outputs.extend(s.collect_outputs(j, s.amount, rand::random::<[u8; 32]>));
    }
    let tx = chain.spend(&tranche_inputs(&s, &due, false), outputs, None).expect("collecting what is due");
    let pairs: Vec<([u8; 32], [u8; 32])> = (0..s.count).map(|j| (s.payout_tag(j), payee_pk)).collect();
    let payouts = payouts_in(&tx, &pairs, "test");
    assert_eq!(payouts.iter().map(|p| p.value).sum::<u64>(), 2 * s.owed(s.amount));

    // The payer stops the stream: what is not yet due comes back.
    let rest: Vec<(u32, u32, u64)> = live_tranches(&s, chain.live());
    assert!(rest.iter().all(|(j, _, _)| s.due(*j) > chain.0.height));
    let inputs = tranche_inputs(&s, &rest, true);
    let total: u64 = rest.iter().map(|c| c.2).sum();
    let fee = fee_for(reveal_size(&inputs, midstate_walletd::bond::mss_sig_len(2), 8));
    chain.spend(&inputs, to_plain(total - fee), Some((&mut payer, payer_pk))).expect("a cancel");
    assert!(live_tranches(&s, chain.live()).is_empty());

    // And the payee sweeps what it was paid.
    let got: u64 = payouts.iter().map(|p| p.value).sum();
    let sweep = payout_inputs(&payouts);
    let fee = fee_for(reveal_size(&sweep, midstate_walletd::bond::mss_sig_len(2), 8));
    chain.spend(&sweep, to_plain(got - fee), Some((&mut payee, payee_pk))).expect("the payee's sweep");
}

#[test]
fn a_stream_that_cannot_be_cancelled_cannot_be() {
    let dir = TempDir::new().unwrap();
    let mut payer = wallet(&dir, "payer.dat");
    let mut chain = Chain::at(HEIGHT);
    chain.give(&mut payer, 1 << 40);
    let payer_pk = tree(&mut payer);
    let s = PaymentStream {
        payer_pk: None,
        recipient_pk: [0x77; 32],
        first: HEIGHT + DAY,
        period: DAY,
        amount: 1 << 33,
        count: 3,
        secret: rand::random(),
    };
    chain.fund(&mut payer, s.funding_outputs());
    let all = live_tranches(&s, chain.live());
    let mut inputs = tranche_inputs(&s, &all, false);
    for i in &mut inputs {
        i.opening = midstate_walletd::plans::Opening::Signed(vec!["01".into()]);
    }
    let r = chain.spend(&inputs, to_plain((1 << 33) * 3 - 4096), Some((&mut payer, payer_pk)));
    assert!(r.is_err(), "taken back");
    // Nor collected early.
    let r = chain.spend(&tranche_inputs(&s, &all[..1], false), s.collect_outputs(0, 1 << 33, rand::random::<[u8; 32]>), None);
    assert!(r.is_err(), "collected before it was due");
}
