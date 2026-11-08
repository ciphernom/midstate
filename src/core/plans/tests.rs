//! Inheritance plans and payment streams through consensus: every path, and
//! the spends each must refuse.

use super::*;
use crate::core::mmr::UtxoAccumulator;
use crate::core::mss;
use crate::core::script;
use crate::core::transaction::apply_transaction;
use crate::core::types::{compute_address, compute_commitment, is_v2_at, State, Transaction};
use crate::core::wots;

const HEIGHT: u64 = ACTIVATION_HEIGHT + 10_000;
const PERIOD: u64 = 365 * DAY;

fn state_at(height: u64) -> State {
    State {
        midstate: [0u8; 32],
        coins: UtxoAccumulator::new(),
        commitments: UtxoAccumulator::new(),
        depth: 0,
        target: [0xff; 32],
        height,
        timestamp: 1_790_000_000,
        commitment_heights: im::HashMap::new(),
        expirations: im::OrdMap::new(),
        header_hash: [0u8; 32],
        chain_mmr: crate::core::mmr::MerkleMountainRange::new(),
        burned_wots: UtxoAccumulator::new(),
    }
}

/// An MSS key for tests (a short tree: signing is all that is checked). Only
/// parties that sign get one; keygen is slow in debug builds.
fn key(tag: &[u8]) -> mss::MssKeypair {
    mss::keygen(&hash(tag), 3).unwrap()
}

/// A key nobody in the test signs with.
fn pk(tag: &[u8]) -> [u8; 32] {
    hash(&[b"pk:".as_slice(), tag].concat())
}

fn sign(k: &mut mss::MssKeypair, c: &[u8; 32]) -> Vec<u8> {
    k.sign(c).unwrap().to_bytes()
}

fn coin(state: &mut State, input: &InputReveal) {
    state.coins.insert(input.coin_id(), is_v2_at(state.height));
}

fn commit(state: &mut State, inputs: &[InputReveal], outputs: &[OutputData], salt: &[u8; 32]) -> [u8; 32] {
    let ins: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
    let outs: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();
    let c = compute_commitment(&ins, &outs, salt);
    state.commitments.insert(c, is_v2_at(state.height));
    state.commitment_heights.insert(c, state.height - 1);
    c
}

/// A reveal of `inputs` with the witnesses `wit(commitment)` gives.
fn reveal(
    state: &mut State,
    inputs: Vec<InputReveal>,
    outputs: Vec<OutputData>,
    salt: u8,
    wit: impl FnOnce(&[u8; 32]) -> Vec<Witness>,
) -> Transaction {
    let tx_salt = [salt; 32];
    let c = commit(state, &inputs, &outputs, &tx_salt);
    Transaction::Reveal { inputs, witnesses: wit(&c), outputs, salt: tx_salt }
}

fn ok(state: &State, tx: &Transaction) -> bool {
    apply_transaction(&mut state.clone(), tx).is_ok()
}

fn salter(seed: u8) -> impl FnMut() -> [u8; 32] {
    let mut n = 0u32;
    move || {
        n += 1;
        hash(&[[seed; 4].as_slice(), &n.to_le_bytes()].concat())
    }
}

fn plain(tag: &[u8]) -> [u8; 32] {
    compute_address(&wots::keygen(&hash(tag)))
}

fn plan_for(owner: [u8; 32], heirs: &[([u8; 32], u32)]) -> InheritancePlan {
    InheritancePlan {
        owner_pk: owner,
        heirs: heirs.iter().map(|(k, share)| Heir { pk: *k, share: *share }).collect(),
        start: HEIGHT - 100,
        period: PERIOD,
        secret: hash(b"plan secret"),
    }
}

/// Coins `values` at deadline `k`, on chain.
fn fund_plan(state: &mut State, p: &InheritancePlan, k: u64, values: &[u64]) -> Vec<InputReveal> {
    values
        .iter()
        .enumerate()
        .map(|(n, v)| {
            let i = p.input(k, n as u32, *v);
            coin(state, &i);
            i
        })
        .collect()
}

const HALF: u32 = 32_768;
const THIRD: u32 = 21_845; // three of them are all 65,535 parts

// ── Shape ───────────────────────────────────────────────────────────────────

#[test]
fn scripts_are_well_formed_and_fit() {
    let owner = pk(b"owner");
    let heirs: Vec<[u8; 32]> = (0..MAX_HEIRS as u8).map(|i| pk(&[b'h', i])).collect();
    let p = InheritancePlan {
        owner_pk: owner,
        heirs: heirs.iter().enumerate().map(|(i, k)| Heir { pk: *k, share: if i == 0 { 8_198 } else { 8_191 } }).collect(),
        start: HEIGHT,
        period: PERIOD,
        secret: [7; 32],
    };
    p.check().unwrap();
    let bc = p.script(MAX_EPOCHS);
    assert!(bc.len() <= script::MAX_SCRIPT_SIZE, "{} bytes", bc.len());
    script::validate_structure(&bc, HEIGHT).unwrap();

    let s = PaymentStream {
        payer_pk: Some(owner),
        recipient_pk: heirs[0],
        first: HEIGHT,
        period: 30 * DAY,
        amount: 3 << 30,
        count: 12,
        secret: [8; 32],
    };
    s.check().unwrap();
    script::validate_structure(&s.script(11), HEIGHT).unwrap();
    script::validate_structure(&PaymentStream { payer_pk: None, ..s.clone() }.script(0), HEIGHT).unwrap();
    script::validate_structure(&payout_script(&[1; 32], &owner), HEIGHT).unwrap();
    // Every payout address is distinct: by deadline, by heir, by tranche.
    let mut seen = std::collections::HashSet::new();
    for k in 1..4 {
        for i in 0..p.heirs.len() {
            assert!(seen.insert(p.payout_address(k, i)));
        }
    }
    for j in 0..s.count {
        assert!(seen.insert(s.payout_address(j)));
        assert!(seen.insert(s.address(j)));
    }
}

#[test]
fn terms_round_trip_and_bad_ones_are_refused() {
    let (o, a, b) = (pk(b"o"), pk(b"a"), pk(b"b"));
    let p = plan_for(o, &[(a, HALF), (b, HALF - 1)]);
    p.check().unwrap();
    assert_eq!(InheritancePlan::from_bytes(&p.to_bytes()), Some(p.clone()));
    let mut cut = p.to_bytes();
    cut.pop();
    assert_eq!(InheritancePlan::from_bytes(&cut), None);
    let bad = |q: InheritancePlan| assert!(q.check().is_err() && InheritancePlan::from_bytes(&q.to_bytes()).is_none());
    bad(InheritancePlan { heirs: vec![], ..p.clone() });
    bad(plan_for(o, &[(a, HALF), (b, HALF)])); // leaves nothing for the fee
    bad(plan_for(o, &[(a, HALF)])); // leaves half unowed
    plan_for(o, &[(a, THIRD), (b, THIRD), (pk(b"c"), THIRD)]).check().unwrap();
    bad(plan_for(o, &[(a, 0), (b, (SHARE_DENOM - 1) as u32)]));
    bad(plan_for(o, &[(a, HALF), (a, HALF - 1)]));
    bad(plan_for(o, &[(o, (SHARE_DENOM - 1) as u32)]));
    bad(InheritancePlan { period: DAY - 1, ..p.clone() });
    let many: Vec<[u8; 32]> = (0..=MAX_HEIRS as u8).map(|i| pk(&[b'm', i])).collect();
    let n = many.len() as u32;
    bad(plan_for(o, &many.iter().enumerate().map(|(i, k)| (*k, if i == 0 { 65_535 - (n - 1) } else { 1 })).collect::<Vec<_>>()));

    let s = PaymentStream {
        payer_pk: None,
        recipient_pk: a,
        first: HEIGHT,
        period: 7 * DAY,
        amount: MIN_COIN,
        count: 52,
        secret: [3; 32],
    };
    s.check().unwrap();
    assert_eq!(PaymentStream::from_bytes(&s.to_bytes()), Some(s.clone()));
    let c = PaymentStream { payer_pk: Some(o), ..s.clone() };
    assert_eq!(PaymentStream::from_bytes(&c.to_bytes()), Some(c.clone()));
    for t in [
        PaymentStream { amount: MIN_COIN - 1, ..s.clone() },
        PaymentStream { amount: MIN_COIN * 31, ..s.clone() }, // five coins
        PaymentStream { count: 0, ..s.clone() },
        PaymentStream { count: MAX_TRANCHES + 1, ..s.clone() },
        PaymentStream { amount: MIN_COIN * 15, count: 51, ..s.clone() }, // 204 outputs
        PaymentStream { period: 0, ..s.clone() },
        PaymentStream { payer_pk: Some(a), ..s.clone() },
    ] {
        assert!(t.check().is_err(), "{t:?}");
        assert!(PaymentStream::from_bytes(&t.to_bytes()).is_none());
    }
}

#[test]
fn the_schedule_always_leaves_a_full_period() {
    let p = plan_for(pk(b"o"), &[(pk(b"a"), HALF)]);
    for h in [p.start, p.start + 1, p.start + PERIOD - 1, p.start + PERIOD, p.start + 7 * PERIOD + 3] {
        let k = p.epoch_after(h);
        assert!(p.deadline(k) >= h + PERIOD, "at {h}");
        assert!(k == 1 || p.deadline(k - 1) < h + PERIOD, "at {h}: not the first such deadline");
        assert!(p.epochs_to(h).contains(&k));
    }
}

#[test]
fn the_fee_share_covers_a_transaction_and_the_sums_never_overflow() {
    let p = plan_for(pk(b"o"), &[(pk(b"a"), (SHARE_DENOM - 1) as u32)]);
    for total in [MIN_COIN, MIN_COIN + 1, u64::MAX / 2, u64::MAX] {
        let owed: u64 = p.owed(1, total).iter().map(|(_, v)| v).sum();
        assert!(owed <= total && total - owed >= MIN_COIN / SHARE_DENOM, "{total}");
        let s = PaymentStream {
            payer_pk: None,
            recipient_pk: [1; 32],
            first: 0,
            period: 1,
            amount: MIN_COIN,
            count: 1,
            secret: [0; 32],
        };
        assert!(total - s.owed(total) >= 512);
    }
}

// ── Inheritance ─────────────────────────────────────────────────────────────

#[test]
fn the_owner_spends_at_any_time_with_one_signature_for_every_coin() {
    let mut owner = key(b"owner");
    let p = plan_for(owner.master_pk, &[(pk(b"a"), HALF), (pk(b"b"), HALF - 1)]);
    for height in [HEIGHT, p.deadline(1), p.deadline(1) + 5 * PERIOD] {
        let mut s = state_at(height);
        let coins = fund_plan(&mut s, &p, 1, &[1 << 33, 1 << 30]);
        // A check-in: everything, less a fee, to the next deadline's address.
        let k2 = p.epoch_after(height);
        let outputs = vec![p.output(k2, 0, 1 << 33), p.output(k2, 1, (1 << 30) - (1 << 29)), p.output(k2, 2, 1 << 28)];
        let tx = reveal(&mut s, coins.clone(), outputs, 1, |c| {
            let sig = sign(&mut owner, c);
            vec![InheritancePlan::owner_witness(sig.clone()), InheritancePlan::owner_witness(sig)]
        });
        assert!(ok(&s, &tx), "a check-in at {height}");
        // Someone else's key, or the owner's signature without its flag, cannot.
        let mut thief = mss::keygen(&hash(b"thief"), 2).unwrap();
        let forged = reveal(&mut s, coins.clone(), vec![OutputData::Standard { address: plain(b"t"), value: 1 << 33, salt: [1; 32] }], 2, |c| {
            vec![InheritancePlan::owner_witness(sign(&mut thief, c)); 2]
        });
        assert!(!ok(&s, &forged));
        let unflagged = reveal(&mut s, coins, vec![OutputData::Standard { address: plain(b"t"), value: 1 << 33, salt: [1; 32] }], 3, |c| {
            vec![Witness::ScriptInputs(vec![sign(&mut owner, c)]); 2]
        });
        assert!(!ok(&s, &unflagged));
    }
}

#[test]
fn heirs_are_paid_their_shares_from_the_deadline_and_not_before() {
    let a = pk(b"a");
    let p = plan_for(pk(b"owner"), &[(a, HALF), (pk(b"b"), HALF - 1)]);
    p.check().unwrap();
    let total = (1u64 << 34) + (1 << 31);

    // A block before the deadline, nothing moves without the owner.
    let mut early = state_at(p.deadline(1) - 1);
    let coins = fund_plan(&mut early, &p, 1, &[1 << 34, 1 << 31]);
    let claim = |s: &mut State, outputs: Vec<OutputData>, salt: u8| {
        reveal(s, coins.clone(), outputs, salt, |_| vec![InheritancePlan::claim_witness(); 2])
    };
    let exact = p.claim_outputs(1, total, salter(1));
    let tx = claim(&mut early, exact.clone(), 1);
    assert!(!ok(&early, &tx), "claimed before the deadline");

    let mut s = state_at(p.deadline(1));
    fund_plan(&mut s, &p, 1, &[1 << 34, 1 << 31]);
    let tx = claim(&mut s, exact.clone(), 2);
    assert!(ok(&s, &tx), "anyone may pay the heirs from the deadline");
    let owed = p.owed(1, total);
    assert_eq!(owed[0].1, total / SHARE_DENOM * HALF as u64);
    assert!(total - owed.iter().map(|(_, v)| v).sum::<u64>() >= MIN_COIN / SHARE_DENOM);

    // One unit short for an heir, an heir paid at its plain key's address,
    // or the claim spending only part of the plan with the whole plan's
    // outputs scaled down: refused.
    let mut short = p.owed(1, total);
    short[1].1 -= 1;
    let outs = |o: &[([u8; 32], u64)], seed: u8| -> Vec<OutputData> {
        let mut salt = salter(seed);
        o.iter().flat_map(|(a, v)| super::coins_to(*a, *v, &mut salt)).collect()
    };
    let tx = claim(&mut s, outs(&short, 3), 3);
    assert!(!ok(&s, &tx), "an heir shorted by one unit");
    let mut wrong = p.owed(1, total);
    wrong[0].0 = compute_address(&a);
    let tx = claim(&mut s, outs(&wrong, 4), 4);
    assert!(!ok(&s, &tx), "an heir paid outside its payout address");

    // The builder may keep what the fee does not need, and no more: the
    // outputs cannot exceed the inputs.
    let mut greedy = exact.clone();
    greedy.push(OutputData::Standard { address: plain(b"builder"), value: 1 << 15, salt: [9; 32] });
    let tx = claim(&mut s, greedy, 5);
    assert!(ok(&s, &tx), "the leftover is the builder's");
    let mut greedier = exact;
    greedier.push(OutputData::Standard { address: plain(b"builder"), value: 1 << 30, salt: [9; 32] });
    let tx = claim(&mut s, greedier, 6);
    assert!(!ok(&s, &tx), "more than the leftover");
}

#[test]
fn one_claim_cannot_pay_two_deadlines_or_two_plans_with_the_same_outputs() {
    let (a, b) = (pk(b"a"), pk(b"b"));
    let mum = plan_for(pk(b"mum"), &[(a, HALF), (b, HALF - 1)]);
    let dad = InheritancePlan { secret: hash(b"dad's secret"), ..plan_for(pk(b"dad"), &[(a, HALF), (b, HALF - 1)]) };
    let mut s = state_at(mum.deadline(2));
    let big = fund_plan(&mut s, &mum, 1, &[1 << 34]);
    let small = fund_plan(&mut s, &mum, 2, &[1 << 30]);
    let other = fund_plan(&mut s, &dad, 1, &[1 << 31]);

    // Two deadlines of one plan: each must be paid at its own addresses.
    let only_big = mum.claim_outputs(1, 1 << 34, salter(1));
    let tx = reveal(&mut s, vec![big[0].clone(), small[0].clone()], only_big.clone(), 1, |_| {
        vec![InheritancePlan::claim_witness(); 2]
    });
    assert!(!ok(&s, &tx), "the smaller deadline skimmed");
    let mut both = only_big.clone();
    both.extend(mum.claim_outputs(2, 1 << 30, salter(2)));
    let tx = reveal(&mut s, vec![big[0].clone(), small[0].clone()], both, 2, |_| {
        vec![InheritancePlan::claim_witness(); 2]
    });
    assert!(ok(&s, &tx));

    // Two plans with the same heirs: the same.
    let tx = reveal(&mut s, vec![big[0].clone(), other[0].clone()], only_big.clone(), 3, |_| {
        vec![InheritancePlan::claim_witness(); 2]
    });
    assert!(!ok(&s, &tx), "the other plan skimmed");
    let mut both = only_big;
    both.extend(dad.claim_outputs(1, 1 << 31, salter(3)));
    let tx = reveal(&mut s, vec![big[0].clone(), other[0].clone()], both, 4, |_| {
        vec![InheritancePlan::claim_witness(); 2]
    });
    assert!(ok(&s, &tx));
}

#[test]
fn an_heir_sweeps_its_payout_with_its_key_and_nobody_else_can() {
    let mut a = mss::keygen(&hash(b"a"), 2).unwrap();
    let p = plan_for(pk(b"owner"), &[(a.master_pk, HALF), (pk(b"b"), HALF - 1)]);
    let mut s = state_at(p.deadline(1));
    let payout = payout_input(&p.payout_tag(1, 0), &a.master_pk, 1 << 30, [5; 32]);
    coin(&mut s, &payout);
    let out = vec![OutputData::Standard { address: plain(b"a's wallet"), value: 1 << 29, salt: [6; 32] }];
    let mut thief = mss::keygen(&hash(b"thief"), 2).unwrap();
    let tx = reveal(&mut s, vec![payout.clone()], out.clone(), 1, |c| vec![payout_witness(sign(&mut thief, c))]);
    assert!(!ok(&s, &tx));
    let tx = reveal(&mut s, vec![payout], out, 2, |c| vec![payout_witness(sign(&mut a, c))]);
    assert!(ok(&s, &tx));
}

// ── Streams ─────────────────────────────────────────────────────────────────

fn stream(payer: Option<[u8; 32]>, recipient: [u8; 32]) -> PaymentStream {
    PaymentStream {
        payer_pk: payer,
        recipient_pk: recipient,
        first: HEIGHT,
        period: 30 * DAY,
        amount: (1 << 32) + (1 << 30),
        count: 6,
        secret: hash(b"stream secret"),
    }
}

fn fund_stream(state: &mut State, st: &PaymentStream) {
    for o in st.funding_outputs() {
        if let OutputData::Standard { address, value, salt } = o {
            state.coins.insert(compute_coin_id(&address, value, &salt), is_v2_at(state.height));
        }
    }
}

#[test]
fn funding_lays_out_every_tranche_where_the_recipient_looks() {
    let st = stream(None, pk(b"r"));
    let outs = st.funding_outputs();
    assert_eq!(outs.len(), st.count as usize * st.tranche_coins().len());
    assert_eq!(outs.iter().map(|o| o.value()).sum::<u64>(), st.total());
    let ids: Vec<[u8; 32]> = (0..st.count).flat_map(|j| st.coins(j)).map(|(_, _, id)| id).collect();
    for o in &outs {
        assert!(ids.contains(&o.coin_id().unwrap()));
    }
}

#[test]
fn a_tranche_is_collected_for_the_recipient_once_due_and_not_before() {
    let r = pk(b"recipient");
    let st = stream(Some(pk(b"payer")), r);
    let j = 2;
    let ins = |st: &PaymentStream| (0..st.tranche_coins().len() as u32).map(|n| st.input(j, n)).collect::<Vec<_>>();
    let collect = |s: &mut State, outputs: Vec<OutputData>, salt: u8| {
        reveal(s, ins(&st), outputs, salt, |_| vec![st.collect_witness(); 2])
    };
    let mut early = state_at(st.due(j) - 1);
    fund_stream(&mut early, &st);
    let tx = collect(&mut early, st.collect_outputs(j, st.amount, salter(1)), 1);
    assert!(!ok(&early, &tx), "collected early");

    let mut s = state_at(st.due(j));
    fund_stream(&mut s, &st);
    let tx = collect(&mut s, st.collect_outputs(j, st.amount, salter(2)), 2);
    assert!(ok(&s, &tx), "anyone may collect for the recipient once due");

    let short = super::coins_to(st.payout_address(j), st.owed(st.amount) - 1, &mut salter(3));
    let tx = collect(&mut s, short, 3);
    assert!(!ok(&s, &tx), "a unit short");
    let elsewhere = super::coins_to(compute_address(&r), st.owed(st.amount), &mut salter(4));
    let tx = collect(&mut s, elsewhere, 4);
    assert!(!ok(&s, &tx), "paid outside the payout address");
    let next = st.collect_outputs(j + 1, st.amount, salter(5));
    let tx = collect(&mut s, next, 5);
    assert!(!ok(&s, &tx), "paid to another tranche's payout address");
}

#[test]
fn two_tranches_cannot_share_one_payment() {
    let st = stream(None, pk(b"r"));
    let mut s = state_at(st.due(5));
    fund_stream(&mut s, &st);
    let inputs: Vec<InputReveal> = (0..2u32)
        .flat_map(|j| (0..st.tranche_coins().len() as u32).map(move |n| (j, n)))
        .map(|(j, n)| st.input(j, n))
        .collect();
    let n = inputs.len();
    let one = st.collect_outputs(0, st.amount, salter(1));
    let tx = reveal(&mut s, inputs.clone(), one.clone(), 1, |_| vec![st.collect_witness(); n]);
    assert!(!ok(&s, &tx), "tranche 1 skimmed");
    let mut both = one;
    both.extend(st.collect_outputs(1, st.amount, salter(2)));
    let tx = reveal(&mut s, inputs, both, 2, |_| vec![st.collect_witness(); n]);
    assert!(ok(&s, &tx));
}

#[test]
fn a_payer_takes_back_what_is_not_collected_only_if_it_could_cancel() {
    let mut payer = key(b"payer");
    let st = stream(Some(payer.master_pk), pk(b"recipient"));
    let fixed = PaymentStream { payer_pk: None, secret: hash(b"fixed"), ..st.clone() };
    for height in [st.due(0) - 1, st.due(3) + 1] {
        let mut s = state_at(height);
        fund_stream(&mut s, &st);
        fund_stream(&mut s, &fixed);
        let back = vec![OutputData::Standard { address: plain(b"payer"), value: 1 << 32, salt: [2; 32] }];
        let ins: Vec<InputReveal> = (0..2).map(|n| st.input(3, n)).collect();
        let tx = reveal(&mut s, ins, back.clone(), 1, |c| {
            let sig = sign(&mut payer, c);
            vec![PaymentStream::cancel_witness(sig.clone()), PaymentStream::cancel_witness(sig)]
        });
        assert!(ok(&s, &tx), "a cancellable tranche, at {height}");
        let ins: Vec<InputReveal> = (0..2).map(|n| fixed.input(3, n)).collect();
        let tx = reveal(&mut s, ins.clone(), back.clone(), 2, |c| {
            let sig = sign(&mut payer, c);
            vec![PaymentStream::cancel_witness(sig.clone()), PaymentStream::cancel_witness(sig)]
        });
        assert!(!ok(&s, &tx), "a stream that cannot be cancelled was");
        let tx = reveal(&mut s, ins, back, 3, |c| {
            let sig = sign(&mut payer, c);
            vec![Witness::ScriptInputs(vec![sig.clone()]), Witness::ScriptInputs(vec![sig])]
        });
        assert!(!ok(&s, &tx), "nor with a bare signature");
    }
}
