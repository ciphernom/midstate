//! MSB-1 burns end to end: a burn built by the wallet's planning code and
//! signed by a real wallet, applied with `apply_transaction` (after the
//! relay's fee floor) to a stand-in chain, then read by the node's burn index
//! from the block it would sit in.

use midstate::core::burns::{self, Backing, Kind, Purpose};
use midstate::core::mmr::{MerkleMountainRange, UtxoAccumulator};
use midstate::core::transaction::apply_transaction;
use midstate::core::types::{block_reward, is_v2_at, Batch, Extension, OutputData, State, Transaction};
use midstate::storage::burn_index;
use midstate::wallet::Wallet;
use midstate_walletd::sendplan;
use tempfile::TempDir;

const HEIGHT: u64 = 600_000;

fn state() -> State {
    State {
        midstate: [0; 32],
        coins: UtxoAccumulator::new(),
        commitments: UtxoAccumulator::new(),
        depth: 0,
        target: [0xff; 32],
        height: HEIGHT,
        timestamp: 0,
        commitment_heights: Default::default(),
        expirations: Default::default(),
        chain_mmr: MerkleMountainRange::new(),
        header_hash: [0; 32],
        burned_wots: UtxoAccumulator::new(),
        coinbase_window: Default::default(),
    }
}

#[test]
fn a_wallet_burn_passes_consensus_and_the_index_reads_it_as_backing() {
    let dir = TempDir::new().unwrap();
    let mut w = Wallet::create(&dir.path().join("w.dat"), b"pass").unwrap();
    let mut s = state();
    let id = w.import_coin(rand::random(), 1 << 40, rand::random(), None).unwrap();
    assert!(s.coins.insert(id, is_v2_at(s.height)));

    // A day of issuance for nick.mds, and a notarization for the fee alone,
    // in one send.
    let nick = Backing::name("nick", "for the shop").unwrap();
    let doc = Backing::new(Purpose::NOTARY, *blake3::hash(b"a contract").as_bytes(), "contract v1").unwrap();
    let amount = burns::cost(1_440, HEIGHT);
    let mut outputs = nick.outputs(amount);
    outputs.push(OutputData::DataBurn { payload: doc.payload(), value_burned: 0 });
    let live: Vec<[u8; 32]> = w.coins().iter().map(|c| c.coin_id).collect();
    let plan = sendplan::plan_fixed_outputs(&mut w, &live, outputs).unwrap();
    let change: u64 = plan.outputs.iter().filter(|o| matches!(o, OutputData::Standard { .. })).map(|o| o.value()).sum();
    assert_eq!(change + amount + plan.fee, 1 << 40, "the burn is paid for, not taken from the change");
    let (c, _) = w.prepare_commit(&plan.input_coin_ids, &plan.outputs, plan.change_seeds, false, false).unwrap();
    let pending = w.find_pending(&c).unwrap().clone();
    s.commitments.insert(c, is_v2_at(s.height));
    s.commitment_heights.insert(c, s.height - 1);
    let (inputs, witnesses) = w.sign_reveal(&pending).unwrap();
    let tx = Transaction::Reveal { inputs, witnesses, outputs: pending.outputs.clone(), salt: pending.salt };
    let bytes = bincode::serialized_size(&tx).unwrap() as u128;
    assert!((tx.fee() as u128) * 1_024 >= 10 * bytes, "pays the relay floor");
    apply_transaction(&mut s, &tx).expect("a burn is an ordinary send");

    // The block it lands in, as the burn index reads it.
    let batch = Batch {
        prev_midstate: [0; 32],
        transactions: vec![tx],
        extension: Extension { nonce: 0, final_hash: [1; 32] },
        coinbase: vec![],
        timestamp: 1_700_000_000,
        target: [0xff; 32],
        state_root: [0; 32],
        prev_header_hash: [0; 32],
    };
    let rows = burn_index::rows_of(&batch);
    assert_eq!(rows.len(), 2, "one row a payload, its outputs summed");
    let backing = rows.iter().find(|r| Kind::of(&r.payload) == Kind::Backing && r.value > 0).unwrap();
    assert_eq!(Backing::parse(&backing.payload), Some(nick));
    assert_eq!(backing.value, amount);
    assert!((burns::weight(backing.value, HEIGHT) - 1_440.0).abs() < 1e-9, "a day of issuance weighs 1,440 blocks");
    assert_eq!(amount, 1_440 * block_reward(HEIGHT));
    let notary = rows.iter().find(|r| r.value == 0).unwrap();
    assert_eq!(Backing::parse(&notary.payload), Some(doc));
}
