//! Solo-mining MDW from the wallet, end to end on midwimble's devnet:
//! the wallet registers its bond against a midstate node's RPC (a stand-in
//! serving midwimble's devnet midstate), then the local pool's merged mining
//! runs through walletd's merge endpoint, and the block it finds lands on
//! the embedded node carrying the bond's registration and paying the wallet.
//!
//! `cargo test -p midstate-walletd --features devnet --test mwb_mining`
#![cfg(feature = "devnet")]

use axum::extract::{Path, State};
use axum::routing::get;
use axum::{Json, Router};
use midstate_walletd::bond::{BondRecord, RegStatus};
use midstate_walletd::mwb_account::{BlockView, MwbAccount};
use midstate_walletd::mwb_mining::{
    bond_coin, miner_bond, mining_key_of, mining_secret, register_once, ChainNow, MergeServer,
};
use midstate_walletd::mwb_node::{start_mwb_node, MwbNodeConfig};
use midwimble::core::auxpow::{bytes32, ParentTemplate};
use midwimble::core::bond::{
    est_midstate_height, min_mining_bond, min_remaining_bond_lock, registration_cap, registration_min_bond,
    registration_work, BondCoin, BondScript,
};
use midwimble::core::extension::create_extension_with;
use midwimble::core::relay::devnet::DevnetMidstate;
use midwimble::core::state::{calculate_work, current_timestamp};
use midwimble::core::types::{hash, PowAlgo};
use midwimble::core::mw::stealth::WalletKeys;
use midwimble::rpc::RpcClient;
use serde_json::{json, Value};
use std::sync::Arc;

/// Serves a devnet midstate the way midstate's node RPC answers the three
/// requests registration makes.
async fn serve_midstate(chain: Arc<DevnetMidstate>, coin: BondCoin) -> String {
    async fn state(State((c, _)): State<(Arc<DevnetMidstate>, BondCoin)>) -> Json<Value> {
        Json(json!({ "height": c.height() + 1 }))
    }
    async fn proof(State((c, coin)): State<(Arc<DevnetMidstate>, BondCoin)>, Path(id): Path<String>) -> Json<Value> {
        assert_eq!(id, hex::encode(coin.coin_id()), "asked about the bond coin");
        let p = c.prove(&coin);
        Json(json!({
            "height": p.midstate_height,
            "coins_root": hex::encode(p.roots.coins),
            "commitments_root": hex::encode(p.roots.commitments),
            "chain_mmr_root": hex::encode(p.roots.chain_mmr),
            "burned_wots_root": p.roots.burned_wots.map(hex::encode),
            "proof": serde_json::to_value(&p.smt).unwrap(),
            "state_root": hex::encode(p.roots.state_root(p.midstate_height).unwrap()),
        }))
    }
    async fn headers(
        State((c, _)): State<(Arc<DevnetMidstate>, BondCoin)>,
        Path((start, count)): Path<(u64, u64)>,
    ) -> Json<Value> {
        let run = &c.mined;
        let mut prev = run.prev;
        let mut out = Vec::new();
        for (i, h) in run.headers.iter().enumerate() {
            let height = run.start + i as u64;
            if height >= start && height < start + count {
                out.push(json!({
                    "post_tx_midstate": hex::encode(h.post_tx_midstate),
                    "state_root": hex::encode(h.state_root),
                    "timestamp": h.timestamp,
                    "target": hex::encode(h.target),
                    "extension": { "nonce": h.nonce, "final_hash": hex::encode(h.final_hash) },
                    "prev_header_hash": hex::encode(prev),
                }));
            }
            prev = h.final_hash;
        }
        Json(json!({ "headers": out }))
    }
    let app = Router::new()
        .route("/state", get(state))
        .route("/utxo_proof/{id}", get(proof))
        .route("/headers/{start}/{count}", get(headers))
        .with_state((chain, coin));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_wallet_registers_its_bond_and_merge_mines_mdw_with_it() {
    let dir = tempfile::tempdir().unwrap();
    let node = start_mwb_node(MwbNodeConfig {
        data_dir: dir.path().join("midwimble"),
        p2p_port: 0,
        bootstrap: Vec::new(),
        explorer_port: None,
        midstate_rpc: None,
    })
    .await
    .expect("the node starts");
    let h = node.handle.clone();
    let now = current_timestamp();
    let target = h.state().target;

    // The wallet's mining key, from its phrase, and a bond locked on it.
    let secret = mining_secret(&hash(b"a recovery phrase's master seed"));
    let key = mining_key_of(&secret);
    let coin = BondCoin {
        script: BondScript {
            mining_key: key,
            bonded_until: est_midstate_height(now) + min_remaining_bond_lock(now) + 10_000,
            owner_pk: hash(b"the bond's owner key"),
        },
        value: registration_min_bond(&target, now).max(min_mining_bond(now)),
        salt: hash(b"the bond's salt"),
    };

    // Midstate, as its node would serve it: the coin locked in a header
    // (where the proof is taken), then a day of midwimble's work above it.
    let mut chain = DevnetMidstate::new();
    chain.lock(&coin);
    chain.mine(1);
    let required = registration_work(&target);
    let per = calculate_work(&chain.target).min(registration_cap(&target));
    let burial = (required + required / 4).div_ceil(per.max(1)) as u64 + 1;
    let proving = chain.clone();
    let mut halfway = chain.clone();
    halfway.mine(burial / 2);
    chain.mine(burial);
    let chain_now = ChainNow::of(&h.state(), now);
    let run = |served: DevnetMidstate, held| async move {
        let rpc = serve_midstate(Arc::new(served), coin).await;
        tokio::task::spawn_blocking(move || register_once(coin, held, &rpc, &chain_now)).await.unwrap()
    };

    // The first run takes the proof at midstate's tip: nothing above it yet.
    let (file, status) = run(proving, None).await.expect("a first run takes the proof");
    assert_eq!((status.phase.as_str(), status.progress), ("burying", 0), "{status:?}");
    let proof = file.proof.clone().expect("the proof is kept");
    // Halfway buried: it waits, and says how far it has got.
    let (file, status) = run(halfway, Some(file)).await.expect("a later run carries on");
    assert_eq!(status.phase, "burying", "{status:?}");
    assert!(status.progress > 0 && status.progress < 100, "{status:?}");
    // Fully buried: the same proof, now ready.
    let (file, status) = run(chain, Some(file)).await.expect("the last run finishes it");
    assert_eq!(status.phase, "ready", "{status:?}");
    assert_eq!(file.proof.as_ref(), Some(&proof), "a live proof is kept, not retaken");

    // The wallet's record, and the bond it mines with.
    let record = BondRecord {
        mining_key: key,
        bonded_until: coin.script.bonded_until,
        owner_pk: coin.script.owner_pk,
        owner: Default::default(),
        value: coin.value,
        salt: coin.salt,
        label: String::new(),
        created_at: now,
        fund_commitment: None,
        unbond_commitment: None,
        unbond_sig: None,
        owner_next_leaf: 0,
        seen_live: true,
        bond_file: Some(file),
        reg: RegStatus { phase: "ready".into(), ..status },
    };
    assert_eq!(bond_coin(&record), coin);
    let bond = miner_bond(&secret, &key, &[record], now).expect("a ready bond on this key mines");
    assert_eq!(bond.bond_id, coin.coin_id());

    // The local pool's view: ask for a job, plant its commitment in the
    // coinbase's last output, search, hand the share back.
    let merge = MergeServer::start(h.clone(), Some(bond)).await.unwrap();
    let keys = WalletKeys::random();
    let payout = keys.address().encode();
    let rpc = merge.rpc();
    let height_before = h.state().height;
    let found = tokio::task::spawn_blocking(move || {
        let mw = RpcClient::new(rpc);
        let state = mw.get("/state").unwrap();
        assert!(state["tip"].is_string(), "the pool refreshes jobs on this");
        // No run of over-cap blocks at the tip: nothing to build past.
        assert!(state.get("orphaning").is_some_and(Value::is_null), "{state}");
        let job = mw
            .post("/merge/job", &json!({ "payouts": [{ "address": payout, "weight": 1 }] }))
            .expect("/merge/job");
        assert_eq!((&job["prev_hash"], &job["orphans"]), (&state["tip"], &json!(0)), "a job on the tip");
        let commitment = bytes32(&job["commitment"]).unwrap();
        let target = bytes32(&job["target"]).unwrap();
        let mining_hash = bytes32(&job["mining_hash"]).unwrap();
        let algo = PowAlgo::from_name(job["pow"].as_str().unwrap()).unwrap();
        let batch = json!({
            "prev_midstate": hex::encode(hash(b"midstate's previous midstate")),
            "transactions": [],
            "coinbase": [
                { "address": hex::encode([0x22u8; 32]), "value": 1u64 << 20, "salt": hex::encode(hash(b"pool root")) },
                { "address": hex::encode([0x22u8; 32]), "value": 1u64 << 21, "salt": hex::encode(commitment) },
            ],
            "state_root": hex::encode(hash(b"midstate state root")),
            "prev_header_hash": hex::encode(hash(b"midstate tip")),
            "timestamp": current_timestamp(),
            "target": hex::encode([0xffu8; 32]),
        });
        let parent = ParentTemplate::from_midstate_json(&batch, 1).unwrap();
        let parent_mining = parent.proof(0, [0; 32]).parent_mining_hash(&mining_hash);
        let (nonce, final_hash) = (0u64..)
            .find_map(|n| {
                let ext = create_extension_with(algo, parent_mining, n);
                (ext.final_hash < target).then_some((n, ext.final_hash))
            })
            .unwrap();
        // A job the endpoint never handed out is refused.
        let stale = mw.post_raw(
            "/merge/found",
            &json!({ "mining_hash": hex::encode([9u8; 32]), "batch_template": batch, "commit_index": 1,
                     "nonce": nonce, "final_hash": hex::encode(final_hash) }),
        );
        assert!(stale.is_ok_and(|(code, _)| code == 400));
        mw.post(
            "/merge/found",
            &json!({ "mining_hash": job["mining_hash"], "batch_template": batch, "commit_index": 1,
                     "nonce": nonce, "final_hash": hex::encode(final_hash) }),
        )
        .expect("/merge/found")
    })
    .await
    .unwrap();
    assert_eq!(found["accepted"], true, "{found}");
    assert_eq!(found["merged_mined"], true, "{found}");

    // The block is the chain's tip, the bond rode in with it, and it pays
    // the address the pool named. (The node publishes its new state just
    // after it answers, so give it a moment.)
    for _ in 0..100 {
        if h.state().height > height_before {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    let state = h.state();
    assert_eq!(state.height, height_before + 1);
    assert!(state.bonds.contains_key(&coin.coin_id()), "midwimble now holds the bond");
    let block = h.storage().load_batch(height_before).unwrap().unwrap();
    let mut account = MwbAccount::starting_at(height_before);
    let delta = account.scan(&keys, &BlockView::of(height_before, &block)).unwrap();
    assert_eq!(delta.received.len(), 1, "the payout");
    assert!(delta.received[0].1, "a coinbase output: MDW mined");

    // A request naming another host is refused, so no web page reaches it.
    let refused = tokio::task::spawn_blocking({
        let addr = merge.rpc();
        move || {
            use std::io::{Read, Write};
            let mut s = std::net::TcpStream::connect(&addr).unwrap();
            write!(s, "GET /state HTTP/1.1\r\nHost: evil.example\r\nConnection: close\r\n\r\n").unwrap();
            let mut out = String::new();
            s.read_to_string(&mut out).unwrap();
            out
        }
    })
    .await
    .unwrap();
    assert!(refused.starts_with("HTTP/1.1 403"), "{refused}");

    // Without a bond the endpoint refuses jobs, and a pool mines midstate alone.
    merge.set_bond(None);
    let rpc = merge.rpc();
    let refused = tokio::task::spawn_blocking(move || {
        RpcClient::new(rpc).post_raw("/merge/job", &json!({ "payouts": [] }))
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(refused.0, 400);
    h.shutdown();
}
