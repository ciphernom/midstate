//! The USDC leg against a real chain: a permit signed by this wallet's code,
//! checked by the token's own `ecrecover`; a lock and its claim, a bid and
//! its cancel, through `MidstateTokenSwap`; every event read back as the
//! wallet reads it. Nothing here is simulated.
//!
//! Skipped unless pointed at a chain where the contract is deployed (see
//! `tools/token-swap`), with an account holding a little ETH and USDC:
//!
//! ```text
//! DEVNET_RPC=https://sepolia.base.org DEVNET_KEY=0x… \
//! DEVNET_SWAP=0x… DEVNET_TOKEN=0x036CbD53842c5426634e7929541eC2318f3dCF7e \
//! cargo test -p midstate-walletd --test usdc_devnet -- --nocapture
//! ```
//!
//! It moves two millionths of a dollar, and gets both back.

use midstate_walletd::base::{self, BaseClient, ChainConfig, Settlement};
use midstate_walletd::eth_trade::Asset;
use midstate_walletd::evm::{parse_address, EvmKey};
use std::time::Duration;

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

async fn mined(c: &BaseClient, tx: [u8; 32]) -> base::Receipt {
    for _ in 0..120 {
        if let Some(r) = c.receipt(&tx).await.expect("a receipt") {
            return r;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    panic!("{} never mined", hex::encode(tx));
}

#[tokio::test]
async fn a_usdc_lock_and_a_bid_go_through_the_token_contract() {
    let (Some(rpc), Some(key), Some(swap), Some(token)) = (env("DEVNET_RPC"), env("DEVNET_KEY"), env("DEVNET_SWAP"), env("DEVNET_TOKEN")) else {
        eprintln!("skipped: set DEVNET_RPC, DEVNET_KEY, DEVNET_SWAP and DEVNET_TOKEN to run against a chain");
        return;
    };
    let secret: [u8; 32] = hex::decode(key.trim_start_matches("0x")).unwrap().try_into().unwrap();
    let key = EvmKey::from_secret(&secret).unwrap();
    let probe = BaseClient::new(ChainConfig { rpc_url: rpc.clone(), chain_id: 0, contract: swap.clone(), ..ChainConfig::default() }).unwrap();
    let chain_id = probe.chain_id().await.unwrap();
    let cfg = ChainConfig { rpc_url: rpc, chain_id, contract: swap.clone(), confirmations: 0, usdc_contract: swap.clone(), usdc_token: token };
    let c = BaseClient::for_asset(cfg, Asset::Usdc).unwrap();
    assert_eq!(c.contract(), parse_address(&swap).unwrap());
    let before = c.token_balance(&key.address).await.unwrap();
    assert!(before >= 2, "the account needs two millionths of a dollar of the token");

    // A lock, paid in with a permit in the same transaction; claimed.
    let preimage: [u8; 32] = rand::random();
    let hashlock = midstate::core::types::hash(&preimage);
    let (_, now) = c.head().await.unwrap();
    let permit = c.permit(&key, 1, now).await.unwrap();
    let data = base::token_lock_call(hashlock, key.address, 1, 600, Some(&permit)).unwrap();
    let r = mined(&c, c.send(&key, 0, data).await.unwrap()).await;
    assert!(r.success, "the lock reverted");
    let id = base::locked_swap_id(&r.logs, &c.contract()).expect("a Locked event from the contract");
    assert!(base::locked_swap_id(&r.logs, &[0; 20]).is_none());
    let s = c.get_swap(id).await.unwrap().expect("the swap, in storage");
    assert_eq!((s.amount, s.hashlock, s.settled, s.refund_to), (1, hashlock, false, key.address));
    assert_eq!(c.token_balance(&key.address).await.unwrap(), before - 1, "exactly the amount moved");
    let r = mined(&c, c.claim(&key, id, preimage).await.unwrap()).await;
    assert!(r.success, "the claim reverted");
    let ev = r.logs.iter().filter(|l| l.address == c.contract()).find_map(base::revealed_preimage);
    assert_eq!(ev, Some((id, preimage)), "the claim publishes the secret");
    assert_eq!(c.find_settlement(id, r.block.saturating_sub(5), r.block).await.unwrap(), Some(Settlement::Claimed(preimage)));
    assert_eq!(c.token_balance(&key.address).await.unwrap(), before, "back to the beneficiary: us");

    // A bid, escrowed with a permit; cancelled.
    let h2 = midstate::core::types::hash(&rand::random::<[u8; 32]>());
    let (_, now) = c.head().await.unwrap();
    let permit = c.permit(&key, 1, now).await.unwrap();
    let data = base::token_bid_call(h2, 1, 4096, [7; 32], 3_600, 0, Some(&permit)).unwrap();
    let r = mined(&c, c.send(&key, 0, data).await.unwrap()).await;
    assert!(r.success, "the bid reverted");
    let bid = base::created_bid_id(&r.logs, &c.contract()).expect("a BidCreated event");
    let b = c.get_bid(bid).await.unwrap().expect("the bid, in storage");
    assert_eq!((b.amount, b.mds_amount, b.hashlock, b.maker), (1, 4096, h2, key.address));
    let r = mined(&c, c.cancel_bid(&key, bid).await.unwrap()).await;
    assert!(r.success, "the cancel reverted");
    assert!(c.get_bid(bid).await.unwrap().unwrap().settled);
    assert_eq!(c.token_balance(&key.address).await.unwrap(), before, "every millionth back");
    eprintln!("lock {} and bid {} went through {}", hex::encode(id), hex::encode(bid), swap);
}
