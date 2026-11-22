//! MSB-1 in the wallet (`midstate::core::burns`): burning to back a name, a
//! key, a Base account, a pool or a hub, or to notarize a document; and
//! reading what anything has had burned for it from the node's burn index
//! (`midstate::storage::burn_index`), for the views that show it:
//!
//! * names (the Names page and Find), and payments to them;
//! * sealed messages: a conversation whose sender's name is backed below the
//!   inbox's bar is a message request;
//! * the ETH market: a seller's account and verified name, together;
//! * channel hubs, listed most-backed first.
//!
//! A child of `service`, sharing the actor's state, as `service_names` is.

use super::Service;
use crate::api::{BackingView, BurnRequest, BurnsView, NotaryMark, NotaryView, RecentBurnView, WalletEvent};
use anyhow::{anyhow, bail, Result};
use midstate::core::burns::{self, Backing, Purpose};
use midstate::core::names;
use midstate::core::OutputData;
use midstate::storage::burn_index;

/// The most-backed lists the Burns page shows.
const TOP: usize = 10;

pub(super) fn backing_view(s: &burn_index::Subject) -> BackingView {
    BackingView {
        purpose: s.purpose.word(),
        subject: burns::subject_text(s.purpose, &s.subject),
        blocks: s.backing,
        text: burns::blocks_text(s.backing),
        tier: burns::tier(s.backing).unwrap_or_default().into(),
        total: s.total,
        count: s.count,
        first: s.first,
        last: s.last,
        label: s.label.clone(),
    }
}

fn nothing(purpose: Purpose, subject: &[u8; 32]) -> BackingView {
    // A name's backing counts from its registration: burns from before it
    // may exist, but do not count.
    let text = if purpose == Purpose::NAME { "nothing burned since it was registered" } else { "nothing burned yet" };
    BackingView { purpose: purpose.word(), subject: burns::subject_text(purpose, subject), text: text.into(), ..BackingView::default() }
}

impl Service {
    /// The backing that counts for `subject` as `purpose`, now: for a name,
    /// what was burned since its current registration began
    /// (`burn_index::counts_from`). `None` if nothing counts, or if the
    /// indexes cannot say.
    pub(super) fn backing_of(&self, purpose: Purpose, subject: &[u8; 32]) -> Option<BackingView> {
        burn_index::counted(self.node.storage.db(), purpose, subject).ok().flatten().map(|s| backing_view(&s))
    }

    /// Blocks of backing behind `purpose`'s `subject`: zero for none.
    pub(super) fn blocks_for(&self, purpose: Purpose, subject: &[u8; 32]) -> f64 {
        self.backing_of(purpose, subject).map_or(0.0, |b| b.blocks)
    }

    /// What backs the name `name` (`nick` or `nick.mds`).
    pub(super) fn name_backing(&self, name: &str) -> Option<BackingView> {
        self.backing_of(Purpose::NAME, &burns::name_subject(name).ok()?)
    }

    /// Blocks of backing behind a Base account and the name it verified as.
    pub(super) fn seller_backing(&self, account: &[u8; 20], name: &str) -> f64 {
        let named = if name.is_empty() { 0.0 } else { self.name_backing(name).map_or(0.0, |b| b.blocks) };
        self.blocks_for(Purpose::EVM, &burns::evm_subject(account)) + named
    }

    /// Burn for something: back it, or notarize a document. Returns the send's
    /// commitment; the burn goes out as an ordinary send would.
    pub(super) async fn burn(&mut self, req: BurnRequest) -> Result<String> {
        self.ready_to_spend()?;
        let p = Purpose::parse(&req.purpose).ok_or_else(|| anyhow!("no purpose {:?}", req.purpose))?;
        let b = Backing::new(p, burns::parse_subject(p, &req.subject)?, &req.label)?;
        if req.units == 0 && !p.is_notary() {
            bail!("backing is what it costs: burn something");
        }
        let mut outputs = b.outputs(req.units);
        if outputs.is_empty() {
            // A notarization for the fee alone: the payload is the record.
            outputs.push(OutputData::DataBurn { payload: b.payload(), value_burned: 0 });
        }
        let state = self.node.get_state().await;
        let to = format!("{} {}", p.word(), b.subject_text());
        let what = if p.is_notary() { format!("notarizing {}", &b.subject_text()[..16]) } else { format!("backing {}", b.subject_text()) };
        let c = self.commit_outputs(&state, outputs, None, &to, what).await?;
        self.broadcast_commit(c).await?;
        let _ = self.events.send(WalletEvent::WalletChanged);
        Ok(hex::encode(c))
    }

    /// The Burns page.
    pub(super) fn burns_view(&self) -> Result<BurnsView> {
        let db = self.node.storage.db();
        let m = burn_index::meta(db)?;
        let mut v = BurnsView { tip: m.next, reward: midstate::core::block_reward(m.next), ready: m.ready(), ..BurnsView::default() };
        if !v.ready {
            v.note = if m.missing != 0 {
                format!("This node does not have block {}, so it cannot count burns past it.", m.missing - 1)
            } else {
                format!("Counting the chain's burns: block {} of {}.", m.next, m.chain_len)
            };
        }
        // This wallet's own: its names, then its Base account.
        let held: Vec<String> = self.names.jobs.iter().filter(|j| j.status == "registered" && !j.renter).map(|j| j.name.clone()).collect();
        for n in held {
            let Ok(s) = burns::name_subject(&n) else { continue };
            v.mine.push(self.backing_of(Purpose::NAME, &s).unwrap_or_else(|| nothing(Purpose::NAME, &s)));
        }
        if let Ok(k) = self.evm_key() {
            let s = burns::evm_subject(&k.address);
            v.mine.push(self.backing_of(Purpose::EVM, &s).unwrap_or_else(|| nothing(Purpose::EVM, &s)));
        }
        // A wallet that routes payments is a hub: its channel identity is
        // what payers pick it by.
        if let (true, Some(pk)) = (self.book.hub.forward, self.book.identity_pk) {
            v.mine.push(self.backing_of(Purpose::HUB, &pk).unwrap_or_else(|| nothing(Purpose::HUB, &pk)));
        }
        if let Ok((years, trailing)) = burn_index::years(db) {
            v.issued = trailing.issued;
            v.burned = trailing.burned.burned();
            v.names_burned = trailing.burned.names;
            v.backing_burned = trailing.burned.backing_total();
            v.other_burned = trailing.burned.dex.saturating_add(trailing.burned.other);
            if let Some(y) = years.last() {
                v.year = y.year + 1;
                v.year_issued = y.issued;
                v.year_burned = y.burned.burned();
            }
        }
        if let Ok(held) = burn_index::held(db) {
            v.held = held.iter().map(|(_, b, _)| b).sum();
            v.upkeep_share = burns::upkeep(v.held) / midstate::core::BLOCKS_PER_YEAR as f64;
        }
        let top = |p: Purpose| burn_index::top_counted(db, p, TOP).map(|l| l.iter().map(backing_view).collect()).unwrap_or_default();
        v.top_names = top(Purpose::NAME);
        v.top_pools = top(Purpose::POOL);
        v.top_hubs = top(Purpose::HUB);
        v.recent = burn_index::recent(db, 20, 5_000)
            .map(|l| {
                l.iter()
                    .map(|r| RecentBurnView {
                        height: r.height,
                        time: r.time,
                        purpose: r.purpose.word(),
                        subject: burns::subject_text(r.purpose, &r.subject),
                        value: r.value,
                        blocks: burns::weight(r.value, r.height),
                        label: r.label.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(v)
    }

    /// Every notarization of a document's hash (64 hex characters), under
    /// either hash.
    pub(super) fn notary(&self, hash: &str) -> Result<NotaryView> {
        let s = burns::parse_subject(Purpose::NOTARY, hash)?;
        let db = self.node.storage.db();
        let mut marks = Vec::new();
        for p in [Purpose::NOTARY, Purpose::NOTARY_SHA256] {
            for b in burn_index::history(db, p, &s, 1_000)? {
                marks.push(NotaryMark { purpose: p.word(), height: b.height, time: b.time, value: b.value, label: b.label });
            }
        }
        marks.sort_by_key(|m| m.height);
        Ok(NotaryView { hash: hex::encode(s), marks })
    }

    /// The backing a conversation's sender needs to reach the inbox, if the
    /// inbox has a bar (`None`: every message does).
    pub(super) fn inbox_bar(&self) -> Option<f64> {
        self.ident.book.inbox_bar
    }

    pub(super) fn set_inbox_bar(&mut self, bar: Option<f64>) {
        self.ident.book.inbox_bar = bar.filter(|b| b.is_finite() && *b >= 0.0);
        self.ident.book.save(&self.wallet_path);
        let _ = self.events.send(WalletEvent::WalletChanged);
    }

    /// Whether a conversation with `peer` (`nick`, verified or not) is a
    /// message request: the inbox has a bar, and the sender is anonymous,
    /// unverified, or backed below it. Conversations this wallet opened never
    /// are.
    pub(super) fn is_request(&self, peer: &str, verified: bool, opened_by_me: bool, backing: f64) -> bool {
        let Some(bar) = self.inbox_bar() else { return false };
        !opened_by_me && (peer.is_empty() || !verified || backing < bar || names::normalize(peer).is_err())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_subject_with_nothing_burned_says_so() {
        let s = burns::name_subject("nick").unwrap();
        let v = nothing(Purpose::NAME, &s);
        assert_eq!((v.purpose.as_str(), v.subject.as_str(), v.blocks, v.tier.as_str()), ("name", "nick.mds", 0.0, ""));
        let sub = burn_index::Subject {
            purpose: Purpose::NAME,
            subject: s,
            backing: 1_500.0,
            at: 10,
            total: 1,
            count: 1,
            first: 1,
            last: 1,
            label: "x".into(),
        };
        let b = backing_view(&sub);
        assert_eq!((b.tier.as_str(), b.text.as_str()), ("well backed", "1.0 days of issuance"));
    }
}
