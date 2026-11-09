//! Mining bonds. A bond locks one coin until a block height, naming a
//! midwimble mining key: midwimble lets that key sign its blocks while enough
//! of the lock remains, and only this wallet's recovery phrase can spend the
//! coin once it unlocks. This is the CLI's `generate-mss` → `send` →
//! `spend-script` flow, with the owner key and the script handled here.
//!
//! The tab is meant to be read at a glance. A new bond is three choices (who
//! mines with it, how much, for how long) and one button, and registration
//! starts by itself. Each bond is a card showing where it is on Lock →
//! Register → Mine → Unbond, with the one thing to do next.

use crate::app::App;
use crate::bridge::Action;
use crate::theme::{self, compact_units, short_hex};
use eframe::egui::{self, CornerRadius, FontId, Frame, Margin, RichText, Sense, Stroke, TextEdit, Ui};
use midstate_walletd::api::BondView;
use midstate_walletd::bond::{midwimble, parse_mining_key};
use midstate_walletd::co_bond;
use midstate_walletd::rental;

/// A test bond: the smallest coin, unlocking this many blocks out.
const TEST_EXP: u32 = 10;
const TEST_BLOCKS: u64 = 10;
/// Mining times offered, in days. The longest midwimble takes is added.
const DAYS: [u64; 3] = [30, 90, 180];
/// At most this many sizes, from the smallest midwimble registers now.
const MAX_SIZES: usize = 4;
/// Line width of the cards' borders and the track.
const HAIR: f32 = 1.0;

pub fn show(app: &mut App, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    theme::heading(ui, "Bonds");

    let height = app.sync.as_ref().map(|s| s.height).unwrap_or(0);
    let blocked = app
        .sync
        .as_ref()
        .map(|s| s.is_syncing || s.restore_pass > 0)
        .unwrap_or(true);

    let views = ["Your bonds", "Rent"];
    theme::segmented(ui, &views, &mut app.bonds_view);
    ui.add_space(8.0);
    if blocked {
        theme::hint(ui, "Bonds can be locked, rented and unbonded once the node has synced.");
    }
    let notice = if app.bonds_view == 1 { app.rental_notice.clone() } else { app.bond_notice.clone() };
    status_lines(app, ui, &notice);
    if app.bonds_view == 1 {
        super::rent::show(app, ui, &ctx, height, blocked);
        return;
    }

    let now = theme::now_secs();
    let bonds = app.bonds.clone();
    for b in &bonds {
        bond_card(app, ui, &ctx, b, blocked, now);
        ui.add_space(8.0);
    }

    if bonds.is_empty() || app.bond_form_open {
        new_bond(app, ui, &ctx, height, blocked);
    } else if ui.add(primary("+  New bond")).clicked() {
        app.bond_form_open = true;
        app.bond_confirm = false;
    }

    ui.add_space(16.0);
    import_form(app, ui, &ctx);
}

/// The spinner while an action runs, then its outcome.
fn status_lines(app: &App, ui: &mut Ui, notice: &str) {
    if app.busy {
        ui.horizontal(|ui| {
            ui.spinner();
            theme::hint(ui, "working…");
        });
    }
    if !notice.is_empty() {
        ui.label(RichText::new(notice).size(12.5).color(theme::ink()));
    }
    if !app.error.is_empty() {
        ui.label(RichText::new(&app.error).size(12.5).color(theme::muted()));
    }
    if app.busy || !notice.is_empty() || !app.error.is_empty() {
        ui.add_space(6.0);
    }
}

// ── A new bond ──────────────────────────────────────────────────────────

fn new_bond(app: &mut App, ui: &mut Ui, ctx: &egui::Context, height: u64, blocked: bool) {
    let now = theme::now_secs();
    let rules = app.bond_rules.clone();
    let spendable = app.balance.as_ref().map(|b| b.confirmed).unwrap_or(0);
    let wallet_key = rules.as_ref().map(|r| r.mining_key.clone()).unwrap_or_default();
    // A wallet made before recovery phrases has no mining key of its own.
    let wallet_can_mine = !wallet_key.is_empty() || rules.is_none();
    if !wallet_can_mine && app.bond_key_mode != 1 {
        app.bond_key_mode = 1;
        if app.bond_key.trim().is_empty() {
            app.bond_key = co_bond::DEFAULT_POOL.into();
        }
    }

    // A co-bond names a pool by its address, which the wallet asks for the
    // pool's mining key and terms (and hands the bond in to), or by a bare
    // mining key, for a pool that takes co-bond files.
    let pool_text = app.bond_key.trim().to_string();
    let by_key = parse_mining_key(&pool_text).is_ok();
    let base = (app.bond_key_mode == 1 && !by_key).then(|| co_bond::pool_base(&pool_text).ok()).flatten();
    if let Some(b) = &base {
        // Once the address has stood still for a moment.
        let t = app.t0.elapsed().as_secs_f64();
        if app.pool_asked != *b && t - app.pool_edit_at > 0.7 {
            app.pool_asked = b.clone();
            app.go(ctx, Action::PoolTerms { url: b.clone() });
        }
    }
    let terms = base
        .as_ref()
        .and_then(|b| app.pool_terms.as_ref().filter(|(u, _)| u == b).map(|(_, r)| r.clone()));
    let pool_key = match &terms {
        Some(Ok(t)) if t.accepting => t.mining_key.clone().filter(|k| parse_mining_key(k).is_ok()),
        _ => None,
    };
    let host = base.as_deref().map(co_bond::pool_host).unwrap_or_default();

    // The sizes: what midwimble registers now, doubling up to the most its
    // price can ever ask. A bigger bond mines no more than the smallest.
    let needed = rules.as_ref().map_or(midwimble::min_bond(now), |r| r.registration_min);
    let ceiling = rules.as_ref().map_or(midwimble::MAX_BOND, |r| r.max_bond).max(needed);
    let sizes: Vec<u32> = (exp_at_least(needed)..=exp_at_least(ceiling)).take(MAX_SIZES).collect();
    if !sizes.contains(&app.bond_exp) {
        app.bond_exp = sizes[0];
    }

    // The mining times: a month, three, six, and the longest midwimble
    // takes (a lock at most a year out, less its month of unbonding).
    let longest = midwimble::horizon(now).saturating_sub(midwimble::until_for_days(now, 0)) / 1440;
    let mut days: Vec<u64> = DAYS.iter().copied().filter(|d| *d < longest).collect();
    if longest > 0 {
        days.push(longest);
    }
    if !days.is_empty() && !days.contains(&app.bond_days) {
        app.bond_days = days.iter().copied().find(|d| *d >= app.bond_days).unwrap_or(days[days.len() - 1]);
    }

    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.label(RichText::new("New bond").font(theme::font_medium(15.0)));
            if !app.bonds.is_empty() {
                theme::right_aligned(ui, |ui| {
                    if ui.small_button("Close").clicked() {
                        app.bond_form_open = false;
                        app.bond_confirm = false;
                    }
                });
            }
        });
        ui.add_space(6.0);

        // 1. Who mines with it.
        section(ui, "Who mines with it");
        let w = (ui.available_width() - 10.0) / 2.0;
        ui.horizontal(|ui| {
            let caption = if wallet_can_mine { "Mine MDW on this computer" } else { "Needs a wallet with a recovery phrase" };
            if choice(ui, "who_wallet", w, app.bond_key_mode == 0, wallet_can_mine, "This wallet", caption) {
                app.bond_key_mode = 0;
                app.bond_confirm = false;
            }
            if choice(ui, "who_pool", w, app.bond_key_mode == 1, true, "A pool (co-bond)", "Back a pool's mining") {
                if app.bond_key_mode != 1 && app.bond_key.trim().is_empty() {
                    app.bond_key = co_bond::DEFAULT_POOL.into();
                }
                app.bond_key_mode = 1;
                app.bond_confirm = false;
            }
        });
        if app.bond_key_mode == 1 {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                let r = ui.add(
                    TextEdit::singleline(&mut app.bond_key)
                        .hint_text("the pool's address, or its mining key")
                        .font(egui::TextStyle::Monospace)
                        .desired_width((ui.available_width() - 10.0) * 0.7),
                );
                if r.changed() {
                    app.bond_confirm = false;
                    app.pool_edit_at = app.t0.elapsed().as_secs_f64();
                }
                ui.add(
                    TextEdit::singleline(&mut app.bond_label)
                        .hint_text("pool name (optional)")
                        .desired_width(ui.available_width()),
                );
            });
            pool_line(app, ui, &pool_text, by_key, base.as_deref(), terms.as_ref());
        }

        // 2. How much.
        ui.add_space(10.0);
        ui.horizontal(|ui| {
            section(ui, "Size");
            theme::right_aligned(ui, |ui| {
                ui.label(theme::mono(format!("available {}", compact_units(spendable))).size(11.0).color(theme::muted()));
            });
        });
        let n = sizes.len() + 1;
        let w = ((ui.available_width() - 10.0 * (n - 1) as f32) / n as f32).min(210.0);
        ui.horizontal(|ui| {
            for &e in &sizes {
                let v = 1u64 << e;
                let caption = if v >= spendable {
                    "more than you have"
                } else if e == sizes[0] {
                    "minimum"
                } else {
                    "mines the same"
                };
                let picked = !app.bond_test && app.bond_exp == e;
                if choice(ui, ("size", e), w, picked, v < spendable, &compact_units(v), caption) {
                    app.bond_exp = e;
                    app.bond_test = false;
                    app.bond_confirm = false;
                }
            }
            let test = 1u64 << TEST_EXP;
            let caption = format!("{}, back in minutes", compact_units(test));
            if choice(ui, "size_test", w, app.bond_test, test < spendable, "Test", &caption) {
                app.bond_test = true;
                app.bond_confirm = false;
            }
        });

        // 3. For how long.
        ui.add_space(10.0);
        section(ui, "Mining time");
        let n = days.len().max(1);
        let w = ((ui.available_width() - 10.0 * (n - 1) as f32) / n as f32).min(210.0);
        ui.horizontal(|ui| {
            for &d in &days {
                let back = back_at(midwimble::until_for_days(now, d), height, now);
                let picked = !app.bond_test && app.bond_days == d;
                let caption = format!("back ~{}", theme::fmt_date(back));
                if choice(ui, ("days", d), w, picked, !app.bond_test, &months_label(d), &caption) {
                    app.bond_days = d;
                    app.bond_confirm = false;
                }
            }
        });

        // What it comes to, and the button.
        let value = 1u64 << if app.bond_test { TEST_EXP } else { app.bond_exp };
        let until = if app.bond_test { height + TEST_BLOCKS } else { midwimble::until_for_days(now, app.bond_days) };
        let back = back_at(until, height, now);
        let key = match app.bond_key_mode {
            0 => wallet_key.clone(),
            _ if by_key => pool_text.clone(),
            _ => pool_key.clone().unwrap_or_default(),
        };
        // Handed in to the pool by the wallet: not a test bond, which no
        // pool registers.
        let hand_in = (app.bond_key_mode == 1 && !by_key && !app.bond_test).then(|| base.clone()).flatten();
        let key_ok = parse_mining_key(&key).is_ok();
        let affordable = value < spendable;
        let ready = !app.busy && !blocked && key_ok && affordable && (app.bond_test || !days.is_empty());

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            fact(ui, "Locks", &compact_units(value));
            if app.bond_test {
                fact(ui, "Mines", "no");
                fact(ui, "Back", &format!("in ~{}", about(TEST_BLOCKS)));
            } else {
                fact(ui, "Mines", &format!("~{} days", app.bond_days));
                fact(ui, "Back", &format!("~{}", theme::fmt_date(back)));
            }
            theme::right_aligned(ui, |ui| {
                if app.bond_confirm {
                    if ui.add_enabled(ready, primary("Confirm")).clicked() {
                        let label = if app.bond_test {
                            "test".to_string()
                        } else if app.bond_key_mode == 1 {
                            app.bond_label.trim().to_string()
                        } else {
                            String::new()
                        };
                        // Midwimble can count it: have the wallet register it
                        // as soon as the lock confirms, unless a pool does.
                        let register = !app.bond_test
                            && hand_in.is_none()
                            && value >= midwimble::min_bond(now)
                            && rules.as_ref().is_some_and(|r| r.rpc_on);
                        app.busy = true;
                        app.bond_confirm = false;
                        app.error.clear();
                        app.bond_notice.clear();
                        app.go(
                            ctx,
                            Action::CreateBond { mining_key: key.clone(), value, until, label, register, pool: hand_in.clone() },
                        );
                    }
                    if ui.button("Cancel").clicked() {
                        app.bond_confirm = false;
                    }
                } else if ui.add_enabled(ready, primary(&format!("Lock {}", compact_units(value)))).clicked() {
                    app.bond_confirm = true;
                }
            });
        });

        if app.bond_confirm {
            let mut line = if app.bond_test {
                format!("Locks {} for about {}. It does not mine.", compact_units(value), about(TEST_BLOCKS))
            } else {
                format!("The coin can't move until ~{}.", theme::fmt_date(back))
            };
            if hand_in.is_some() {
                line.push_str(&format!(" It mines for {host}, and the wallet hands it in there once the lock confirms."));
            } else if app.bond_key_mode == 1 {
                line.push_str(&format!(" It mines for the pool key {}.", short_hex(&key, 8)));
            }
            ui.label(RichText::new(line).size(12.5).color(theme::ink()));
        } else if !affordable && spendable > 0 {
            ui.horizontal(|ui| {
                theme::hint(ui, &format!("You have {}.", compact_units(spendable)));
                if !app.bond_test && ui.small_button("Rent a bond instead").clicked() {
                    app.bonds_view = 1;
                }
            });
        } else if !affordable {
            ui.horizontal(|ui| {
                theme::hint(ui, "Nothing to lock yet.");
                if ui.small_button("Rent a bond instead").clicked() {
                    app.bonds_view = 1;
                }
            });
        }
    });
}

/// Under the pool field: what the pool takes and gives, or why it can't be
/// used.
fn pool_line(
    app: &mut App,
    ui: &mut Ui,
    text: &str,
    by_key: bool,
    base: Option<&str>,
    terms: Option<&Result<co_bond::PoolTerms, String>>,
) {
    let pct = |v: f64| format!("{}%", format!("{v:.2}").trim_end_matches('0').trim_end_matches('.'));
    match (base, terms) {
        _ if by_key => theme::hint(ui, "A mining key: once the bond is ready, save its co-bond file and send it to the pool."),
        (None, _) if text.is_empty() => {}
        (None, _) => theme::hint(ui, "Give the pool's address, like rpc.cypherpunk.gold/pool, or its mining key."),
        (Some(b), None) => {
            ui.horizontal(|ui| {
                ui.spinner();
                theme::hint(ui, &format!("Asking {}…", co_bond::pool_host(b)));
            });
        }
        (Some(b), Some(Err(e))) => {
            ui.horizontal(|ui| {
                theme::hint(ui, &format!("Couldn't use {}: {e}.", co_bond::pool_host(b)));
                if ui.small_button("Try again").clicked() {
                    app.pool_asked.clear();
                }
            });
        }
        (Some(b), Some(Ok(t))) if !t.accepting || t.mining_key.is_none() => {
            let why = t.note.clone().unwrap_or_else(|| "it gives no mining key".into());
            theme::hint(ui, &format!("{} isn't taking co-bonds: {why}.", co_bond::pool_host(b)));
        }
        (Some(b), Some(Ok(t))) => {
            // The fee with this bond too, counting the co-bonds this wallet
            // already has signing for the pool.
            let signing = app.bonds.iter().filter(|x| x.pool == b && x.reg_phase == "on_chain" && x.status == "locked").count();
            let with_this = (t.fee_percent - t.fee_cut * (signing as f64 + 1.0)).max(0.0);
            let mut gives = format!("MDW fee {}", pct(t.fee_percent));
            if t.fee_cut > 0.0 {
                gives.push_str(&format!(", {} with this bond", pct(with_this)));
            }
            if t.reward_percent > 0.0 {
                gives.push_str(&format!(" · co-bonders share {} of its fee", pct(t.reward_percent)));
            }
            ui.horizontal(|ui| {
                theme::badge(ui, "takes co-bonds", theme::ink());
                ui.label(RichText::new(co_bond::pool_host(b)).size(12.5).color(theme::ink()));
                theme::hint(ui, &gives);
            });
            let pays = t.fee_cut > 0.0 || t.reward_percent > 0.0;
            let mdw = app.mwb_account.as_ref().and_then(|a| a.address.clone());
            if pays && mdw.is_none() {
                theme::hint(ui, "Add an MDW account on the Receive tab to get these: the wallet hands the bond in with its address.");
            }
        }
    }
}

/// The exponent of the smallest coin holding at least `v`.
fn exp_at_least(v: u64) -> u32 {
    v.max(1).checked_next_power_of_two().map_or(63, u64::trailing_zeros)
}

/// "1 month", "3 months", "11 months"; "7 days" under half a month.
pub(super) fn months_label(days: u64) -> String {
    match (days + 15) / 30 {
        0 => format!("{days} days"),
        1 => "1 month".into(),
        m => format!("{m} months"),
    }
}

/// About how long `blocks` take, a block a minute: "10 minutes", "an hour".
fn about(blocks: u64) -> String {
    match blocks.max(1) {
        1 => "a minute".into(),
        m @ 2..=59 => format!("{m} minutes"),
        60..=119 => "an hour".into(),
        m @ 120..=2879 => format!("{} hours", m / 60),
        m => format!("{} days", m / 1440),
    }
}

/// About when a coin locked until `until` can move again (unix seconds).
fn back_at(until: u64, height: u64, now: u64) -> u64 {
    now + rental::locked_days(until, height, now) * 86_400
}

// ── Bonds ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Step {
    Done,
    Now,
    Next,
    /// Not for this bond: a test bond never registers or mines.
    Skip,
}

enum Next {
    Register,
    CoBondFile,
    Unbond,
    Forget,
}

/// Where a bond stands, read off its view.
struct Stage {
    /// How far along Lock → Register → Mine → Unbond it is; 4 when done.
    at: usize,
    skip_mining: bool,
    badge: &'static str,
    line: String,
    next: Option<Next>,
    progress: Option<f32>,
}

fn stage(b: &BondView, now: u64) -> Stage {
    let unlock = if b.blocks_left < 2 * 1440 {
        format!("unlocks in about {}", about(b.blocks_left))
    } else {
        format!("unlocks ~{}", theme::fmt_date(now + b.blocks_left * 60))
    };
    let counting = b.counts_until > now;
    let rented = b.rental == "rented";
    let co_bond = !b.wallet_key && b.rental.is_empty();
    // A co-bond handed to a pool by its address: the pool registers it.
    let pool = (!b.pool.is_empty()).then(|| co_bond::pool_host(&b.pool));
    let mut s = Stage { at: 0, skip_mining: b.below_min, badge: "", line: String::new(), next: None, progress: None };
    let set = |s: &mut Stage, at: usize, badge: &'static str, line: String| {
        s.at = at;
        s.badge = badge;
        s.line = line;
    };
    match b.status.as_str() {
        "not_funded" => {
            set(&mut s, 0, "failed", "The lock never confirmed.".into());
            s.next = b.can_remove.then_some(Next::Forget);
        }
        "locking" => {
            let line = match &pool {
                Some(host) => format!("Waiting for the send to confirm. Then it goes to {host}."),
                None => "Waiting for the send to confirm.".into(),
            };
            set(&mut s, 0, "locking", line);
        }
        "unbonding" => set(&mut s, 3, "unbonding", "Coming back to your balance.".into()),
        "unbonded" => {
            let line = if rented { "Its owner has taken the coin back." } else { "Back in your balance." };
            set(&mut s, 4, "done", line.into());
            s.next = b.can_remove.then_some(Next::Forget);
        }
        "unlocked" => {
            let line = if rented { "Its lock is over: its owner takes the coin back." } else { "Unbond it to bring the coin back." };
            set(&mut s, 3, "unlocked", line.into());
            s.next = b.can_unbond.then_some(Next::Unbond);
        }
        _ if b.below_min => set(&mut s, 3, "test", format!("A test bond: it doesn't mine, and {unlock}.")),
        _ if b.rental == "rented_out" => set(&mut s, 2, "rented out", format!("Earning rent: the renter mines with it. It {unlock}.")),
        _ if !counting && b.reg_phase == "on_chain" => set(&mut s, 3, "done mining", format!("Finished mining. It {unlock}.")),
        _ if !counting => {
            s.skip_mining = true;
            set(&mut s, 3, "locked", format!("Too short to mine. It {unlock}."));
        }
        _ => match b.reg_phase.as_str() {
            "to_pool" => set(&mut s, 1, "handing in", b.reg_note.clone()),
            "pool" => {
                set(&mut s, 1, "pool registering", b.reg_note.clone());
                if (1..100).contains(&b.reg_progress) {
                    s.progress = Some(b.reg_progress as f32 / 100.0);
                }
            }
            "on_chain" => {
                let until = theme::fmt_date(b.counts_until);
                let line = if rented {
                    format!("Mining for you until ~{until}.")
                } else if let Some(host) = &pool {
                    format!("Mining with {host} until ~{until}. It {unlock}.")
                } else if co_bond {
                    format!("Mining with your pool until ~{until}. It {unlock}.")
                } else {
                    format!("Mines when you mine, until ~{until}. It {unlock}.")
                };
                set(&mut s, 2, "mining", line);
            }
            "ready" if co_bond && pool.is_none() => {
                set(&mut s, 1, "send to pool", "Save the co-bond file and send it to your pool.".into());
                s.next = Some(Next::CoBondFile);
            }
            "ready" => set(
                &mut s,
                2,
                "ready",
                "Ready: start mining on the Node tab, and your first MDW block registers it.".into(),
            ),
            "burying" => {
                set(&mut s, 1, "registering", format!("Registering: about a day in all, {}% done.", b.reg_progress));
                s.progress = Some(b.reg_progress as f32 / 100.0);
            }
            "waiting" => set(&mut s, 1, "registering", "Registering: proving the lock.".into()),
            "failed" => set(&mut s, 1, "retrying", "Registering hit a problem; the wallet tries again by itself.".into()),
            _ if b.can_register => {
                set(&mut s, 1, "not registered", "Register it so midwimble counts it.".into());
                s.next = Some(Next::Register);
            }
            _ => set(&mut s, 1, "locked", format!("It {unlock}.")),
        },
    }
    s
}

fn bond_card(app: &mut App, ui: &mut Ui, ctx: &egui::Context, b: &BondView, blocked: bool, now: u64) {
    let st = stage(b, now);
    let rpc_on = app.bond_rules.as_ref().is_none_or(|r| r.rpc_on);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.label(RichText::new(compact_units(b.value)).font(theme::font_mono_medium(16.0)).color(theme::ink()));
            let quiet = matches!(st.badge, "failed" | "retrying" | "done");
            theme::badge(ui, st.badge, if quiet { theme::muted() } else { theme::ink() });
            theme::right_aligned(ui, |ui| {
                ui.label(RichText::new(who(b)).size(12.0).color(theme::muted()));
            });
        });
        ui.add_space(2.0);
        let steps: Vec<(&str, Step)> = ["Lock", "Register", "Mine", "Unbond"]
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let step = if st.skip_mining && (i == 1 || i == 2) {
                    Step::Skip
                } else if i < st.at {
                    Step::Done
                } else if i == st.at {
                    Step::Now
                } else {
                    Step::Next
                };
                (*name, step)
            })
            .collect();
        track(ui, &steps);
        if let Some(p) = st.progress {
            theme::progress_bar(ui, p);
        }
        ui.label(RichText::new(&st.line).size(12.5).color(theme::bright()));
        if b.reg_phase == "failed" && !b.reg_note.is_empty() {
            theme::hint(ui, &b.reg_note);
        }
        if matches!(st.next, Some(Next::Register)) && !rpc_on {
            theme::hint(ui, "Registering needs this app's node RPC, which is off (MIDSTATE_RPC_PORT=none).");
        }

        let open_id = egui::Id::new(("bond_details", &b.id));
        let mut open = ui.data_mut(|d| *d.get_temp_mut_or(open_id, false));
        ui.horizontal(|ui| {
            if ui.small_button(if open { "Hide details" } else { "Details" }).clicked() {
                open = !open;
                ui.data_mut(|d| d.insert_temp(open_id, open));
            }
            theme::right_aligned(ui, |ui| {
                let idle = !app.busy;
                let go = |app: &mut App, a: Action| {
                    app.busy = true;
                    app.error.clear();
                    app.bond_notice.clear();
                    app.go(ctx, a);
                };
                match st.next {
                    Some(Next::Unbond) => {
                        if ui.add_enabled(idle && !blocked, primary("Unbond")).clicked() {
                            go(app, Action::Unbond { id: b.id.clone() });
                        }
                    }
                    Some(Next::CoBondFile) => {
                        if ui.add_enabled(idle, primary("Save co-bond file")).clicked() {
                            go(app, Action::CopyCoBondFile { id: b.id.clone() });
                        }
                    }
                    Some(Next::Register) => {
                        if ui.add_enabled(idle && rpc_on, primary("Register")).clicked() {
                            go(app, Action::RegisterBond { id: b.id.clone() });
                        }
                    }
                    Some(Next::Forget) => {
                        if ui.add_enabled(idle, egui::Button::new("Forget")).clicked() {
                            go(app, Action::RemoveBond { id: b.id.clone() });
                        }
                    }
                    None => {}
                }
            });
        });
        if open {
            details(app, ui, ctx, b);
        }
    });
}

/// Whose miner the bond is for, as the card's corner says it.
fn who(b: &BondView) -> String {
    let who = match b.rental.as_str() {
        "rented" => "rented".to_string(),
        "rented_out" => "rented out".to_string(),
        _ if b.wallet_key => "this wallet".to_string(),
        _ if !b.pool.is_empty() => format!("co-bond · {}", co_bond::pool_host(&b.pool)),
        _ => format!("co-bond · {}", short_hex(&b.mining_key, 6)),
    };
    if b.label.is_empty() || b.label == "test" {
        who
    } else {
        format!("{} · {who}", b.label)
    }
}

/// Everything about a bond, for whoever wants it: keys, ids, and the ways
/// to take it elsewhere.
fn details(app: &mut App, ui: &mut Ui, ctx: &egui::Context, b: &BondView) {
    ui.add_space(4.0);
    egui::Grid::new(("bond_grid", &b.id)).num_columns(3).spacing([12.0, 4.0]).show(ui, |ui| {
        let row = |ui: &mut Ui, k: &str, v: String, copy: Option<&str>| {
            ui.label(RichText::new(k).size(11.5).color(theme::muted()));
            ui.label(theme::mono(v).size(11.5));
            if let Some(c) = copy {
                if ui.small_button("Copy").clicked() {
                    ui.ctx().copy_text(c.to_string());
                }
            } else {
                ui.label("");
            }
            ui.end_row();
        };
        row(ui, "unlocks at block", b.bonded_until.to_string(), None);
        if b.counts_until > 0 {
            row(ui, "counts for mining until", theme::fmt_dt(b.counts_until), None);
        }
        row(ui, "mining key", short_hex(&b.mining_key, 10), Some(&b.mining_key));
        row(ui, "owner key", short_hex(&b.owner_pk, 10), Some(&b.owner_pk));
        row(ui, "bond id", short_hex(&b.id, 10), Some(&b.id));
        row(ui, "salt", short_hex(&b.salt, 10), Some(&b.salt));
    });
    if !b.reg_note.is_empty() && b.reg_phase != "failed" {
        theme::hint(ui, &b.reg_note);
    }
    ui.horizontal(|ui| {
        if ui.small_button("Copy for another wallet").clicked() {
            ui.ctx().copy_text(b.export_line.clone());
        }
        let co_bond = !b.wallet_key && b.rental.is_empty() && b.pool.is_empty();
        if co_bond && b.reg_phase == "on_chain" && ui.add_enabled(!app.busy, egui::Button::new("Save co-bond file").small()).clicked() {
            app.busy = true;
            app.error.clear();
            app.go(ctx, Action::CopyCoBondFile { id: b.id.clone() });
        }
        if b.reg_phase != "on_chain" && !b.below_min && b.pool.is_empty() && ui.small_button("Copy register command").clicked() {
            ui.ctx().copy_text(b.register_cmd.clone());
        }
    });
}

/// Saves a bond's co-bond file where people look for downloads, under a
/// name of its own, and says where.
pub fn save_co_bond_file(id: &str, text: &str) -> std::io::Result<std::path::PathBuf> {
    let dir = dirs::download_dir()
        .or_else(dirs::home_dir)
        .ok_or_else(|| std::io::Error::other("no downloads folder"))?;
    let path = dir.join(format!("midstate-co-bond-{}.json", id.get(..12).unwrap_or(id)));
    std::fs::write(&path, text)?;
    Ok(path)
}

// ── Import ──────────────────────────────────────────────────────────────

fn import_form(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    egui::CollapsingHeader::new(RichText::new("Import a bond").size(12.5).color(theme::muted()))
        .id_salt("bond_import")
        .show(ui, |ui| {
            theme::hint(ui, "A bond this phrase made elsewhere: midwimble's bond.json, or a line another wallet copied.");
            ui.add(
                TextEdit::multiline(&mut app.bond_import)
                    .hint_text("bond.json, or <script> <value> <salt>")
                    .font(egui::TextStyle::Monospace)
                    .desired_rows(3)
                    .desired_width(f32::INFINITY),
            );
            ui.horizontal(|ui| {
                ui.add(
                    TextEdit::singleline(&mut app.bond_import_label)
                        .hint_text("label (optional)")
                        .desired_width(ui.available_width() - 90.0),
                );
                let can = !app.busy && !app.bond_import.trim().is_empty();
                if ui.add_enabled(can, egui::Button::new("Import")).clicked() {
                    app.busy = true;
                    app.error.clear();
                    app.bond_notice.clear();
                    app.go(
                        ctx,
                        Action::ImportBond {
                            text: app.bond_import.trim().to_string(),
                            label: app.bond_import_label.trim().to_string(),
                        },
                    );
                }
            });
        });
}

// ── Pieces shared with the Rent view ────────────────────────────────────

/// A small capital label over a group of choices.
pub(super) fn section(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text.to_uppercase()).font(FontId::monospace(10.0)).color(theme::muted()));
}

/// One choice of a group, drawn as a card with a title and a caption. The
/// picked one is inked in, as the segmented control is. True when clicked.
pub(super) fn choice(
    ui: &mut Ui,
    id: impl std::hash::Hash,
    width: f32,
    picked: bool,
    enabled: bool,
    title: &str,
    caption: &str,
) -> bool {
    let (fill, stroke, fg, sub) = match (picked, enabled) {
        (true, true) => (theme::ink(), theme::ink(), theme::on_ink(), theme::on_ink().gamma_multiply(0.7)),
        (true, false) => (theme::panel2(), theme::bright(), theme::muted(), theme::muted()),
        (false, true) => (theme::panel2(), theme::border2(), theme::ink(), theme::muted()),
        (false, false) => (theme::panel2(), theme::border(), theme::faint(), theme::faint()),
    };
    let rect = Frame::default()
        .fill(fill)
        .stroke(Stroke::new(HAIR, stroke))
        .corner_radius(CornerRadius::ZERO)
        .inner_margin(Margin::symmetric(12, 8))
        .show(ui, |ui| {
            ui.set_width((width - 26.0).max(40.0));
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 2.0;
                ui.label(RichText::new(title).font(theme::font_medium(13.5)).color(fg));
                ui.label(RichText::new(if caption.is_empty() { " " } else { caption }).size(11.0).color(sub));
            });
        })
        .response
        .rect;
    let resp = ui.interact(rect, egui::Id::new(("bond_choice", id)), if enabled { Sense::click() } else { Sense::hover() });
    if !enabled {
        return false;
    }
    if !picked && resp.hovered() {
        ui.painter().rect_stroke(rect, CornerRadius::ZERO, Stroke::new(HAIR, theme::ink()), egui::StrokeKind::Inside);
    }
    resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked()
}

/// A labelled figure, for a row summing up what a button will do.
pub(super) fn fact(ui: &mut Ui, label: &str, value: &str) {
    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 1.0;
        ui.label(RichText::new(label.to_uppercase()).font(FontId::monospace(10.0)).color(theme::muted()));
        ui.label(RichText::new(value).font(theme::font_mono_medium(15.0)).color(theme::ink()));
    });
    ui.add_space(18.0);
}

/// The one button that moves things on.
pub(super) fn primary(text: &str) -> egui::Button<'static> {
    egui::Button::new(RichText::new(text.to_string()).font(theme::font_medium(13.0)).color(theme::on_ink()))
        .fill(theme::ink())
        .stroke(Stroke::new(HAIR, theme::ink()))
}

/// Lock → Register → Mine → Unbond, as squares on a line: filled behind,
/// ringed where the bond is now, hollow ahead. Monochrome, as the send
/// timeline is.
fn track(ui: &mut Ui, steps: &[(&str, Step)]) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        for (i, (name, step)) in steps.iter().enumerate() {
            if i > 0 {
                let (rect, _) = ui.allocate_exact_size(egui::vec2(18.0, 16.0), Sense::hover());
                let lit = matches!(step, Step::Done | Step::Now);
                let col = if lit { theme::bright() } else { theme::border2() };
                ui.painter().hline(rect.x_range(), rect.center().y, Stroke::new(HAIR, col));
            }
            let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 16.0), Sense::hover());
            let sq = egui::Rect::from_center_size(rect.center(), egui::vec2(8.0, 8.0));
            let p = ui.painter();
            match step {
                Step::Done => {
                    p.rect_filled(sq, CornerRadius::ZERO, theme::bright());
                }
                Step::Now => {
                    p.rect_filled(sq, CornerRadius::ZERO, theme::ink());
                    p.rect_stroke(sq.expand(2.5), CornerRadius::ZERO, Stroke::new(HAIR, theme::ink()), egui::StrokeKind::Inside);
                }
                Step::Next => {
                    p.rect_stroke(sq, CornerRadius::ZERO, Stroke::new(HAIR, theme::faint()), egui::StrokeKind::Inside);
                }
                Step::Skip => {
                    p.hline(sq.x_range(), sq.center().y, Stroke::new(HAIR, theme::faint()));
                }
            }
            let text = RichText::new(*name);
            let text = match step {
                Step::Done => text.font(FontId::monospace(11.0)).color(theme::bright()),
                Step::Now => text.font(theme::font_mono_medium(11.0)).color(theme::ink()),
                Step::Next => text.font(FontId::monospace(11.0)).color(theme::faint()),
                Step::Skip => text.font(FontId::monospace(11.0)).color(theme::faint()).strikethrough(),
            };
            ui.label(text);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bond(status: &str, reg: &str) -> BondView {
        BondView {
            status: status.into(),
            reg_phase: reg.into(),
            value: 1 << 37,
            blocks_left: 90 * 1440,
            counts_until: theme::now_secs() + 60 * 86_400,
            ..Default::default()
        }
    }

    #[test]
    fn sizes_and_times_read_as_people_say_them() {
        assert_eq!(exp_at_least(1 << 37), 37);
        assert_eq!(exp_at_least((1 << 37) + 1), 38);
        assert_eq!(exp_at_least(0), 0);
        assert_eq!(months_label(30), "1 month");
        assert_eq!(months_label(90), "3 months");
        assert_eq!(months_label(334), "11 months");
        assert_eq!(months_label(7), "7 days");
        assert_eq!(about(0), "a minute");
        assert_eq!(about(10), "10 minutes");
        assert_eq!(about(60), "an hour");
        assert_eq!(about(1500), "25 hours");
        assert_eq!(about(3 * 1440), "3 days");
    }

    #[test]
    fn the_longest_time_offered_fits_midwimble_s_horizon() {
        let now = theme::now_secs();
        let longest = midwimble::horizon(now).saturating_sub(midwimble::until_for_days(now, 0)) / 1440;
        assert!(longest > 180, "a year's horizon leaves room for six months");
        assert!(midwimble::until_for_days(now, longest) <= midwimble::horizon(now));
        assert!(midwimble::until_for_days(now, longest + 1) > midwimble::horizon(now));
    }

    #[test]
    fn each_bond_has_one_place_and_one_next_thing() {
        let now = theme::now_secs();
        let s = stage(&bond("locking", "waiting"), now);
        assert_eq!((s.at, s.badge), (0, "locking"));

        let mut b = bond("locked", "");
        b.can_register = true;
        let s = stage(&b, now);
        assert_eq!((s.at, s.badge), (1, "not registered"));
        assert!(matches!(s.next, Some(Next::Register)));

        let mut b = bond("locked", "ready");
        b.wallet_key = false;
        let s = stage(&b, now);
        assert_eq!(s.badge, "send to pool");
        assert!(matches!(s.next, Some(Next::CoBondFile)));

        let mut b = bond("locked", "ready");
        b.wallet_key = true;
        assert!(stage(&b, now).next.is_none(), "a ready bond of this wallet waits for mining");

        assert_eq!(stage(&bond("locked", "on_chain"), now).badge, "mining");

        let mut b = bond("locked", "on_chain");
        b.counts_until = 0;
        assert_eq!(stage(&b, now).at, 3, "past its mining, waiting to unlock");

        let mut b = bond("unlocked", "on_chain");
        b.can_unbond = true;
        assert!(matches!(stage(&b, now).next, Some(Next::Unbond)));

        let mut b = bond("locked", "");
        b.below_min = true;
        let s = stage(&b, now);
        assert!(s.skip_mining && s.badge == "test");

        // A co-bond handed to a pool: the pool registers it, so there is no
        // co-bond file to save.
        let mut b = bond("locked", "pool");
        b.pool = "https://rpc.cypherpunk.gold/pool".into();
        b.reg_progress = 40;
        let s = stage(&b, now);
        assert_eq!((s.at, s.badge), (1, "pool registering"));
        assert!(s.next.is_none() && s.progress.is_some());
        let mut b = bond("locked", "ready");
        b.pool = "https://rpc.cypherpunk.gold/pool".into();
        assert!(stage(&b, now).next.is_none());
        assert_eq!(who(&b), "co-bond · rpc.cypherpunk.gold");
    }
}
