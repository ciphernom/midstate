//! Mining bonds. A bond locks one coin until a block height, naming a
//! midwimble mining key: midwimble lets that key sign its blocks while enough
//! of the lock remains, and only this wallet's recovery phrase can spend the
//! coin once it unlocks. This is the CLI's `generate-mss` → `send` →
//! `spend-script` flow, with the owner key and the script handled here.

use crate::app::App;
use crate::bridge::Action;
use crate::theme::{self, compact_units, short_hex};
use eframe::egui::{self, RichText, TextEdit, Ui};
use midstate_walletd::bond::{blocks_human, midwimble, secs_human};

/// Smallest amount offered: a rehearsal coin.
const MIN_EXP: u32 = 10;

pub fn show(app: &mut App, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    theme::heading(ui, "Bonds");

    let height = app.sync.as_ref().map(|s| s.height).unwrap_or(0);
    let blocked = app
        .sync
        .as_ref()
        .map(|s| s.is_syncing || s.restore_pass > 0)
        .unwrap_or(true);

    let views = ["Your bonds", "Bonds for rent"];
    theme::segmented(ui, &views, &mut app.bonds_view);
    ui.add_space(8.0);
    if app.bonds_view == 1 {
        super::rent::show(app, ui, &ctx, height, blocked);
    } else {
        lock_form(app, ui, &ctx, height, blocked);
        ui.add_space(10.0);
        bond_list(app, ui, &ctx, blocked);
        ui.add_space(10.0);
        import_form(app, ui, &ctx);
    }

    if app.busy {
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.spinner();
            theme::hint(ui, "working…");
        });
    }
    if !app.bond_notice.is_empty() {
        ui.label(RichText::new(&app.bond_notice).size(12.0).color(theme::bright()));
    }
    if !app.error.is_empty() {
        ui.label(RichText::new(&app.error).size(12.0).color(theme::muted()));
    }
}

// ── Lock ────────────────────────────────────────────────────────────────

fn lock_form(app: &mut App, ui: &mut Ui, ctx: &egui::Context, height: u64, blocked: bool) {
    let now = theme::now_secs();
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Lock a mining bond").font(theme::font_medium(14.0)));
        theme::hint(
            ui,
            "One coin, locked until a block height, naming a midwimble mining key. Midwimble lets \
             that key sign blocks while enough of the lock remains. Only this wallet's recovery \
             phrase can spend the coin, and only once it unlocks.",
        );
        ui.add_space(6.0);

        ui.label(RichText::new("Mining key").size(12.0).color(theme::muted()));
        let modes = ["This wallet mines with it", "Another key (a pool's co-bond)"];
        if theme::segmented(ui, &modes, &mut app.bond_key_mode) {
            app.bond_confirm = false;
        }
        let wallet_key = app.bond_rules.as_ref().map(|r| r.mining_key.clone()).unwrap_or_default();
        if app.bond_key_mode == 0 && wallet_key.is_empty() && app.bond_rules.is_some() {
            theme::hint(
                ui,
                "This wallet predates recovery phrases, so it has no mining key of its own: lock on \
                 another key, or mine from a wallet made with a phrase.",
            );
        } else if app.bond_key_mode == 0 {
            ui.horizontal(|ui| {
                ui.label(theme::mono(short_hex(&wallet_key, 14)).size(11.5).color(theme::bright()));
                if !wallet_key.is_empty() && ui.small_button("Copy").clicked() {
                    ui.ctx().copy_text(wallet_key.clone());
                }
            });
            theme::hint(
                ui,
                "This wallet's own mining key, from your recovery phrase. Once the bond registers, mining \
                 on this machine's pool (Node tab) mines MDW with it, and a restored wallet mines with it \
                 again. Every bond on it signs as the same key, which links them on chain.",
            );
        } else {
            ui.add(
                TextEdit::singleline(&mut app.bond_key)
                    .hint_text("your pool's mining key to co-bond, or one `midwimble bond new` printed")
                    .font(egui::TextStyle::Monospace)
                    .desired_width(f32::INFINITY),
            );
        }

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Amount").size(12.0).color(theme::muted()));
            let spendable = app.balance.as_ref().map(|b| b.confirmed).unwrap_or(0);
            let top = if spendable == 0 { MIN_EXP } else { (63 - spendable.leading_zeros()).max(MIN_EXP) };
            let needed = app.bond_rules.as_ref().map_or(midwimble::min_bond(now), |r| r.registration_min);
            if app.bond_exp < MIN_EXP {
                app.bond_exp = needed.trailing_zeros();
            }
            egui::ComboBox::from_id_salt("bond_amount")
                .selected_text(theme::mono(compact_units(1u64 << app.bond_exp)).size(11.5))
                .show_ui(ui, |ui| {
                    for e in MIN_EXP..=top.max(app.bond_exp).max(needed.trailing_zeros()) {
                        let v = 1u64 << e;
                        let note = if v < needed {
                            "  ·  rehearsal only"
                        } else if v > spendable {
                            "  ·  more than you hold"
                        } else {
                            ""
                        };
                        let text = format!("{}{note}", compact_units(v));
                        if ui.selectable_label(app.bond_exp == e, theme::mono(text).size(11.5)).clicked() {
                            app.bond_exp = e;
                        }
                    }
                });
            theme::hint(ui, "one coin, so a power of two");
        });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Unlocks at block").size(12.0).color(theme::muted()));
            let r = ui.add(
                TextEdit::singleline(&mut app.bond_until)
                    .hint_text("height")
                    .font(egui::TextStyle::Monospace)
                    .desired_width(110.0),
            );
            if r.changed() {
                app.bond_until.retain(|c| c.is_ascii_digit());
                app.bond_confirm = false;
            }
            theme::hint(ui, "or count for:");
            for (label, days) in [("30 days", 30u64), ("90 days", 90), ("180 days", 180), ("360 days", 360)] {
                if ui.small_button(label).clicked() {
                    app.bond_until = midwimble::until_for_days(now, days).to_string();
                    app.bond_confirm = false;
                }
            }
        });

        let until: Option<u64> = app.bond_until.parse().ok();
        let value = 1u64 << app.bond_exp;
        if let Some(until) = until {
            let needed = app.bond_rules.as_ref().map_or(midwimble::min_bond(now), |r| r.registration_min);
            for line in readout(until, value, needed, height, now) {
                theme::hint(ui, &line);
            }
        }

        ui.add_space(4.0);
        ui.add(
            TextEdit::singleline(&mut app.bond_label)
                .hint_text("label (optional)")
                .desired_width(f32::INFINITY),
        );

        ui.add_space(6.0);
        let key_text = if app.bond_key_mode == 0 { wallet_key.clone() } else { app.bond_key.clone() };
        let key_ok = key_text.trim().trim_start_matches("mining key").trim().len() == 64;
        let until_ok = until.is_some_and(|u| u > height && u <= midwimble::horizon(now));
        theme::right_aligned(ui, |ui| {
            let can = !app.busy && !blocked && key_ok && until_ok;
            if app.bond_confirm {
                if ui.button("Cancel").clicked() {
                    app.bond_confirm = false;
                }
                if ui.add_enabled(can, egui::Button::new("Confirm and lock")).clicked() {
                    app.busy = true;
                    app.bond_confirm = false;
                    app.error.clear();
                    app.bond_notice.clear();
                    app.go(
                        ctx,
                        Action::CreateBond {
                            mining_key: key_text.trim().to_string(),
                            value,
                            until: until.unwrap_or(0),
                            label: app.bond_label.trim().to_string(),
                        },
                    );
                }
            } else {
                if ui.add_enabled(can, egui::Button::new("Lock bond")).clicked() {
                    app.bond_confirm = true;
                }
                if ui.add_enabled(!app.busy && !blocked, egui::Button::new("Rehearse the exit")).clicked() {
                    app.bond_exp = MIN_EXP;
                    app.bond_until = (height + 10).to_string();
                    app.bond_label = "rehearsal".into();
                    app.bond_confirm = false;
                }
            }
        });
        if app.bond_confirm {
            let key = key_text.trim().trim_start_matches("mining key").trim().to_string();
            theme::hint(
                ui,
                &format!(
                    "Lock {} until block {} on mining key {}? The coin cannot move before then. \
                     Check the key: a bond on the wrong key is locked money that cannot mine.",
                    compact_units(value),
                    until.unwrap_or(0),
                    short_hex(&key, 10)
                ),
            );
        }
        if blocked {
            theme::hint(ui, "Bonds can be locked once the node is synced and any restore scan has finished.");
        }
        theme::hint(
            ui,
            "Before locking a real bond, rehearse the exit: lock a small coin a few blocks out and \
             unbond it, to see this wallet open its own bonds.",
        );
    });
}

/// What a lock height means, line by line. `needed` is what a new bond
/// must hold to register now.
fn readout(until: u64, value: u64, needed: u64, height: u64, now: u64) -> Vec<String> {
    let mut out = Vec::new();
    if until <= height {
        out.push(format!("That is not in the future: the chain is at block {height}."));
        return out;
    }
    let horizon = midwimble::horizon(now);
    if until > horizon {
        out.push(format!(
            "Too far ahead: midwimble registers bonds locked at most a year out (block {horizon} today)."
        ));
        return out;
    }
    out.push(format!("Unlocks in about {} (block {until}).", blocks_human(until - height)));
    if value < needed {
        out.push(format!(
            "Below the {} a new bond needs to register now: it will never count for mining. Fine for a \
             rehearsal.",
            compact_units(needed)
        ));
        return out;
    }
    // It registers about a day after it is locked.
    let registered = now + midwimble::REGISTRATION_DELAY_BLOCKS * 60;
    let days = midwimble::min_remaining_lock(now) / 1440;
    match midwimble::counts_until(until, now) {
        Some(t) if t > registered => out.push(format!(
            "Midwimble counts it for mining for about {} after it registers (about a day from now). It \
             stops counting {days} days before unlocking, by a clock that runs a week ahead.",
            secs_human(t - registered)
        )),
        _ => out.push(format!(
            "Too short to count for mining: midwimble stops counting a bond once {days} days or fewer of \
             its lock remain, by a clock that runs a week ahead. Fine for a rehearsal."
        )),
    }
    out
}

// ── Bonds ───────────────────────────────────────────────────────────────

fn bond_list(app: &mut App, ui: &mut Ui, ctx: &egui::Context, blocked: bool) {
    if app.bonds.is_empty() {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            theme::hint(ui, "No bonds yet.");
        });
        return;
    }
    let bonds = app.bonds.clone();
    for b in &bonds {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                let title = if b.label.is_empty() { "Bond".to_string() } else { b.label.clone() };
                ui.label(RichText::new(title).font(theme::font_medium(14.0)));
                ui.label(theme::mono(compact_units(b.value)).color(theme::ink()));
                theme::badge(ui, &b.status.replace('_', " "), theme::ink());
                match b.rental.as_str() {
                    "rented" => theme::badge(ui, "rented", theme::muted()),
                    "rented_out" => theme::badge(ui, "rented out", theme::muted()),
                    _ => {}
                }
            });
            theme::hint(ui, &b.status_text);
            ui.add_space(4.0);

            let row = |ui: &mut Ui, k: &str, v: &str, copy: Option<&str>| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(k).size(11.5).color(theme::muted()));
                    ui.label(theme::mono(v).size(11.5));
                    if let Some(c) = copy {
                        if ui.small_button("Copy").clicked() {
                            ui.ctx().copy_text(c.to_string());
                        }
                    }
                });
            };
            row(ui, "unlocks at block", &b.bonded_until.to_string(), None);
            row(ui, "mining key", &short_hex(&b.mining_key, 10), Some(&b.mining_key));
            row(ui, "owner key", &short_hex(&b.owner_pk, 10), Some(&b.owner_pk));
            row(ui, "bond id", &short_hex(&b.id, 10), Some(&b.id));
            row(ui, "salt", &short_hex(&b.salt, 10), Some(&b.salt));

            registration(app, ui, ctx, b);

            theme::right_aligned(ui, |ui| {
                if b.can_remove && ui.add_enabled(!app.busy, egui::Button::new("Forget")).clicked() {
                    app.busy = true;
                    app.error.clear();
                    app.go(ctx, Action::RemoveBond { id: b.id.clone() });
                }
                if b.can_unbond
                    && ui.add_enabled(!app.busy && !blocked, egui::Button::new("Unbond")).clicked()
                {
                    app.busy = true;
                    app.error.clear();
                    app.bond_notice.clear();
                    app.go(ctx, Action::Unbond { id: b.id.clone() });
                }
                if ui.small_button("Copy for another wallet").clicked() {
                    ui.ctx().copy_text(b.export_line.clone());
                }
            });
        });
        ui.add_space(6.0);
    }
}

/// A bond's registration with midwimble: where it stands, and the button
/// that starts (or hurries) it.
fn registration(app: &mut App, ui: &mut Ui, ctx: &egui::Context, b: &midstate_walletd::api::BondView) {
    let phase = b.reg_phase.as_str();
    if phase.is_empty() && !b.can_register {
        return;
    }
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label(RichText::new("Midwimble").size(12.0).color(theme::muted()));
        let (badge, colour) = match phase {
            "" => ("not registered", theme::muted()),
            "waiting" => ("registering", theme::bright()),
            "burying" => ("burying the proof", theme::bright()),
            "ready" => ("ready to register", theme::ink()),
            "on_chain" => ("registered", theme::ink()),
            _ => ("retrying", theme::muted()),
        };
        theme::badge(ui, badge, colour);
        if b.wallet_key {
            theme::badge(ui, "mines from this wallet", theme::muted());
        }
    });
    if phase == "burying" {
        theme::progress_bar(ui, b.reg_progress as f32 / 100.0);
    }
    if !b.reg_note.is_empty() {
        theme::hint(ui, &b.reg_note);
    } else if phase.is_empty() {
        theme::hint(
            ui,
            "Register it so midwimble counts it: the wallet takes a proof of the lock from your node, waits \
             for a day of midwimble's work above it, and then the first block mined with it carries it in.",
        );
    }
    if phase == "ready" && !b.wallet_key {
        theme::hint(
            ui,
            "It names another key, so it registers in a block that key's owner mines: give the pool its \
             co-bond file.",
        );
    }
    let rpc_off = app.bond_rules.as_ref().is_some_and(|r| !r.rpc_on);
    if b.can_register && rpc_off {
        theme::hint(
            ui,
            "Registering reads midstate through this app's node RPC, which is off this session \
             (MIDSTATE_RPC_PORT=none). Turn it on to register here, or use the command line below.",
        );
    }
    let co_bond = matches!(phase, "ready" | "on_chain") && !b.wallet_key;
    if co_bond || b.can_register {
        theme::right_aligned(ui, |ui| {
            if co_bond && ui.add_enabled(!app.busy, egui::Button::new("Copy co-bond file")).clicked() {
                app.busy = true;
                app.error.clear();
                app.go(ctx, Action::CopyCoBondFile { id: b.id.clone() });
            }
            if b.can_register {
                let label = if phase.is_empty() { "Register with midwimble" } else { "Check now" };
                if ui.add_enabled(!app.busy && !rpc_off, egui::Button::new(label)).clicked() {
                    app.busy = true;
                    app.error.clear();
                    app.bond_notice.clear();
                    app.go(ctx, Action::RegisterBond { id: b.id.clone() });
                }
            }
        });
    }
    if phase == "on_chain" {
        return;
    }
    egui::CollapsingHeader::new(RichText::new("Register with midwimble's command line instead").size(11.0))
        .id_salt(format!("cli_{}", b.id))
        .show(ui, |ui| {
            theme::hint(ui, "Run it once the lock confirms, and again a day later to finish.");
            ui.add(egui::Label::new(theme::mono(&b.register_cmd).size(10.5).color(theme::bright())).wrap());
            if ui.small_button("Copy command").clicked() {
                ui.ctx().copy_text(b.register_cmd.clone());
            }
        });
}

// ── Import ──────────────────────────────────────────────────────────────

fn import_form(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Import a bond").font(theme::font_medium(14.0)));
        theme::hint(
            ui,
            "A bond this phrase made elsewhere (after a restore, or in the web wallet): paste \
             midwimble's bond.json once `bond register` has filled it in, or the \
             `<script> <value> <salt>` line another wallet copied.",
        );
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
