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

    lock_form(app, ui, &ctx, height, blocked);
    ui.add_space(10.0);
    bond_list(app, ui, &ctx, blocked);
    ui.add_space(10.0);
    import_form(app, ui, &ctx);

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
        ui.add(
            TextEdit::singleline(&mut app.bond_key)
                .hint_text("the key `midwimble bond new` printed, or your pool's to co-bond")
                .font(egui::TextStyle::Monospace)
                .desired_width(f32::INFINITY),
        );

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Amount").size(12.0).color(theme::muted()));
            let spendable = app.balance.as_ref().map(|b| b.confirmed).unwrap_or(0);
            let top = if spendable == 0 { MIN_EXP } else { (63 - spendable.leading_zeros()).max(MIN_EXP) };
            if app.bond_exp < MIN_EXP {
                app.bond_exp = midwimble::MIN_MINING_BOND.trailing_zeros();
            }
            egui::ComboBox::from_id_salt("bond_amount")
                .selected_text(theme::mono(compact_units(1u64 << app.bond_exp)).size(11.5))
                .show_ui(ui, |ui| {
                    for e in MIN_EXP..=top.max(app.bond_exp) {
                        let v = 1u64 << e;
                        let note = if v < midwimble::MIN_MINING_BOND {
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
            for line in readout(until, value, height, now) {
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
        let key_ok = app.bond_key.trim().trim_start_matches("mining key").trim().len() == 64;
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
                            mining_key: app.bond_key.trim().to_string(),
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
            let key = app.bond_key.trim().trim_start_matches("mining key").trim().to_string();
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

/// What a lock height means, line by line.
fn readout(until: u64, value: u64, height: u64, now: u64) -> Vec<String> {
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
    if value < midwimble::MIN_MINING_BOND {
        out.push(format!(
            "Below midwimble's minimum bond of {}: it will never count for mining. Fine for a rehearsal.",
            compact_units(midwimble::MIN_MINING_BOND)
        ));
        return out;
    }
    // It registers about a day after it is locked.
    let registered = now + midwimble::REGISTRATION_DELAY_BLOCKS * 60;
    match midwimble::counts_until(until) {
        Some(t) if t > registered => out.push(format!(
            "Midwimble counts it for mining for about {} after it registers (about a day from now). \
             It stops counting two days before unlocking, by a clock that runs a week ahead.",
            secs_human(t - registered)
        )),
        _ => out.push(
            "Too short to count for mining: midwimble stops counting a bond once fewer than two days \
             of its lock remain, by a clock that runs a week ahead. Fine for a rehearsal."
                .into(),
        ),
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

            if matches!(b.status.as_str(), "locking" | "locked") && !b.below_min {
                ui.add_space(4.0);
                theme::hint(
                    ui,
                    "Register it with midwimble: run this once the lock confirms, and again a day \
                     later to finish. For a co-bond, use the file `midwimble bond new --mining-pubkey` \
                     made, then give that file to the pool.",
                );
                ui.horizontal(|ui| {
                    ui.add(
                        egui::Label::new(theme::mono(&b.register_cmd).size(10.5).color(theme::bright()))
                            .wrap(),
                    );
                });
                theme::right_aligned(ui, |ui| {
                    if ui.small_button("Copy command").clicked() {
                        ui.ctx().copy_text(b.register_cmd.clone());
                    }
                });
            }

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
