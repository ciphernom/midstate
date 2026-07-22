//! Coins. Every coin is a fixed power-of-two denomination bound to one key,
//! so a wallet's shape — how many coins, in what sizes, on how many keys —
//! determines what it can actually spend. That is what this tab shows, with
//! the tools that reshape it kept one click away rather than in the way.

use crate::app::{App, CoinsTab};
use crate::bridge::Action;
use crate::theme::{self, short_hex, units};
use eframe::egui::{self, FontId, RichText, TextEdit, Ui};

pub fn show(app: &mut App, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    theme::heading(ui, "Coins");

    let frag = app.coins.iter().filter(|c| c.live && c.kind != "mss").count();
    let l1 = if frag > 8 { format!("Housekeeping ({frag})") } else { "Housekeeping".to_string() };
    let labels = ["Holdings", l1.as_str(), "Advanced"];
    let mut ix = match app.coins_tab {
        CoinsTab::Holdings => 0,
        CoinsTab::Housekeeping => 1,
        CoinsTab::Advanced => 2,
    };
    if theme::segmented(ui, &labels, &mut ix) {
        app.coins_tab = [CoinsTab::Holdings, CoinsTab::Housekeeping, CoinsTab::Advanced][ix];
        app.error.clear();
        app.coins_notice.clear();
    }
    ui.add_space(10.0);

    match app.coins_tab {
        CoinsTab::Holdings => holdings(app, ui, &ctx),
        CoinsTab::Housekeeping => housekeeping(app, ui, &ctx),
        CoinsTab::Advanced => advanced(app, ui, &ctx),
    }

    if app.busy {
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.spinner();
            theme::hint(ui, "working…");
        });
    }
    if !app.coins_notice.is_empty() {
        ui.label(RichText::new(&app.coins_notice).size(12.0).color(theme::bright()));
    }
    if !app.error.is_empty() {
        ui.label(RichText::new(&app.error).size(12.0).color(theme::muted()));
    }
}

// ── Holdings ────────────────────────────────────────────────────────────

fn holdings(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    if app.coins.is_empty() {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            theme::hint(ui, "No coins yet.");
        });
        return;
    }

    // Composition: the denominations you hold decide what you can pay without
    // change, so it is worth seeing at a glance.
    let live: Vec<_> = app.coins.iter().filter(|c| c.live && !c.in_flight).collect();
    let total: u64 = live.iter().map(|c| c.value).sum();
    let spendable = live.iter().filter(|c| !c.wots_signed).count();
    let reusable = live.iter().filter(|c| c.kind == "mss").count();

    ui.columns(4, |c| {
        theme::stat(&mut c[0], "Value", &units(total), "units");
        theme::stat(&mut c[1], "Coins", &units(live.len() as u64), "live");
        theme::stat(&mut c[2], "Spendable now", &units(spendable as u64), "");
        theme::stat(&mut c[3], "On reusable keys", &units(reusable as u64), "");
    });

    // Denomination histogram.
    let mut denoms: std::collections::BTreeMap<u64, usize> = Default::default();
    for c in &live {
        *denoms.entry(c.value).or_insert(0) += 1;
    }
    if denoms.len() > 1 {
        ui.add_space(4.0);
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Denominations").font(theme::font_medium(13.0)));
            let peak = denoms.values().copied().max().unwrap_or(1) as f32;
            for (v, n) in denoms.iter().rev().take(14) {
                ui.horizontal(|ui| {
                    ui.label(theme::mono(format!("{:>18}", units(*v))).size(11.0));
                    let w = (ui.available_width() - 60.0).max(20.0) * (*n as f32 / peak);
                    let (rect, _) = ui.allocate_exact_size(
                        egui::vec2(w.max(1.0), 9.0),
                        egui::Sense::hover(),
                    );
                    ui.painter().rect_filled(rect, 0.0, theme::bright());
                    ui.label(theme::mono(format!("{n}")).size(11.0).color(theme::muted()));
                });
            }
        });
    }

    // The table.
    ui.add_space(6.0);
    let mut export: Option<String> = None;
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        egui::Grid::new("coins").num_columns(6).spacing([18.0, 7.0]).striped(true).show(ui, |ui| {
            for c in ["Value", "Coin id", "Address", "Key", "State", ""] {
                ui.label(
                    RichText::new(c.to_uppercase())
                        .font(FontId::monospace(10.0))
                        .color(theme::muted()),
                );
            }
            ui.end_row();
            for c in &app.coins {
                ui.label(theme::mono(units(c.value)).size(12.5));
                ui.label(theme::mono(short_hex(&c.coin_id, 6)).color(theme::muted()).size(11.5));
                ui.label(theme::mono(short_hex(&c.address, 10)).color(theme::muted()).size(11.5));
                theme::badge(
                    ui,
                    if c.kind == "mss" { "reusable" } else { "one-time" },
                    if c.kind == "mss" { theme::ink() } else { theme::muted() },
                );
                if c.in_flight {
                    theme::badge(ui, "in flight", theme::muted());
                } else if !c.live {
                    theme::badge(ui, "off-chain", theme::muted());
                } else if c.wots_signed {
                    theme::badge(ui, "signed", theme::muted());
                } else {
                    theme::badge(ui, "spendable", theme::ink());
                }
                if ui.button(RichText::new("export").size(10.5)).clicked() {
                    export = Some(c.coin_id.clone());
                }
                ui.end_row();
            }
        });
    });
    if let Some(id) = export {
        app.go(ctx, Action::ExportCoin { id });
    }

    if let Some(e) = app.coin_export.clone() {
        ui.add_space(6.0);
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Coin export").font(theme::font_medium(14.0)));
            theme::hint(ui, "Anyone with these three values controls this coin. Handle like cash.");
            for (k, v) in [("value", units(e.value)), ("seed", e.seed.clone()), ("salt", e.salt.clone())] {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(k).font(FontId::monospace(10.0)).color(theme::muted()));
                    ui.label(theme::mono(&v).size(11.0));
                    if ui.button(RichText::new("copy").size(10.5)).clicked() {
                        ui.ctx().copy_text(v.clone());
                    }
                });
            }
            theme::right_aligned(ui, |ui| {
                if ui.button("Close").clicked() {
                    app.coin_export = None;
                }
            });
        });
    }
}

// ── Housekeeping ────────────────────────────────────────────────────────

fn housekeeping(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    let syncing = app.sync.as_ref().map(|s| s.is_syncing).unwrap_or(true);

    ui.columns(2, |cols| {
        // Defrag
        theme::panel_frame().show(&mut cols[0], |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Defrag").font(theme::font_medium(14.0)));
            theme::hint(
                ui,
                "Sweeps many small one-time coins into a single reusable address. One batch \
                 per run; the wallet rescans first so no sibling coin is stranded.",
            );
            ui.horizontal(|ui| {
                theme::hint(ui, "max inputs");
                let r = ui.add(
                    TextEdit::singleline(&mut app.defrag_max)
                        .desired_width(56.0)
                        .font(egui::TextStyle::Monospace),
                );
                if r.changed() {
                    app.defrag_max.retain(|c| c.is_ascii_digit());
                }
                theme::right_aligned(ui, |ui| {
                    if ui
                        .add_enabled(!app.busy && !syncing, egui::Button::new("Run defrag"))
                        .clicked()
                    {
                        let max = app.defrag_max.parse::<usize>().unwrap_or(40).clamp(2, 200);
                        app.busy = true;
                        app.error.clear();
                        app.coins_notice.clear();
                        app.go(ctx, Action::Defrag { max_inputs: max });
                    }
                });
            });
        });

        // Consolidate — same footprint as defrag.
        theme::panel_frame().show(&mut cols[1], |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Consolidate an address").font(theme::font_medium(14.0)));
            theme::hint(
                ui,
                "Spends every live coin at one address together, into a fresh reusable \
                 address. Only addresses holding two or more coins can be consolidated.",
            );

            let mut groups: std::collections::HashMap<String, (usize, u64)> = Default::default();
            for coin in app.coins.iter().filter(|c| c.live && !c.in_flight) {
                let e = groups.entry(coin.address.clone()).or_insert((0, 0));
                e.0 += 1;
                e.1 += coin.value;
            }
            let mut candidates: Vec<(String, usize, u64)> = groups
                .into_iter()
                .filter(|(_, (n, _))| *n >= 2)
                .map(|(a, (n, v))| (a, n, v))
                .collect();
            candidates.sort_by(|a, b| b.2.cmp(&a.2));

            if candidates.is_empty() {
                theme::hint(ui, "No address currently holds two or more live coins.");
                app.consolidate_addr.clear();
                return;
            }
            ui.horizontal(|ui| {
                let label = if app.consolidate_addr.is_empty() {
                    "choose an address…".to_string()
                } else {
                    short_hex(&app.consolidate_addr, 8)
                };
                egui::ComboBox::from_id_salt("consolidate_addr")
                    .selected_text(theme::mono(label).size(11.5))
                    .width(ui.available_width() - 110.0)
                    .show_ui(ui, |ui| {
                        for (addr, n, val) in &candidates {
                            let text =
                                format!("{}  ·  {} coins  ·  {}", short_hex(addr, 8), n, units(*val));
                            let sel = &app.consolidate_addr == addr;
                            if ui.selectable_label(sel, theme::mono(text).size(11.5)).clicked() {
                                app.consolidate_addr = addr.clone();
                            }
                        }
                    });
                let can = !app.busy && !syncing && !app.consolidate_addr.trim().is_empty();
                if ui.add_enabled(can, egui::Button::new("Consolidate")).clicked() {
                    app.busy = true;
                    app.error.clear();
                    app.coins_notice.clear();
                    app.go(
                        ctx,
                        Action::Consolidate { address: app.consolidate_addr.trim().to_string() },
                    );
                }
            });
        });
    });

    if syncing {
        theme::hint(ui, "Coin housekeeping unlocks when the node reaches the chain tip.");
    }

    ui.add_space(8.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        theme::hint(
            ui,
            "Why this exists: a one-time key can sign once, so every coin sharing an address \
             must be spent in the same transaction — and a normal send tops out at 256 inputs. \
             Sweeping coins onto a reusable address keeps your wallet spendable. Progress \
             appears on the Send tab.",
        );
    });
}

// ── Advanced ────────────────────────────────────────────────────────────

fn advanced(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Import a coin").font(theme::font_medium(14.0)));
        theme::hint(ui, "From another wallet's export: seed, value and salt.");
        ui.add(
            TextEdit::singleline(&mut app.import_seed)
                .hint_text("seed (64 hex)")
                .font(egui::TextStyle::Monospace)
                .desired_width(f32::INFINITY),
        );
        ui.horizontal(|ui| {
            let r = ui.add(
                TextEdit::singleline(&mut app.import_value)
                    .hint_text("value")
                    .font(egui::TextStyle::Monospace)
                    .desired_width(130.0),
            );
            if r.changed() {
                app.import_value.retain(|c| c.is_ascii_digit());
            }
            ui.add(
                TextEdit::singleline(&mut app.import_label)
                    .hint_text("label (optional)")
                    .desired_width(f32::INFINITY),
            );
        });
        ui.add(
            TextEdit::singleline(&mut app.import_salt)
                .hint_text("salt (64 hex)")
                .font(egui::TextStyle::Monospace)
                .desired_width(f32::INFINITY),
        );
        theme::right_aligned(ui, |ui| {
            let can = !app.busy
                && app.import_seed.trim().len() == 64
                && app.import_salt.trim().len() == 64
                && app.import_value.parse::<u64>().is_ok();
            if ui.add_enabled(can, egui::Button::new("Import coin")).clicked() {
                app.busy = true;
                app.error.clear();
                let label = {
                    let l = app.import_label.trim();
                    if l.is_empty() { None } else { Some(l.to_string()) }
                };
                app.go(
                    ctx,
                    Action::ImportCoin {
                        seed: app.import_seed.trim().to_string(),
                        value: app.import_value.parse().unwrap_or(0),
                        salt: app.import_salt.trim().to_string(),
                        label,
                    },
                );
            }
        });
    });

    ui.add_space(8.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Abandon an address").font(theme::font_medium(14.0)));
        theme::hint(
            ui,
            "Removes this wallet's records for coins at an address. Wallet-local only — the \
             chain is unaffected, and the coins still exist. For quarantined or unrecoverable \
             addresses that clutter your balance.",
        );
        ui.add(
            TextEdit::singleline(&mut app.abandon_addr)
                .hint_text("address")
                .font(egui::TextStyle::Monospace)
                .desired_width(f32::INFINITY),
        );
        theme::right_aligned(ui, |ui| {
            let can = !app.busy && !app.abandon_addr.trim().is_empty();
            if ui.add_enabled(can, egui::Button::new("Abandon records")).clicked() {
                app.busy = true;
                app.error.clear();
                app.go(ctx, Action::AbandonAddress { address: app.abandon_addr.trim().to_string() });
            }
        });
    });
}
