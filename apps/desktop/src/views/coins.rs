use crate::app::App;
use crate::bridge::Action;
use crate::theme::{self, short_hex, units};
use eframe::egui::{self, RichText, TextEdit, Ui};

pub fn show(app: &mut App, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    theme::heading(ui, "Coins");
    theme::hint(
        ui,
        "Every coin is a fixed power-of-two denomination bound to one key. A coin marked \
         SIGNED has used its one-time signature and will only move together with its siblings.",
    );
    ui.add_space(4.0);

    // ── Housekeeping: defrag + consolidate ──────────────────────────────
    let syncing = app.sync.as_ref().map(|s| s.is_syncing).unwrap_or(true);
    ui.columns(2, |cols| {
        theme::panel_frame().show(&mut cols[0], |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Defrag").font(theme::font_medium(14.0)));
            theme::hint(
                ui,
                "Sweeps many small single-use coins into one fresh reusable address. \
                 One batch per run; the app rescans first so no sibling coin is burned.",
            );
            ui.horizontal(|ui| {
                theme::hint(ui, "max inputs");
                let r = ui.add(TextEdit::singleline(&mut app.defrag_max).desired_width(60.0).font(egui::TextStyle::Monospace));
                if r.changed() {
                    app.defrag_max.retain(|c| c.is_ascii_digit());
                }
                theme::right_aligned(ui, |ui| {
                    let can = !app.busy && !syncing;
                    if ui.add_enabled(can, egui::Button::new("Run defrag")).clicked() {
                        let max = app.defrag_max.parse::<usize>().unwrap_or(40).clamp(2, 200);
                        app.busy = true;
                        app.error.clear();
                        app.coins_notice.clear();
                        app.go(&ctx, Action::Defrag { max_inputs: max });
                    }
                });
            });
        });
        theme::panel_frame().show(&mut cols[1], |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Consolidate an address").font(theme::font_medium(14.0)));
            theme::hint(
                ui,
                "Spends every live coin at one address together (single-witness sweep) \
                 into a fresh reusable address. Rescans that address first.",
            );
            // Only addresses actually holding two or more live coins can be
            // consolidated, so offer exactly those rather than a blank field.
            // Owned keys: the dropdown writes back into `app`, so nothing here
            // may keep borrowing `app.coins`.
            let mut groups: std::collections::HashMap<String, (usize, u64)> =
                std::collections::HashMap::new();
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
            } else {
                let label = if app.consolidate_addr.is_empty() {
                    "choose an address…".to_string()
                } else {
                    short_hex(&app.consolidate_addr, 10)
                };
                egui::ComboBox::from_id_salt("consolidate_addr")
                    .selected_text(theme::mono(label).size(12.0))
                    .width(ui.available_width() - 8.0)
                    .show_ui(ui, |ui| {
                        for (addr, n, val) in &candidates {
                            let text =
                                format!("{}  ·  {} coins  ·  {}", short_hex(addr, 10), n, units(*val));
                            let sel = &app.consolidate_addr == addr;
                            if ui.selectable_label(sel, theme::mono(text).size(12.0)).clicked() {
                                app.consolidate_addr = addr.clone();
                            }
                        }
                    });
            }
            theme::right_aligned(ui, |ui| {
                let can = !app.busy && !syncing && !app.consolidate_addr.trim().is_empty();
                if ui.add_enabled(can, egui::Button::new("Consolidate")).clicked() {
                    app.busy = true;
                    app.error.clear();
                    app.coins_notice.clear();
                    app.go(&ctx, Action::Consolidate { address: app.consolidate_addr.trim().to_string() });
                }
            });
        });
    });
    if syncing {
        theme::hint(ui, "Coin housekeeping unlocks when the node reaches the chain tip.");
    }
    if app.busy {
        ui.horizontal(|ui| {
            ui.spinner();
            theme::hint(ui, "working — rescanning and planning…");
        });
    }
    if !app.coins_notice.is_empty() {
        ui.label(RichText::new(&app.coins_notice).size(12.0).color(theme::bright()));
    }
    if !app.error.is_empty() {
        ui.label(RichText::new(&app.error).size(12.0).color(theme::muted()));
    }

    // ── The coin table ──────────────────────────────────────────────────
    ui.add_space(6.0);
    let mut export: Option<String> = None;
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        if app.coins.is_empty() {
            theme::hint(ui, "No coins yet.");
            return;
        }
        egui::Grid::new("coins").num_columns(6).spacing([20.0, 8.0]).striped(true).show(ui, |ui| {
            for c in ["Value", "Coin id", "Address", "Key", "State", ""] {
                ui.label(RichText::new(c.to_uppercase()).font(egui::FontId::monospace(10.0)).color(theme::muted()));
            }
            ui.end_row();
            for c in &app.coins {
                ui.label(theme::mono(units(c.value)));
                ui.label(theme::mono(short_hex(&c.coin_id, 6)).color(theme::muted()).size(12.0));
                ui.label(theme::mono(short_hex(&c.address, 10)).color(theme::muted()).size(12.0));
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
                if ui.button(RichText::new("export").size(11.0)).clicked() {
                    export = Some(c.coin_id.clone());
                }
                ui.end_row();
            }
        });
    });
    if let Some(id) = export {
        app.go(&ctx, Action::ExportCoin { id });
    }

    // ── Export result ───────────────────────────────────────────────────
    if let Some(e) = app.coin_export.clone() {
        ui.add_space(6.0);
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Coin export").font(theme::font_medium(14.0)));
            theme::hint(ui, "Anyone with these three values controls this coin. Handle like cash.");
            for (k, v) in [("value", units(e.value)), ("seed", e.seed.clone()), ("salt", e.salt.clone())] {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(k).font(egui::FontId::monospace(10.0)).color(theme::muted()));
                    ui.label(theme::mono(&v).size(11.5));
                    if ui.button(RichText::new("copy").size(11.0)).clicked() {
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

    // ── Advanced ────────────────────────────────────────────────────────
    ui.add_space(6.0);
    egui::CollapsingHeader::new("Advanced").show(ui, |ui| {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Import a coin").font(theme::font_medium(13.0)));
            theme::hint(ui, "From another wallet's export: seed, value, and salt.");
            ui.add(TextEdit::singleline(&mut app.import_seed).hint_text("seed (64 hex)").font(egui::TextStyle::Monospace).desired_width(f32::INFINITY));
            ui.horizontal(|ui| {
                let r = ui.add(TextEdit::singleline(&mut app.import_value).hint_text("value").font(egui::TextStyle::Monospace).desired_width(140.0));
                if r.changed() { app.import_value.retain(|c| c.is_ascii_digit()); }
                ui.add(TextEdit::singleline(&mut app.import_label).hint_text("label (optional)").desired_width(f32::INFINITY));
            });
            ui.add(TextEdit::singleline(&mut app.import_salt).hint_text("salt (64 hex)").font(egui::TextStyle::Monospace).desired_width(f32::INFINITY));
            theme::right_aligned(ui, |ui| {
                let can = !app.busy && app.import_seed.trim().len() == 64 && app.import_salt.trim().len() == 64 && app.import_value.parse::<u64>().is_ok();
                if ui.add_enabled(can, egui::Button::new("Import coin")).clicked() {
                    app.busy = true;
                    app.error.clear();
                    let label = { let l = app.import_label.trim(); if l.is_empty() { None } else { Some(l.to_string()) } };
                    app.go(&ctx, Action::ImportCoin {
                        seed: app.import_seed.trim().to_string(),
                        value: app.import_value.parse().unwrap_or(0),
                        salt: app.import_salt.trim().to_string(),
                        label,
                    });
                }
            });

            ui.add_space(10.0);
            ui.label(RichText::new("Abandon an address").font(theme::font_medium(13.0)));
            theme::hint(
                ui,
                "Removes this wallet's records for coins at an address. Wallet-local only \
                 — the chain is unaffected. For quarantined or unrecoverable addresses.",
            );
            ui.horizontal(|ui| {
                ui.add(TextEdit::singleline(&mut app.abandon_addr).hint_text("address").font(egui::TextStyle::Monospace).desired_width(f32::INFINITY));
            });
            theme::right_aligned(ui, |ui| {
                let can = !app.busy && !app.abandon_addr.trim().is_empty();
                if ui.add_enabled(can, egui::Button::new("Abandon records")).clicked() {
                    app.busy = true;
                    app.error.clear();
                    app.go(&ctx, Action::AbandonAddress { address: app.abandon_addr.trim().to_string() });
                }
            });
        });
    });
}
