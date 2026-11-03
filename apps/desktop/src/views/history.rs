//! Transaction history. The chain stores bare coin ids — no amounts, no
//! addresses — so everything here is reconstructed from what this wallet
//! still holds. Values appear where they are genuinely known; the shape of a
//! transaction (how many coins in, how many out, how many ours) is always
//! exact.
//!
//! Layout note: this deliberately does not use `egui::Grid`. A grid sizes its
//! columns from content and collapses them when a cell is wide, which squashed
//! timestamps and amounts into vertical strips. Fixed-width cells laid out by
//! hand keep the table a table, and let an expanded row span the full width.

use crate::app::App;
use crate::theme::{self, ago, fmt_dt, short_hex, units};
use crate::views::mdw::{self as m, mdw};
use eframe::egui::{self, Align, FontId, Layout, RichText, TextEdit, Ui};
use midstate_walletd::api::{HistoryView, MwbHistoryView};

const FILTERS: [&str; 5] = ["All", "Received", "Sent", "Mined", "Housekeeping"];
const MDW_FILTERS: [&str; 5] = ["All", "Received", "Sent", "Mined", "Swaps"];

const W_WHEN: f32 = 128.0;
const W_TYPE: f32 = 104.0;
const W_VALUE: f32 = 132.0;
const W_FEE: f32 = 64.0;
const W_SHAPE: f32 = 116.0;
/// An MDW entry's block and state ("maturing · block 26,400").
const W_WHERE: f32 = 196.0;
const ROW_H: f32 = 34.0;

fn matches_filter(kind: &str, ix: usize) -> bool {
    match ix {
        1 => kind == "received",
        2 => kind == "sent",
        3 => kind == "coinbase",
        4 => kind == "consolidate" || kind == "mixed",
        _ => true,
    }
}

fn kind_label(kind: &str) -> &str {
    match kind {
        "coinbase" => "mined",
        "consolidate" => "consolidated",
        other => other,
    }
}

/// One fixed-width cell. Without an explicit size, egui lets content dictate
/// column width and the table stops lining up.
fn cell(ui: &mut Ui, w: f32, add: impl FnOnce(&mut Ui)) {
    ui.allocate_ui_with_layout(
        egui::vec2(w, ROW_H),
        Layout::left_to_right(Align::Center),
        |ui| {
            ui.set_width(w);
            add(ui);
        },
    );
}

pub fn show(app: &mut App, ui: &mut Ui) {
    theme::heading(ui, "History");
    theme::segmented(ui, &crate::views::send::ASSETS, &mut app.hist_asset);
    ui.add_space(6.0);
    if app.hist_asset == 1 {
        history_mdw(app, ui);
        return;
    }

    ui.horizontal(|ui| {
        theme::segmented(ui, &FILTERS, &mut app.hist_filter);
        ui.add_space(12.0);
        ui.add(
            TextEdit::singleline(&mut app.hist_search)
                .hint_text("find a coin id")
                .font(egui::TextStyle::Monospace)
                .desired_width(180.0),
        );
        if !app.hist_search.is_empty() && ui.button(RichText::new("clear").size(11.0)).clicked() {
            app.hist_search.clear();
        }
    });
    ui.add_space(6.0);

    let needle = app.hist_search.trim().to_lowercase();
    let rows: Vec<(usize, HistoryView)> = app
        .history
        .iter()
        .enumerate()
        .filter(|(_, h)| matches_filter(&h.kind, app.hist_filter))
        .filter(|(_, h)| {
            needle.is_empty()
                || h.inputs.iter().chain(h.outputs.iter()).any(|id| id.starts_with(&needle))
        })
        .map(|(i, h)| (i, h.clone()))
        .collect();

    let in_val: u64 = rows
        .iter()
        .filter(|(_, h)| h.kind == "received" || h.kind == "coinbase")
        .map(|(_, h)| h.amount)
        .sum();
    let fees: u64 = rows.iter().map(|(_, h)| h.fee).sum();
    ui.horizontal(|ui| {
        theme::hint(ui, &format!("{} transaction(s)", rows.len()));
        if in_val > 0 {
            theme::hint(ui, &format!("· {} received", units(in_val)));
        }
        if fees > 0 {
            theme::hint(ui, &format!("· {} paid in fees", units(fees)));
        }
    });
    ui.add_space(4.0);

    if rows.is_empty() {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            theme::hint(
                ui,
                if app.history.is_empty() {
                    "Nothing recorded yet. Received coins and completed sends appear here."
                } else {
                    "No transactions match this filter."
                },
            );
        });
        return;
    }

    let mut toggle: Option<usize> = None;
    let open = app.hist_open;
    let explorer = m::mds_explorer(app);

    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());

        // Header
        ui.horizontal(|ui| {
            for (label, w) in [
                ("When", W_WHEN),
                ("Type", W_TYPE),
                ("Value", W_VALUE),
                ("Fee", W_FEE),
                ("Coins", W_SHAPE),
            ] {
                cell(ui, w, |ui| {
                    ui.label(
                        RichText::new(label.to_uppercase())
                            .font(FontId::monospace(9.5))
                            .color(theme::muted()),
                    );
                });
            }
        });
        ui.separator();

        for (i, h) in &rows {
            let incoming = h.kind == "received" || h.kind == "coinbase";

            ui.horizontal(|ui| {
                cell(ui, W_WHEN, |ui| {
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = 0.0;
                        ui.label(theme::mono(fmt_dt(h.timestamp)).size(11.0));
                        ui.label(RichText::new(ago(h.timestamp)).size(9.5).color(theme::faint()));
                    });
                });
                cell(ui, W_TYPE, |ui| {
                    theme::badge(
                        ui,
                        kind_label(&h.kind),
                        if incoming { theme::ink() } else { theme::muted() },
                    );
                });
                cell(ui, W_VALUE, |ui| {
                    if incoming && h.amount > 0 {
                        ui.label(theme::mono(format!("+{}", units(h.amount))).size(12.0).color(theme::ink()))
                            .on_hover_text("Value that arrived in this wallet.");
                    } else if let Some(sent) = h.sent {
                        ui.label(theme::mono(format!("-{}", units(sent))).size(12.0).color(theme::ink()))
                            .on_hover_text(match &h.to {
                                Some(to) => format!("Sent to {to}"),
                                None => "Sent out of this wallet.".into(),
                            });
                    } else if !incoming {
                        // The chain records no amounts, so a send made before
                        // walletd started keeping its own ledger cannot be
                        // priced after the fact. Say so rather than showing the
                        // leftover change and calling it the amount.
                        ui.label(theme::mono("—").color(theme::faint()))
                            .on_hover_text("Amount not recorded — this send predates value tracking.");
                    } else {
                        ui.label(theme::mono("—").color(theme::faint()));
                    }
                });
                cell(ui, W_FEE, |ui| {
                    ui.label(
                        theme::mono(if h.fee > 0 { units(h.fee) } else { "—".into() })
                            .color(theme::muted())
                            .size(11.5),
                    );
                });
                cell(ui, W_SHAPE, |ui| {
                    ui.label(
                        theme::mono(format!("{} in → {} out", h.n_in, h.n_out))
                            .size(11.0)
                            .color(theme::muted()),
                    );
                });
                let is_open = open == Some(*i);
                if ui
                    .button(RichText::new(if is_open { "hide" } else { "detail" }).size(10.5))
                    .clicked()
                {
                    toggle = Some(*i);
                }
            });

            // Expanded detail spans the full table width, outside any column.
            if open == Some(*i) {
                detail(ui, h, explorer.as_deref());
            }
            ui.separator();
        }
    });

    if let Some(i) = toggle {
        app.hist_open = if app.hist_open == Some(i) { None } else { Some(i) };
    }

    theme::hint(
        ui,
        "Transactions publish every amount on-chain, but this wallet's history file keeps only \
         coin ids. Values shown here are remembered as the wallet sees them, so a past entry \
         does not change when you spend today. Anything still blank can be rebuilt from your \
         block store — Settings, Rebuild history amounts.",
    );
}

fn detail(ui: &mut Ui, h: &HistoryView, explorer: Option<&str>) {
    let left = h.n_out.saturating_sub(h.ours_out);
    ui.add_space(2.0);
    egui::Frame::default()
        .fill(theme::panel2())
        .stroke(egui::Stroke::new(1.0, theme::border()))
        .inner_margin(egui::Margin::symmetric(12, 10))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal_wrapped(|ui| {
                theme::hint(ui, &format!("{} spent", h.n_in));
                theme::hint(ui, &format!("· {} created", h.n_out));
                theme::hint(ui, &format!("· {} ours", h.ours_out));
                if left > 0 {
                    theme::hint(ui, &format!("· {left} left the wallet"));
                }
                if h.fee > 0 {
                    theme::hint(ui, &format!("· fee {}", units(h.fee)));
                }
                if h.change > 0 {
                    theme::hint(ui, &format!("· {} change", units(h.change)));
                }
            });
            if let Some(to) = &h.to {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("TO").font(FontId::monospace(9.0)).color(theme::muted()));
                    ui.label(theme::mono(to).size(11.0).color(theme::bright()));
                    if ui.button(RichText::new("copy").size(10.0)).clicked() {
                        ui.ctx().copy_text(to.clone());
                    }
                });
            }
            if !h.inputs.is_empty() {
                ui.add_space(4.0);
                coin_list(ui, "SPENT", &h.inputs);
            }
            if !h.outputs.is_empty() {
                ui.add_space(4.0);
                coin_list(ui, "CREATED", &h.outputs);
            }
            if let (Some(base), Some(first)) = (explorer, h.outputs.first().or(h.inputs.first())) {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    m::explorer_button(
                        ui,
                        "open in explorer",
                        Some(m::mds_page(base, first)),
                        "Look this transaction's coins up in midstate's explorer, served by your own node",
                    );
                    theme::hint(ui, "midstate's explorer, on your own node");
                });
            }
        });
}

fn coin_list(ui: &mut Ui, label: &str, ids: &[String]) {
    ui.label(RichText::new(label).font(FontId::monospace(9.0)).color(theme::muted()));
    // Full width available here, so wrapping produces rows of chips rather
    // than one character per line.
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(4.0, 3.0);
        for id in ids {
            if ui
                .add(
                    egui::Button::new(
                        RichText::new(short_hex(id, 6)).font(FontId::monospace(10.0)),
                    )
                    .min_size(egui::vec2(96.0, 18.0)),
                )
                .on_hover_text("click to copy the full coin id")
                .clicked()
            {
                ui.ctx().copy_text(id.clone());
            }
        }
    });
}

// ── MDW ─────────────────────────────────────────────────────────────────

fn matches_mdw(h: &MwbHistoryView, ix: usize) -> bool {
    match ix {
        1 => h.kind == "received",
        2 => h.kind == "sent",
        3 => h.kind == "mined",
        4 => h.kind == "swap" || h.kind == "refund" || h.kind == "swap lock",
        _ => true,
    }
}

/// The MDW account's history. Unlike midstate, midwimble keeps amounts
/// hidden on chain, so every figure here is this wallet's own: what its keys
/// opened, and what it paid.
fn history_mdw(app: &mut App, ui: &mut Ui) {
    if app.mwb_account.as_ref().is_some_and(|a| !a.enabled) {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            theme::hint(ui, "This wallet has no MDW account yet: add it from the Receive tab.");
        });
        return;
    }
    theme::segmented(ui, &MDW_FILTERS, &mut app.mwb_hist_filter);
    ui.add_space(6.0);
    let rows: Vec<(usize, MwbHistoryView)> = app
        .mwb_history
        .iter()
        .enumerate()
        .filter(|(_, h)| matches_mdw(h, app.mwb_hist_filter))
        .map(|(i, h)| (i, h.clone()))
        .collect();
    let received: u64 = rows.iter().filter(|(_, h)| h.incoming).map(|(_, h)| h.amount).sum();
    let paid: u64 = rows.iter().filter(|(_, h)| !h.incoming && h.status != "failed").map(|(_, h)| h.amount).sum();
    let fees: u64 = rows.iter().filter(|(_, h)| h.status != "failed").map(|(_, h)| h.fee).sum();
    ui.horizontal(|ui| {
        theme::hint(ui, &format!("{} entr{}", rows.len(), if rows.len() == 1 { "y" } else { "ies" }));
        if received > 0 {
            theme::hint(ui, &format!("· {} in", mdw(received)));
        }
        if paid > 0 {
            theme::hint(ui, &format!("· {} out", mdw(paid)));
        }
        if fees > 0 {
            theme::hint(ui, &format!("· {} in fees", mdw(fees)));
        }
    });
    ui.add_space(4.0);
    if rows.is_empty() {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            theme::hint(
                ui,
                if app.mwb_history.is_empty() {
                    "Nothing yet. MDW you receive, send or trade appears here."
                } else {
                    "Nothing matches this filter."
                },
            );
        });
        return;
    }

    let explorer = m::mwb_explorer(app);
    let open = app.mwb_hist_open;
    let mut toggle: Option<usize> = None;
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            for (label, w) in [("When", W_WHEN), ("Type", W_TYPE), ("Amount", W_VALUE), ("Fee", W_FEE), ("Where", W_WHERE)] {
                cell(ui, w, |ui| {
                    ui.label(RichText::new(label.to_uppercase()).font(FontId::monospace(9.5)).color(theme::muted()));
                });
            }
        });
        ui.separator();
        for (i, h) in &rows {
            ui.horizontal(|ui| {
                cell(ui, W_WHEN, |ui| {
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = 0.0;
                        if h.time > 0 {
                            ui.label(theme::mono(fmt_dt(h.time)).size(11.0));
                            ui.label(RichText::new(ago(h.time)).size(9.5).color(theme::faint()));
                        } else {
                            ui.label(theme::mono("—").color(theme::faint()));
                        }
                    });
                });
                cell(ui, W_TYPE, |ui| {
                    theme::badge(ui, m::kind_label(&h.kind), if h.incoming { theme::ink() } else { theme::muted() });
                });
                cell(ui, W_VALUE, |ui| {
                    let sign = if h.incoming { "+" } else { "-" };
                    let colour = if h.status == "failed" { theme::faint() } else { theme::ink() };
                    ui.label(theme::mono(format!("{sign}{}", m::plain(h.amount))).size(12.0).color(colour))
                        .on_hover_text(mdw(h.amount));
                });
                cell(ui, W_FEE, |ui| {
                    ui.label(
                        theme::mono(if h.fee > 0 { m::plain(h.fee) } else { "—".into() })
                            .color(theme::muted())
                            .size(11.5),
                    );
                });
                cell(ui, W_WHERE, |ui| {
                    let colour = if h.status == "pending" { theme::bright() } else { theme::muted() };
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = 0.0;
                        ui.label(theme::mono(m::status_text(h)).size(11.0).color(colour));
                        if let Some(settled) = m::settled_text(h) {
                            ui.label(RichText::new(settled).size(9.5).color(theme::faint()))
                                .on_hover_text(m::FULL_WEIGHT_HINT);
                        }
                    });
                });
                let is_open = open == Some(*i);
                if ui.button(RichText::new(if is_open { "hide" } else { "detail" }).size(10.5)).clicked() {
                    toggle = Some(*i);
                }
                m::explorer_button(
                    ui,
                    "explorer",
                    explorer.as_deref().and_then(|e| m::history_page(e, h)),
                    "Open it in midwimble's explorer, served by your own node",
                );
            });
            if open == Some(*i) {
                mdw_detail(ui, h);
            }
            ui.separator();
        }
    });
    if let Some(i) = toggle {
        app.mwb_hist_open = if app.mwb_hist_open == Some(i) { None } else { Some(i) };
    }
    theme::hint(
        ui,
        "Midwimble hides amounts on chain: each output is a commitment only its owner can open. \
         These figures are what this wallet's keys opened and what it paid; the explorer shows \
         the same transactions without them. Change from your own payments is left out.",
    );
}

fn mdw_detail(ui: &mut Ui, h: &MwbHistoryView) {
    ui.add_space(2.0);
    egui::Frame::default()
        .fill(theme::panel2())
        .stroke(egui::Stroke::new(1.0, theme::border()))
        .inner_margin(egui::Margin::symmetric(12, 10))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            let line = |ui: &mut Ui, label: &str, value: &str| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(label).font(FontId::monospace(9.0)).color(theme::muted()));
                    ui.label(theme::mono(value).size(11.0).color(theme::bright()));
                    if ui.button(RichText::new("copy").size(10.0)).clicked() {
                        ui.ctx().copy_text(value.to_string());
                    }
                });
            };
            ui.horizontal_wrapped(|ui| {
                theme::hint(ui, &mdw(h.amount));
                if h.fee > 0 {
                    theme::hint(ui, &format!("· fee {}", mdw(h.fee)));
                }
                theme::hint(ui, &format!("· {}", m::status_line(h)));
            });
            if let Some(to) = &h.to {
                line(ui, "TO", to);
            }
            if let Some(o) = &h.output {
                line(ui, "OUTPUT", o);
            }
            if let Some(k) = &h.kernel {
                line(ui, "KERNEL", k);
            }
            if let Some(s) = &h.swap {
                line(ui, "SWAP", s);
            }
        });
}
