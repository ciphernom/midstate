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
use eframe::egui::{self, Align, FontId, Layout, RichText, TextEdit, Ui};
use midstate_walletd::api::HistoryView;

const FILTERS: [&str; 5] = ["All", "Received", "Sent", "Mined", "Housekeeping"];

const W_WHEN: f32 = 128.0;
const W_TYPE: f32 = 104.0;
const W_VALUE: f32 = 132.0;
const W_FEE: f32 = 64.0;
const W_SHAPE: f32 = 116.0;
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
                    if h.amount > 0 {
                        ui.label(
                            theme::mono(format!("{}{}", if incoming { "+" } else { "~" }, units(h.amount)))
                                .size(12.0)
                                .color(theme::ink()),
                        )
                        .on_hover_text(if incoming {
                            "Value that arrived in this wallet."
                        } else {
                            "Change that came back to this wallet — not the amount sent."
                        });
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
                detail(ui, h);
            }
            ui.separator();
        }
    });

    if let Some(i) = toggle {
        app.hist_open = if app.hist_open == Some(i) { None } else { Some(i) };
    }

    theme::hint(
        ui,
        "This chain records only coin ids, so a value can be shown when this wallet holds the \
         matching coin. What left in a send is known to the recipient, not to the ledger.",
    );
}

fn detail(ui: &mut Ui, h: &HistoryView) {
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
            });
            if !h.inputs.is_empty() {
                ui.add_space(4.0);
                coin_list(ui, "SPENT", &h.inputs);
            }
            if !h.outputs.is_empty() {
                ui.add_space(4.0);
                coin_list(ui, "CREATED", &h.outputs);
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
