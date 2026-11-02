//! MDS <-> MDW: atomic swaps with midwimble.
//!
//! Orders are advertised on midstate's chat bus and stand while the maker's
//! wallet is online. Taking one starts an encrypted negotiation between the
//! two wallets; once every signature is exchanged the maker locks first, the
//! taker second, and each wallet watches both chains until it has claimed
//! what it bought or taken its own coins back. Nothing here needs either
//! side to trust the other: the locks only open for the secret the trade
//! reveals, or for their owner after their deadline.

use crate::app::{App, Tab};
use crate::bridge::Action;
use crate::theme::{self, compact_units, short_hex};
use crate::views::mdw::{self as m, mdw, parse_mdw, COIN};
use eframe::egui::{self, FontId, RichText, TextEdit, Ui};
use midstate_walletd::api::{MwbBookEntryView, MwbSwapView};

const TABS: [&str; 3] = ["Order book", "My orders", "Swaps"];

/// MDW per gMDS: both sides of every order in comparable figures.
fn price(mds: u64, mdw_units: u64) -> String {
    if mds == 0 {
        return "—".into();
    }
    let per = mdw_units as f64 / COIN as f64 / (mds as f64 / theme::UNIT_MULS[3] as f64);
    let s = format!("{per:.6}");
    format!("{} MDW/gMDS", s.trim_end_matches('0').trim_end_matches('.'))
}

pub fn show(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    theme::hint(
        ui,
        "Atomic swaps with midwimble. Neither chain was changed for this: a cut-and-choose \
         proof ties midstate's hash locks to midwimble's adaptor signatures, so either both \
         sides get paid or both get their coins back. Orders live on the chat bus while their \
         maker is online.",
    );
    ui.add_space(6.0);
    let enabled = account_strip(app, ui, ctx);
    ui.add_space(8.0);
    if !enabled {
        return;
    }

    let live = app.mwb_swaps.iter().filter(|s| !s.settled).count();
    if live > 0 && app.mwb_tab != 2 {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                theme::badge(ui, "in progress", theme::ink());
                ui.label(
                    RichText::new(format!(
                        "{live} swap(s) running — keep the wallet open until they finish; it acts on \
                         their deadlines for you."
                    ))
                    .size(12.0),
                );
                if ui.button(RichText::new("view").size(11.0)).clicked() {
                    app.mwb_tab = 2;
                }
            });
        });
        ui.add_space(6.0);
    }

    let mut ix = app.mwb_tab.min(2);
    if theme::segmented(ui, &TABS, &mut ix) {
        app.mwb_tab = ix;
    }
    ui.add_space(8.0);
    if !app.mwb_notice.is_empty() {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new(&app.mwb_notice).size(12.0).color(theme::bright()));
        });
        ui.add_space(6.0);
    }
    match app.mwb_tab {
        0 => {
            place_panel(app, ui, ctx);
            book(app, ui, ctx);
        }
        1 => my_orders(app, ui, ctx),
        _ => swaps(app, ui),
    }
}

/// The MDW account in a line, or the form that adds it. Returns whether it
/// exists.
fn account_strip(app: &mut App, ui: &mut Ui, ctx: &egui::Context) -> bool {
    if !m::account_gate(app, ui, ctx) {
        return false;
    }
    let Some(a) = app.mwb_account.clone() else { return false };
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        m::node_line(app, ui);
        ui.horizontal(|ui| {
            ui.label(theme::mono(mdw(a.spendable)).size(13.0).color(theme::ink()));
            if a.reserved > 0 {
                ui.label(theme::mono(format!("+ {} locked", mdw(a.reserved))).size(11.0).color(theme::muted()));
            }
            if a.immature > 0 {
                ui.label(theme::mono(format!("+ {} maturing", mdw(a.immature))).size(11.0).color(theme::muted()));
            }
            theme::right_aligned(ui, |ui| {
                if ui.button(RichText::new("Send").size(11.0)).clicked() {
                    app.tab = Tab::Send;
                    app.send_asset = 1;
                }
                if ui.button(RichText::new("Receive").size(11.0)).clicked() {
                    app.tab = Tab::Receive;
                    app.recv_asset = 1;
                }
            });
        });
    });
    true
}

fn takeable(e: &MwbBookEntryView, tip: u64) -> bool {
    !e.mine && e.expires > tip
}

fn book(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    let tip = app.sync.as_ref().map_or(0, |s| s.height);
    for (side, title, blurb) in [
        ("sell_mds", "Selling MDS", "Pay MDW, receive MDS. The seller locks first; your MDW moves only once their MDS is locked on chain."),
        ("sell_mdw", "Selling MDW", "Pay MDS, receive MDW. The seller locks first; your MDS moves only once their MDW is locked on chain."),
    ] {
        ui.label(RichText::new(title).font(theme::font_medium(14.0)).color(theme::ink()));
        theme::hint(ui, blurb);
        let mut rows: Vec<MwbBookEntryView> = app.mwb_book.iter().filter(|e| e.side == side).cloned().collect();
        rows.sort_by(|a, b| {
            let pa = a.mdw_amount as f64 / a.mds_amount.max(1) as f64;
            let pb = b.mdw_amount as f64 / b.mds_amount.max(1) as f64;
            pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
        });
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            if rows.is_empty() {
                theme::hint(ui, "No orders heard on the bus right now.");
                return;
            }
            egui::Grid::new(format!("mwb_book_{side}")).num_columns(5).spacing([18.0, 7.0]).striped(true).show(ui, |ui| {
                for h in ["Price", "MDS", "MDW", "Stands for", ""] {
                    ui.label(RichText::new(h.to_uppercase()).font(FontId::monospace(9.5)).color(theme::muted()));
                }
                ui.end_row();
                for e in &rows {
                    ui.label(theme::mono(price(e.mds_amount, e.mdw_amount)).size(11.5).color(theme::ink()));
                    ui.label(theme::mono(compact_units(e.mds_amount)).size(11.5));
                    ui.label(theme::mono(mdw(e.mdw_amount)).size(11.5));
                    ui.label(theme::mono(format!("{} blocks", e.expires.saturating_sub(tip))).size(11.0).color(theme::muted()));
                    if e.mine {
                        theme::badge(ui, "yours", theme::ink());
                    } else if takeable(e, tip) {
                        let armed = app.mwb_take_armed.as_deref() == Some(e.order_id.as_str());
                        let label = if armed { "Confirm" } else { "Take" };
                        if ui.add_enabled(!app.busy, egui::Button::new(label)).clicked() {
                            if armed {
                                app.busy = true;
                                app.error.clear();
                                app.mwb_take_armed = None;
                                app.go(ctx, Action::MwbTake { order_id: e.order_id.clone() });
                            } else {
                                app.mwb_take_armed = Some(e.order_id.clone());
                            }
                        }
                    } else {
                        ui.label("");
                    }
                    ui.end_row();
                }
            });
        });
        ui.add_space(8.0);
    }
    if let Some(id) = app.mwb_take_armed.clone() {
        if let Some(e) = app.mwb_book.iter().find(|e| e.order_id == id) {
            let (give, get) = if e.side == "sell_mds" {
                (mdw(e.mdw_amount), compact_units(e.mds_amount))
            } else {
                (compact_units(e.mds_amount), mdw(e.mdw_amount))
            };
            theme::hint(
                ui,
                &format!(
                    "Taking this order gives {give} for {get}, less a small fee on each side's claim. \
                     Press Confirm to start; keep the wallet open until the swap finishes."
                ),
            );
        }
    }
}

fn place_panel(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    let syncing = app.sync.as_ref().map(|s| s.is_syncing).unwrap_or(true);
    egui::CollapsingHeader::new("Make an order")
        .id_salt("mwb_place")
        .default_open(app.mwb_orders.iter().all(|o| o.status != "open"))
        .show(ui, |ui| {
            theme::panel_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                theme::segmented(ui, &["Sell MDS for MDW", "Sell MDW for MDS"], &mut app.mwb_side);
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    theme::hint(ui, "MDS");
                    ui.add(TextEdit::singleline(&mut app.mwb_mds).hint_text("amount").font(egui::TextStyle::Monospace).desired_width(130.0));
                    theme::unit_selector(ui, &mut app.mwb_mds_unit);
                });
                ui.horizontal(|ui| {
                    theme::hint(ui, "MDW");
                    ui.add(TextEdit::singleline(&mut app.mwb_mdw).hint_text("0.0").font(egui::TextStyle::Monospace).desired_width(130.0));
                });
                ui.horizontal(|ui| {
                    theme::hint(ui, "order stands for");
                    let r = ui.add(TextEdit::singleline(&mut app.mwb_life).font(egui::TextStyle::Monospace).desired_width(70.0));
                    if r.changed() {
                        app.mwb_life.retain(|c| c.is_ascii_digit());
                    }
                    let blocks: u64 = app.mwb_life.parse().unwrap_or(0);
                    theme::hint(ui, &format!("midstate blocks (~{:.1} h)", blocks as f64 / 60.0));
                });
                let mds = theme::parse_in_unit(&app.mwb_mds, app.mwb_mds_unit);
                let mdw_units = parse_mdw(&app.mwb_mdw);
                match (&mds, &mdw_units) {
                    (Ok(a), Ok(b)) => {
                        ui.label(RichText::new(format!("≈ {}", price(*a, *b))).size(12.0).color(theme::bright()));
                    }
                    (Err(e), _) | (_, Err(e)) if !e.is_empty() => {
                        ui.label(RichText::new(e).size(12.0).color(theme::amber()));
                    }
                    _ => {}
                }
                theme::hint(
                    ui,
                    if app.mwb_side == 0 {
                        "When someone takes it, your wallet negotiates the swap with theirs, then locks \
                         your MDS. You claim their MDW, which hands them the secret to claim your MDS. \
                         If they never lock, your MDS comes back after about five hours."
                    } else {
                        "When someone takes it, your wallet negotiates the swap with theirs, then locks \
                         your MDW. You claim their MDS, which hands them the secret to claim your MDW. \
                         If they never lock, your MDW comes back after about five hours."
                    },
                );
                theme::right_aligned(ui, |ui| {
                    let can = !app.busy && !syncing && mds.is_ok() && mdw_units.is_ok();
                    if ui.add_enabled(can, egui::Button::new("Publish order")).clicked() {
                        app.busy = true;
                        app.error.clear();
                        app.go(
                            ctx,
                            Action::MwbPlaceOrder {
                                side: if app.mwb_side == 0 { "sell_mds".into() } else { "sell_mdw".into() },
                                mds_amount: mds.clone().unwrap_or(0),
                                mdw_amount: mdw_units.clone().unwrap_or(0),
                                lifetime_blocks: app.mwb_life.parse().unwrap_or(1_440),
                            },
                        );
                    }
                });
            });
        });
    ui.add_space(8.0);
}

fn my_orders(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    theme::hint(ui, "Orders you made. Open ones are re-announced every few blocks while the wallet runs.");
    ui.add_space(4.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        if app.mwb_orders.is_empty() {
            theme::hint(ui, "No orders yet: make one from the Order book tab.");
            return;
        }
        let orders = app.mwb_orders.clone();
        egui::Grid::new("mwb_orders").num_columns(5).spacing([18.0, 7.0]).striped(true).show(ui, |ui| {
            for h in ["Selling", "For", "Status", "Expires", ""] {
                ui.label(RichText::new(h.to_uppercase()).font(FontId::monospace(9.5)).color(theme::muted()));
            }
            ui.end_row();
            for o in &orders {
                let (sell, buy) = if o.side == "sell_mds" {
                    (compact_units(o.mds_amount), mdw(o.mdw_amount))
                } else {
                    (mdw(o.mdw_amount), compact_units(o.mds_amount))
                };
                ui.label(theme::mono(sell).size(11.5));
                ui.label(theme::mono(buy).size(11.5));
                theme::badge(ui, &o.status, if o.status == "open" || o.status == "taken" { theme::ink() } else { theme::muted() });
                ui.label(theme::mono(format!("block {}", theme::units(o.expires))).size(11.0).color(theme::muted()));
                if o.status == "open" {
                    if ui.add_enabled(!app.busy, egui::Button::new("Cancel")).clicked() {
                        app.busy = true;
                        app.go(ctx, Action::MwbCancelOrder { id: o.order_id.clone() });
                    }
                } else {
                    ui.label("");
                }
                ui.end_row();
            }
        });
    });
}

fn swaps(app: &mut App, ui: &mut Ui) {
    theme::hint(
        ui,
        "Each swap has two deadlines, one per chain, and the wallet acts on them for you: it \
         claims when it can, and takes your coins back if the other side goes quiet. Keep it \
         open while a swap runs; closing it for longer than the deadlines is the one way to lose \
         one.",
    );
    ui.add_space(4.0);
    if app.mwb_swaps.is_empty() {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            theme::hint(ui, "No swaps yet.");
        });
        return;
    }
    for s in &app.mwb_swaps {
        swap_card(ui, s);
        ui.add_space(6.0);
    }
}

fn swap_card(ui: &mut Ui, s: &MwbSwapView) {
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        let amount = |coin: &str| if coin == "MDS" { compact_units(s.mds_amount) } else { mdw(s.mdw_amount) };
        ui.horizontal_wrapped(|ui| {
            theme::badge(ui, &s.phase, if s.settled { theme::muted() } else { theme::ink() });
            theme::badge(ui, &s.role, theme::muted());
            ui.label(theme::mono(format!("give {}", amount(&s.gives))).size(12.0));
            theme::hint(ui, "·");
            ui.label(theme::mono(format!("get {}", amount(&s.gets))).size(12.0));
            ui.label(theme::mono(short_hex(&s.id, 6)).size(10.5).color(theme::faint()));
        });
        if !s.note.is_empty() {
            ui.label(RichText::new(&s.note).size(11.5).color(theme::muted()));
        }
        if !s.settled && s.reveal_height > 0 {
            ui.label(
                RichText::new(format!(
                    "Deadlines: {} block {} · {} block {}",
                    s.reveal_chain,
                    theme::units(s.reveal_height),
                    s.follow_chain,
                    theme::units(s.follow_height)
                ))
                .size(11.0)
                .color(theme::faint()),
            );
        }
    });
}
