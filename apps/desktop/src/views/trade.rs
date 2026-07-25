//! Cross-chain trading against the Base atomic-swap contract.
//!
//! This view is read-only for now: it shows the wallet's Base account and the
//! live order book, but does not place or fill orders. Everything displayed is
//! independently verifiable — bids escrow their ETH in the contract, and asks
//! are backed by midstate coins the wallet checks are still unspent — so
//! nothing here relies on an indexer or a counterparty's word.

use crate::app::App;
use crate::bridge::Action;
use crate::theme::{self, short_hex, units};
use eframe::egui::{self, FontId, RichText, TextEdit, Ui};
use midstate_walletd::api::DexConfigView;

const TABS: [&str; 6] = ["Buy orders", "Sell orders", "Expired", "Trades", "Swaps", "Start a swap"];

/// Base's public explorer. Everything the wallet claims about the ETH leg is
/// independently checkable there, so it should always be one click away.
fn basescan_address(addr: &str) -> String {
    format!("https://basescan.org/address/{addr}")
}
fn basescan_tx(tx: &str) -> String {
    format!("https://basescan.org/tx/{tx}")
}
fn open_in_browser(url: &str) {
    #[cfg(target_os = "linux")]
    let cmd = "xdg-open";
    #[cfg(target_os = "macos")]
    let cmd = "open";
    #[cfg(target_os = "windows")]
    let cmd = "explorer";
    let _ = std::process::Command::new(cmd).arg(url).spawn();
}

/// A small "verify on Basescan" affordance.
fn explorer_link(ui: &mut Ui, label: &str, url: String) {
    if ui
        .add(egui::Button::new(RichText::new(label).size(10.0)))
        .on_hover_text("Open on Basescan to verify independently")
        .clicked()
    {
        open_in_browser(&url);
    }
}

/// Wei is unreadable at 18 decimals; show ETH with enough places to still
/// distinguish small orders.
fn eth(wei_str: &str) -> String {
    let wei: u128 = wei_str.parse().unwrap_or(0);
    let whole = wei / 1_000_000_000_000_000_000;
    let frac = wei % 1_000_000_000_000_000_000;
    if whole > 0 {
        format!("{whole}.{:06} ETH", frac / 1_000_000_000_000)
    } else if frac >= 1_000_000_000_000 {
        format!("0.{:06} ETH", frac / 1_000_000_000_000)
    } else {
        format!("{wei} wei")
    }
}

/// Price is wei per MDS unit. Always rendered in the SAME unit across the
/// column — switching between wei and gwei part-way down makes a correctly
/// sorted list look shuffled.
fn price(p: f64) -> String {
    if p <= 0.0 {
        return "—".into();
    }
    let gwei = p / 1_000_000_000.0;
    let s = if gwei >= 1.0 {
        format!("{gwei:.3}")
    } else {
        format!("{gwei:.9}")
    };
    let s = if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    };
    format!("{s} gwei")
}

pub fn show(app: &mut App, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    theme::heading(ui, "Trade");
    theme::hint(
        ui,
        "Atomic swaps between MDS and ETH on Base. One hash-lock governs both \
         chains, so either the whole trade completes or both sides get their money \
         back — there is no point at which a counterparty holds both.",
    );
    ui.add_space(6.0);

    account_strip(app, ui, &ctx);
    ui.add_space(8.0);

    let mut ix = app.dex_tab.min(5);
    if theme::segmented(ui, &TABS, &mut ix) {
        app.dex_tab = ix;
    }
    ui.add_space(8.0);

    if app.dex_tab == 5 {
        wizard(app, ui, &ctx);
        return;
    }
    if app.dex_tab == 4 {
        swaps_panel(app, ui);
        return;
    }

    // Swaps run against deadlines, so surface them wherever you are on this tab.
    let live_swaps = app.swaps.iter().filter(|s| !s.settled).count();
    if live_swaps > 0 && app.dex_tab != 4 {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                theme::badge(ui, "in progress", theme::ink());
                ui.label(
                    RichText::new(format!(
                        "{live_swaps} swap(s) running — the wallet is watching them for you.",
                    ))
                    .size(12.0),
                );
                if ui.button(RichText::new("view").size(11.0)).clicked() {
                    app.dex_tab = 4;
                }
            });
        });
        ui.add_space(6.0);
    }

    match app.book.clone() {
        None => {
            theme::panel_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                theme::hint(ui, "No book loaded yet — press Refresh to scan both chains.");
            });
        }
        Some(b) => {
            if let Some(err) = &b.last_error {
                theme::panel_frame().show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(RichText::new(err).size(12.0).color(theme::amber()));
                    theme::hint(ui, "Check the endpoint under Connection below.");
                });
                ui.add_space(6.0);
            }
            let tip = app.sync.as_ref().map(|s| s.height).unwrap_or(0);
            match app.dex_tab {
                0 => {
                    place_bid_panel(app, ui, &ctx);
                    bids(app, ui, &b)
                }
                1 => {
                    place_ask_panel(app, ui, &ctx, tip);
                    asks(app, ui, &ctx, &b, tip, false)
                }
                2 => asks(app, ui, &ctx, &b, tip, true),
                3 => trades_panel(ui, &b),
                _ => {}
            }
            ui.add_space(4.0);
            // What the scan actually decoded. Without this an empty book is
            // ambiguous: no activity, or a decoder seeing nothing it knows.
            theme::panel_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                theme::hint(
                    ui,
                    &format!(
                        "Base scanned to block {} · midstate to height {}",
                        units(b.base_cursor),
                        units(b.mds_cursor)
                    ),
                );
                let total = b.bids_created + b.bids_closed + b.locks + b.claims;
                if total == 0 && b.announcements == 0 {
                    theme::hint(
                        ui,
                        "No contract activity in the scanned range — widen the window or set \
                         a start block below to reach further back.",
                    );
                } else {
                    theme::hint(
                        ui,
                        &format!(
                            "Decoded {} bid(s) created, {} closed, {} lock(s), {} claim(s), \
                             {} midstate announcement(s).",
                            b.bids_created, b.bids_closed, b.locks, b.claims, b.announcements
                        ),
                    );
                }
                if b.undecoded_logs > 0 {
                    ui.label(
                        RichText::new(format!(
                            "{} contract log(s) did not match any known event — the ABI may be \
                             out of step with the deployed contract.",
                            b.undecoded_logs
                        ))
                        .size(12.0)
                        .color(theme::amber()),
                    );
                }
            });
        }
    }

    ui.add_space(8.0);
    connection(app, ui, &ctx);
}

// ── Account ─────────────────────────────────────────────────────────────

fn account_strip(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        let Some(a) = app.evm.clone() else {
            theme::hint(ui, "Reading your Base account…");
            return;
        };

        if a.missing_key {
            ui.label(
                RichText::new("This wallet has no Base account.")
                    .font(theme::font_medium(14.0))
                    .color(theme::ink()),
            );
            theme::hint(
                ui,
                "It was created before cross-chain support existed. The key is derived from \
                 your recovery phrase, which the wallet no longer holds — restore from your \
                 phrase into a new wallet to get one.",
            );
            return;
        }

        ui.horizontal(|ui| {
            ui.label(RichText::new("base account").font(FontId::monospace(9.5)).color(theme::muted()));
            ui.label(theme::mono(&a.address).size(11.5).color(theme::bright()));
            if ui.button(RichText::new("copy").size(10.0)).clicked() {
                ui.ctx().copy_text(a.address.clone());
            }
            explorer_link(ui, "view on Basescan", basescan_address(&a.address));
        });
        ui.horizontal(|ui| {
            ui.label(RichText::new("contract").font(FontId::monospace(9.5)).color(theme::muted()));
            ui.label(theme::mono(short_hex(&a.contract, 10)).size(11.0).color(theme::muted()));
            explorer_link(ui, "view on Basescan", basescan_address(&a.contract));
        });
        ui.horizontal(|ui| {
            match &a.balance_wei {
                Some(w) => ui.label(theme::mono(eth(w)).size(12.5)),
                None => ui.label(
                    RichText::new("balance unavailable — endpoint unreachable")
                        .size(12.0)
                        .color(theme::muted()),
                ),
            };
            theme::right_aligned(ui, |ui| {
                let label = if app.dex_syncing { "Scanning…" } else { "Refresh book" };
                if ui.add_enabled(!app.dex_syncing, egui::Button::new(label)).clicked() {
                    app.dex_syncing = true;
                    app.go(ctx, Action::SyncOrderBook);
                }
            });
        });
        theme::hint(
            ui,
            "Same recovery phrase, standard derivation path — this account also opens in \
             MetaMask. Swaps need a little ETH here for gas.",
        );
    });
}

// ── Book ────────────────────────────────────────────────────────────────

fn bids(app: &mut App, ui: &mut Ui, b: &midstate_walletd::api::OrderBookView) {
    theme::hint(
        ui,
        "People offering ETH for your MDS. The ETH is already escrowed in the contract, so \
         each of these is funded — fill one by locking MDS against the same hash.",
    );
    ui.add_space(4.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        if b.bids.is_empty() {
            theme::hint(ui, "No open buy orders in the scanned range.");
            return;
        }
        egui::Grid::new("bids").num_columns(6).spacing([18.0, 7.0]).striped(true).show(ui, |ui| {
            for h in ["Price", "Paying", "For", "Bond", "Maker", ""] {
                ui.label(RichText::new(h.to_uppercase()).font(FontId::monospace(9.5)).color(theme::muted()));
            }
            ui.end_row();
            for x in &b.bids {
                ui.label(theme::mono(price(x.price)).size(11.5).color(theme::ink()));
                ui.label(theme::mono(eth(&x.wei)).size(11.5));
                ui.label(theme::mono(units(x.mds_amount)).size(11.5));
                ui.label(
                    theme::mono(if x.fill_bond == "0" { "none".into() } else { eth(&x.fill_bond) })
                        .size(11.0)
                        .color(theme::muted()),
                );
                ui.horizontal(|ui| {
                    ui.label(theme::mono(short_hex(&x.maker, 8)).size(11.0).color(theme::muted()));
                    explorer_link(ui, "↗", basescan_address(&x.maker));
                });
                if x.mine {
                    theme::badge(ui, "yours", theme::ink());
                } else if x.reserved {
                    theme::badge(ui, "reserved", theme::muted());
                } else if !x.takeable {
                    theme::badge(ui, "expired", theme::muted());
                } else {
                    ui.label("");
                }
                ui.end_row();
            }
        });
    });
    let _ = app;
}

/// Publish a sell order. Funding and announcement go out together, which is
/// what lets a buyer verify the order is backed before touching it.
fn place_ask_panel(app: &mut App, ui: &mut Ui, ctx: &egui::Context, tip: u64) {
    let syncing = app.sync.as_ref().map(|s| s.is_syncing).unwrap_or(true);
    egui::CollapsingHeader::new("Sell MDS — publish an order")
        .id_salt("place_ask")
        .default_open(app.my_orders.is_empty())
        .show(ui, |ui| {
            theme::panel_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    theme::hint(ui, "sell");
                    let r = ui.add(
                        TextEdit::singleline(&mut app.ask_mds)
                            .hint_text("amount")
                            .font(egui::TextStyle::Monospace)
                            .desired_width(130.0),
                    );
                    if r.changed() {
                        app.ask_mds.retain(|c| c.is_ascii_digit());
                    }
                    theme::unit_selector(ui, &mut app.ask_mds_unit);
                });
                ui.horizontal(|ui| {
                    theme::hint(ui, "for");
                    let r2 = ui.add(
                        TextEdit::singleline(&mut app.ask_wei)
                            .hint_text("amount")
                            .font(egui::TextStyle::Monospace)
                            .desired_width(160.0),
                    );
                    if r2.changed() {
                        app.ask_wei.retain(|c| c.is_ascii_digit());
                    }
                    theme::segmented(ui, &ETH_UNITS, &mut app.ask_eth_unit);
                });
                ui.horizontal(|ui| {
                    theme::hint(ui, "offer stands for");
                    let r3 = ui.add(
                        TextEdit::singleline(&mut app.ask_life)
                            .font(egui::TextStyle::Monospace)
                            .desired_width(70.0),
                    );
                    if r3.changed() {
                        app.ask_life.retain(|c| c.is_ascii_digit());
                    }
                    let blocks: u64 = app.ask_life.parse().unwrap_or(0);
                    theme::hint(ui, &format!("blocks (~{:.1} days)", blocks as f64 / 1440.0));
                });

                let mds: u64 = app.ask_mds.parse().unwrap_or(0);
                let wei: u128 = app.ask_wei.parse().unwrap_or(0);
                if mds > 0 && wei > 0 {
                    let parts = {
                        let mut n = 0;
                        let mut v = mds;
                        while v > 0 {
                            if v & 1 == 1 {
                                n += 1;
                            }
                            v >>= 1;
                        }
                        n
                    };
                    ui.label(
                        RichText::new(format!(
                            "≈ {} per unit · splits into {parts} independently sellable piece(s)",
                            price(wei as f64 / mds as f64)
                        ))
                        .size(12.0)
                        .color(theme::bright()),
                    );
                }
                theme::hint(
                    ui,
                    "Your coins are locked behind a covenant that pays out only against a secret \
                     you hold, and the order is announced in the same transaction — so a buyer \
                     can confirm it is really funded. Nothing is committed until someone pays; \
                     if nobody does, the lock expires and the coins return to you.",
                );
                theme::right_aligned(ui, |ui| {
                    let can = !app.busy && !syncing && mds > 0 && wei > 0;
                    if ui.add_enabled(can, egui::Button::new("Publish order")).clicked() {
                        app.busy = true;
                        app.error.clear();
                        app.ask_notice.clear();
                        app.go(
                            ctx,
                            Action::PlaceAsk {
                                mds_amount: mds,
                                wei_amount: app.ask_wei.clone(),
                                lifetime_blocks: app.ask_life.parse().unwrap_or(4320),
                            },
                        );
                    }
                });
                if syncing {
                    theme::hint(ui, "Publishing unlocks when the node reaches the chain tip.");
                }
                if !app.ask_notice.is_empty() {
                    ui.label(RichText::new(&app.ask_notice).size(12.0).color(theme::bright()));
                }
                if !app.error.is_empty() {
                    ui.label(RichText::new(&app.error).size(12.0).color(theme::muted()));
                }
            });

            if !app.my_orders.is_empty() {
                ui.add_space(6.0);
                theme::panel_frame().show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(RichText::new("Your published orders").font(theme::font_medium(13.0)));
                    let mut reclaim: Option<String> = None;
                    for o in &app.my_orders {
                        let expired = tip > 0 && o.timeout_height <= tip;
                        // An order is only offered once its reveal is mined —
                        // the covenant coins and the announcement burns are
                        // both in that transaction.
                        let (badge, tone) = if expired {
                            ("expired", theme::muted())
                        } else if o.on_chain {
                            ("live", theme::ink())
                        } else if o.stage == "confirmed" {
                            ("mined", theme::bright())
                        } else {
                            ("publishing", theme::amber())
                        };
                        ui.horizontal_wrapped(|ui| {
                            theme::badge(ui, badge, tone);
                            ui.label(theme::mono(units(o.mds_amount)).size(11.5));
                            theme::hint(ui, "for");
                            ui.label(theme::mono(eth(&o.wei_amount)).size(11.5));
                            ui.label(
                                RichText::new(format!(
                                    "{} unit(s) · expires h{}",
                                    o.units, o.timeout_height
                                ))
                                .size(11.0)
                                .color(theme::muted()),
                            );
                        });
                        if expired {
                            ui.horizontal(|ui| {
                                ui.label(theme::mono("   ").size(11.0));
                                if ui
                                    .add_enabled(!app.busy, egui::Button::new(RichText::new("reclaim coins").size(11.0)))
                                    .on_hover_text("Sweep whatever went unsold back into your wallet")
                                    .clicked()
                                {
                                    reclaim = Some(o.group_id.clone());
                                }
                                theme::hint(ui, "unsold units can be swept back now");
                            });
                        }
                        if !o.on_chain && !expired {
                            ui.label(
                                RichText::new(format!("   {}", o.detail))
                                    .size(11.0)
                                    .color(theme::muted()),
                            );
                            if o.stage == "confirmed" {
                                theme::hint(
                                    ui,
                                    "   Mined. It appears in the book below at the next scan.",
                                );
                            } else {
                                theme::hint(
                                    ui,
                                    "   Nothing is on-chain until the reveal is mined — the coins \
                                     and the announcement travel in that same transaction. Follow \
                                     it on the Send tab.",
                                );
                            }
                        }
                    }
                    if let Some(g) = reclaim {
                        app.busy = true;
                        app.error.clear();
                        app.ask_notice.clear();
                        app.go(ctx, Action::ReclaimOrder { group_id: g });
                    }
                    theme::hint(
                        ui,
                        "The secrets that release these orders are stored with your wallet. \
                         Losing the wallet file means you cannot be paid for them — though the \
                         coins still come back to you once the order expires.",
                    );
                });
            }
        });
    ui.add_space(6.0);
}

fn asks(
    app: &mut App,
    ui: &mut Ui,
    ctx: &egui::Context,
    b: &midstate_walletd::api::OrderBookView,
    tip: u64,
    want_expired: bool,
) {
    // Deliberately not an egui::Grid. A grid sizes columns from content and
    // clips anything wider, so the per-unit take buttons wrapped one-per-line
    // inside a narrow cell. Fixed-width cells keep the table aligned and let
    // the expanded row span the full width.
    const W_PRICE: f32 = 150.0;
    const W_SELL: f32 = 120.0;
    const W_WANT: f32 = 130.0;
    const W_UNITS: f32 = 64.0;
    const W_EXP: f32 = 110.0;
    const ROW_H: f32 = 26.0;

    fn cell(ui: &mut Ui, w: f32, add: impl FnOnce(&mut Ui)) {
        ui.allocate_ui_with_layout(
            egui::vec2(w, ROW_H),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.set_width(w);
                add(ui);
            },
        );
    }

    theme::hint(
        ui,
        if want_expired {
            "Orders whose lock has already run out. The maker can reclaim these at any moment, \
             so they cannot be filled — kept here only so you can see what was offered."
        } else {
            "People offering MDS for ETH. Each is backed by midstate coins this wallet has \
             verified are still unspent; partially filled orders show their remaining units."
        },
    );
    ui.add_space(4.0);

    let rows: Vec<&midstate_walletd::api::AskView> = b
        .asks
        .iter()
        .filter(|x| ((tip > 0 && x.timeout_height <= tip)) == want_expired)
        .collect();

    let mut take: Option<(String, usize)> = None;
    let mut request: Option<(String, u64)> = None;

    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        if rows.is_empty() {
            theme::hint(
                ui,
                if want_expired {
                    "No expired orders in the scanned range."
                } else {
                    "No open sell orders in the scanned range. Announcements are published when \
                     an order is created, so older orders may sit outside the window."
                },
            );
            return;
        }

        ui.horizontal(|ui| {
            for (label, w) in [
                ("Price", W_PRICE),
                ("Selling", W_SELL),
                ("Wants", W_WANT),
                ("Units", W_UNITS),
                ("Expires", W_EXP),
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

        for x in &rows {
            let tone = if want_expired { theme::faint() } else { theme::ink() };
            ui.horizontal(|ui| {
                cell(ui, W_PRICE, |ui| {
                    ui.label(theme::mono(price(x.price)).size(11.5).color(tone));
                });
                cell(ui, W_SELL, |ui| {
                    ui.label(theme::mono(units(x.mds_value)).size(11.5).color(tone));
                });
                cell(ui, W_WANT, |ui| {
                    ui.label(theme::mono(eth(&x.wei)).size(11.5).color(tone));
                });
                cell(ui, W_UNITS, |ui| {
                    ui.label(
                        theme::mono(format!("{}/{}", x.live_units, x.total_units))
                            .size(11.0)
                            .color(theme::muted()),
                    );
                });
                cell(ui, W_EXP, |ui| {
                    ui.label(
                        theme::mono(format!("h{}", units(x.timeout_height)))
                            .size(11.0)
                            .color(if want_expired { theme::amber() } else { theme::muted() }),
                    );
                });
                if x.mine {
                    theme::badge(ui, "yours", theme::ink());
                }
                if !x.mine && !want_expired {
                    // Whether settlement can be instant is a property of the
                    // lane between you and this maker, not of the order.
                    let (label, tone, why) = match x.route.as_str() {
                        "direct" => (
                            "instant",
                            theme::ink(),
                            format!(
                                "This maker has a channel to you with {} spendable — settlement \
                                 is immediate.",
                                units(x.route_capacity)
                            ),
                        ),
                        "hub" => (
                            "maybe instant",
                            theme::bright(),
                            "You have inbound channel capacity, so a hub may be able to route \
                             this. If it cannot, the trade falls back to the on-chain route."
                                .into(),
                        ),
                        _ => (
                            "on-chain",
                            theme::muted(),
                            "No channel points at you from this maker, so settlement takes two \
                             rounds on a 60-second chain. Only they can open one — you can ask."
                                .into(),
                        ),
                    };
                    ui.label(
                        RichText::new(label).size(10.0).color(tone),
                    )
                    .on_hover_text(why);
                }
                explorer_link(ui, "maker", basescan_address(&x.maker_evm));
            });

            // Offer the one action that can turn this into an instant trade.
            // A buyer cannot open the lane themselves — value only flows from
            // the channel's sender — so asking is the whole mechanism.
            if !x.mine && !want_expired && x.route == "none" && !x.maker_mds_pk.is_empty() {
                ui.horizontal_wrapped(|ui| {
                    theme::hint(ui, "no channel from this seller —");
                    let want = x.mds_value.max(4096);
                    if ui
                        .add_enabled(!app.busy, egui::Button::new(RichText::new("ask them to open one").size(10.5)))
                        .on_hover_text(
                            "Sends a request over the chat network. They fund it if their \
                             settings allow; every later trade with them is then instant.",
                        )
                        .clicked()
                    {
                        request = Some((x.maker_mds_pk.clone(), want));
                    }
                    theme::hint(ui, "or take it on-chain below");
                });
            }

            // Full width here, so the units lay out as a row of chips.
            if !x.mine && !want_expired && !x.units.is_empty() {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0);
                    theme::hint(ui, "take");
                    for u in x.units.iter().take(12) {
                        let label = format!("{} for {}", units(u.mds), eth(&u.wei));
                        if ui
                            .add(egui::Button::new(RichText::new(label).size(10.5)))
                            .on_hover_text(
                                "Escrows your ETH against this unit's hash. If the seller never \
                                 claims it, the escrow refunds itself.",
                            )
                            .clicked()
                        {
                            take = Some((x.group_id.clone(), u.index));
                        }
                    }
                });
            }
            ui.separator();
        }
    });

    if let Some((g, u)) = take {
        app.busy = true;
        app.error.clear();
        app.ask_notice.clear();
        app.go(ctx, Action::TakeAsk { group_id: g, unit: u });
    }
    if let Some((peer, capacity)) = request {
        app.busy = true;
        app.error.clear();
        app.ask_notice =
            "Channel requested. If they accept, it appears on the Channels tab and this order \
             becomes instant."
                .into();
        app.go(ctx, Action::RequestChannel { peer, capacity });
    }
}

// ── Connection ──────────────────────────────────────────────────────────

fn connection(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    egui::CollapsingHeader::new("Connection")
        .id_salt("dex_conn")
        .show(ui, |ui| {
            let Some(mut next) = app.dex_cfg.clone() else {
                theme::hint(ui, "Loading…");
                return;
            };
            // Compare against what walletd actually holds. Comparing against a
            // clone of the edit buffer makes the Save button flash for a single
            // frame and then vanish, which is what it did.
            let saved = app.dex_cfg_saved.clone().unwrap_or_else(|| next.clone());
            theme::panel_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    theme::hint(ui, "rpc");
                    ui.add(
                        TextEdit::singleline(&mut next.rpc_url)
                            .font(egui::TextStyle::Monospace)
                            .desired_width(f32::INFINITY),
                    );
                });
                ui.horizontal(|ui| {
                    theme::hint(ui, "contract");
                    ui.add(
                        TextEdit::singleline(&mut next.contract)
                            .font(egui::TextStyle::Monospace)
                            .desired_width(f32::INFINITY),
                    );
                });
                ui.horizontal(|ui| {
                    theme::hint(ui, "chain id");
                    let mut cid = next.chain_id.to_string();
                    if ui
                        .add(TextEdit::singleline(&mut cid).font(egui::TextStyle::Monospace).desired_width(80.0))
                        .changed()
                    {
                        cid.retain(|c| c.is_ascii_digit());
                        next.chain_id = cid.parse().unwrap_or(next.chain_id);
                    }
                    theme::hint(ui, "confirmations");
                    let mut cf = next.confirmations.to_string();
                    if ui
                        .add(TextEdit::singleline(&mut cf).font(egui::TextStyle::Monospace).desired_width(60.0))
                        .changed()
                    {
                        cf.retain(|c| c.is_ascii_digit());
                        next.confirmations = cf.parse().unwrap_or(next.confirmations);
                    }
                    theme::hint(ui, "scan window");
                    let mut sw = next.scan_window.to_string();
                    if ui
                        .add(TextEdit::singleline(&mut sw).font(egui::TextStyle::Monospace).desired_width(100.0))
                        .changed()
                    {
                        sw.retain(|c| c.is_ascii_digit());
                        next.scan_window = sw.parse().unwrap_or(next.scan_window);
                    }
                    theme::hint(ui, "blocks");
                });
                ui.horizontal(|ui| {
                    theme::hint(ui, "or start at block");
                    let mut sb = next.start_block.to_string();
                    if ui
                        .add(TextEdit::singleline(&mut sb).font(egui::TextStyle::Monospace).desired_width(120.0))
                        .changed()
                    {
                        sb.retain(|c| c.is_ascii_digit());
                        next.start_block = sb.parse().unwrap_or(next.start_block);
                    }
                    theme::hint(ui, "0 = use the window. Base mines ~30k blocks a day.");
                });
                theme::hint(
                    ui,
                    "The wallet refuses to sign if the endpoint reports a different chain id \
                     than configured, so a misdirected RPC cannot quietly put a transaction \
                     on the wrong network. Changing any of this clears the book and rescans.",
                );
                let dirty = changed(&saved, &next);
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(dirty && !app.busy, egui::Button::new("Save and rescan"))
                        .clicked()
                    {
                        app.busy = true;
                        app.error.clear();
                        app.go(ctx, Action::SetDexConfig { cfg: next.clone() });
                    }
                    if dirty {
                        theme::hint(ui, "unsaved changes");
                        if ui.button(RichText::new("revert").size(11.0)).clicked() {
                            next = saved.clone();
                        }
                    }
                });
                if !app.error.is_empty() {
                    ui.label(RichText::new(&app.error).size(12.0).color(theme::muted()));
                }
            });
            app.dex_cfg = Some(next);
        });
}

fn changed(a: &DexConfigView, b: &DexConfigView) -> bool {
    a.rpc_url != b.rpc_url
        || a.contract != b.contract
        || a.chain_id != b.chain_id
        || a.confirmations != b.confirmations
        || a.scan_window != b.scan_window
}

// ── Guided swap ─────────────────────────────────────────────────────────

const SIDES: [&str; 2] = ["Buy MDS", "Sell MDS"];
const RAILS: [&str; 2] = ["Instant (channel)", "On-chain"];
/// Ether denominations, same idea as the MDS selector.
const ETH_UNITS: [&str; 3] = ["wei", "gwei", "ETH"];

/// What happens, in order. Derived from the two choices alone, so it can be
/// shown before anything is filled in — which is when someone new needs it.
fn walkthrough(selling: bool, instant: bool) -> Vec<(&'static str, String)> {
    let mds_leg = if instant {
        "through your payment channel — instant, and nothing touches the chain"
    } else {
        "as an on-chain lock — two confirmations on a 60-second chain"
    };
    if selling {
        vec![
            ("A secret is created",
             "Your wallet picks a random secret and publishes only its hash. Both chains lock \
              against that same hash, and nobody can work backwards from it.".into()),
            ("You lock your MDS",
             format!("The MDS is locked {mds_leg}. If the buyer never pays, it comes straight \
                      back to you when the lock expires.")),
            ("They escrow the ETH",
             "The buyer puts their ETH into the Base contract against the same hash. You can \
              verify the amount before doing anything else.".into()),
            ("You take the ETH",
             "Claiming it publishes the secret. This is the first moment either side is \
              committed to anything.".into()),
            ("They take the MDS",
             "The buyer reads the secret from that claim and unlocks the MDS. If they never \
              do, your lock expires and the MDS returns to you anyway.".into()),
        ]
    } else {
        vec![
            ("The seller creates a secret",
             "They pick a random secret and show you only its hash. Both legs lock to it.".into()),
            ("They lock the MDS",
             format!("The MDS is locked {mds_leg}. Your wallet checks the amount and the \
                      deadline before you put up anything.")),
            ("You escrow the ETH",
             "Your ETH goes into the Base contract, refundable to you if the trade stalls \
              after this point.".into()),
            ("They take the ETH",
             "Claiming it publishes the secret onto Base, where anyone can read it.".into()),
            ("You get the MDS automatically",
             "Your wallet watches for that secret and uses it to unlock the MDS. If the \
              seller never claims, your ETH refunds itself after the deadline.".into()),
        ]
    }
}

fn wizard(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    let tip = app.sync.as_ref().map(|s| s.height).unwrap_or(0);
    let selling = app.swap_side == 1;

    // ── 1 · Direction ───────────────────────────────────────────────────
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("1 · What do you want to do?").font(theme::font_medium(14.0)));
        let mut side = app.swap_side;
        if theme::segmented(ui, &SIDES, &mut side) {
            app.swap_side = side;
            app.swap_quote = None;
        }
        // Selling and buying are genuinely different acts here, and pretending
        // otherwise is what made "who is my counterparty?" unanswerable.
        theme::hint(
            ui,
            if selling {
                "Selling means publishing an offer. There is no counterparty yet — anyone \
                 holding ETH can take it, and you find out who when they do."
            } else {
                "Buying means taking someone's published offer. That order's maker is your \
                 counterparty, and picking the order picks them."
            },
        );
    });

    // ── Selling: this is order publication, not a two-party negotiation ──
    if selling {
        ui.add_space(6.0);
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("2 · How selling works").font(theme::font_medium(14.0)));
            for (i, (title, body)) in walkthrough(true, false).iter().enumerate() {
                ui.horizontal_wrapped(|ui| {
                    ui.label(theme::mono(format!("{}", i + 1)).size(12.0).color(theme::muted()));
                    ui.label(RichText::new(*title).size(12.5).color(theme::ink()));
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label(theme::mono("   ").size(11.0));
                    ui.label(RichText::new(body).size(11.5).color(theme::muted()));
                });
                ui.add_space(3.0);
            }
            theme::hint(
                ui,
                "Because there is no counterparty yet, there is nothing to negotiate and no \
                 channel to arrange. You publish; someone takes it; the swap runs itself.",
            );
        });
        ui.add_space(6.0);
        place_ask_panel(app, ui, ctx, tip);
        return;
    }

    // ── Buying: pick an order, which fixes the counterparty ─────────────
    ui.add_space(6.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("2 · Pick an order").font(theme::font_medium(14.0)));

        let live: Vec<&midstate_walletd::api::AskView> = app
            .book
            .as_ref()
            .map(|b| {
                b.asks
                    .iter()
                    .filter(|x| tip == 0 || x.timeout_height > tip)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        if live.is_empty() {
            theme::hint(
                ui,
                "No live sell orders in the scanned range. Widen the scan under Connection, or \
                 wait for someone to publish one.",
            );
        } else {
            // Pick a UNIT, not an order. Units are what actually get taken —
            // each has its own hash and price — and choosing one here is what
            // gives the Start button something concrete to execute.
            let label = match (&app.swap_peer, app.swap_unit) {
                (g, Some(_)) if !g.is_empty() => {
                    format!("{} MDS for {}", units(app.swap_mds.parse().unwrap_or(0)), eth(&app.swap_eth))
                }
                _ => {
                    let n: usize = live.iter().map(|x| x.units.len()).sum();
                    format!("{n} unit(s) available…")
                }
            };
            egui::ComboBox::from_id_salt("pick_ask")
                .selected_text(theme::mono(label).size(12.0))
                .width(ui.available_width() - 8.0)
                .show_ui(ui, |ui| {
                    for x in &live {
                        if x.mine {
                            continue; // taking your own order would be a no-op
                        }
                        for u in &x.units {
                            let text = format!(
                                "{} for {}  ·  {}  ·  order {}",
                                units(u.mds),
                                eth(&u.wei),
                                if x.route == "direct" { "instant" } else { "on-chain" },
                                short_hex(&x.group_id, 6)
                            );
                            let sel = app.swap_peer == x.group_id && app.swap_unit == Some(u.index);
                            if ui.selectable_label(sel, theme::mono(text).size(11.5)).clicked() {
                                app.swap_peer = x.group_id.clone();
                                app.swap_maker_pk = x.maker_mds_pk.clone();
                                app.swap_unit = Some(u.index);
                                app.swap_mds = u.mds.to_string();
                                app.swap_eth = u.wei.clone();
                                // The rail is a property of the route, not a
                                // preference — pick the one that can work.
                                app.swap_rail = if x.route == "direct" { 0 } else { 1 };
                                app.swap_quote = None;
                            }
                        }
                    }
                });
            theme::hint(
                ui,
                "Choosing a unit fills in its terms and picks the route that can actually carry \
                 it. The seller is whoever published the order — you never type their key.",
            );
        }
    });

    // ── 3 · Rail ────────────────────────────────────────────────────────
    ui.add_space(6.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("3 · How fast").font(theme::font_medium(14.0)));
        let mut rail = app.swap_rail;
        if theme::segmented(ui, &RAILS, &mut rail) {
            app.swap_rail = rail;
            app.swap_quote = None;
        }
        ui.add_space(4.0);
        // Make the trade-off unmissable rather than burying it in one clause.
        let (speed, needs) = if app.swap_rail == 0 {
            (
                "Seconds. The MDS moves through a payment channel, so that leg never touches \
                 the chain.",
                "Needs an open channel with enough room toward the seller.",
            )
        } else {
            (
                "Several minutes. The MDS leg is locked and unlocked on-chain, which costs two \
                 rounds on a 60-second chain.",
                "Needs nothing extra — works with any counterparty.",
            )
        };
        ui.label(RichText::new(speed).size(12.5).color(theme::ink()));
        ui.label(RichText::new(needs).size(11.5).color(theme::muted()));
        ui.add_space(4.0);
        for (i, (title, body)) in walkthrough(false, app.swap_rail == 0).iter().enumerate() {
            ui.horizontal_wrapped(|ui| {
                ui.label(theme::mono(format!("{}", i + 1)).size(12.0).color(theme::muted()));
                ui.label(RichText::new(*title).size(12.5).color(theme::ink()));
            });
            ui.horizontal_wrapped(|ui| {
                ui.label(theme::mono("   ").size(11.0));
                ui.label(RichText::new(body).size(11.5).color(theme::muted()));
            });
            ui.add_space(3.0);
        }
    });

    // ── 4 · Terms ───────────────────────────────────────────────────────
    ui.add_space(6.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("4 · Terms").font(theme::font_medium(14.0)));
        ui.horizontal(|ui| {
            theme::hint(ui, "you receive");
            let r = ui.add(
                TextEdit::singleline(&mut app.swap_mds)
                    .hint_text("amount")
                    .font(egui::TextStyle::Monospace)
                    .desired_width(120.0),
            );
            if r.changed() {
                app.swap_mds.retain(|c| c.is_ascii_digit());
                app.swap_quote = None;
            }
            if theme::unit_selector(ui, &mut app.swap_mds_unit) {
                app.swap_quote = None;
            }
        });
        ui.horizontal(|ui| {
            theme::hint(ui, "you pay");
            let r = ui.add(
                TextEdit::singleline(&mut app.swap_eth)
                    .hint_text("amount")
                    .font(egui::TextStyle::Monospace)
                    .desired_width(160.0),
            );
            if r.changed() {
                app.swap_eth.retain(|c| c.is_ascii_digit());
                app.swap_quote = None;
            }
            if theme::segmented(ui, &ETH_UNITS, &mut app.swap_eth_unit) {
                app.swap_quote = None;
            }
        });
        ui.horizontal(|ui| {
            theme::hint(ui, "escrow lasts");
            let r = ui.add(
                TextEdit::singleline(&mut app.swap_hours)
                    .font(egui::TextStyle::Monospace)
                    .desired_width(50.0),
            );
            if r.changed() {
                app.swap_hours.retain(|c| c.is_ascii_digit());
                app.swap_quote = None;
            }
            theme::hint(ui, "hour(s) before an unfinished swap refunds itself");
        });
        theme::right_aligned(ui, |ui| {
            let ok = app.swap_mds.parse::<u64>().unwrap_or(0) > 0;
            if ui.add_enabled(ok && !app.busy, egui::Button::new("Check")).clicked() {
                app.busy = true;
                app.error.clear();
                let hours = app.swap_hours.parse::<u64>().unwrap_or(1).clamp(1, 168);
                app.go(
                    ctx,
                    Action::SwapQuote {
                        side: "buy".into(),
                        rail: if app.swap_rail == 1 { "onchain".into() } else { "submarine".into() },
                        mds_amount: app.swap_mds.parse().unwrap_or(0),
                        wei_amount: if app.swap_eth.is_empty() { "0".into() } else { app.swap_eth.clone() },
                        peer_mds_pk: app.swap_maker_pk.clone(),
                        eth_refund_secs: hours * 3600,
                    },
                );
            }
        });
    });

    let Some(q) = app.swap_quote.clone() else {
        ui.add_space(6.0);
        theme::hint(ui, "Press Check when the terms look right. Nothing is sent until you confirm.");
        return;
    };

    // ── 5 · Readiness ───────────────────────────────────────────────────
    ui.add_space(6.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("5 · Before you start").font(theme::font_medium(14.0)));
        for c in &q.checks {
            ui.horizontal_wrapped(|ui| {
                theme::badge(
                    ui,
                    if c.ok { "ok" } else { "needed" },
                    if c.ok { theme::ink() } else { theme::amber() },
                );
                ui.label(RichText::new(&c.label).size(12.5).color(theme::ink()));
                ui.label(RichText::new(&c.detail).size(11.5).color(theme::muted()));
            });
            if let Some(fix) = &c.fix {
                ui.label(RichText::new(format!("   → {fix}")).size(11.5).color(theme::bright()));
                // A missing channel is the one blocker the wallet could clear
                // for you, so offer the shortcut instead of a lecture.
                if c.label.contains("Channel") && app.swap_rail == 0 {
                    ui.horizontal(|ui| {
                        ui.label(theme::mono("   ").size(11.0));
                        if ui.button(RichText::new("use the on-chain route instead").size(11.0)).clicked() {
                            app.swap_rail = 1;
                            app.swap_quote = None;
                        }
                    });
                }
            }
        }
    });

    // ── 6 · Deadlines ───────────────────────────────────────────────────
    ui.add_space(6.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("6 · Deadlines").font(theme::font_medium(14.0)));
        match (&q.timings, &q.timing_error) {
            (Some(t), _) => {
                theme::hint(
                    ui,
                    "The two chains keep time differently — Base counts seconds, midstate counts \
                     blocks. The ETH side always expires first, so whoever moves second still \
                     has room to finish.",
                );
                ui.label(
                    theme::mono(format!(
                        "ETH refundable after {} minute(s) · MDS lock releases at height {}",
                        t.eth_refund_secs / 60,
                        t.mds_timeout_height
                    ))
                    .size(12.0),
                );
                ui.label(
                    RichText::new(format!(
                        "{} minutes of slack between them, after allowing for block-time drift.",
                        t.margin_secs / 60
                    ))
                    .size(11.5)
                    .color(theme::muted()),
                );
            }
            (None, Some(e)) => {
                ui.label(RichText::new(e).size(12.0).color(theme::amber()));
            }
            _ => {}
        }
    });

    ui.add_space(6.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        if q.ready {
            ui.label(
                RichText::new("These terms would be safe to run.")
                    .size(12.5)
                    .color(theme::ink()),
            );
        } else {
            ui.label(
                RichText::new("Not ready — clear the items marked NEEDED above.")
                    .size(12.5)
                    .color(theme::amber()),
            );
        }
        let chosen = app.swap_unit.filter(|_| !app.swap_peer.is_empty());
        let can = q.ready && chosen.is_some() && !app.busy;
        if ui
            .add_enabled(can, egui::Button::new(RichText::new("Start swap").strong()))
            .on_disabled_hover_text(if chosen.is_none() {
                "Pick a unit above first."
            } else {
                "Clear the items marked NEEDED before starting."
            })
            .clicked()
        {
            if let Some(unit) = chosen {
                app.busy = true;
                app.error.clear();
                app.ask_notice.clear();
                app.go(
                    ctx,
                    Action::TakeAsk { group_id: app.swap_peer.clone(), unit },
                );
            }
        }
        theme::hint(
            ui,
            "This escrows your ETH against the seller's hash. Nothing else moves until they \
             claim it — and if they never do, the escrow refunds itself to you.",
        );
        if !app.ask_notice.is_empty() {
            ui.label(RichText::new(&app.ask_notice).size(12.0).color(theme::bright()));
        }
        if !app.error.is_empty() {
            ui.label(RichText::new(&app.error).size(12.0).color(theme::muted()));
        }
    });
}

// ── Live swaps ──────────────────────────────────────────────────────────

fn swaps_panel(app: &mut App, ui: &mut Ui) {
    theme::hint(
        ui,
        "Swaps in progress. Each step has a deadline, and the wallet acts on them for you — \
         claiming when the secret appears, refunding if a counterparty goes quiet. Leaving \
         this app closed for a long time is the only way to lose one.",
    );
    ui.add_space(4.0);

    if app.swaps.is_empty() {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            theme::hint(ui, "No swaps yet. Take an order from the Sell orders tab to start one.");
        });
        return;
    }

    let now = theme::now_secs();
    for s in &app.swaps {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal_wrapped(|ui| {
                theme::badge(
                    ui,
                    &s.phase,
                    if s.settled { theme::muted() } else { theme::ink() },
                );
                theme::badge(ui, &s.role, theme::muted());
                ui.label(theme::mono(units(s.mds_value)).size(12.0));
                theme::hint(ui, "for");
                ui.label(theme::mono(eth(&s.wei)).size(12.0));
                if let Some(tx) = &s.tx {
                    if !tx.is_empty() {
                        explorer_link(ui, "transaction", basescan_tx(tx));
                    }
                }
            });
            if !s.detail.is_empty() {
                ui.label(RichText::new(&s.detail).size(11.5).color(theme::muted()));
            }
            if !s.settled && s.eth_deadline > now {
                let mins = (s.eth_deadline - now) / 60;
                ui.label(
                    RichText::new(format!(
                        "Escrow refunds itself in {mins} minute(s) if the swap does not complete."
                    ))
                    .size(11.0)
                    .color(theme::faint()),
                );
            }
        });
        ui.add_space(6.0);
    }
}

// ── Placing a buy order ─────────────────────────────────────────────────

/// Escrow ETH as a resting bid. Inverted from a sell order: here you hold the
/// secret, so the escrow only ever pays out against something you released.
fn place_bid_panel(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    egui::CollapsingHeader::new("Buy MDS — place an order")
        .id_salt("place_bid")
        .default_open(app.my_bids.is_empty())
        .show(ui, |ui| {
            theme::panel_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    theme::hint(ui, "buy");
                    let r = ui.add(
                        TextEdit::singleline(&mut app.bid_mds)
                            .hint_text("MDS")
                            .font(egui::TextStyle::Monospace)
                            .desired_width(130.0),
                    );
                    if r.changed() {
                        app.bid_mds.retain(|c| c.is_ascii_digit());
                    }
                    theme::hint(ui, "for");
                    let r2 = ui.add(
                        TextEdit::singleline(&mut app.bid_wei)
                            .hint_text("wei")
                            .font(egui::TextStyle::Monospace)
                            .desired_width(160.0),
                    );
                    if r2.changed() {
                        app.bid_wei.retain(|c| c.is_ascii_digit());
                    }
                });
                ui.horizontal(|ui| {
                    theme::hint(ui, "open for");
                    let r = ui.add(
                        TextEdit::singleline(&mut app.bid_hours)
                            .font(egui::TextStyle::Monospace)
                            .desired_width(60.0),
                    );
                    if r.changed() {
                        app.bid_hours.retain(|c| c.is_ascii_digit());
                    }
                    theme::hint(ui, "hour(s) · seller's stake");
                    let r2 = ui.add(
                        TextEdit::singleline(&mut app.bid_bond)
                            .font(egui::TextStyle::Monospace)
                            .desired_width(110.0),
                    );
                    if r2.changed() {
                        app.bid_bond.retain(|c| c.is_ascii_digit());
                    }
                    theme::hint(ui, "wei");
                });

                let mds: u64 = app.bid_mds.parse().unwrap_or(0);
                let wei: u128 = app.bid_wei.parse().unwrap_or(0);
                if mds > 0 && wei > 0 {
                    ui.label(
                        RichText::new(format!("≈ {} per unit", price(wei as f64 / mds as f64)))
                            .size(12.0)
                            .color(theme::bright()),
                    );
                }
                theme::hint(
                    ui,
                    "Your ETH is escrowed in the contract, so sellers can see the offer is real. \
                     You hold the secret that releases it: a seller locks MDS against your hash, \
                     you take the MDS, and that reveals the secret they need to collect the ETH. \
                     Nothing pays out otherwise — cancel any time before someone reserves it.",
                );
                theme::hint(
                    ui,
                    "The seller's stake is forfeited to you if they reserve your order and then \
                     fail to deliver. Zero disables it.",
                );
                theme::right_aligned(ui, |ui| {
                    let can = !app.busy && mds > 0 && wei > 0;
                    if ui.add_enabled(can, egui::Button::new("Escrow buy order")).clicked() {
                        app.busy = true;
                        app.error.clear();
                        app.ask_notice.clear();
                        app.go(
                            ctx,
                            Action::PlaceBid {
                                mds_amount: mds,
                                wei: app.bid_wei.clone(),
                                ttl_secs: app.bid_hours.parse::<u64>().unwrap_or(24).clamp(1, 2160) * 3600,
                                fill_bond: if app.bid_bond.is_empty() { "0".into() } else { app.bid_bond.clone() },
                            },
                        );
                    }
                });
                if !app.ask_notice.is_empty() {
                    ui.label(RichText::new(&app.ask_notice).size(12.0).color(theme::bright()));
                }
                if !app.error.is_empty() {
                    ui.label(RichText::new(&app.error).size(12.0).color(theme::muted()));
                }
            });

            if !app.my_bids.is_empty() {
                ui.add_space(6.0);
                theme::panel_frame().show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(RichText::new("Your buy orders").font(theme::font_medium(13.0)));
                    let mut cancel: Option<String> = None;
                    for b in &app.my_bids {
                        ui.horizontal_wrapped(|ui| {
                            theme::badge(
                                ui,
                                &b.status,
                                if b.status == "open" { theme::ink() } else { theme::muted() },
                            );
                            ui.label(theme::mono(units(b.mds_amount)).size(11.5));
                            theme::hint(ui, "for");
                            ui.label(theme::mono(eth(&b.wei)).size(11.5));
                            if !b.tx.is_empty() {
                                explorer_link(ui, "transaction", basescan_tx(&b.tx));
                            }
                            if b.status == "open"
                                && ui
                                    .add_enabled(!app.busy, egui::Button::new(RichText::new("cancel").size(10.5)))
                                    .on_hover_text("Returns your ETH, if nobody has reserved it")
                                    .clicked()
                            {
                                cancel = Some(b.bid_id.clone());
                            }
                        });
                    }
                    if let Some(id) = cancel {
                        app.busy = true;
                        app.error.clear();
                        app.go(ctx, Action::CancelBid { bid_id: id });
                    }
                    theme::hint(
                        ui,
                        "These fill themselves. When a seller locks MDS against your order, this \
                         wallet spots their announcement, checks it pays you enough, and collects \
                         it — which is what releases the secret they need to take the ETH. Leave \
                         the app running, or cancel before the deadline to get your ETH back.",
                    );
                });
            }
        });
    ui.add_space(6.0);
}

// ── Completed trades ────────────────────────────────────────────────────

/// What the market has actually done, as opposed to what it is offering.
fn trades_panel(ui: &mut Ui, b: &midstate_walletd::api::OrderBookView) {
    theme::hint(
        ui,
        "Trades that settled, rebuilt from the contract's own events — every one is a real \
         payout, not an offer. Reconstructed from the scanned range, so widening the window \
         under Connection reveals more history.",
    );
    ui.add_space(4.0);

    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        if b.trades.is_empty() {
            theme::hint(ui, "No completed trades in the scanned range.");
            return;
        }
        egui::Grid::new("trades").num_columns(5).spacing([20.0, 7.0]).striped(true).show(ui, |ui| {
            for h in ["Block", "Type", "MDS", "ETH", "Price"] {
                ui.label(
                    RichText::new(h.to_uppercase()).font(FontId::monospace(9.5)).color(theme::muted()),
                );
            }
            ui.end_row();
            for t in &b.trades {
                ui.label(theme::mono(units(t.block)).size(11.0).color(theme::muted()));
                theme::badge(
                    ui,
                    if t.kind == "buy" { "bid filled" } else { "ask taken" },
                    theme::muted(),
                );
                ui.label(match t.mds {
                    Some(m) => theme::mono(units(m)).size(11.5),
                    // The settlement event names only an id; the size is known
                    // only if the matching announcement was in range.
                    None => theme::mono("—").size(11.5).color(theme::faint()),
                });
                ui.label(
                    theme::mono(if t.wei == "0" { "—".to_string() } else { eth(&t.wei) })
                        .size(11.5),
                );
                ui.label(match t.price {
                    Some(p) => theme::mono(price(p)).size(11.5).color(theme::ink()),
                    None => theme::mono("—").size(11.0).color(theme::faint()),
                });
                ui.end_row();
            }
        });
    });
}
