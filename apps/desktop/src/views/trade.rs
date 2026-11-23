//! Trading MDS for ETH or USDC on Base, wallet to wallet.
//!
//! Every trade is an atomic swap against a contract on Base:
//! `MidstateAtomicSwap` for ETH, `MidstateTokenSwap` for USDC. Both sides
//! are paid, or both get their money back. The two markets are one page in
//! two currencies, each with three tabs:
//!
//! * **Buy MDS**: sell orders heard on the bus (each seller's wallet locks
//!   the MDS for one buyer when asked), and posting a buy order, whose ETH
//!   or USDC waits in the contract for a seller.
//! * **Sell MDS**: buy orders escrowed on Base, and posting a sell order.
//! * **My trades**: what is under way, and what finished, in both.

use crate::app::App;
use crate::bridge::Action;
use crate::theme::{self, short_hex, units};
use eframe::egui::{self, FontId, RichText, TextEdit, Ui};
use midstate_walletd::api::{BidView, DexConfigView, EthAdvertView, EthSwapView};

const TABS: [&str; 3] = ["Buy MDS", "Sell MDS", "My trades"];
const GMDS: u128 = 1 << 30;
const WEI: u128 = 1_000_000_000_000_000_000;
/// USDC's smallest unit: a millionth of a dollar.
const MICRO: u128 = 1_000_000;
const ETH: &str = "ETH";
const USDC: &str = "USDC";

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

/// Wei as ETH, with enough places to tell small amounts apart.
fn eth(wei_str: &str) -> String {
    eth_of(wei_str.parse().unwrap_or(0))
}

fn eth_of(wei: u128) -> String {
    if wei == 0 {
        return "0 ETH".into();
    }
    let whole = wei / WEI;
    let frac = format!("{:018}", wei % WEI);
    let frac = frac.trim_end_matches('0');
    let shown = if frac.is_empty() { whole.to_string() } else { format!("{whole}.{}", &frac[..frac.len().min(8)]) };
    format!("{shown} ETH")
}

/// Millionths of a dollar as USDC: cents always, more places only when
/// they say something.
fn usdc_of(micro: u128) -> String {
    let (whole, frac) = (micro / MICRO, micro % MICRO);
    let mut f = format!("{frac:06}");
    while f.len() > 2 && f.ends_with('0') {
        f.pop();
    }
    let whole = whole.to_string();
    let mut grouped = String::new();
    for (i, c) in whole.chars().enumerate() {
        if i > 0 && (whole.len() - i) % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(c);
    }
    format!("{grouped}.{f} USDC")
}

/// An amount in `asset`'s smallest unit, as that currency.
fn amount_of(asset: &str, v: u128) -> String {
    if asset == USDC {
        usdc_of(v)
    } else {
        eth_of(v)
    }
}

fn amount(asset: &str, v: &str) -> String {
    amount_of(asset, v.parse().unwrap_or(0))
}

/// A price, smallest units per 2^30 units, per gMDS.
fn per_mds(asset: &str, wei_per_gmds: &str) -> String {
    amount(asset, wei_per_gmds)
}

/// A price in smallest units per unit, per gMDS.
fn per_mds_f(asset: &str, wei_per_unit: f64) -> String {
    if wei_per_unit <= 0.0 {
        return "—".into();
    }
    amount_of(asset, (wei_per_unit * GMDS as f64).round() as u128)
}

/// Units as gMDS (2^30 units), as the wallet shows amounts this size.
fn gmds(units: u64) -> String {
    let x = units as f64 / GMDS as f64;
    let s = format!("{x:.6}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    format!("{s} gMDS")
}

/// gMDS typed by a person, in units.
fn parse_mds(s: &str) -> Option<u64> {
    let x: f64 = s.trim().parse().ok()?;
    (x > 0.0 && x < 1e9).then(|| (x * GMDS as f64).round() as u64)
}

/// ETH typed by a person, in wei, exactly.
fn parse_eth(s: &str) -> Option<u128> {
    let t = s.trim();
    let (i, f) = t.split_once('.').unwrap_or((t, ""));
    if (i.is_empty() && f.is_empty()) || f.len() > 18 || !i.chars().all(|c| c.is_ascii_digit()) || !f.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let whole: u128 = if i.is_empty() { 0 } else { i.parse().ok()? };
    let frac: u128 = if f.is_empty() { 0 } else { format!("{f:0<18}").parse().ok()? };
    whole.checked_mul(WEI)?.checked_add(frac).filter(|v| *v > 0)
}

/// USDC typed by a person, in millionths of a dollar, exactly.
fn parse_usdc(s: &str) -> Option<u128> {
    let t = s.trim().trim_start_matches('$').replace(',', "");
    let (i, f) = t.split_once('.').unwrap_or((&t, ""));
    if (i.is_empty() && f.is_empty()) || f.len() > 6 || !i.chars().all(|c| c.is_ascii_digit()) || !f.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let whole: u128 = if i.is_empty() { 0 } else { i.parse().ok()? };
    let frac: u128 = if f.is_empty() { 0 } else { format!("{f:0<6}").parse().ok()? };
    whole.checked_mul(MICRO)?.checked_add(frac).filter(|v| *v > 0)
}

/// An amount of `asset` typed by a person, in its smallest unit.
fn parse_amount(asset: &str, s: &str) -> Option<u128> {
    if asset == USDC {
        parse_usdc(s)
    } else {
        parse_eth(s)
    }
}

/// The currency of the market on screen.
fn asset_of(app: &App) -> &'static str {
    if app.market == 1 {
        USDC
    } else {
        ETH
    }
}

/// What `units` cost at `wei_per_gmds`, rounded up as the seller's wallet does.
fn cost(units: u64, wei_per_gmds: u128) -> u128 {
    (units as u128 * wei_per_gmds).div_ceil(GMDS)
}

fn field(ui: &mut Ui, text: &mut String, hint: &str, width: f32) -> egui::Response {
    ui.add(TextEdit::singleline(text).hint_text(hint).font(egui::TextStyle::Monospace).desired_width(width))
}

fn caption(ui: &mut Ui, t: &str) {
    ui.label(RichText::new(t.to_uppercase()).font(FontId::monospace(9.5)).color(theme::muted()));
}

pub fn show(app: &mut App, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    theme::heading(ui, "Trade");
    let mut market = app.market.min(2);
    if theme::segmented(ui, &["MDS ↔ ETH (Base)", "MDS ↔ USDC (Base)", "MDS ↔ MDW (midwimble)"], &mut market) {
        // Prices typed in one currency mean nothing in the other.
        if (market == 1) != (app.market == 1) {
            app.eth_sell_price.clear();
            app.bid_wei.clear();
            app.bid_bond = "0".into();
            app.eth_buy = None;
            app.eth_fill = None;
        }
        app.market = market;
        app.mwb_last_poll = 0.0;
        app.eth_last_poll = 0.0;
    }
    ui.add_space(6.0);
    if app.market == 2 {
        crate::views::trade_mdw::show(app, ui, &ctx);
        return;
    }
    let asset = asset_of(app);
    theme::hint(
        ui,
        &format!(
            "Swap MDS and {asset} on Base, wallet to wallet. Every trade is atomic: both sides are paid, \
             or both get their money back. Keep the app open while a trade runs."
        ),
    );
    ui.add_space(6.0);
    account_strip(app, ui, &ctx);
    ui.add_space(8.0);
    if asset == USDC && !app.evm.as_ref().is_some_and(|a| a.usdc_ready) {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("USDC trading is not switched on yet").font(theme::font_medium(14.0)));
            theme::hint(
                ui,
                "USDC trades run through their own contract on Base (MidstateTokenSwap), the same \
                 protocol as ETH's for one token. Once it is deployed, its address goes under \
                 Connection below, and this market opens.",
            );
        });
        ui.add_space(10.0);
        connection(app, ui, &ctx);
        return;
    }

    let running = app.eth_market.as_ref().map_or(0, |m| m.swaps.iter().filter(|s| !s.settled).count())
        + app.swaps.iter().filter(|s| !s.settled).count();
    let mine = if running > 0 { format!("My trades ({running})") } else { "My trades".into() };
    let labels = [TABS[0], TABS[1], mine.as_str()];
    let mut ix = app.eth_tab.min(2);
    if theme::segmented(ui, &labels, &mut ix) {
        app.eth_tab = ix;
    }
    ui.add_space(8.0);
    if !app.eth_notice.is_empty() {
        ui.label(RichText::new(&app.eth_notice).size(12.0).color(theme::bright()));
    }
    if !app.error.is_empty() {
        ui.label(RichText::new(&app.error).size(12.0).color(theme::muted()));
    }
    if let Some(e) = app.eth_market.as_ref().and_then(|m| m.base_error.clone()) {
        theme::hint(ui, &format!("Base is not answering right now ({e}). Trades wait and retry; nothing is lost."));
    }
    match app.eth_tab {
        0 => buy_view(app, ui, &ctx, asset),
        1 => sell_view(app, ui, &ctx, asset),
        _ => my_trades(app, ui, &ctx),
    }
    ui.add_space(10.0);
    connection(app, ui, &ctx);
}

// ── Buy MDS ─────────────────────────────────────────────────────────────────

fn buy_view(app: &mut App, ui: &mut Ui, ctx: &egui::Context, asset: &str) {
    let sells: Vec<EthAdvertView> = app.eth_market.as_ref().map(|m| m.sells.iter().filter(|a| a.asset == asset).cloned().collect()).unwrap_or_default();
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Sell orders").font(theme::font_medium(14.0)));
        theme::hint(
            ui,
            &format!(
                "Buy instantly from a seller whose wallet is online. The seller locks the MDS so that only \
                 you can claim it; your {asset} is locked only after that, and comes back to you if the seller \
                 never claims it."
            ),
        );
        ui.add_space(4.0);
        if sells.iter().all(|a| a.mine) {
            theme::hint(ui, "No sell orders right now. Post a buy order below and a seller can fill it while you are away.");
        }
        egui::Grid::new("eth_sells").num_columns(4).spacing([18.0, 7.0]).striped(true).show(ui, |ui| {
            for h in ["Price per gMDS", "Available", "Seller", ""] {
                caption(ui, h);
            }
            ui.end_row();
            for a in sells.iter().filter(|a| !a.mine) {
                ui.label(theme::mono(per_mds(asset, &a.wei_per_gmds)).size(11.5).color(theme::ink()));
                ui.label(theme::mono(gmds(a.mds)).size(11.5));
                ui.horizontal(|ui| {
                    // A name only when the seller's account signed it and the
                    // name's Base record is that account; the account is what pays.
                    if !a.seller_name.is_empty() {
                        ui.label(RichText::new(&a.seller_name).strong().size(12.0))
                            .on_hover_text(format!("{} signed for this name, and the name's Base account is {}", a.seller, a.seller));
                    }
                    // What the seller burned to back its account and name:
                    // a cost a throwaway account would not carry.
                    if let Some(t) = midstate_walletd::burns::tier(a.seller_backing) {
                        theme::badge(ui, t, theme::gold());
                    }
                    if a.seller_backing > 0.0 {
                        ui.label(RichText::new(midstate_walletd::burns::blocks_text(a.seller_backing)).size(10.5).color(theme::muted()))
                            .on_hover_text("MDS burned to back this seller's Base account and name (MSB-1), in blocks of issuance, halving yearly");
                    }
                    ui.label(theme::mono(short_hex(&a.seller, 8)).size(11.0).color(theme::muted()));
                    explorer_link(ui, "↗", basescan_address(&a.seller));
                });
                let open = app.eth_buy.as_deref() == Some(a.order_id.as_str());
                if ui.add_enabled(!app.busy, egui::Button::new(if open { "Close" } else { "Buy" })).clicked() {
                    app.eth_buy = if open { None } else { Some(a.order_id.clone()) };
                    app.eth_buy_amount = format!("{}", a.mds as f64 / GMDS as f64);
                }
                ui.end_row();
            }
        });
        if let Some(a) = sells.iter().find(|a| Some(a.order_id.as_str()) == app.eth_buy.as_deref()).cloned() {
            ui.add_space(6.0);
            buy_form(app, ui, ctx, &a);
        }
    });
    ui.add_space(8.0);
    place_bid_panel(app, ui, ctx, asset);
    if let Some(b) = app.book.clone() {
        ui.add_space(8.0);
        egui::CollapsingHeader::new("Recent trades").id_salt(("eth_recent", asset)).show(ui, |ui| trades_panel(ui, &b, asset));
    }
}

fn buy_form(app: &mut App, ui: &mut Ui, ctx: &egui::Context, a: &EthAdvertView) {
    let asset = a.asset.as_str();
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        let price: u128 = a.wei_per_gmds.parse().unwrap_or(0);
        ui.horizontal(|ui| {
            theme::hint(ui, "buy");
            field(ui, &mut app.eth_buy_amount, "gMDS", 120.0);
            theme::hint(ui, &format!("gMDS of {} available, at {} per gMDS", gmds(a.mds), per_mds(asset, &a.wei_per_gmds)));
        });
        let units = parse_mds(&app.eth_buy_amount);
        let ok = units.is_some_and(|u| u >= a.min && u <= a.mds);
        match units {
            Some(u) if ok => {
                let gas = if asset == USDC { "plus a little ETH for gas" } else { "plus Base gas" };
                ui.label(RichText::new(format!("You pay {} ({gas}) for {}.", amount_of(asset, cost(u, price)), gmds(u))).size(13.0).color(theme::bright()));
            }
            Some(_) => theme::hint(ui, &format!("Between {} and {}.", gmds(a.min), gmds(a.mds))),
            None => theme::hint(ui, "Enter an amount of gMDS."),
        }
        let how = if asset == USDC {
            "What happens: the seller's wallet locks the MDS for you (about two blocks), your wallet \
             checks it on chain, then locks your USDC, in one transaction that approves exactly that \
             amount and no more. The seller claims the USDC, which hands your wallet the key to the \
             MDS. If anything stalls, your USDC comes back after an hour."
        } else {
            "What happens: the seller's wallet locks the MDS for you (about two blocks), your wallet \
             checks it on chain, then locks your ETH. The seller claims the ETH, which hands your wallet \
             the key to the MDS. If anything stalls, your ETH comes back after an hour."
        };
        theme::hint(ui, how);
        theme::right_aligned(ui, |ui| {
            if ui.add_enabled(ok && !app.busy, egui::Button::new("Confirm buy")).clicked() {
                app.busy = true;
                app.error.clear();
                app.eth_notice.clear();
                app.go(ctx, Action::EthBuy { order_id: a.order_id.clone(), mds: units.unwrap_or(0) });
                app.eth_buy = None;
                app.eth_tab = 2;
            }
        });
    });
}

fn place_bid_panel(app: &mut App, ui: &mut Ui, ctx: &egui::Context, asset: &str) {
    egui::CollapsingHeader::new("Post a buy order")
        .id_salt(("eth_post_bid", asset))
        .default_open(false)
        .show(ui, |ui| {
            theme::panel_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                theme::hint(
                    ui,
                    &format!(
                        "Your {asset} waits in the contract until a seller fills the order, so it works while you \
                         are away. A seller reserves it, locks the MDS to you, and your wallet collects it, \
                         which pays them. Cancel any time before someone reserves it."
                    ),
                );
                egui::Grid::new("eth_bid_form").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
                    theme::hint(ui, "amount");
                    ui.horizontal(|ui| {
                        field(ui, &mut app.bid_mds, "gMDS", 120.0);
                        theme::hint(ui, "gMDS");
                    });
                    ui.end_row();
                    theme::hint(ui, "price");
                    ui.horizontal(|ui| {
                        field(ui, &mut app.bid_wei, if asset == USDC { "2.50" } else { "0.001" }, 120.0);
                        theme::hint(ui, &format!("{asset} per gMDS"));
                    });
                    ui.end_row();
                    theme::hint(ui, "for");
                    ui.horizontal(|ui| {
                        field(ui, &mut app.bid_hours, "24", 60.0);
                        theme::hint(ui, "hours");
                    });
                    ui.end_row();
                    theme::hint(ui, "seller's bond");
                    ui.horizontal(|ui| {
                        field(ui, &mut app.bid_bond, "0", 120.0);
                        theme::hint(ui, &format!("{asset}, kept by you if a seller reserves and never delivers"));
                    });
                    ui.end_row();
                });
                let units = parse_mds(&app.bid_mds);
                let price = parse_amount(asset, &app.bid_wei);
                let bond = if app.bid_bond.trim().is_empty() { Some(0) } else { parse_amount(asset, &app.bid_bond).or(Some(0)) };
                let total = units.zip(price).map(|(u, p)| cost(u, p));
                if let (Some(u), Some(t)) = (units, total) {
                    let how = if asset == USDC { ", approved and escrowed in one transaction" } else { "" };
                    ui.label(RichText::new(format!("Escrows {} for {}{how}.", amount_of(asset, t), gmds(u))).size(13.0).color(theme::bright()));
                }
                theme::right_aligned(ui, |ui| {
                    let can = !app.busy && total.is_some_and(|t| t > 0);
                    if ui.add_enabled(can, egui::Button::new("Escrow buy order")).clicked() {
                        app.busy = true;
                        app.error.clear();
                        app.eth_notice.clear();
                        app.go(
                            ctx,
                            Action::PlaceBid {
                                asset: asset.to_string(),
                                mds_amount: units.unwrap_or(0),
                                wei: total.unwrap_or(0).to_string(),
                                ttl_secs: app.bid_hours.trim().parse::<u64>().unwrap_or(24).clamp(1, 2160) * 3600,
                                fill_bond: bond.unwrap_or(0).to_string(),
                            },
                        );
                    }
                });
            });
        });
}

// ── Sell MDS ────────────────────────────────────────────────────────────────

fn sell_view(app: &mut App, ui: &mut Ui, ctx: &egui::Context, asset: &str) {
    let bids: Vec<BidView> = app.book.as_ref().map(|b| b.bids.iter().filter(|b| b.asset == asset).cloned().collect()).unwrap_or_default();
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Buy orders").font(theme::font_medium(14.0)));
        theme::hint(
            ui,
            &format!(
                "{asset} already escrowed on Base, waiting for MDS. Sell into one: your wallet reserves it, \
                 locks the MDS so it can only go to that buyer, and collects the {asset} when the buyer takes \
                 the MDS. If the buyer never does, your MDS comes back."
            ),
        );
        ui.add_space(4.0);
        let open: Vec<&BidView> = bids.iter().filter(|b| b.takeable && !b.mine).collect();
        if open.is_empty() {
            theme::hint(ui, "No open buy orders right now. Post a sell order below.");
        }
        egui::Grid::new("eth_bids").num_columns(5).spacing([18.0, 7.0]).striped(true).show(ui, |ui| {
            for h in ["Price per gMDS", "Wants", "Pays", "Bond", ""] {
                caption(ui, h);
            }
            ui.end_row();
            for b in &open {
                ui.label(theme::mono(per_mds_f(asset, b.price)).size(11.5).color(theme::ink()));
                ui.label(theme::mono(gmds(b.mds_amount)).size(11.5));
                ui.label(theme::mono(amount(asset, &b.wei)).size(11.5));
                ui.label(theme::mono(if b.fill_bond == "0" { "none".into() } else { amount(asset, &b.fill_bond) }).size(11.0).color(theme::muted()));
                let armed = app.eth_fill.as_deref() == Some(b.bid_id.as_str());
                if ui.add_enabled(!app.busy, egui::Button::new(if armed { "Close" } else { "Sell" })).clicked() {
                    app.eth_fill = if armed { None } else { Some(b.bid_id.clone()) };
                }
                ui.end_row();
            }
        });
        if let Some(b) = open.iter().find(|b| Some(b.bid_id.as_str()) == app.eth_fill.as_deref()) {
            ui.add_space(6.0);
            theme::panel_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new(format!("Sell {} for {}.", gmds(b.mds_amount), amount(asset, &b.wei))).size(13.0).color(theme::bright()));
                let bond = if b.fill_bond == "0" {
                    "no bond".to_string()
                } else {
                    format!("a bond of {}, returned when you are paid", amount(asset, &b.fill_bond))
                };
                let gas = if asset == USDC { "plus a little ETH for gas" } else { "plus gas" };
                theme::hint(
                    ui,
                    &format!(
                        "Your wallet reserves the order on Base ({bond}, {gas}), then locks the MDS and a \
                         little more for the buyer's claim fee. The reservation lasts four hours."
                    ),
                );
                theme::right_aligned(ui, |ui| {
                    if ui.add_enabled(!app.busy, egui::Button::new("Confirm sell")).clicked() {
                        app.busy = true;
                        app.error.clear();
                        app.eth_notice.clear();
                        app.go(ctx, Action::EthFillBid { bid_id: b.bid_id.clone() });
                        app.eth_fill = None;
                        app.eth_tab = 2;
                    }
                });
            });
        }
    });
    ui.add_space(8.0);
    sell_order_form(app, ui, ctx, asset);
}

fn sell_order_form(app: &mut App, ui: &mut Ui, ctx: &egui::Context, asset: &str) {
    egui::CollapsingHeader::new("Post a sell order")
        .id_salt(("eth_post_sell", asset))
        .default_open(false)
        .show(ui, |ui| {
            theme::panel_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                theme::hint(
                    ui,
                    "Nothing is locked until a buyer asks; then your wallet locks just what they asked for, \
                     claimable only by them. Buyers see the order while this app is open.",
                );
                egui::Grid::new("eth_sell_form").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
                    theme::hint(ui, "amount");
                    ui.horizontal(|ui| {
                        field(ui, &mut app.eth_sell_mds, "gMDS", 120.0);
                        theme::hint(ui, "gMDS");
                    });
                    ui.end_row();
                    theme::hint(ui, "price");
                    ui.horizontal(|ui| {
                        field(ui, &mut app.eth_sell_price, if asset == USDC { "2.50" } else { "0.001" }, 120.0);
                        theme::hint(ui, &format!("{asset} per gMDS"));
                    });
                    ui.end_row();
                    theme::hint(ui, "smallest sale");
                    ui.horizontal(|ui| {
                        field(ui, &mut app.eth_sell_min, "0.1", 120.0);
                        theme::hint(ui, "gMDS");
                    });
                    ui.end_row();
                    theme::hint(ui, "for");
                    ui.horizontal(|ui| {
                        field(ui, &mut app.eth_sell_hours, "24", 60.0);
                        theme::hint(ui, "hours");
                    });
                    ui.end_row();
                });
                let units = parse_mds(&app.eth_sell_mds);
                let price = parse_amount(asset, &app.eth_sell_price);
                let min = parse_mds(&app.eth_sell_min).unwrap_or(1 << 20);
                if let (Some(u), Some(p)) = (units, price) {
                    ui.label(RichText::new(format!("{} for {} in all.", gmds(u), amount_of(asset, cost(u, p)))).size(13.0).color(theme::bright()));
                }
                theme::right_aligned(ui, |ui| {
                    let can = !app.busy && units.is_some() && price.is_some();
                    if ui.add_enabled(can, egui::Button::new("Post sell order")).clicked() {
                        let hours = app.eth_sell_hours.trim().parse::<u64>().unwrap_or(24).clamp(1, 720);
                        app.busy = true;
                        app.error.clear();
                        app.eth_notice.clear();
                        app.go(
                            ctx,
                            Action::EthSell { asset: asset.to_string(), mds: units.unwrap_or(0), wei_per_gmds: price.unwrap_or(0).to_string(), min, life: hours * 60 },
                        );
                    }
                });
            });
        });
}

// ── My trades ───────────────────────────────────────────────────────────────

fn kind_label(k: &str) -> &'static str {
    match k {
        "buy" => "buying",
        "sell" => "selling",
        "fill" => "selling into an order",
        "bid" => "your buy order",
        _ => "trade",
    }
}

fn swap_row(ui: &mut Ui, s: &EthSwapView) {
    ui.horizontal_wrapped(|ui| {
        theme::badge(ui, kind_label(&s.kind), if s.settled { theme::muted() } else { theme::ink() });
        ui.label(theme::mono(gmds(s.mds)).size(11.5));
        theme::hint(ui, "for");
        ui.label(theme::mono(amount(&s.asset, &s.wei)).size(11.5));
        if !s.counterparty.is_empty() {
            theme::hint(ui, "with");
            if !s.counterparty_name.is_empty() {
                ui.label(RichText::new(&s.counterparty_name).strong().size(12.0));
            }
            ui.label(theme::mono(short_hex(&s.counterparty, 8)).size(11.0).color(theme::muted()));
        }
        if let Some(tx) = &s.base_tx {
            explorer_link(ui, "Base tx ↗", basescan_tx(tx));
        }
    });
    ui.label(RichText::new(&s.note).size(12.0).color(if s.settled { theme::muted() } else { theme::bright() }));
}

fn my_trades(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    let m = app.eth_market.clone().unwrap_or_default();
    let live: Vec<&EthSwapView> = m.swaps.iter().filter(|s| !s.settled).collect();
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Under way").font(theme::font_medium(14.0)));
        if live.is_empty() {
            theme::hint(ui, "Nothing under way.");
        } else {
            theme::hint(ui, "Each step has a deadline, and the wallet acts on them for you. Keep the app open until these finish.");
        }
        for s in &live {
            ui.add_space(4.0);
            swap_row(ui, s);
        }
    });

    let mut cancel_sell: Option<String> = None;
    let mut cancel_bid: Option<String> = None;
    if m.orders.iter().any(|o| o.status == "open") || app.my_bids.iter().any(|b| b.status == "open") {
        ui.add_space(8.0);
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Your open orders").font(theme::font_medium(14.0)));
            for o in m.orders.iter().filter(|o| o.status == "open") {
                ui.horizontal_wrapped(|ui| {
                    theme::badge(ui, "selling", theme::ink());
                    ui.label(theme::mono(gmds(o.mds)).size(11.5));
                    theme::hint(ui, &format!("at {} per gMDS", per_mds(&o.asset, &o.wei_per_gmds)));
                    if o.sold > 0 {
                        theme::hint(ui, &format!("· {} sold", gmds(o.sold)));
                    }
                    if o.running > 0 {
                        theme::hint(ui, &format!("· {} sale(s) under way", o.running));
                    }
                    theme::hint(ui, &format!("· ends at block {}", o.expires));
                    if ui.add_enabled(!app.busy, egui::Button::new(RichText::new("cancel").size(10.5))).clicked() {
                        cancel_sell = Some(o.id.clone());
                    }
                });
            }
            for b in app.my_bids.iter().filter(|b| b.status == "open") {
                ui.horizontal_wrapped(|ui| {
                    theme::badge(ui, "buying", theme::ink());
                    ui.label(theme::mono(gmds(b.mds_amount)).size(11.5));
                    theme::hint(ui, "for");
                    ui.label(theme::mono(amount(&b.asset, &b.wei)).size(11.5));
                    if !b.tx.is_empty() {
                        explorer_link(ui, "Base tx ↗", basescan_tx(&b.tx));
                    }
                    if !b.bid_id.is_empty()
                        && ui
                            .add_enabled(!app.busy, egui::Button::new(RichText::new("cancel").size(10.5)))
                            .on_hover_text(format!("Returns your {}, if no seller has reserved it", b.asset))
                            .clicked()
                    {
                        cancel_bid = Some(b.bid_id.clone());
                    }
                });
            }
        });
    }
    if let Some(id) = cancel_sell {
        app.busy = true;
        app.error.clear();
        app.go(ctx, Action::EthCancelSell { id });
    }
    if let Some(id) = cancel_bid {
        app.busy = true;
        app.error.clear();
        app.go(ctx, Action::CancelBid { bid_id: id });
    }

    let done: Vec<&EthSwapView> = m.swaps.iter().filter(|s| s.settled).collect();
    if !done.is_empty() {
        ui.add_space(8.0);
        egui::CollapsingHeader::new(format!("Finished ({})", done.len())).id_salt("eth_done").show(ui, |ui| {
            for s in done {
                swap_row(ui, s);
                ui.add_space(4.0);
            }
        });
    }
    if !app.my_orders.is_empty() {
        // Sell orders from before this market: their coins come back once
        // each order expires.
        let tip = app.sync.as_ref().map(|s| s.height).unwrap_or(0);
        let mut reclaim: Option<String> = None;
        ui.add_space(8.0);
        egui::CollapsingHeader::new(format!("Earlier sell orders ({})", app.my_orders.len())).id_salt("old_orders").show(ui, |ui| {
            theme::hint(ui, "Published before sell orders locked for one buyer at a time. Unsold coins can be swept back once an order expires.");
            for o in &app.my_orders {
                let expired = tip > 0 && o.timeout_height <= tip;
                ui.horizontal_wrapped(|ui| {
                    theme::badge(ui, if expired { "expired" } else { "open" }, if expired { theme::muted() } else { theme::ink() });
                    ui.label(theme::mono(gmds(o.mds_amount)).size(11.5));
                    theme::hint(ui, &format!("· expires at block {}", o.timeout_height));
                    if expired
                        && ui
                            .add_enabled(!app.busy, egui::Button::new(RichText::new("reclaim coins").size(10.5)))
                            .on_hover_text("Sweep whatever went unsold back into your wallet")
                            .clicked()
                    {
                        reclaim = Some(o.group_id.clone());
                    }
                });
            }
        });
        if let Some(g) = reclaim {
            app.busy = true;
            app.error.clear();
            app.go(ctx, Action::ReclaimOrder { group_id: g });
        }
    }
    if !app.swaps.is_empty() {
        ui.add_space(8.0);
        egui::CollapsingHeader::new("Earlier trades").id_salt("eth_legacy").show(ui, |ui| {
            theme::hint(ui, "Trades from before sell orders locked for one buyer at a time.");
            for s in &app.swaps {
                ui.horizontal_wrapped(|ui| {
                    theme::badge(ui, &s.phase, if s.settled { theme::muted() } else { theme::ink() });
                    ui.label(theme::mono(gmds(s.mds_value)).size(11.5));
                    ui.label(theme::mono(eth(&s.wei)).size(11.5));
                    ui.label(RichText::new(&s.detail).size(11.5).color(theme::muted()));
                });
            }
        });
    }
}

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
        let usdc = app.market == 1;
        let contract = if usdc { a.usdc_contract.clone() } else { a.contract.clone() };
        if !contract.is_empty() {
            ui.horizontal(|ui| {
                ui.label(RichText::new("contract").font(FontId::monospace(9.5)).color(theme::muted()));
                ui.label(theme::mono(short_hex(&contract, 10)).size(11.0).color(theme::muted()));
                explorer_link(ui, "view on Basescan", basescan_address(&contract));
            });
        }
        ui.horizontal(|ui| {
            match &a.balance_wei {
                Some(w) => ui.label(theme::mono(eth(w)).size(12.5)),
                None => ui.label(
                    RichText::new("balance unavailable — endpoint unreachable")
                        .size(12.0)
                        .color(theme::muted()),
                ),
            };
            if usdc {
                match &a.usdc_balance {
                    Some(u) => ui.label(theme::mono(amount(USDC, u)).size(12.5)),
                    None if a.usdc_ready => ui.label(RichText::new("USDC balance unavailable").size(12.0).color(theme::muted())),
                    None => ui.label(RichText::new("")),
                };
            }
            theme::right_aligned(ui, |ui| {
                let label = if app.dex_syncing { "Scanning…" } else { "Refresh" };
                if ui.add_enabled(!app.dex_syncing, egui::Button::new(label)).clicked() {
                    app.dex_syncing = true;
                    app.go(ctx, Action::SyncOrderBook);
                }
            });
        });
        theme::hint(
            ui,
            if usdc {
                "Same recovery phrase, standard derivation path — this account also opens in \
                 MetaMask. USDC trades still need a little ETH here for gas."
            } else {
                "Same recovery phrase, standard derivation path — this account also opens in \
                 MetaMask. Swaps need a little ETH here for gas."
            },
        );
    });
}


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
                    theme::hint(ui, "USDC contract");
                    ui.add(
                        TextEdit::singleline(&mut next.usdc_contract)
                            .font(egui::TextStyle::Monospace)
                            .hint_text("MidstateTokenSwap, once deployed")
                            .desired_width(f32::INFINITY),
                    );
                });
                ui.horizontal(|ui| {
                    theme::hint(ui, "USDC token");
                    ui.add(
                        TextEdit::singleline(&mut next.usdc_token)
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
        || a.start_block != b.start_block
        || a.usdc_contract != b.usdc_contract
        || a.usdc_token != b.usdc_token
}


/// What the market has actually done, as opposed to what it is offering.
fn trades_panel(ui: &mut Ui, b: &midstate_walletd::api::OrderBookView, asset: &str) {
    theme::hint(
        ui,
        "Trades that settled, rebuilt from the contract's own events — every one is a real \
         payout, not an offer. Reconstructed from the scanned range, so widening the window \
         under Connection reveals more history.",
    );
    ui.add_space(4.0);

    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        let trades: Vec<&midstate_walletd::api::TradeView> = b.trades.iter().filter(|t| t.asset == asset).collect();
        if trades.is_empty() {
            theme::hint(ui, "No completed trades in the scanned range.");
            return;
        }
        egui::Grid::new(("trades", asset)).num_columns(5).spacing([20.0, 7.0]).striped(true).show(ui, |ui| {
            for h in ["Block", "Type", "Amount", asset, "Price per gMDS"] {
                ui.label(
                    RichText::new(h.to_uppercase()).font(FontId::monospace(9.5)).color(theme::muted()),
                );
            }
            ui.end_row();
            for t in trades {
                ui.label(theme::mono(units(t.block)).size(11.0).color(theme::muted()));
                theme::badge(
                    ui,
                    if t.kind == "buy" { "bid filled" } else { "ask taken" },
                    theme::muted(),
                );
                ui.label(match t.mds {
                    Some(m) => theme::mono(gmds(m)).size(11.5),
                    // The settlement event names only an id; the size is known
                    // only if the matching announcement was in range.
                    None => theme::mono("—").size(11.5).color(theme::faint()),
                });
                ui.label(
                    theme::mono(if t.wei == "0" { "—".to_string() } else { amount(asset, &t.wei) })
                        .size(11.5),
                );
                ui.label(match t.price {
                    Some(p) => theme::mono(per_mds_f(asset, p)).size(11.5).color(theme::ink()),
                    None => theme::mono("—").size(11.0).color(theme::faint()),
                });
                ui.end_row();
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usdc_reads_and_parses_as_dollars() {
        assert_eq!(usdc_of(12_500_000), "12.50 USDC");
        assert_eq!(usdc_of(1), "0.000001 USDC");
        assert_eq!(usdc_of(1_234_567_890_000), "1,234,567.89 USDC");
        assert_eq!(usdc_of(0), "0.00 USDC");
        assert_eq!(parse_usdc("2.5"), Some(2_500_000));
        assert_eq!(parse_usdc("$1,000.01"), Some(1_000_010_000));
        assert_eq!(parse_usdc(".000001"), Some(1));
        assert_eq!(parse_usdc("0.0000001"), None, "finer than a millionth");
        assert_eq!(parse_usdc("0"), None);
        assert_eq!(parse_usdc("abc"), None);
        // The same text is a very different amount in the other currency.
        assert_eq!(parse_amount(USDC, "1"), Some(1_000_000));
        assert_eq!(parse_amount(ETH, "1"), Some(WEI));
        assert_eq!(amount(USDC, "2500000"), "2.50 USDC");
        assert_eq!(per_mds_f(USDC, 2_500_000.0 / GMDS as f64), "2.50 USDC");
    }
}
