//! Names (MSN-1): find any name and see what can be done with it, keep the
//! ones this wallet holds, and trade them on the market.

use crate::app::App;
use crate::bridge::Action;
use crate::theme;
use eframe::egui::{self, RichText, TextEdit, Ui};
use midstate_walletd::api::{NameAction, NameCard, NameEdit};
use midstate_walletd::names::normalize;

const DAY: u64 = 1_440;

/// An amount in gMDS, for people.
pub fn gmds(units: u64) -> String {
    let x = units as f64 / (1u64 << 30) as f64;
    if units % (1 << 30) == 0 { format!("{x:.0} gMDS") } else { format!("{x:.4} gMDS") }
}

/// gMDS typed by a person, in units.
fn parse_gmds(s: &str) -> Option<u64> {
    let x: f64 = s.trim().parse().ok()?;
    (x > 0.0 && x < 1e9).then(|| (x * (1u64 << 30) as f64).round() as u64)
}

fn days(blocks: u64) -> String {
    match blocks / DAY {
        0 => "under a day".into(),
        1 => "1 day".into(),
        d if d < 730 => format!("{d} days"),
        d => format!("{:.1} years", d as f64 / 365.0),
    }
}

fn short(name: &str) -> String {
    name.trim_end_matches(".mds").to_string()
}

/// What backs a name (MSB-1 burns), and a way to add to it.
fn backing_row(app: &mut App, ui: &mut Ui, name: &str, b: Option<&midstate_walletd::api::BackingView>) {
    ui.horizontal(|ui| {
        match b {
            Some(b) => {
                if !b.tier.is_empty() {
                    theme::badge(ui, &b.tier, theme::gold());
                }
                theme::hint(ui, &format!("backed by {} ({} burned)", b.text, gmds(b.total)));
            }
            None => theme::hint(ui, "no backing burned for it"),
        }
        if ui.small_button("Back…").on_hover_text("Burn MDS to back this name: wallets show it to payers and to everyone it messages").clicked() {
            app.burn_form = crate::views::burns::BurnForm::backing("name", name);
            app.tab = crate::app::Tab::Burns;
        }
    });
}

pub fn show(app: &mut App, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    theme::heading(ui, "Names");
    theme::hint(
        ui,
        "A name like nick.mds takes payments in MDS and MDW from any wallet that reads names. Holding \
         one burns its tier's price each year — platinum 64, gold 8, silver 1, bronze 0.125 gMDS — \
         dearer for short names and names in demand.",
    );
    find(app, ui, &ctx);
    yours(app, ui, &ctx);
    market(app, ui, &ctx);
    recent(app, ui);
}

/// One box for any name: what it is, what it costs, and what to do.
fn find(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            let r = ui.add(TextEdit::singleline(&mut app.name_input).hint_text("find a name").desired_width(220.0));
            ui.label(".mds");
            let y = ui.add(egui::DragValue::new(&mut app.name_years).range(1..=10).suffix(" yr"));
            let name = app.name_input.trim().to_string();
            if (r.changed() || y.changed()) && normalize(&name).is_ok() {
                app.name_confirm = None;
                app.go(ctx, Action::FindName { name, years: app.name_years });
            }
        });
        let typed = normalize(&app.name_input).ok();
        if let (Err(e), false) = (normalize(&app.name_input), app.name_input.trim().is_empty()) {
            ui.label(RichText::new(e.to_string()).color(theme::red()).size(12.0));
        }
        let Some(card) = app.name_card.clone().filter(|c| Some(short(&c.name)) == typed) else { return };
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new(&card.name).strong().size(16.0));
            theme::badge(ui, &card.tier, theme::muted());
            theme::badge(ui, &card.state, theme::gold());
        });
        let left = card.expires.saturating_sub(card.tip);
        let line = match card.state.as_str() {
            "available" => format!("Free to register: {} a year.", gmds(card.rate)),
            "unclaimed" => format!("A premium name no one has claimed yet: {} a year.", gmds(card.rate)),
            "lapsed" => format!("Its holder let it lapse; anyone may claim it: {} a year.", gmds(card.rate)),
            "for sale" => format!("For sale at {}. Held for {} more.", gmds(card.price), days(left)),
            "for rent" => format!("For rent: {} for {} at a time.", gmds(card.price), days(card.rent_days * DAY)),
            "rented" => "Rented out right now.".into(),
            "yours" => format!("This wallet's. Held for {} more.", days(left)),
            _ => format!("Held, for {} more. Not offered.", days(left)),
        };
        theme::hint(ui, &line);
        if let Some(p) = &card.pays {
            ui.label(RichText::new(format!("Pays {} — {}", theme::grouped_hash(p, 24), card.pays_kind)).size(12.0));
        }
        let mut can = Vec::new();
        if card.identity {
            can.push("its wallet hands out a fresh address for each payment");
        }
        if card.sealed {
            can.push("it takes sealed messages");
        }
        if !can.is_empty() {
            theme::hint(ui, &format!("{}.", can.join("; ")));
        }
        for r in card.records.iter().filter(|r| r.kind != "script") {
            ui.label(RichText::new(format!("{:<12}{}", r.kind, r.value)).monospace().size(10.5).color(theme::muted()));
        }
        backing_row(app, ui, &card.name, card.backing.as_ref());
        if card.sealed && ui.small_button("Message…").clicked() {
            app.sealed_to = card.name.clone();
            app.tab = crate::app::Tab::Chat;
        }
        // The action this state allows, behind a confirm that says what it spends.
        let (label, act, spend) = match card.state.as_str() {
            "available" | "unclaimed" | "lapsed" => (
                format!("Register for {} yr", card.years),
                "register",
                format!("Burns {}, plus a network fee of about {} units. Held about {}.", gmds(card.burn), card.fee, days(card.years * 525_600)),
            ),
            "for sale" => (
                "Buy".into(),
                "buy",
                format!("Pays {} to its holder, plus a network fee of about {} units. Yours for the {} it has left; renew it after.", gmds(card.price), card.fee, days(left)),
            ),
            "for rent" => (
                "Rent".into(),
                "rent",
                format!(
                    "Pays {} to its holder, plus a network fee of about {} units. Payments to it come to you for {}; its holder keeps it.",
                    gmds(card.price),
                    card.fee,
                    days(card.rent_until.saturating_sub(card.tip))
                ),
            ),
            _ => (String::new(), "", String::new()),
        };
        ui.horizontal(|ui| {
            if !act.is_empty() && app.name_confirm.is_none() && ui.add_enabled(!app.busy, egui::Button::new(RichText::new(&label).strong())).clicked() {
                app.name_confirm = Some(act);
            }
            if !matches!(card.state.as_str(), "yours" | "available") && app.name_confirm.is_none() {
                ui.add(TextEdit::singleline(&mut app.watch_max).hint_text("max gMDS").desired_width(70.0));
                if ui.button("Watch").on_hover_text("Claim it the block it comes free; buy it if offered for at most the max").clicked() {
                    app.busy = true;
                    let max_price = parse_gmds(&app.watch_max);
                    app.go(ctx, Action::WatchName { name: short(&card.name), years: card.years, max_price });
                }
            }
        });
        if let Some(act) = app.name_confirm {
            confirm_box(app, ui, ctx, &card, act, &spend);
        }
    });
}

fn confirm_box(app: &mut App, ui: &mut Ui, ctx: &egui::Context, card: &NameCard, act: &'static str, spend: &str) {
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.label(RichText::new(spend).size(12.5));
        ui.horizontal(|ui| {
            if ui.add_enabled(!app.busy, egui::Button::new(RichText::new("Confirm").strong())).clicked() {
                app.busy = true;
                app.error.clear();
                let name = short(&card.name);
                app.go(
                    ctx,
                    match act {
                        "register" => Action::RegisterName { name, years: card.years },
                        "buy" => Action::TakeName { name, rent: false },
                        _ => Action::TakeName { name, rent: true },
                    },
                );
            }
            if ui.button("Cancel").clicked() {
                app.name_confirm = None;
            }
        });
    });
    if !app.error.is_empty() {
        ui.label(RichText::new(&app.error).color(theme::red()));
    }
}

/// The names this wallet holds, rents or watches, and what it can do with them.
fn yours(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    theme::heading(ui, "Your names");
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        if app.names.is_empty() {
            theme::hint(ui, "None yet. Find a name above to register it, or buy or rent one on the market.");
        }
        let tip = app.sync.as_ref().map_or(0, |s| s.height);
        for n in app.names.clone() {
            ui.separator();
            ui.horizontal(|ui| {
                ui.label(RichText::new(&n.name).strong());
                theme::badge(ui, &n.tier, theme::muted());
                let gold = matches!(n.status.as_str(), "registered" | "rented");
                theme::badge(ui, &n.status, if gold { theme::gold() } else { theme::muted() });
                if let Some(p) = &n.pays {
                    ui.label(RichText::new(format!("pays {}", theme::grouped_hash(p, 24))).monospace().size(11.0));
                    theme::hint(ui, &n.pays_kind);
                }
                if n.sealed {
                    theme::badge(ui, "messages", theme::muted());
                }
            });
            if n.expires > tip {
                let renter = if n.renter { "You rent it. " } else { "" };
                theme::hint(ui, &format!("{renter}Held for {} more. {}", days(n.expires - tip), n.deal));
            }
            for r in n.records.iter().filter(|r| r.kind != "script") {
                ui.label(RichText::new(format!("{:<12}{}", r.kind, r.value)).monospace().size(10.5).color(theme::muted()));
            }
            if n.status == "registered" {
                backing_row(app, ui, &n.name, n.backing.as_ref());
            }
            if !n.note.is_empty() {
                theme::hint(ui, &n.note);
            }
            let name = short(&n.name);
            if n.status == "watching" {
                if ui.small_button("Stop watching").clicked() {
                    app.go(ctx, Action::UnwatchName { name });
                }
                continue;
            }
            if n.status != "registered" || n.renter {
                continue;
            }
            ui.horizontal(|ui| {
                let mut on = n.auto_renew;
                if ui.checkbox(&mut on, "Renew automatically").changed() {
                    app.go(ctx, Action::NameAct { name: name.clone(), act: NameAction::AutoRenew(on) });
                }
                if ui.add_enabled(!app.busy, egui::Button::new("Renew a year")).clicked() {
                    app.busy = true;
                    app.go(ctx, Action::NameAct { name: name.clone(), act: NameAction::Renew { years: 1 } });
                }
                if ui.small_button("Edit records").clicked() {
                    app.name_edit = Some((
                        name.clone(),
                        NameEdit {
                            mdw: n.mdw,
                            identity: n.identity,
                            evm: n.evm,
                            text: n.text.clone(),
                            pay_to: n.pay_to.clone(),
                            script: String::new(),
                            sealed: n.sealed,
                        },
                    ));
                    app.go(ctx, Action::LoadNameTargets);
                }
                if n.deal.is_empty() {
                    if ui.small_button("Sell…").clicked() {
                        app.name_offer = Some((name.clone(), false, String::new(), String::new()));
                    }
                    if ui.small_button("Rent out…").clicked() {
                        app.name_offer = Some((name.clone(), true, String::new(), "30".into()));
                    }
                    if ui.small_button("Give…").clicked() {
                        app.name_give = Some((name.clone(), String::new()));
                    }
                } else if !n.deal.starts_with("rented") && ui.small_button("Stop offering").clicked() {
                    app.busy = true;
                    app.go(ctx, Action::NameAct { name: name.clone(), act: NameAction::Unlist });
                }
            });
            offer_box(app, ui, ctx, &name);
            give_box(app, ui, ctx, &name);
            edit_box(app, ui, ctx, &name);
        }
    });
}

fn offer_box(app: &mut App, ui: &mut Ui, ctx: &egui::Context, name: &str) {
    let Some((_, rent, price, days_text)) = app.name_offer.as_mut().filter(|(o, ..)| o == name) else { return };
    let rent = *rent;
    ui.horizontal(|ui| {
        ui.label(if rent { "Rent for" } else { "Sell for" });
        ui.add(TextEdit::singleline(price).hint_text("gMDS").desired_width(80.0));
        ui.label("gMDS");
        if rent {
            ui.label("per");
            ui.add(TextEdit::singleline(days_text).desired_width(40.0));
            ui.label("days");
        }
    });
    let price = parse_gmds(price);
    let days: Option<u64> = if rent { days_text.trim().parse().ok().filter(|d| *d > 0) } else { Some(0) };
    let note = if rent {
        "Whoever rents it gets its payments until the rental ends; then the wallet takes it back. You cannot change it meanwhile."
    } else {
        "Whoever pays first gets it, and the payment comes to a fresh address of this wallet's."
    };
    theme::hint(ui, note);
    ui.horizontal(|ui| {
        if ui.add_enabled(price.is_some() && days.is_some() && !app.busy, egui::Button::new("Offer it")).clicked() {
            let (price, days) = (price.unwrap(), days.unwrap());
            let act = if rent { NameAction::RentOut { price, days } } else { NameAction::Sell { price } };
            app.busy = true;
            app.go(ctx, Action::NameAct { name: name.to_string(), act });
        }
        if ui.button("Cancel").clicked() {
            app.name_offer = None;
        }
    });
}

fn give_box(app: &mut App, ui: &mut Ui, ctx: &egui::Context, name: &str) {
    let Some((_, to)) = app.name_give.as_mut().filter(|(g, _)| g == name) else { return };
    ui.horizontal(|ui| {
        ui.label("Give to");
        ui.add(TextEdit::singleline(to).hint_text("their name.mds, or a control key").desired_width(260.0));
    });
    theme::hint(ui, "Their wallet finds it and sets it up. This cannot be undone.");
    let to = to.trim().to_string();
    ui.horizontal(|ui| {
        if ui.add_enabled(!to.is_empty() && !app.busy, egui::Button::new("Give it")).clicked() {
            app.busy = true;
            app.go(ctx, Action::NameAct { name: name.to_string(), act: NameAction::Give { to } });
        }
        if ui.button("Cancel").clicked() {
            app.name_give = None;
        }
    });
}

fn edit_box(app: &mut App, ui: &mut Ui, ctx: &egui::Context, name: &str) {
    let targets = app.name_targets.clone();
    let Some((_, e)) = app.name_edit.as_mut().filter(|(e, _)| e == name) else { return };
    // Where MDS payments go: this wallet's own key, or a shared address.
    let mut shared = !e.pay_to.trim().is_empty();
    ui.horizontal(|ui| {
        ui.label("Payments go to");
        if ui.radio(!shared, "this wallet").clicked() {
            shared = false;
            e.pay_to.clear();
            e.script.clear();
        }
        if ui.radio(shared, "a vault, multisig or plan").clicked() && !shared {
            shared = true;
            e.pay_to = " ".into();
        }
    });
    if shared {
        ui.add(TextEdit::singleline(&mut e.pay_to).hint_text("its midstate address").desired_width(f32::INFINITY));
        ui.add(
            TextEdit::singleline(&mut e.script)
                .hint_text("its script, hex (optional: published so everyone sees what kind of address it is)")
                .desired_width(f32::INFINITY),
        );
        if !targets.is_empty() {
            ui.horizontal_wrapped(|ui| {
                theme::hint(ui, "or one of this wallet's:");
                for t in &targets {
                    if ui.small_button(&t.label).on_hover_text(&t.kind).clicked() {
                        e.pay_to = t.address.clone();
                        e.script = t.script.clone();
                    }
                }
            });
        }
        if !e.pay_to.trim().is_empty() {
            match midstate_walletd::describe_target(&e.pay_to, &e.script) {
                Ok(kind) => theme::hint(ui, &format!("Payments to {name}.mds go to this {kind}. The identity handshake is skipped for a shared address.")),
                Err(err) => {
                    ui.label(RichText::new(err).color(theme::red()).size(12.0));
                }
            }
        }
    }
    ui.checkbox(&mut e.mdw, "Take MDW (this wallet's midwimble address)");
    ui.checkbox(&mut e.identity, "Identity key (payers ask it for a fresh address each time; invoices; channels)");
    ui.checkbox(&mut e.sealed, "Take sealed messages (publishes this wallet's ML-KEM key hash)");
    ui.checkbox(&mut e.evm, "Base account (this wallet's)");
    ui.add(TextEdit::singleline(&mut e.text).hint_text("a line of text, e.g. a website").desired_width(f32::INFINITY));
    ui.horizontal(|ui| {
        if ui.add_enabled(!app.busy, egui::Button::new("Save")).clicked() {
            app.busy = true;
            app.error.clear();
            let mut edit = app.name_edit.as_ref().map(|(_, e)| e.clone()).unwrap_or_default();
            edit.pay_to = edit.pay_to.trim().to_string();
            app.go(ctx, Action::NameAct { name: name.to_string(), act: NameAction::Edit(edit) });
        }
        if ui.button("Cancel").clicked() {
            app.name_edit = None;
        }
    });
}

/// Every offer, searchable and sorted; taking one opens it in Find.
fn market(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    theme::heading(ui, "Market");
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.add(TextEdit::singleline(&mut app.market_query).hint_text("search").desired_width(160.0));
            theme::segmented(ui, &["All", "For sale", "For rent"], &mut app.market_kind);
            theme::segmented(ui, &["Cheapest", "Dearest", "Ending soonest"], &mut app.market_sort);
        });
        let q = app.market_query.trim().to_ascii_lowercase();
        let mut list: Vec<_> = app
            .name_market
            .iter()
            .filter(|m| q.is_empty() || m.name.contains(&q))
            .filter(|m| match app.market_kind {
                1 => m.kind == "sale",
                2 => m.kind == "rent",
                _ => true,
            })
            .cloned()
            .collect();
        match app.market_sort {
            1 => list.sort_by_key(|m| std::cmp::Reverse(m.price)),
            2 => list.sort_by_key(|m| m.expires),
            _ => list.sort_by_key(|m| m.price),
        }
        if list.is_empty() {
            theme::hint(ui, if app.name_market.is_empty() { "No names are offered right now." } else { "Nothing matches." });
        }
        let tip = app.sync.as_ref().map_or(0, |s| s.height);
        for m in list {
            ui.horizontal(|ui| {
                ui.label(RichText::new(&m.name).strong());
                theme::badge(ui, &m.tier, theme::muted());
                // Backing goes with the name to its next holder.
                if let Some(t) = midstate_walletd::burns::tier(m.backing) {
                    theme::badge(ui, t, theme::gold());
                }
                let terms = if m.kind == "rent" {
                    format!("{} for {}", gmds(m.price), days(m.span))
                } else {
                    format!("{} to buy", gmds(m.price))
                };
                ui.label(terms);
                theme::hint(ui, &format!("held {} more", days(m.expires.saturating_sub(tip))));
                if m.mine {
                    theme::hint(ui, "your offer");
                } else if ui.button(if m.kind == "rent" { "Rent…" } else { "Buy…" }).clicked() {
                    app.name_input = short(&m.name);
                    app.name_confirm = None;
                    app.go(ctx, Action::FindName { name: short(&m.name), years: app.name_years });
                }
            });
        }
    });
}

fn recent(app: &mut App, ui: &mut Ui) {
    if app.name_deals.is_empty() {
        return;
    }
    theme::heading(ui, "Recent sales and rentals");
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        for d in &app.name_deals {
            let what = if d.kind == "rent" { "rented for" } else { "sold for" };
            ui.label(RichText::new(format!("{} {what} {} at block {}", d.name, gmds(d.price), d.height)).size(12.0));
        }
    });
}
