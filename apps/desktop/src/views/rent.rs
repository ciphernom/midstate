//! Bonds for rent (walletd `rental`): renting a mining bond for this
//! wallet's miner, and offering this wallet's coins as bonds for others.
//! Neither side waits on the other: an offer is a coin on chain that anyone
//! can take by paying its rent, and that only its owner can ever spend.

use super::bonds::{choice, fact, primary, section};
use crate::app::App;
use crate::bridge::Action;
use crate::theme::{self, compact_units, short_hex, UNIT_NAMES};
use eframe::egui::{self, RichText, TextEdit, Ui};
use midstate_walletd::api::RentalsView;
use midstate_walletd::bond::midwimble;

/// Units the rent is typed in: kMDS, mMDS, gMDS.
const RENT_UNITS: [usize; 3] = [1, 2, 3];
/// Mining times an offer can give, in days, besides the longest.
const DAYS: [u64; 4] = [7, 30, 90, 180];

pub fn show(app: &mut App, ui: &mut Ui, ctx: &egui::Context, height: u64, blocked: bool) {
    let Some(r) = app.rentals.clone() else {
        ui.horizontal(|ui| {
            ui.spinner();
            theme::hint(ui, "Loading bonds for rent…");
        });
        return;
    };
    market(app, ui, ctx, &r, blocked);
    if !r.rentals.is_empty() {
        ui.add_space(10.0);
        my_rentals(ui, &r);
    }
    ui.add_space(10.0);
    offer_form(app, ui, ctx, &r, height, blocked);
    if !r.offers.is_empty() {
        ui.add_space(10.0);
        my_offers(app, ui, ctx, &r, blocked);
    }
}

fn title(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).font(theme::font_medium(15.0)));
}

// ── Renting ─────────────────────────────────────────────────────────────

fn market(app: &mut App, ui: &mut Ui, ctx: &egui::Context, r: &RentalsView, blocked: bool) {
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        title(ui, "Rent a bond");
        theme::hint(ui, "Mine MDW without locking coins of your own. Pay once; the bond mines for this wallet.");
        if r.mining_key.is_empty() {
            theme::hint(ui, "Renting needs a wallet made with a recovery phrase.");
        }
        if r.scanned < r.window {
            ui.horizontal(|ui| {
                ui.spinner();
                theme::hint(ui, &format!("Reading offers off the chain: {}%", r.scanned * 100 / r.window.max(1)));
            });
        }
        if r.listings.is_empty() {
            ui.add_space(4.0);
            theme::hint(ui, "No bonds for rent right now.");
            return;
        }
        let can = !app.busy && !blocked && !r.mining_key.is_empty();
        for l in &r.listings {
            ui.add_space(4.0);
            ui.separator();
            let picked = app.rent_pick.as_deref() == Some(l.id.as_str());
            ui.horizontal(|ui| {
                ui.label(RichText::new(compact_units(l.bond)).font(theme::font_mono_medium(14.0)).color(theme::ink()));
                ui.label(theme::mono(format!("{} days of mining", l.mining_days)).size(12.0));
                ui.label(theme::mono(format!("rent {}", compact_units(l.rent))).size(12.0));
                ui.label(
                    theme::mono(format!("{} a day", compact_units(l.rent_per_day)))
                        .size(11.0)
                        .color(theme::muted()),
                );
                if l.yours {
                    theme::badge(ui, "yours", theme::muted());
                }
                if l.takeable && !picked {
                    theme::right_aligned(ui, |ui| {
                        if ui.add_enabled(can, egui::Button::new("Rent")).clicked() {
                            app.rent_pick = Some(l.id.clone());
                            app.rental_notice.clear();
                        }
                    });
                }
            });
            if !l.takeable && !l.yours {
                theme::hint(ui, &format!("Can't rent: {}.", l.why_not));
            }
            if picked && l.takeable {
                ui.label(
                    RichText::new(format!(
                        "Pays {} once, plus a small fee. Mines for ~{} days, from about a day after it confirms.",
                        compact_units(l.rent),
                        l.mining_days
                    ))
                    .size(12.5)
                    .color(theme::ink()),
                );
                theme::right_aligned(ui, |ui| {
                    if ui.add_enabled(can, primary("Confirm and pay")).clicked() {
                        app.busy = true;
                        app.error.clear();
                        app.rental_notice.clear();
                        app.go(ctx, Action::TakeOffer { id: l.id.clone() });
                    }
                    if ui.button("Back").clicked() {
                        app.rent_pick = None;
                    }
                });
            }
        }
    });
}

fn my_rentals(ui: &mut Ui, r: &RentalsView) {
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        title(ui, "Bonds you rent");
        for b in &r.rentals {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new(compact_units(b.bond)).font(theme::font_mono_medium(14.0)).color(theme::ink()));
                theme::badge(ui, &b.status, if b.status == "lost" { theme::muted() } else { theme::ink() });
                ui.label(theme::mono(format!("rent {}", compact_units(b.rent))).size(12.0));
                if b.status != "lost" {
                    ui.label(theme::mono(format!("{} days of mining left", b.mining_days)).size(12.0));
                }
            });
            theme::hint(ui, &b.note);
            if b.status == "rented" && !b.reg_note.is_empty() {
                theme::hint(ui, &b.reg_note);
            }
        }
    });
}

// ── Offering ────────────────────────────────────────────────────────────

fn offer_form(app: &mut App, ui: &mut Ui, ctx: &egui::Context, r: &RentalsView, height: u64, blocked: bool) {
    let now = theme::now_secs();
    let spendable = app.balance.as_ref().map(|b| b.confirmed).unwrap_or(0);
    let mut days: Vec<u64> = DAYS.iter().copied().filter(|d| (r.min_days..r.max_days).contains(d)).collect();
    if r.max_days >= r.min_days && r.max_days > 0 {
        days.push(r.max_days);
    }
    let chosen: Option<u64> = app.rent_days.parse().ok().filter(|d| days.contains(d));
    if chosen.is_none() {
        if let Some(d) = days.iter().copied().find(|d| *d >= 30).or(days.last().copied()) {
            app.rent_days = d.to_string();
        }
    }
    let chosen: Option<u64> = app.rent_days.parse().ok().filter(|d| days.contains(d));

    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        title(ui, "Rent out a bond");
        theme::hint(
            ui,
            &format!(
                "Earn rent on {}. The renter mines with it; only you can ever spend the coin.",
                compact_units(r.bond)
            ),
        );
        ui.add_space(6.0);

        section(ui, "Mining time");
        let n = days.len().max(1);
        let w = ((ui.available_width() - 10.0 * (n - 1) as f32) / n as f32).min(180.0);
        ui.horizontal(|ui| {
            for &d in &days {
                let until = midwimble::until_for_days(now, d);
                let back = now + midstate_walletd::rental::locked_days(until, height, now) * 86_400;
                let caption = format!("back ~{}", theme::fmt_date(back));
                if choice(ui, ("rent_days", d), w, chosen == Some(d), true, &super::bonds::months_label(d), &caption) {
                    app.rent_days = d.to_string();
                    app.rent_confirm = false;
                }
            }
        });

        ui.add_space(10.0);
        section(ui, "Rent, for the whole time");
        ui.horizontal(|ui| {
            let e = ui.add(
                TextEdit::singleline(&mut app.rent_price)
                    .hint_text("amount")
                    .font(egui::TextStyle::Monospace)
                    .desired_width(160.0),
            );
            if e.changed() {
                app.rent_confirm = false;
            }
            let labels: Vec<&str> = RENT_UNITS.iter().map(|u| UNIT_NAMES[*u]).collect();
            let mut pick = RENT_UNITS.iter().position(|u| *u == app.rent_unit).unwrap_or(2);
            if theme::segmented(ui, &labels, &mut pick) {
                app.rent_unit = RENT_UNITS[pick];
                app.rent_confirm = false;
            }
        });

        let rent = theme::parse_in_unit(&app.rent_price, app.rent_unit).ok().filter(|v| *v > 0);
        let until = chosen.map(|d| midwimble::until_for_days(now, d));
        let back = until.map(|u| now + midstate_walletd::rental::locked_days(u, height, now) * 86_400);
        let affordable = spendable > r.bond;
        let ready = chosen.is_some() && rent.is_some() && affordable && !app.busy && !blocked;

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            fact(ui, "Locks", &compact_units(r.bond));
            fact(ui, "Back", &back.map_or("—".into(), |b| format!("~{}", theme::fmt_date(b))));
            let per_day = match (rent, chosen) {
                (Some(v), Some(d)) => compact_units(v / d.max(1)),
                _ => "—".into(),
            };
            fact(ui, "A day", &per_day);
            theme::right_aligned(ui, |ui| {
                if app.rent_confirm {
                    if ui.add_enabled(ready, primary("Confirm")).clicked() {
                        app.busy = true;
                        app.error.clear();
                        app.rental_notice.clear();
                        app.go(ctx, Action::PostOffer { days: chosen.unwrap_or(0), rent: rent.unwrap_or(0) });
                    }
                    if ui.button("Cancel").clicked() {
                        app.rent_confirm = false;
                    }
                } else if ui.add_enabled(ready, primary("Offer for rent")).clicked() {
                    app.rent_confirm = true;
                }
            });
        });
        if app.rent_confirm {
            if let Some(b) = back {
                ui.label(
                    RichText::new(format!(
                        "Locks {} until ~{}. It comes back to you then, rented or not.",
                        compact_units(r.bond),
                        theme::fmt_date(b)
                    ))
                    .size(12.5)
                    .color(theme::ink()),
                );
            }
        } else if !affordable {
            theme::hint(ui, &format!("You have {}: not enough to offer a bond.", compact_units(spendable)));
        }
    });
}

fn my_offers(app: &mut App, ui: &mut Ui, ctx: &egui::Context, r: &RentalsView, blocked: bool) {
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        title(ui, "Your offers");
        for o in &r.offers {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new(compact_units(o.bond)).font(theme::font_mono_medium(14.0)).color(theme::ink()));
                let quiet = matches!(o.status.as_str(), "cancelled" | "failed");
                theme::badge(ui, &o.status, if quiet { theme::muted() } else { theme::ink() });
                ui.label(theme::mono(format!("rent {}", compact_units(o.rent))).size(12.0));
                if o.status == "listed" {
                    ui.label(theme::mono(format!("{} days of mining", o.mining_days)).size(12.0));
                }
                if !o.taken_by.is_empty() {
                    ui.label(theme::mono(format!("to {}", short_hex(&o.taken_by, 6))).size(11.0).color(theme::muted()));
                }
                if o.can_cancel {
                    theme::right_aligned(ui, |ui| {
                        if ui.add_enabled(!app.busy && !blocked, egui::Button::new("Cancel offer")).clicked() {
                            app.busy = true;
                            app.error.clear();
                            app.rental_notice.clear();
                            app.go(ctx, Action::CancelOffer { id: o.id.clone() });
                        }
                    });
                }
            });
            theme::hint(ui, &o.note);
        }
    });
}
