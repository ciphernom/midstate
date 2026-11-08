//! Bonds for rent (walletd `rental`): renting a mining bond for this
//! wallet's miner, and offering this wallet's coins as bonds for others.
//! Neither side waits on the other: an offer is a coin on chain that anyone
//! can take by paying its rent, and that only its owner can ever spend.

use crate::app::App;
use crate::bridge::Action;
use crate::theme::{self, compact_units, short_hex, UNIT_NAMES};
use eframe::egui::{self, RichText, TextEdit, Ui};
use midstate_walletd::api::RentalsView;
use midstate_walletd::bond::midwimble;

/// Units the rent is typed in: kMDS, mMDS, gMDS.
const RENT_UNITS: [usize; 3] = [1, 2, 3];

pub fn show(app: &mut App, ui: &mut Ui, ctx: &egui::Context, height: u64, blocked: bool) {
    let Some(r) = app.rentals.clone() else {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.spinner();
                theme::hint(ui, "Loading bonds for rent…");
            });
        });
        return;
    };
    market(app, ui, ctx, &r, blocked);
    ui.add_space(10.0);
    offer_form(app, ui, ctx, &r, height, blocked);
    if !r.offers.is_empty() {
        ui.add_space(10.0);
        my_offers(app, ui, ctx, &r, blocked);
    }
    if !r.rentals.is_empty() {
        ui.add_space(10.0);
        my_rentals(ui, &r);
    }
    if !app.rental_notice.is_empty() {
        ui.add_space(6.0);
        ui.label(RichText::new(&app.rental_notice).size(12.0).color(theme::bright()));
    }
}

// ── Renting ─────────────────────────────────────────────────────────────

fn market(app: &mut App, ui: &mut Ui, ctx: &egui::Context, r: &RentalsView, blocked: bool) {
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Rent a bond").font(theme::font_medium(14.0)));
        theme::hint(
            ui,
            "Mine MDW at full weight without locking coins of your own. You pay the rent once, in one send \
             of yours. The bond names this wallet's mining key, so your miner signs with it until its lock \
             runs down, and its owner gets the coin back then.",
        );
        if r.mining_key.is_empty() {
            theme::hint(
                ui,
                "This wallet predates recovery phrases, so it has no mining key to rent a bond for. Make a \
                 wallet with a phrase to rent one.",
            );
        }
        if r.scanned < r.window {
            ui.horizontal(|ui| {
                ui.spinner();
                theme::hint(
                    ui,
                    &format!("Reading offers off the chain: {}% of the last ten weeks.", r.scanned * 100 / r.window.max(1)),
                );
            });
        }
        if r.listings.is_empty() {
            ui.add_space(4.0);
            theme::hint(ui, "No bonds for rent right now.");
            return;
        }
        for l in &r.listings {
            ui.add_space(6.0);
            ui.separator();
            ui.horizontal(|ui| {
                ui.label(theme::mono(compact_units(l.bond)).color(theme::ink()));
                theme::hint(ui, "·");
                ui.label(theme::mono(format!("{} days of mining", l.mining_days)).size(12.0));
                theme::hint(ui, "·");
                ui.label(theme::mono(format!("rent {}", compact_units(l.rent))).size(12.0));
                ui.label(
                    theme::mono(format!("{} a day", compact_units(l.rent_per_day)))
                        .size(11.0)
                        .color(theme::muted()),
                );
                if l.yours {
                    theme::badge(ui, "yours", theme::muted());
                }
            });
            let picked = app.rent_pick.as_deref() == Some(l.id.as_str());
            if !l.takeable {
                if !l.yours {
                    theme::hint(ui, &format!("Can't rent: {}.", l.why_not));
                }
                continue;
            }
            if picked {
                theme::hint(
                    ui,
                    &format!(
                        "You pay {} and a small fee, once. The bond mines for this wallet for about {} days, \
                         starting about a day after the send confirms, while midwimble registers it. Keep \
                         mining on (Node tab) to use it.",
                        compact_units(l.rent),
                        l.mining_days
                    ),
                );
            }
            theme::right_aligned(ui, |ui| {
                let can = !app.busy && !blocked && !r.mining_key.is_empty();
                if picked {
                    if ui.add_enabled(can, egui::Button::new("Confirm and pay")).clicked() {
                        app.busy = true;
                        app.error.clear();
                        app.rental_notice.clear();
                        app.go(ctx, Action::TakeOffer { id: l.id.clone() });
                    }
                    if ui.small_button("Back").clicked() {
                        app.rent_pick = None;
                    }
                } else if ui.add_enabled(can, egui::Button::new("Rent")).clicked() {
                    app.rent_pick = Some(l.id.clone());
                    app.rental_notice.clear();
                }
            });
        }
    });
}

fn my_rentals(ui: &mut Ui, r: &RentalsView) {
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Bonds you rent").font(theme::font_medium(14.0)));
        for b in &r.rentals {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                theme::badge(ui, &b.status, if b.status == "lost" { theme::muted() } else { theme::ink() });
                ui.label(theme::mono(compact_units(b.bond)).color(theme::ink()));
                theme::hint(ui, "·");
                ui.label(theme::mono(format!("rent {}", compact_units(b.rent))).size(12.0));
                if b.status != "lost" {
                    theme::hint(ui, "·");
                    ui.label(theme::mono(format!("{} days of mining left", b.mining_days)).size(12.0));
                }
                ui.label(theme::mono(short_hex(&b.id, 6)).size(10.5).color(theme::faint()));
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
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Rent out a bond").font(theme::font_medium(14.0)));
        theme::hint(
            ui,
            &format!(
                "Earn rent on coins you hold. The wallet locks one coin of {} (the bond midwimble registers \
                 now) where anyone can rent it at your price. The renter mines with it; nobody but you can \
                 ever spend the coin, and it comes back to you when the lock ends. The rent arrives in this \
                 wallet when someone rents it, whether or not the wallet is open.",
                compact_units(r.bond)
            ),
        );
        ui.add_space(6.0);

        ui.horizontal(|ui| {
            ui.label(RichText::new("Days of mining").size(12.0).color(theme::muted()));
            let e = ui.add(
                TextEdit::singleline(&mut app.rent_days)
                    .hint_text("30")
                    .font(egui::TextStyle::Monospace)
                    .desired_width(60.0),
            );
            if e.changed() {
                app.rent_days.retain(|c| c.is_ascii_digit());
                app.rent_confirm = false;
            }
            theme::hint(ui, &format!("{} to {}", r.min_days, r.max_days));
        });
        ui.horizontal(|ui| {
            ui.label(RichText::new("Rent").size(12.0).color(theme::muted()));
            let e = ui.add(
                TextEdit::singleline(&mut app.rent_price)
                    .hint_text("for the whole rental")
                    .font(egui::TextStyle::Monospace)
                    .desired_width(140.0),
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

        let days: Option<u64> = app.rent_days.parse().ok().filter(|d| (r.min_days..=r.max_days).contains(d));
        let rent = theme::parse_in_unit(&app.rent_price, app.rent_unit).ok().filter(|v| *v > 0);
        if let Some(d) = days {
            let until = midwimble::until_for_days(now, d);
            let locked = midstate_walletd::rental::locked_days(until, height, now);
            let mut line = format!(
                "Your coin is locked for about {locked} days in all: the {d} days of mining, a day while it \
                 registers, and the five weeks of lock midwimble wants left on a bond it counts."
            );
            if let Some(v) = rent {
                line.push_str(&format!(" That is {} a day of mining.", compact_units(v / d)));
            }
            theme::hint(ui, &line);
        } else if !app.rent_days.is_empty() {
            theme::hint(ui, &format!("Choose {} to {} days.", r.min_days, r.max_days));
        }
        let spendable = app.balance.as_ref().map(|b| b.confirmed).unwrap_or(0);
        if spendable < r.bond {
            theme::hint(
                ui,
                &format!("You need {} spendable to offer a bond; you have {}.", compact_units(r.bond), compact_units(spendable)),
            );
        }

        ui.add_space(6.0);
        let ready = days.is_some() && rent.is_some() && spendable >= r.bond && !app.busy && !blocked;
        theme::right_aligned(ui, |ui| {
            let label = if app.rent_confirm {
                format!("Confirm: lock {}", compact_units(r.bond))
            } else {
                "Offer for rent".to_string()
            };
            if ui.add_enabled(ready, egui::Button::new(label)).clicked() {
                if app.rent_confirm {
                    app.busy = true;
                    app.error.clear();
                    app.rental_notice.clear();
                    app.go(ctx, Action::PostOffer { days: days.unwrap(), rent: rent.unwrap() });
                } else {
                    app.rent_confirm = true;
                }
            }
            if app.rent_confirm && ui.small_button("Back").clicked() {
                app.rent_confirm = false;
            }
        });
    });
}

fn my_offers(app: &mut App, ui: &mut Ui, ctx: &egui::Context, r: &RentalsView, blocked: bool) {
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Your offers").font(theme::font_medium(14.0)));
        for o in &r.offers {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                theme::badge(ui, &o.status, if matches!(o.status.as_str(), "cancelled" | "failed") { theme::muted() } else { theme::ink() });
                ui.label(theme::mono(compact_units(o.bond)).color(theme::ink()));
                theme::hint(ui, "·");
                ui.label(theme::mono(format!("rent {}", compact_units(o.rent))).size(12.0));
                if o.status == "listed" {
                    theme::hint(ui, "·");
                    ui.label(theme::mono(format!("{} days of mining", o.mining_days)).size(12.0));
                }
                if !o.taken_by.is_empty() {
                    theme::hint(ui, "·");
                    ui.label(theme::mono(format!("to {}", short_hex(&o.taken_by, 6))).size(11.0).color(theme::muted()));
                }
            });
            theme::hint(ui, &o.note);
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
        }
    });
}
