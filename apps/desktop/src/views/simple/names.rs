//! Names in the simple view: the ones this wallet has, and getting one.
//! Selling, renting, records and backing are in the full app.

use super::{big_button, button, card, field, gmds, label, quiet};
use crate::app::App;
use crate::bridge::Action;
use crate::theme;
use eframe::egui::{RichText, Ui};
use midstate_walletd::api::{NameAction, NameView};
use midstate_walletd::names::normalize;

/// Blocks in a day, at one a minute.
const DAY: u64 = 1_440;

/// What a name of this wallet's is doing, in a few words.
pub fn state(n: &NameView, tip: u64) -> String {
    match n.status.as_str() {
        "registering" => "Being registered…".into(),
        "updating" => "Updating…".into(),
        _ if n.renter => format!("Rented · {} left", days(n.expires.saturating_sub(tip))),
        // The checkbox beside it says so.
        _ if n.auto_renew => String::new(),
        _ => format!("Ends in {}", days(n.expires.saturating_sub(tip))),
    }
}

fn days(blocks: u64) -> String {
    match blocks / DAY {
        0 => "under a day".into(),
        1 => "1 day".into(),
        d => format!("{d} days"),
    }
}

/// Names to list: held, rented or on their way, not ones only watched or
/// already gone.
pub fn listed(names: &[NameView]) -> Vec<NameView> {
    names
        .iter()
        .filter(|n| matches!(n.status.as_str(), "registering" | "registered" | "updating" | "rented"))
        .cloned()
        .collect()
}

pub fn show(app: &mut App, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    super::title(ui, "Names");
    let tip = app.sync.as_ref().map_or(0, |s| s.height);

    let mine = listed(&app.names);
    if !mine.is_empty() {
        card(ui, |ui| {
            label(ui, if mine.len() == 1 { "Your name" } else { "Your names" });
            for (i, n) in mine.iter().enumerate() {
                if i > 0 {
                    ui.separator();
                }
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label(RichText::new(&n.name).font(theme::font_mono_medium(18.0)).color(theme::ink()));
                        let st = state(n, tip);
                        if !st.is_empty() {
                            quiet(ui, &st);
                        }
                    });
                    theme::right_aligned(ui, |ui| {
                        let now = ui.input(|i| i.time);
                        let copied = app.name_copied_at.is_some_and(|t| now - t < 1.5) && app.name_input_copied == n.name;
                        if button(ui, if copied { "Copied" } else { "Copy" }, true).clicked() {
                            ui.ctx().copy_text(n.name.clone());
                            app.name_copied_at = Some(now);
                            app.name_input_copied = n.name.clone();
                        }
                        if n.status == "registered" && !n.renter {
                            let mut on = n.auto_renew;
                            if ui.checkbox(&mut on, "Renew automatically").changed() {
                                app.go(&ctx, Action::NameAct { name: n.name.trim_end_matches(".mds").to_string(), act: NameAction::AutoRenew(on) });
                            }
                        }
                    });
                });
            }
        });
        ui.add_space(12.0);
    }

    card(ui, |ui| {
        label(ui, "Get a name");
        quiet(ui, "People can pay and message a name instead of a long address.");
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            let r = field(ui, &mut app.name_input, "yourname", 260.0);
            ui.label(RichText::new(".mds").font(theme::font_medium(16.0)).color(theme::muted()));
            let typed = app.name_input.trim().to_string();
            if r.changed() && normalize(&typed).is_ok() {
                app.name_confirm = None;
                app.error.clear();
                app.go(&ctx, Action::FindName { name: typed, years: 1 });
            }
        });
        let typed = normalize(&app.name_input).ok();
        if let (Err(e), false) = (normalize(&app.name_input), app.name_input.trim().is_empty()) {
            ui.add_space(4.0);
            quiet(ui, &e.to_string());
        }
        let Some(c) = app.name_card.clone().filter(|c| Some(c.name.trim_end_matches(".mds").to_string()) == typed) else { return };
        ui.add_space(10.0);
        let (line, act, spend): (String, &'static str, u64) = match c.state.as_str() {
            "available" | "unclaimed" | "lapsed" => (format!("{} is free · {} a year", c.name, gmds(c.rate)), "register", c.burn),
            "for sale" => (format!("{} is for sale · {}", c.name, gmds(c.price)), "buy", c.price),
            "yours" => (format!("{} is yours", c.name), "", 0),
            _ => (format!("{} is taken", c.name), "", 0),
        };
        ui.label(RichText::new(line).size(16.0).color(theme::ink()));
        if act.is_empty() {
            return;
        }
        ui.add_space(10.0);
        match app.name_confirm {
            None => {
                let text = if act == "buy" { "Buy it" } else { "Get it for a year" };
                if big_button(ui, text, true, !app.busy).clicked() {
                    app.name_confirm = Some(act);
                }
            }
            Some(_) => {
                let what = if act == "buy" { format!("Buy {} for {}?", c.name, gmds(spend)) } else { format!("Get {} for a year for {}?", c.name, gmds(spend)) };
                ui.label(RichText::new(what).size(15.0).color(theme::ink()));
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 10.0;
                    if big_button(ui, if app.busy { "Working…" } else { "Confirm" }, true, !app.busy).clicked() {
                        app.busy = true;
                        app.error.clear();
                        let name = c.name.trim_end_matches(".mds").to_string();
                        app.go(&ctx, if act == "buy" { Action::TakeName { name, rent: false } } else { Action::RegisterName { name, years: 1 } });
                    }
                    if big_button(ui, "Cancel", false, !app.busy).clicked() {
                        app.name_confirm = None;
                    }
                });
            }
        }
        if !app.error.is_empty() {
            ui.add_space(6.0);
            quiet(ui, &app.error.clone());
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(status: &str, auto_renew: bool, renter: bool, expires: u64) -> NameView {
        NameView {
            name: "nick.mds".into(),
            status: status.into(),
            note: String::new(),
            pays: None,
            expires,
            tier: "silver".into(),
            deal: String::new(),
            renter,
            auto_renew,
            records: vec![],
            pays_kind: String::new(),
            sealed: false,
            backing: None,
            mdw: false,
            identity: false,
            evm: false,
            text: String::new(),
            pay_to: String::new(),
        }
    }

    #[test]
    fn names_in_plain_words() {
        let tip = 100_000;
        assert_eq!(state(&n("registered", true, false, tip + 400 * DAY), tip), "", "its checkbox says it renews");
        assert_eq!(state(&n("registered", false, false, tip + 40 * DAY), tip), "Ends in 40 days");
        assert_eq!(state(&n("registered", false, false, tip + 10), tip), "Ends in under a day");
        assert_eq!(state(&n("rented", false, true, tip + DAY), tip), "Rented · 1 day left");
        assert_eq!(state(&n("registering", false, false, 0), tip), "Being registered…");
        let all = vec![n("registered", true, false, 1), n("watching", false, false, 1), n("lost", false, false, 1), n("registering", false, false, 1)];
        assert_eq!(listed(&all).len(), 2, "watched and lost names are not this wallet's");
    }
}
