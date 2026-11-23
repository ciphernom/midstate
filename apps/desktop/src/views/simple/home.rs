//! Home: what this wallet holds, the two things people come to do, and
//! anything under way. Nothing else.

use super::{active_sends, big_button, card, gmds, quiet, send_card};
use crate::app::{App, Tab};
use crate::theme;
use eframe::egui::{RichText, Ui};

pub fn show(app: &mut App, ui: &mut Ui) {
    ui.add_space(18.0);
    let b = app.balance.clone();
    ui.label(RichText::new("Balance").size(13.0).color(theme::muted()));
    ui.horizontal(|ui| {
        let v = b.as_ref().map(|b| super::amount(b.confirmed)).unwrap_or_else(|| "—".into());
        ui.label(RichText::new(v).font(theme::font_mono_medium(46.0)).color(theme::ink()));
        ui.label(RichText::new(super::UNIT).font(theme::font_medium(18.0)).color(theme::muted()));
    });
    if let Some(b) = &b {
        // Money that is this wallet's but cannot be spent this minute.
        let coming = b.unconfirmed + b.immature;
        if coming > 0 {
            quiet(ui, &format!("+ {} on its way", gmds(coming)));
        }
    }
    if super::has_mdw(app) {
        if let Some(a) = &app.mwb_account {
            quiet(ui, &format!("and {}", crate::views::mdw::mdw(a.spendable)));
        }
    }

    ui.add_space(20.0);
    // With nothing to send, receiving is the thing to do.
    let empty = b.as_ref().is_some_and(|b| b.confirmed == 0 && b.unconfirmed == 0 && b.immature == 0);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 10.0;
        if empty {
            if big_button(ui, "Receive", true, true).clicked() {
                app.tab = Tab::Receive;
                app.error.clear();
            }
            if big_button(ui, "Send", false, true).clicked() {
                app.tab = Tab::Send;
                app.error.clear();
            }
        } else {
            if big_button(ui, "Send", true, true).clicked() {
                app.tab = Tab::Send;
                app.error.clear();
            }
            if big_button(ui, "Receive", false, true).clicked() {
                app.tab = Tab::Receive;
                app.error.clear();
            }
        }
    });

    // Under way: catching up, restoring, sending.
    let syncing = app.sync.as_ref().is_some_and(|s| s.is_syncing);
    let restoring = app.sync.as_ref().is_some_and(|s| s.restore_pass > 0);
    let sends = active_sends(app);
    if syncing || restoring || !sends.is_empty() {
        ui.add_space(22.0);
    }
    if syncing {
        card(ui, |ui| {
            let frac = app.sync_fraction().unwrap_or(0.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new("Catching up with the network").size(15.0).color(theme::ink()));
                theme::right_aligned(ui, |ui| {
                    let left = app.sync_eta_secs().map(|s| format!("{} left · ", theme::fmt_duration(s))).unwrap_or_default();
                    ui.label(RichText::new(format!("{left}{}%", (frac * 100.0).floor())).size(13.0).color(theme::muted()));
                });
            });
            ui.add_space(8.0);
            theme::progress_bar(ui, frac);
            ui.add_space(8.0);
            quiet(ui, "You can receive now. Sending starts when this finishes.");
        });
        ui.add_space(8.0);
    } else if restoring {
        card(ui, |ui| {
            let s = app.sync.clone().unwrap_or_default();
            let frac = if s.height > 0 { s.restore_scan_pos as f32 / s.height as f32 } else { 0.0 };
            ui.horizontal(|ui| {
                ui.label(RichText::new("Finding your coins").size(15.0).color(theme::ink()));
                theme::right_aligned(ui, |ui| {
                    ui.label(RichText::new(format!("{}%", (frac * 100.0).floor())).size(13.0).color(theme::muted()));
                });
            });
            ui.add_space(8.0);
            theme::progress_bar(ui, frac);
        });
        ui.add_space(8.0);
    }
    for s in &sends {
        send_card(app, ui, s);
    }

    // What happened lately.
    ui.add_space(22.0);
    let rows = super::activity::rows(app);
    ui.horizontal(|ui| {
        ui.label(RichText::new("Recent activity").font(theme::font_medium(15.0)).color(theme::ink()));
        if !rows.is_empty() {
            theme::right_aligned(ui, |ui| {
                if ui.link(RichText::new("See all").size(13.0)).clicked() {
                    app.tab = Tab::History;
                }
            });
        }
    });
    ui.add_space(6.0);
    if rows.is_empty() {
        card(ui, |ui| {
            ui.add_space(6.0);
            ui.label(RichText::new("Nothing here yet").size(15.0).color(theme::ink()));
            quiet(ui, "Money you receive will show up here.");
            ui.add_space(6.0);
        });
    } else {
        card(ui, |ui| {
            super::activity::list(ui, &rows[..rows.len().min(5)], false);
        });
    }
}
