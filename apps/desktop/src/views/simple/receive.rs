//! Receive: an address to share — one that can be given out again and
//! again — and this wallet's name, if it has one.

use super::{big_button, button, card, label, quiet};
use crate::app::App;
use crate::bridge::Action;
use crate::theme;
use eframe::egui::{self, RichText, Ui};
use midstate_walletd::api::{AddressInfo, NameView};

/// Signatures a reusable address should still have before it is handed out.
/// Below this the simple view makes a new one rather than share one that is
/// about to run out.
const SPARE: u64 = 16;

/// The address to show: the one on screen if it is reusable and has room,
/// otherwise the reusable one with the most room.
pub fn pick(current: Option<&AddressInfo>, all: &[AddressInfo]) -> Option<AddressInfo> {
    let roomy = |a: &&AddressInfo| a.kind == "mss" && a.remaining_sigs.unwrap_or(0) >= SPARE;
    current
        .filter(roomy)
        .or_else(|| all.iter().filter(roomy).max_by_key(|a| a.remaining_sigs.unwrap_or(0)))
        .cloned()
}

/// Names that send their payments to this wallet.
pub fn own_names(names: &[NameView]) -> Vec<String> {
    names
        .iter()
        .filter(|n| matches!(n.status.as_str(), "registered" | "updating") && !n.renter && n.pay_to.is_empty())
        .map(|n| n.name.clone())
        .collect()
}

pub fn show(app: &mut App, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    super::title(ui, "Receive");
    if super::has_mdw(app) {
        if theme::segmented(ui, &[super::UNIT, "MDW"], &mut app.simple_asset) {
            app.copied_at = None;
        }
        ui.add_space(14.0);
    } else {
        app.simple_asset = 0;
    }

    let (addr, names) = if app.simple_asset == 1 {
        (app.mwb_account.as_ref().and_then(|a| a.address.clone()), Vec::new())
    } else {
        match pick(app.current_addr.as_ref(), &app.addresses) {
            Some(a) => {
                if app.current_addr.as_ref().map(|c| &c.address) != Some(&a.address) {
                    app.current_addr = Some(a.clone());
                }
                (Some(a.address), own_names(&app.names))
            }
            None => {
                // None yet, or all used up: make one, once per visit.
                if !app.simple_addr_asked && !app.busy {
                    app.simple_addr_asked = true;
                    app.busy = true;
                    app.error.clear();
                    app.go(&ctx, Action::NewAddress { mss: true, label: None });
                }
                (None, own_names(&app.names))
            }
        }
    };

    card(ui, |ui| {
        let Some(addr) = addr else {
            ui.add_space(30.0);
            ui.vertical_centered(|ui| {
                if app.busy {
                    ui.spinner();
                    ui.add_space(8.0);
                    quiet(ui, "Making your address…");
                } else if !app.error.is_empty() {
                    quiet(ui, "Your address could not be made.");
                    ui.add_space(8.0);
                    if button(ui, "Try again", true).clicked() {
                        app.simple_addr_asked = false;
                        app.error.clear();
                    }
                } else {
                    quiet(ui, "Getting your address…");
                }
            });
            ui.add_space(30.0);
            return;
        };
        crate::views::receive::ensure_qr(app, &ctx, &addr);
        ui.horizontal_top(|ui| {
            if let Some((_, tex)) = &app.qr {
                ui.image(egui::load::SizedTexture::new(tex.id(), egui::vec2(196.0, 196.0)));
            }
            ui.add_space(18.0);
            ui.vertical(|ui| {
                ui.add_space(4.0);
                label(ui, if app.simple_asset == 1 { "Your MDW address" } else { "Your address" });
                ui.add(egui::Label::new(RichText::new(theme::grouped_hash(&addr, 4)).monospace().size(13.5).color(theme::ink())).wrap());
                ui.add_space(12.0);
                let now = ui.input(|i| i.time);
                let copied = app.copied_at.is_some_and(|t| now - t < 1.5);
                if big_button(ui, if copied { "Copied" } else { "Copy address" }, true, true).clicked() {
                    ui.ctx().copy_text(addr.clone());
                    app.copied_at = Some(now);
                }
            });
        });
    });

    if let Some(name) = names.first() {
        ui.add_space(10.0);
        card(ui, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    label(ui, "Your name");
                    ui.label(RichText::new(name).font(theme::font_mono_medium(18.0)).color(theme::ink()));
                });
                theme::right_aligned(ui, |ui| {
                    let now = ui.input(|i| i.time);
                    let copied = app.name_copied_at.is_some_and(|t| now - t < 1.5) && app.name_input_copied == *name;
                    if button(ui, if copied { "Copied" } else { "Copy name" }, true).clicked() {
                        ui.ctx().copy_text(name.clone());
                        app.name_copied_at = Some(now);
                        app.name_input_copied = name.clone();
                    }
                });
            });
            quiet(ui, "People can pay your name instead of the address.");
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a(addr: &str, kind: &str, sigs: Option<u64>) -> AddressInfo {
        AddressInfo { address: addr.into(), kind: kind.into(), label: None, remaining_sigs: sigs, used: false }
    }

    #[test]
    fn only_reusable_addresses_with_room_are_shown() {
        let all = vec![a("w", "wots", None), a("low", "mss", Some(3)), a("m1", "mss", Some(900)), a("m2", "mss", Some(1000))];
        // The one on screen stays while it has room.
        assert_eq!(pick(Some(&all[2]), &all).unwrap().address, "m1");
        // A one-time address is never handed out here: its key may sign once.
        assert_eq!(pick(Some(&all[0]), &all).unwrap().address, "m2");
        // Nearly used up: the roomiest instead.
        assert_eq!(pick(Some(&all[1]), &all).unwrap().address, "m2");
        assert_eq!(pick(None, &all).unwrap().address, "m2");
        // Nothing reusable with room: a new one is needed.
        assert!(pick(None, &all[..2]).is_none());
    }
}
