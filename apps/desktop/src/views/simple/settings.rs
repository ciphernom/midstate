//! Settings in the simple view: updates, appearance, the recovery phrase
//! check, locking, and the way to every other feature.

use super::{button, card, label, quiet};
use crate::app::App;
use crate::bridge::Action;
use crate::theme;
use crate::update::Status;
use eframe::egui::{self, RichText, TextEdit, Ui};

pub fn show(app: &mut App, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    super::title(ui, "Settings");

    updates(app, ui);
    ui.add_space(12.0);

    card(ui, |ui| {
        label(ui, "Appearance");
        let mut mode = theme::mode();
        let before = mode;
        let mut i = match mode {
            theme::ThemeMode::System => 0,
            theme::ThemeMode::Light => 1,
            theme::ThemeMode::Dark => 2,
        };
        theme::segmented(ui, &["Automatic", "Light", "Dark"], &mut i);
        mode = [theme::ThemeMode::System, theme::ThemeMode::Light, theme::ThemeMode::Dark][i];
        if mode != before {
            theme::set_mode(mode);
            theme::save_pref();
        }
    });
    ui.add_space(12.0);

    card(ui, |ui| {
        label(ui, "Recovery phrase");
        quiet(ui, "Type your 24 words to check your written copy is right. Nothing is saved.");
        ui.add_space(6.0);
        ui.add(
            TextEdit::multiline(&mut app.verify_input)
                .desired_rows(2)
                .desired_width(f32::INFINITY)
                .font(egui::TextStyle::Monospace),
        );
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            let words = app.verify_input.split_whitespace().count();
            if button(ui, "Check", !app.busy && words == 24).clicked() {
                app.busy = true;
                app.error.clear();
                app.verify_result = None;
                app.go(&ctx, Action::VerifyPhrase { phrase: app.verify_input.clone() });
            }
            match app.verify_result {
                Some(true) => {
                    ui.label(RichText::new("✓ That is this wallet's phrase").size(13.0).color(theme::ink()));
                }
                Some(false) => quiet(ui, "Those words are not this wallet's phrase"),
                None if words > 0 && words != 24 => quiet(ui, &format!("{words} of 24 words")),
                None => {}
            }
        });
    });
    ui.add_space(12.0);

    card(ui, |ui| {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                label(ui, "Lock");
                quiet(ui, "Close the wallet until the password is typed again.");
            });
            theme::right_aligned(ui, |ui| {
                if button(ui, "Lock now", true).clicked() {
                    app.go(&ctx, Action::Lock);
                }
            });
        });
    });
    ui.add_space(12.0);

    card(ui, |ui| {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                label(ui, "All features");
                quiet(ui, "Trading, names market, mining, bonds, plans and more.");
            });
            theme::right_aligned(ui, |ui| {
                if button(ui, "Show all features", true).clicked() {
                    super::set_simple(app, false);
                }
            });
        });
    });
}

/// Updates: what runs, what is new, and whether new versions install by
/// themselves. Shared by both views' Settings.
pub fn updates(app: &mut App, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    let status = app.updater.status();
    card(ui, |ui| {
        ui.horizontal(|ui| {
            label(ui, "Updates");
            ui.add_space(6.0);
            quiet(ui, &format!("Midstate {}", crate::update::this_version()));
        });
        ui.add_space(2.0);
        match &status {
            Status::Off(why) => quiet(ui, why),
            Status::Waiting | Status::UpToDate => {
                ui.horizontal(|ui| {
                    quiet(ui, if status == Status::UpToDate { "You have the latest version." } else { "Not checked yet." });
                    if button(ui, "Check now", true).clicked() {
                        app.updater.check_now(false);
                    }
                });
            }
            Status::Checking => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    quiet(ui, "Checking…");
                });
            }
            Status::Available { version } => {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("Version {version} is available.")).size(13.0).color(theme::ink()));
                    if button(ui, "Install", true).clicked() {
                        app.updater.check_now(true);
                    }
                });
            }
            Status::Downloading { version, done, total } => {
                let frac = if *total > 0 { *done as f32 / *total as f32 } else { 0.0 };
                quiet(ui, &format!("Downloading version {version}… {}%", (frac * 100.0).floor()));
                ui.add_space(4.0);
                theme::progress_bar(ui, frac);
            }
            Status::Ready { version } => {
                ui.horizontal(|ui| {
                    let v = version.trim_start_matches('v');
                    ui.label(RichText::new(format!("Version {v} is ready.")).size(13.0).color(theme::ink()));
                    if button(ui, "Restart now", true).clicked() {
                        crate::update::request_restart();
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                quiet(ui, "Or it starts the next time you open Midstate.");
            }
            Status::Manual { version, url } => {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("Version {version} is available.")).size(13.0).color(theme::ink()));
                    if button(ui, "Download", true).clicked() {
                        ctx.open_url(egui::OpenUrl::new_tab(url));
                    }
                });
            }
            Status::Failed(e) => {
                ui.horizontal(|ui| {
                    quiet(ui, "Could not check for updates.");
                    if button(ui, "Try again", true).on_hover_text(e.as_str()).clicked() {
                        app.updater.check_now(false);
                    }
                });
            }
        }
        if !matches!(status, Status::Off(_)) {
            ui.add_space(6.0);
            let mut auto = app.updater.auto();
            if ui.checkbox(&mut auto, "Download new versions automatically").changed() {
                app.updater.set_auto(auto);
            }
        }
    });
}
