//! Create / restore / unlock flows. The mnemonic never leaves this process's
//! memory — one concrete reason this frontend is native Rust.

use crate::app::{App, Onboard};
use crate::bridge::Action;
use crate::theme;
use crate::views::simple::{quiet, title};
use eframe::egui::{self, RichText, TextEdit, Ui};
use midstate_walletd::api::WalletStatus;

pub fn show(app: &mut App, ctx: &egui::Context, status: &WalletStatus) {
    egui::CentralPanel::default()
        .frame(egui::Frame::default().fill(theme::bg()).inner_margin(egui::Margin::symmetric(28, 24)))
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.set_max_width(560.0);
                    ui.with_layout(egui::Layout::top_down(egui::Align::LEFT), |ui| {
                        body(app, ui, status);
                    });
                });
            });
        });
}

fn body(app: &mut App, ui: &mut Ui, status: &WalletStatus) {
    let ctx = ui.ctx().clone();
    ui.add_space(30.0);

    if let Some(tex) = app.logo.clone() {
        ui.vertical_centered(|ui| {
            theme::logo(ui, &tex, 108.0, theme::ink());
        });
        ui.add_space(18.0);
    }

    match app.onboard {
        Onboard::Menu if !status.exists => {
            ui.vertical_centered(|ui| {
                ui.label(RichText::new("Welcome to Midstate").font(theme::font_medium(26.0)).color(theme::ink()));
                ui.add_space(6.0);
                ui.label(RichText::new("A wallet that runs on your own computer.").size(15.0).color(theme::muted()));
            });
            ui.add_space(28.0);
            let label = if app.busy { "Preparing…" } else { "Create a new wallet" };
            if wide(ui, label, true, !app.busy).clicked() {
                app.busy = true;
                app.error.clear();
                app.own_phrase = false;
                app.phrase.clear();
                // Mint the phrase first. Nothing is written to disk until the
                // phrase has been shown, confirmed, and a password chosen.
                app.go(&ctx, Action::NewPhrase);
            }
            ui.add_space(8.0);
            if wide(ui, "I already have a wallet", false, true).clicked() {
                app.onboard = Onboard::Restore;
                app.error.clear();
            }
        }
        Onboard::Create => {
            title(ui, "Choose a password");
            quiet(ui, "It opens this wallet on this computer. Your 24 words are the backup, not this password.");
            field(ui, "Password", |ui| {
                password(ui, &mut app.pw);
            });
            field(ui, "Password again", |ui| {
                password(ui, &mut app.pw2);
            });
            ui.add_space(18.0);
            let label = if app.busy { "Creating…" } else { "Create wallet" };
            if wide(ui, label, true, !app.busy).clicked() {
                if app.pw.len() < 8 {
                    app.error = "Use at least 8 characters.".into();
                } else if app.pw != app.pw2 {
                    app.error = "Passwords do not match.".into();
                } else if app.mnemonic.len() != 24 {
                    app.error = "Recovery phrase missing — start again.".into();
                    app.onboard = Onboard::Menu;
                } else {
                    app.busy = true;
                    app.error.clear();
                    app.go(
                        &ctx,
                        Action::Create {
                            password: app.pw.clone(),
                            phrase: app.mnemonic.join(" "),
                        },
                    );
                }
            }
            ui.add_space(10.0);
            if link(ui, "Show the words again") {
                app.onboard = Onboard::Sheet;
                app.error.clear();
            }
        }
        Onboard::Sheet => {
            title(ui, "Write down these 24 words");
            // The single most consequential fact in the whole app: this screen
            // does not come back. Give it its own frame so it cannot be skimmed.
            egui::Frame::default()
                .fill(theme::highlight())
                .stroke(egui::Stroke::new(1.0, theme::ink()))
                .inner_margin(egui::Margin::same(12))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(
                        RichText::new("This is the only time they are shown.")
                            .font(theme::font_medium(14.0))
                            .color(theme::ink()),
                    );
                    ui.label(
                        RichText::new(
                            "They are the only way back into this wallet if this computer is lost. \
                             Anyone who has them can take your money. Keep them on paper, somewhere safe.",
                        )
                        .size(13.0),
                    );
                });
            ui.add_space(12.0);
            egui::Frame::default()
                .fill(theme::bg())
                .stroke(egui::Stroke::new(1.0, theme::gold()))
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(egui::Margin::same(16))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    egui::Grid::new("mnemonic").num_columns(4).spacing([22.0, 9.0]).show(ui, |ui| {
                        for (i, w) in app.mnemonic.iter().enumerate() {
                            ui.horizontal(|ui| {
                                ui.label(RichText::new(format!("{:>2}.", i + 1)).monospace().size(13.0).color(theme::faint()));
                                ui.label(RichText::new(w).monospace().size(15.0));
                            });
                            if (i + 1) % 4 == 0 {
                                ui.end_row();
                            }
                        }
                    });
                });
            ui.add_space(18.0);
            if wide(ui, "I wrote them down", true, true).clicked() {
                app.onboard = Onboard::Confirm;
                app.error.clear();
            }
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 18.0;
                if !app.busy && link(ui, "Show different words") {
                    app.busy = true;
                    app.error.clear();
                    app.go(&ctx, Action::NewPhrase);
                }
                if link(ui, "Use my own words") {
                    app.own_phrase = true;
                    app.phrase.clear();
                    app.error.clear();
                }
            });

            if app.own_phrase {
                ui.add_space(10.0);
                field(ui, "Your own 24-word phrase", |ui| {
                    ui.add(
                        TextEdit::multiline(&mut app.phrase)
                            .desired_rows(3)
                            .desired_width(f32::INFINITY)
                            .font(egui::TextStyle::Monospace),
                    );
                });
                quiet(ui, "The last word is a checksum, so a typo is caught here.");
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        app.own_phrase = false;
                        app.phrase.clear();
                        app.error.clear();
                    }
                    theme::right_aligned(ui, |ui| {
                        let words = app.phrase.split_whitespace().count();
                        if ui
                            .add_enabled(!app.busy && words == 24, egui::Button::new("Use this phrase"))
                            .clicked()
                        {
                            app.busy = true;
                            app.error.clear();
                            app.go(&ctx, Action::CheckPhrase { phrase: app.phrase.clone() });
                        }
                        if words > 0 && words != 24 {
                            theme::hint(ui, &format!("{words} of 24 words"));
                        }
                    });
                });
            }
        }
        Onboard::Confirm => {
            title(ui, "Check your copy");
            quiet(ui, "Type these words from your paper.");
            let mut quiz = std::mem::take(&mut app.quiz);
            for (i, entered) in quiz.iter_mut() {
                field(ui, &format!("Word {}", *i + 1), |ui| {
                    ui.add(
                        TextEdit::singleline(entered)
                            .font(egui::FontId::proportional(16.0))
                            .margin(egui::Margin::symmetric(10, 9))
                            .desired_width(260.0),
                    );
                });
            }
            app.quiz = quiz;
            ui.add_space(18.0);
            if wide(ui, "Confirm", true, true).clicked() {
                let bad = app
                    .quiz
                    .iter()
                    .find(|(i, e)| e.trim().to_lowercase() != app.mnemonic[*i]);
                match bad {
                    Some((i, _)) => {
                        // Wrong word: send them back to read the sheet
                        // again and re-verify. A confirmation you can
                        // brute-force by guessing is not a confirmation.
                        app.error = format!(
                            "Word {} does not match. Read the phrase again and re-confirm.",
                            i + 1
                        );
                        for (_, e) in app.quiz.iter_mut() {
                            e.clear();
                        }
                        app.onboard = Onboard::Sheet;
                    }
                    None => {
                        // Verified. Only now do we ask for a password
                        // and write anything to disk.
                        app.quiz.clear();
                        app.pw.clear();
                        app.pw2.clear();
                        app.error.clear();
                        app.onboard = Onboard::Create;
                    }
                }
            }
            ui.add_space(10.0);
            if link(ui, "Show the words again") {
                app.onboard = Onboard::Sheet;
            }
        }
        Onboard::Restore => {
            title(ui, "Restore your wallet");
            field(ui, "Your 24 words", |ui| {
                ui.add(
                    TextEdit::multiline(&mut app.phrase)
                        .desired_rows(3)
                        .desired_width(f32::INFINITY)
                        .font(egui::TextStyle::Monospace),
                );
            });
            field(ui, "A password for this computer", |ui| {
                password(ui, &mut app.pw);
            });
            ui.add_space(18.0);
            let label = if app.busy { "Restoring…" } else { "Restore" };
            if wide(ui, label, true, !app.busy).clicked() {
                let words = app.phrase.split_whitespace().count();
                if app.pw.len() < 8 {
                    app.error = "Use at least 8 characters.".into();
                } else if words != 24 {
                    app.error =
                        format!("A recovery phrase has 24 words (you entered {words}).");
                } else {
                    app.busy = true;
                    app.error.clear();
                    app.go(
                        &ctx,
                        Action::Restore {
                            password: app.pw.clone(),
                            phrase: app.phrase.trim().to_string(),
                        },
                    );
                }
            }
            ui.add_space(10.0);
            quiet(ui, "Your money appears as the wallet finds it on the network.");
            ui.add_space(6.0);
            if link(ui, "Back") {
                app.onboard = Onboard::Menu;
            }
        }
        _ => {
            // Unlock (also the fallback when a wallet file exists).
            title(ui, "Welcome back");
            field(ui, "Password", |ui| {
                let r = password(ui, &mut app.pw);
                if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) && !app.busy {
                    app.busy = true;
                    app.error.clear();
                    app.go(&ctx, Action::Unlock { password: app.pw.clone() });
                }
            });
            ui.add_space(18.0);
            let label = if app.busy { "Unlocking…" } else { "Unlock" };
            if wide(ui, label, true, !app.busy).clicked() {
                app.busy = true;
                app.error.clear();
                app.go(&ctx, Action::Unlock { password: app.pw.clone() });
            }
        }
    }

    if !app.error.is_empty() {
        ui.add_space(8.0);
        ui.label(RichText::new(&app.error).color(theme::red()));
    }
}

fn field(ui: &mut Ui, name: &str, add: impl FnOnce(&mut Ui)) {
    ui.add_space(10.0);
    ui.label(RichText::new(name).size(13.0).color(theme::muted()));
    ui.add_space(2.0);
    add(ui);
}

/// A password box at the size of the rest of the screen.
fn password(ui: &mut Ui, text: &mut String) -> egui::Response {
    ui.add(
        TextEdit::singleline(text)
            .password(true)
            .font(egui::FontId::proportional(16.0))
            .margin(egui::Margin::symmetric(10, 9))
            .desired_width(f32::INFINITY),
    )
}

/// A button as wide as the column.
fn wide(ui: &mut Ui, text: &str, primary: bool, enabled: bool) -> egui::Response {
    let ink = if primary { theme::on_ink() } else { theme::ink() };
    let b = egui::Button::new(RichText::new(text).font(theme::font_medium(15.0)).color(ink))
        .fill(if primary { theme::ink() } else { egui::Color32::TRANSPARENT })
        .stroke(egui::Stroke::new(1.0, if primary { theme::ink() } else { theme::border2() }))
        .min_size(egui::vec2(ui.available_width(), 46.0));
    // In a row the text sits in the middle of the button.
    ui.horizontal(|ui| ui.add_enabled(enabled, b)).inner
}

/// A quiet link under the main button.
fn link(ui: &mut Ui, text: &str) -> bool {
    ui.link(RichText::new(text).size(13.0).color(theme::muted())).clicked()
}
