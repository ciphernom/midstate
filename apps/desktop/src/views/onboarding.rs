//! Create / restore / unlock flows. The mnemonic never leaves this process's
//! memory — one concrete reason this frontend is native Rust.

use crate::app::{App, Onboard};
use crate::bridge::Action;
use crate::theme;
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
            theme::heading(ui, "Welcome to Midstate");
            ui.label(
                "This app runs a full node on your machine. Your wallet talks only \
                 to your own validated copy of the chain.",
            );
            ui.add_space(14.0);
            let label = if app.busy { "Preparing…" } else { "Create a new wallet" };
            if ui
                .add_enabled(!app.busy, egui::Button::new(RichText::new(label).strong()))
                .clicked()
            {
                app.busy = true;
                app.error.clear();
                app.own_phrase = false;
                app.phrase.clear();
                // Mint the phrase first. Nothing is written to disk until the
                // phrase has been shown, confirmed, and a password chosen.
                app.go(&ctx, Action::NewPhrase);
            }
            if ui.button("Restore from recovery phrase").clicked() {
                app.onboard = Onboard::Restore;
                app.error.clear();
            }
        }
        Onboard::Create => {
            theme::heading(ui, "Set a password");
            theme::hint(
                ui,
                "Phrase confirmed, and now out of reach — it will not be shown again. This \
                 password only encrypts the wallet file on this machine. It is not a second \
                 backup: on a new machine, or after a disk failure, the written phrase is the \
                 only thing that brings your coins back.",
            );
            field(ui, "Password (encrypts the wallet file on this machine)", |ui| {
                ui.add(TextEdit::singleline(&mut app.pw).password(true).desired_width(f32::INFINITY));
            });
            field(ui, "Repeat password", |ui| {
                ui.add(TextEdit::singleline(&mut app.pw2).password(true).desired_width(f32::INFINITY));
            });
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button("Show phrase again").clicked() {
                    app.onboard = Onboard::Sheet;
                    app.error.clear();
                }
                theme::right_aligned(ui, |ui| {
                    let label = if app.busy { "Creating…" } else { "Create wallet" };
                    if ui.add_enabled(!app.busy, egui::Button::new(RichText::new(label).strong())).clicked() {
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
                });
            });
        }
        Onboard::Sheet => {
            theme::heading(ui, "Your recovery phrase");
            ui.label(
                "Write these 24 words down, in order, on paper. Anyone with them controls \
                 your funds.",
            );
            ui.add_space(8.0);
            // The single most consequential fact in the whole app: this screen
            // does not come back. Give it its own frame so it cannot be skimmed.
            egui::Frame::default()
                .fill(theme::highlight())
                .stroke(egui::Stroke::new(1.0, theme::ink()))
                .inner_margin(egui::Margin::same(12))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(
                        RichText::new("This is the only time these words will be shown.")
                            .font(theme::font_medium(14.0))
                            .color(theme::ink()),
                    );
                    ui.label(
                        "The wallet stores a one-way derivation of them, never the words \
                         themselves. They cannot be shown again by this app, a future version \
                         of it, the command-line wallet, or anyone you could ask. If you lose \
                         your written copy, the coins are unrecoverable.",
                    );
                });
            ui.add_space(8.0);
            egui::Frame::default()
                .fill(theme::bg())
                .stroke(egui::Stroke::new(1.0, theme::gold()))
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(egui::Margin::same(14))
                .show(ui, |ui| {
                    egui::Grid::new("mnemonic").num_columns(4).spacing([18.0, 6.0]).show(ui, |ui| {
                        for (i, w) in app.mnemonic.iter().enumerate() {
                            ui.horizontal(|ui| {
                                ui.label(RichText::new(format!("{:>2}.", i + 1)).monospace().color(theme::faint()));
                                ui.label(RichText::new(w).monospace());
                            });
                            if (i + 1) % 4 == 0 {
                                ui.end_row();
                            }
                        }
                    });
                });
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!app.busy, egui::Button::new("Generate a different phrase"))
                    .clicked()
                {
                    app.busy = true;
                    app.error.clear();
                    app.go(&ctx, Action::NewPhrase);
                }
                if ui.button("Use my own phrase").clicked() {
                    app.own_phrase = true;
                    app.phrase.clear();
                    app.error.clear();
                }
                theme::right_aligned(ui, |ui| {
                    if ui.button(RichText::new("I wrote them down").strong()).clicked() {
                        app.onboard = Onboard::Confirm;
                        app.error.clear();
                    }
                });
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
                theme::hint(
                    ui,
                    "It must be a valid BIP39 phrase — the checksum built into the last \
                     word is verified, so a typo is caught here rather than after your \
                     coins are gone.",
                );
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
            theme::heading(ui, "Confirm your copy");
            theme::hint(
                ui,
                "Enter the requested words from your written copy — not from memory. This is \
                 the last point at which the phrase is still on screen; after this it is gone \
                 for good.",
            );
            ui.add_space(6.0);
            let mut quiz = std::mem::take(&mut app.quiz);
            for (i, entered) in quiz.iter_mut() {
                field(ui, &format!("Word {}", *i + 1), |ui| {
                    ui.add(TextEdit::singleline(entered).desired_width(220.0));
                });
            }
            app.quiz = quiz;
            ui.horizontal(|ui| {
                if ui.button("Show phrase again").clicked() {
                    app.onboard = Onboard::Sheet;
                }
                theme::right_aligned(ui, |ui| {
                    if ui.button(RichText::new("Confirm").strong()).clicked() {
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
                });
            });
        }
        Onboard::Restore => {
            theme::heading(ui, "Restore wallet");
            field(ui, "24-word recovery phrase", |ui| {
                ui.add(
                    TextEdit::multiline(&mut app.phrase)
                        .desired_rows(3)
                        .desired_width(f32::INFINITY)
                        .font(egui::TextStyle::Monospace),
                );
            });
            field(ui, "New password for this machine", |ui| {
                ui.add(TextEdit::singleline(&mut app.pw).password(true).desired_width(f32::INFINITY));
            });
            theme::hint(
                ui,
                "After restoring, the wallet derives its first 1,000 keys and scans the \
                 chain from the beginning. Your balance fills in as the scan progresses.",
            );
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button("Back").clicked() {
                    app.onboard = Onboard::Menu;
                }
                theme::right_aligned(ui, |ui| {
                    let label = if app.busy { "Restoring…" } else { "Restore wallet" };
                    if ui.add_enabled(!app.busy, egui::Button::new(RichText::new(label).strong())).clicked() {
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
                });
            });
        }
        _ => {
            // Unlock (also the fallback when a wallet file exists).
            theme::heading(ui, "Unlock wallet");
            field(ui, "Password", |ui| {
                let r = ui.add(
                    TextEdit::singleline(&mut app.pw).password(true).desired_width(f32::INFINITY),
                );
                if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) && !app.busy {
                    app.busy = true;
                    app.error.clear();
                    app.go(&ctx, Action::Unlock { password: app.pw.clone() });
                }
            });
            theme::right_aligned(ui, |ui| {
                let label = if app.busy { "Unlocking…" } else { "Unlock" };
                if ui.add_enabled(!app.busy, egui::Button::new(RichText::new(label).strong())).clicked() {
                    app.busy = true;
                    app.error.clear();
                    app.go(&ctx, Action::Unlock { password: app.pw.clone() });
                }
            });
        }
    }

    if !app.error.is_empty() {
        ui.add_space(8.0);
        ui.label(RichText::new(&app.error).color(theme::red()));
    }
}

fn field(ui: &mut Ui, name: &str, add: impl FnOnce(&mut Ui)) {
    ui.add_space(6.0);
    theme::hint(ui, name);
    add(ui);
}
