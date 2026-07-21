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
            if ui.button(RichText::new("Create a new wallet").strong()).clicked() {
                app.onboard = Onboard::Create;
                app.error.clear();
            }
            if ui.button("Restore from recovery phrase").clicked() {
                app.onboard = Onboard::Restore;
                app.error.clear();
            }
        }
        Onboard::Create => {
            theme::heading(ui, "Create wallet");
            field(ui, "Password (encrypts the wallet file on this machine)", |ui| {
                ui.add(TextEdit::singleline(&mut app.pw).password(true).desired_width(f32::INFINITY));
            });
            field(ui, "Repeat password", |ui| {
                ui.add(TextEdit::singleline(&mut app.pw2).password(true).desired_width(f32::INFINITY));
            });
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button("Back").clicked() {
                    app.onboard = Onboard::Menu;
                }
                theme::right_aligned(ui, |ui| {
                    let label = if app.busy { "Creating…" } else { "Create wallet" };
                    if ui.add_enabled(!app.busy, egui::Button::new(RichText::new(label).strong())).clicked() {
                        if app.pw.len() < 8 {
                            app.error = "Use at least 8 characters.".into();
                        } else if app.pw != app.pw2 {
                            app.error = "Passwords do not match.".into();
                        } else {
                            app.busy = true;
                            app.error.clear();
                            app.go(&ctx, Action::Create { password: app.pw.clone() });
                        }
                    }
                });
            });
        }
        Onboard::Sheet => {
            theme::heading(ui, "Your recovery phrase");
            ui.label(
                "Write these 24 words down, in order, on paper. They are the only \
                 backup — anyone with them controls your funds, and losing them \
                 makes your funds unrecoverable.",
            );
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
            theme::right_aligned(ui, |ui| {
                if ui.button(RichText::new("I wrote them down").strong()).clicked() {
                    app.onboard = Onboard::Confirm;
                    app.error.clear();
                }
            });
        }
        Onboard::Confirm => {
            theme::heading(ui, "Confirm your copy");
            theme::hint(ui, "Enter the requested words from your written copy.");
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
                                app.error =
                                    format!("Word {} does not match. Check your written copy.", i + 1)
                            }
                            None => {
                                app.mnemonic.clear();
                                app.quiz.clear();
                                app.pw.clear();
                                app.pw2.clear();
                                app.error.clear();
                                app.go(&ctx, Action::LoadStatus);
                                app.reload_wallet(&ctx);
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
