use crate::app::App;
use crate::bridge::Action;
use crate::theme::{self};
use eframe::egui::{self, RichText, TextEdit, Ui};
use midstate_walletd::api::WalletStatus;

pub fn show(app: &mut App, ui: &mut Ui, status: &WalletStatus) {
    let ctx = ui.ctx().clone();
    theme::heading(ui, "Settings");

    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        if let Ok(name) = std::env::var("MIDSTATE_PROFILE") {
            if !name.is_empty() {
                ui.horizontal(|ui| {
                    theme::badge(ui, &format!("profile: {name}"), theme::ink());
                    theme::hint(ui, "a separate wallet, chain copy and node on this machine");
                });
                ui.add_space(6.0);
            }
        }
        theme::hint(ui, "WALLET FILE");
        ui.label(theme::mono(&status.wallet_path).size(11.5).color(theme::muted()));
        ui.add_space(8.0);
        theme::hint(ui, "CHAIN DATA");
        ui.label(
            theme::mono(app.node.as_ref().map(|n| n.data_dir.as_str()).unwrap_or("—"))
                .size(11.5)
                .color(theme::muted()),
        );
        ui.add_space(6.0);
        theme::hint(
            ui,
            "Back up your recovery phrase, not these files. The wallet file is encrypted \
             with your password; the chain data can always be re-synced.",
        );
    });

    theme::heading(ui, "Rescan");
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        theme::hint(
            ui,
            "Re-checks the chain for coins paid to your addresses. Use after restoring on \
             another machine or if a payment seems missing.",
        );
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let r = ui.add(
                TextEdit::singleline(&mut app.rescan_h)
                    .hint_text("from block height")
                    .font(egui::TextStyle::Monospace)
                    .desired_width(200.0),
            );
            if r.changed() {
                app.rescan_h.retain(|c| c.is_ascii_digit());
            }
            if ui.button("Start rescan").clicked() {
                match app.rescan_h.parse::<u64>() {
                    Ok(h) => {
                        app.settings_msg.clear();
                        app.go(&ctx, Action::RescanFrom { height: h });
                        app.rescan_h.clear();
                    }
                    Err(_) => {
                        app.settings_msg =
                            "Enter a block height (0 rescans the whole chain).".into();
                    }
                }
            }
        });
        if !app.settings_msg.is_empty() {
            ui.label(RichText::new(&app.settings_msg).color(theme::green()).size(12.0));
        }
        if !app.error.is_empty() {
            ui.label(RichText::new(&app.error).color(theme::red()).size(12.0));
        }
    });

    theme::heading(ui, "Session");
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            theme::hint(ui, "Save the wallet and require the password again.");
            theme::right_aligned(ui, |ui| {
                if ui.button("Lock wallet").clicked() {
                    app.go(&ctx, Action::Lock);
                }
            });
        });
    });

    theme::heading(ui, "Recovery phrase");
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(
            RichText::new("Your phrase cannot be shown again.")
                .font(theme::font_medium(14.0))
                .color(theme::ink()),
        );
        ui.label(
            "It was displayed once, when this wallet was created. The wallet file stores a \
             one-way derivation of the phrase, never the words — so nothing can recover them: \
             not this app, not a future version of it, not the command-line wallet, not anyone \
             you could ask. If your written copy is gone, it is gone.",
        );
        ui.add_space(8.0);
        theme::hint(
            ui,
            "Without the phrase, this wallet file and its password are your only backup. Copy \
             the file somewhere safe. To get a phrase you actually hold, create a new wallet, \
             write its phrase down, and move your coins to it.",
        );

        ui.add_space(12.0);
        ui.label(RichText::new("Check your written copy").font(theme::font_medium(13.0)));
        theme::hint(
            ui,
            "Type the 24 words to confirm they match this wallet. They are compared and \
             discarded — nothing is stored, and this cannot reveal a phrase you do not \
             already have.",
        );
        ui.add(
            TextEdit::multiline(&mut app.verify_input)
                .desired_rows(3)
                .desired_width(f32::INFINITY)
                .font(egui::TextStyle::Monospace),
        );
        ui.horizontal(|ui| {
            let words = app.verify_input.split_whitespace().count();
            if ui
                .add_enabled(!app.busy && words == 24, egui::Button::new("Check phrase"))
                .clicked()
            {
                app.busy = true;
                app.error.clear();
                app.verify_result = None;
                app.go(&ctx, Action::VerifyPhrase { phrase: app.verify_input.clone() });
            }
            if words > 0 && words != 24 {
                theme::hint(ui, &format!("{words} of 24 words"));
            }
            match app.verify_result {
                Some(true) => {
                    ui.label(
                        RichText::new("Match — this is the phrase for this wallet.")
                            .size(12.0)
                            .color(theme::ink()),
                    );
                }
                Some(false) => {
                    ui.label(
                        RichText::new(
                            "No match. This phrase does not belong to this wallet — check for \
                             transcription errors before relying on it.",
                        )
                        .size(12.0)
                        .color(theme::muted()),
                    );
                }
                None => {}
            }
        });
    });

    theme::heading(ui, "Rebuild history amounts");
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        theme::hint(
            ui,
            "Transactions publish every input and output value on-chain — consensus checks \
             conservation — and each output carries the recipient's address. The wallet's own \
             history file keeps only coin ids, so older transactions can show no amount. This \
             rereads them from your block store and restores both the amounts and who they \
             went to. A match is accepted only when the arithmetic agrees with the fee already \
             recorded, so a wrong guess is never written.",
        );
        theme::right_aligned(ui, |ui| {
            if ui.add_enabled(!app.busy, egui::Button::new("Rebuild from chain")).clicked() {
                app.busy = true;
                app.settings_msg.clear();
                app.go(&ctx, Action::RepairHistory);
            }
        });
    });

    theme::heading(ui, "Appearance");
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        let mut mode = theme::mode();
        let before = mode;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut mode, theme::ThemeMode::System, "Follow system");
            ui.selectable_value(&mut mode, theme::ThemeMode::Dark, "Dark");
            ui.selectable_value(&mut mode, theme::ThemeMode::Light, "Light");
        });
        if mode != before {
            theme::set_mode(mode);
            theme::save_pref();
        }
        theme::hint(
            ui,
            "Ink on paper, either way round — the same tokens the midstate stylesheet uses.",
        );
    });

    theme::heading(ui, "Not in this version");
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        theme::hint(
            ui,
            "Mixing, pruning-license management and multiple wallets still live \
             in the midstate CLI. They share this wallet's format — close this app before \
             pointing the CLI at the same wallet file.",
        );
    });
}
