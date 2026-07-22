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
            "Mixing, pruning-license management, mining, and multiple wallets still live \
             in the midstate CLI. They share this wallet's format — close this app before \
             pointing the CLI at the same wallet file.",
        );
    });
}
