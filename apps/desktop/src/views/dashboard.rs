use crate::app::{App, Tab};
use crate::theme::{self, ago, fmt_duration, send_timeline, short_hex, units};
use eframe::egui::{self, FontId, RichText, Ui};
use midstate_walletd::api::SendStage;

pub fn show(app: &mut App, ui: &mut Ui) {
    theme::heading(ui, "Dashboard");

    ui.columns(3, |cols| {
        let b = app.balance.as_ref();
        theme::stat(
            &mut cols[0],
            "Spendable",
            &b.map(|b| units(b.confirmed)).unwrap_or_else(|| "—".into()),
            "units",
        );
        theme::stat(
            &mut cols[1],
            "In flight",
            &b.map(|b| units(b.in_flight)).unwrap_or_else(|| "—".into()),
            "units",
        );
        theme::stat(
            &mut cols[2],
            "Coins",
            &b.map(|b| units(b.coin_count as u64)).unwrap_or_else(|| "—".into()),
            "",
        );
    });

    if let Some(b) = &app.balance {
        if b.unconfirmed > 0 {
            theme::hint(
                ui,
                &format!(
                    "{} units are not currently in the chain's coin set (reorged or still \
                     confirming). They stay listed under Coins.",
                    units(b.unconfirmed)
                ),
            );
        }
    }

    if app.sync.as_ref().map(|s| s.is_syncing).unwrap_or(false) {
        sync_panel(app, ui);
        primer(app, ui);
    }

    let active: Vec<_> = app
        .sends
        .iter()
        .filter(|s| s.stage != SendStage::Confirmed && s.stage != SendStage::Failed)
        .cloned()
        .collect();
    if !active.is_empty() {
        theme::heading(ui, "Active sends");
        for s in &active {
            theme::panel_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(theme::mono(format!("{} units", units(s.amount))));
                    theme::right_aligned(ui, |ui| {
                        ui.label(theme::mono(short_hex(&s.id, 8)).color(theme::muted()));
                    });
                });
                send_timeline(ui, s);
                theme::hint(ui, &s.detail);
            });
            ui.add_space(4.0);
        }
    }

    theme::heading(ui, "Recent activity");
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        if app.history.is_empty() {
            theme::hint(ui, "No activity yet. Share an address from the Receive tab to get paid.");
            return;
        }
        egui::Grid::new("recent").num_columns(4).spacing([26.0, 8.0]).striped(true).show(ui, |ui| {
            for h in app.history.iter().take(6) {
                let incoming = h.kind == "received" || h.kind == "coinbase";
                theme::badge(ui, &h.kind, if incoming { theme::ink() } else { theme::muted() });
                let sign = if incoming { "+" } else { "\u{2212}" };
                ui.label(theme::mono(if h.amount > 0 {
                    format!("{sign}{}", units(h.amount))
                } else {
                    "—".into()
                }));
                ui.label(
                    theme::mono(if h.fee > 0 { format!("fee {}", units(h.fee)) } else { "—".into() })
                        .color(theme::muted())
                        .size(12.0),
                );
                ui.label(RichText::new(ago(h.timestamp)).color(theme::muted()).size(12.0));
                ui.end_row();
            }
        });
    });
    if !app.history.is_empty() {
        ui.horizontal(|ui| {
            theme::hint(ui, "Full record on the");
            if ui.link(RichText::new("History").size(12.0)).clicked() {
                app.tab = Tab::History;
            }
            theme::hint(ui, "tab.");
        });
    }
}

/// The sync state, made first-class: an ink progress bar against the
/// estimated chain height, with honest phase labeling, rate, and ETA.
fn sync_panel(app: &mut App, ui: &mut Ui) {
    ui.add_space(6.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        let Some(s) = app.sync.clone() else { return };
        let target = s.est_target_height.max(s.height);

        ui.horizontal(|ui| {
            ui.label(
                RichText::new("SYNCING THE CHAIN")
                    .font(FontId::monospace(10.0))
                    .color(theme::muted()),
            );
            theme::right_aligned(ui, |ui| {
                if let Some(frac) = app.sync_fraction() {
                    ui.label(theme::mono(format!("{:.1}%", frac * 100.0)).size(11.0).color(theme::bright()));
                }
            });
        });
        ui.add_space(4.0);
        theme::progress_bar(ui, app.sync_fraction().unwrap_or(0.0));
        ui.add_space(6.0);

        let rate = app.sync_rate();
        let caption = if s.height <= 2 && rate.is_none() {
            format!(
                "Verifying block headers first — the bar moves once whole blocks start \
                 applying. Target: ~{} blocks.",
                units(target)
            )
        } else {
            let mut c = format!("Block {} of ~{}", units(s.height), units(target));
            if let Some(r) = rate {
                c.push_str(&format!(" · {:.0} blocks/s", r));
            }
            if let Some(eta) = app.sync_eta_secs() {
                c.push_str(&format!(" · ~{} left", fmt_duration(eta)));
            }
            c
        };
        theme::hint(ui, &caption);
        theme::hint(
            ui,
            "Every block is verified on this machine. Receiving addresses already work; \
             sending unlocks at the tip.",
        );
    });
}

/// Five short cards that use the sync wait to teach what makes this chain
/// different. Structure over decoration; no motion.
fn primer(app: &mut App, ui: &mut Ui) {
    const CARDS: [(&str, &str); 5] = [
        (
            "Your node is the wallet",
            "This app runs a full Midstate node. Balances, payments, and history come \
             from blocks your own machine verified — no servers, no trusted third party. \
             While it catches up, you can already create addresses and receive; the chain \
             will show those coins as soon as their blocks are reached.",
        ),
        (
            "Keys sign exactly once",
            "Midstate uses post-quantum one-time signatures (WOTS). The moment a key \
             signs, it must never sign again. The wallet enforces this for you: change \
             from every send goes to fresh keys, and an address that has received once \
             is marked \u{201c}used \u{2014} don\u{2019}t share again\u{201d} on the Receive tab.",
        ),
        (
            "Reusable addresses exist too",
            "For anything public \u{2014} payouts, donations, an address you print \u{2014} \
             generate a reusable address instead. It bundles 1,024 one-time keys under a \
             single address, and the Receive tab shows how many signatures it has left.",
        ),
        (
            "Sending is a two-step ritual",
            "A send first posts a sealed commitment on-chain, then reveals it once the \
             commitment is mined. You'll watch both stages on a timeline. It's safe to \
             close the app mid-send \u{2014} it resumes where it left off at next unlock. An \
             optional privacy delay makes the two steps harder to link.",
        ),
        (
            "While you wait",
            "Three useful things you can do right now: write your 24-word recovery \
             phrase on paper if you haven't; open Receive and make your first address; \
             and if you have a CLI wallet, close this app and copy its wallet.dat into \
             this app's data directory (path is under Settings) to bring your coins over.",
        ),
    ];

    ui.add_space(6.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        let page = app.primer_page.min(CARDS.len() - 1);
        let (title, body) = CARDS[page];

        ui.label(
            RichText::new(format!("WHILE YOU SYNC \u{2014} {} / {}", page + 1, CARDS.len()))
                .font(FontId::monospace(10.0))
                .color(theme::faint()),
        );
        ui.add_space(4.0);
        ui.label(RichText::new(title).font(theme::font_medium(15.0)).color(theme::ink()));
        ui.add_space(4.0);
        ui.label(RichText::new(body).color(theme::bright()));
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.add_enabled(page > 0, egui::Button::new("Back")).clicked() {
                app.primer_page = page - 1;
            }
            theme::right_aligned(ui, |ui| {
                let last = page + 1 == CARDS.len();
                if ui.button(if last { "Start over" } else { "Next" }).clicked() {
                    app.primer_page = if last { 0 } else { page + 1 };
                }
            });
        });
    });
}
