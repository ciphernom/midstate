use crate::app::{App, Tab};
use crate::theme::{self, ago, fmt_duration, send_timeline, short_hex, units};
use eframe::egui::{self, FontId, RichText, Ui};
use midstate_walletd::api::SendStage;

pub fn show(app: &mut App, ui: &mut Ui) {
    theme::heading(ui, "Dashboard");

    // ── Money ───────────────────────────────────────────────────────────
    let b = app.balance.clone();
    // Balances run to ten digits; a unit picker makes the headline readable
    // without hiding the exact figure below it.
    ui.horizontal(|ui| {
        theme::hint(ui, "show amounts in");
        theme::unit_selector(ui, &mut app.balance_unit);
        if let Some(bal) = &b {
            ui.label(
                RichText::new(format!("= {} units exactly", units(bal.confirmed)))
                    .size(11.0)
                    .color(theme::faint()),
            );
        }
    });
    ui.add_space(4.0);
    let uu = app.balance_unit;
    let show = |v: u64| theme::in_unit(v, uu);
    ui.columns(4, |c| {
        theme::stat(
            &mut c[0],
            "Spendable",
            &b.as_ref().map(|b| show(b.confirmed)).unwrap_or_else(|| "—".into()),
            theme::UNIT_NAMES[uu.min(3)],
        );
        theme::stat(
            &mut c[1],
            "In flight",
            &b.as_ref().map(|b| show(b.in_flight)).unwrap_or_else(|| "—".into()),
            theme::UNIT_NAMES[uu.min(3)],
        );
        theme::stat(
            &mut c[2],
            "Unconfirmed",
            &b.as_ref().map(|b| show(b.unconfirmed)).unwrap_or_else(|| "—".into()),
            theme::UNIT_NAMES[uu.min(3)],
        );
        theme::stat(
            &mut c[3],
            "Coins",
            &b.as_ref().map(|b| units(b.coin_count as u64)).unwrap_or_else(|| "—".into()),
            "",
        );
    });

    if let Some(b) = &b {
        if b.unconfirmed > 0 {
            theme::hint(
                ui,
                &format!(
                    "{} is not in the chain's coin set right now — either still confirming or \
                     stranded by a reorg. It stays listed under Coins.",
                    units(b.unconfirmed)
                ),
            );
        }
    }

    if app.sync.as_ref().map(|s| s.is_syncing).unwrap_or(false) {
        sync_panel(app, ui);
        primer(app, ui);
    }

    // ── What needs attention ────────────────────────────────────────────
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

    ui.add_space(4.0);
    ui.columns(2, |cols| {
        wallet_health(app, &mut cols[0]);
        network_panel(app, &mut cols[1]);
    });

    // ── Recent activity ─────────────────────────────────────────────────
    ui.add_space(4.0);
    theme::heading(ui, "Recent activity");
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        if app.history.is_empty() {
            theme::hint(ui, "No activity yet. Share an address from the Receive tab to get paid.");
            return;
        }
        egui::Grid::new("recent").num_columns(5).spacing([22.0, 7.0]).striped(true).show(ui, |ui| {
            for h in app.history.iter().take(8) {
                let incoming = h.kind == "received" || h.kind == "coinbase";
                theme::badge(ui, &h.kind, if incoming { theme::ink() } else { theme::muted() });
                ui.label(theme::mono(if h.amount > 0 {
                    format!("{}{}", if incoming { "+" } else { "" }, units(h.amount))
                } else {
                    "—".into()
                }));
                ui.label(
                    theme::mono(if h.fee > 0 { format!("fee {}", units(h.fee)) } else { "—".into() })
                        .color(theme::muted())
                        .size(11.5),
                );
                ui.label(
                    theme::mono(format!("{}→{}", h.n_in, h.n_out))
                        .color(theme::faint())
                        .size(11.0),
                );
                ui.label(RichText::new(ago(h.timestamp)).color(theme::muted()).size(11.5));
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

/// Things that quietly stop a wallet working: exhausted signing keys and
/// coin fragmentation. Both are invisible until they bite, so surface them.
fn wallet_health(app: &mut App, ui: &mut Ui) {
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Wallet").font(theme::font_medium(14.0)));

        let live: Vec<_> = app.coins.iter().filter(|c| c.live && !c.in_flight).collect();
        let one_time = live.iter().filter(|c| c.kind != "mss").count();
        let reusable_addrs = app.addresses.iter().filter(|a| a.kind == "mss").count();
        let sigs: u64 = app.addresses.iter().filter_map(|a| a.remaining_sigs).sum();

        row(ui, "reusable addresses", &format!("{reusable_addrs}"));
        row(ui, "signatures available", &units(sigs));
        row(ui, "one-time coins", &format!("{one_time}"));

        // A normal send caps at 256 inputs, so fragmentation has a hard edge.
        if one_time > 120 {
            theme::hint(
                ui,
                "Coins are fragmenting. A single transaction can spend at most 256 inputs, so \
                 run Defrag on the Coins tab before it becomes a problem.",
            );
        }
        if sigs == 0 && reusable_addrs > 0 {
            theme::hint(ui, "Your reusable addresses are out of signatures — generate a new one.");
        }
        if !app.channels.is_empty() {
            let open = app.channels.iter().filter(|c| c.status == "active").count();
            let locked: u64 = app
                .channels
                .iter()
                .filter(|c| c.status == "active")
                .map(|c| c.my_balance)
                .sum();
            row(ui, "channels", &format!("{open} open · {} yours", units(locked)));
        }
    });
}

/// Node and chain condition, condensed. The full picture lives on the Node tab.
fn network_panel(app: &mut App, ui: &mut Ui) {
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Network").font(theme::font_medium(14.0)));
        let Some(s) = app.sync.clone() else {
            theme::hint(ui, "Starting the node…");
            return;
        };

        ui.horizontal(|ui| {
            theme::badge(
                ui,
                if s.is_syncing { "syncing" } else { "in sync" },
                if s.is_syncing { theme::muted() } else { theme::ink() },
            );
            theme::badge(
                ui,
                &format!("{} peer(s)", s.peer_count),
                if s.peer_count == 0 { theme::muted() } else { theme::ink() },
            );
        });
        row(ui, "height", &units(s.height));
        row(ui, "mempool", &format!("{} waiting", s.mempool));
        row(ui, "settles after", &format!("{} blocks", s.safe_depth));

        if let Some(n) = app.node.clone() {
            let age = theme::now_secs().saturating_sub(n.tip_timestamp);
            row(ui, "last block", &fmt_duration(age));
            row(ui, "chain coins", &units(n.utxo_count as u64));
        }
        if s.peer_count == 0 {
            theme::hint(ui, "No peers yet — the node is still dialing the bootstrap list.");
        }
    });
}

fn row(ui: &mut Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(label.to_uppercase())
                .font(FontId::monospace(9.5))
                .color(theme::muted()),
        );
        theme::right_aligned(ui, |ui| {
            ui.label(theme::mono(value).size(11.5));
        });
    });
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
