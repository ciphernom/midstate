//! The node panel. This app runs a full node in-process, so everything here
//! is read from local consensus state rather than asked of a third party —
//! which is the point, and worth showing rather than just asserting.

use crate::app::App;
use crate::theme::{self, ago, fmt_dt, short_hex, units};
use eframe::egui::{self, FontId, RichText, Ui};

pub fn show(app: &mut App, ui: &mut Ui) {
    theme::heading(ui, "Node");

    let Some(n) = app.node.clone() else {
        theme::hint(ui, "Reading node state…");
        return;
    };
    let syncing = app.sync.as_ref().map(|s| s.is_syncing).unwrap_or(false);

    // Ambient midstate — the chain's whole state compressed to 32 bytes.
    ui.label(
        RichText::new(theme::grouped_hash(&n.midstate, 8))
            .font(FontId::monospace(26.0))
            .color(theme::ambient()),
    );
    ui.add_space(2.0);

    // ── Chain tip ───────────────────────────────────────────────────────
    ui.columns(3, |c| {
        theme::stat(&mut c[0], "Height", &units(n.height), "blocks");
        theme::stat(&mut c[1], "UTXO coins", &units(n.utxo_count as u64), "");
        theme::stat(&mut c[2], "Block reward", &units(n.block_reward), "units");
    });
    ui.columns(3, |c| {
        theme::stat(&mut c[0], "Difficulty", &format!("{}", n.difficulty_bits), "leading zeros");
        theme::stat(&mut c[1], "Open commitments", &units(n.commitment_count as u64), "");
        theme::stat(&mut c[2], "Retired keys", &units(n.burned_count as u64), "one-time");
    });
    ui.add_space(6.0);

    // ── Tip detail ──────────────────────────────────────────────────────
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Chain tip").font(theme::font_medium(14.0)));

        let age = ago(n.tip_timestamp);
        row(ui, "last block", &format!("{}  ·  {}", fmt_dt(n.tip_timestamp), age), None);
        row_copy(ui, "header hash", &n.header_hash);
        row_copy(ui, "midstate", &n.midstate);
        row(ui, "cumulative work", &n.depth, None);
        row(
            ui,
            "confirmation depth",
            &format!("{} blocks", n.safe_depth),
            Some(
                "How deep a transaction must be before this node treats it as settled. It is \
                 estimated from recent chain behaviour, not fixed.",
            ),
        );
        row(
            ui,
            "mempool",
            &format!("{} transaction(s) waiting", n.mempool),
            None,
        );
        if syncing {
            theme::hint(
                ui,
                "Still syncing — these figures describe the chain as far as this node has \
                 verified it, not the network tip.",
            );
        }
    });

    // ── Peers ───────────────────────────────────────────────────────────
    ui.add_space(6.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.label(RichText::new("Peers").font(theme::font_medium(14.0)));
            theme::badge(
                ui,
                &format!("{}", n.peers.len()),
                if n.peers.is_empty() { theme::muted() } else { theme::ink() },
            );
        });
        if n.peers.is_empty() {
            theme::hint(
                ui,
                "No peers connected. The node keeps dialing the bootstrap list — if this \
                 persists, check whether outbound connections on port 9333 are blocked.",
            );
            return;
        }
        theme::hint(ui, "Node identities this wallet is currently connected to.");
        egui::ScrollArea::vertical()
            .max_height(180.0)
            .auto_shrink([false, false])
            .id_salt("peers")
            .show(ui, |ui| {
                for p in &n.peers {
                    ui.horizontal(|ui| {
                        ui.label(theme::mono(short_hex(p, 14)).size(11.0).color(theme::muted()));
                        if ui.button(RichText::new("copy").size(10.0)).clicked() {
                            ui.ctx().copy_text(p.clone());
                        }
                    });
                }
            });
    });

    // ── Local ───────────────────────────────────────────────────────────
    ui.add_space(6.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("This machine").font(theme::font_medium(14.0)));
        row_copy(ui, "data directory", &n.data_dir);
        if let Some(url) = &n.rpc_url {
            row_copy(ui, "rpc", url);
            theme::hint(
                ui,
                "Your own node answers this, so the block explorer below involves no third \
                 party. Other wallets on this machine can point at the same address.",
            );
            if ui.button("Open explorer in browser").clicked() {
                let _ = open_url(url);
            }
        } else {
            theme::hint(ui, "The RPC listener is not enabled for this session.");
        }
    });
}

fn row(ui: &mut Ui, label: &str, value: &str, help: Option<&str>) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(label.to_uppercase())
                .font(FontId::monospace(9.5))
                .color(theme::muted()),
        );
        let l = ui.label(theme::mono(value).size(11.5));
        if let Some(h) = help {
            l.on_hover_text(h);
        }
    });
}

fn row_copy(ui: &mut Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(label.to_uppercase())
                .font(FontId::monospace(9.5))
                .color(theme::muted()),
        );
        ui.label(theme::mono(short_hex(value, 18)).size(11.0).color(theme::bright()));
        if ui.button(RichText::new("copy").size(10.0)).clicked() {
            ui.ctx().copy_text(value.to_string());
        }
    });
}

fn open_url(url: &str) -> std::io::Result<()> {
    #[cfg(target_os = "linux")]
    let cmd = "xdg-open";
    #[cfg(target_os = "macos")]
    let cmd = "open";
    #[cfg(target_os = "windows")]
    let cmd = "explorer";
    std::process::Command::new(cmd).arg(url).spawn().map(|_| ())
}
