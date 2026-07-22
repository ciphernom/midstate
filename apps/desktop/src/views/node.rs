use crate::app::App;
use crate::theme::{self, fmt_dt, grouped_hash, units};
use eframe::egui::{self, RichText, Ui};

pub fn show(app: &mut App, ui: &mut Ui) {
    theme::heading(ui, "Node");

    // The chain tip's midstate rendered as ambient texture — the state this
    // whole app is named after, updating once per block.
    if let Some(s) = &app.sync {
        if !s.midstate.is_empty() {
            ui.label(
                RichText::new(grouped_hash(&s.midstate, 8))
                    .font(egui::FontId::monospace(34.0))
                    .color(theme::ambient()),
            );
            ui.add_space(6.0);
        }
    }

    let s = app.sync.clone();
    ui.columns(3, |cols| {
        theme::stat(&mut cols[0], "Height", &s.as_ref().map(|s| units(s.height)).unwrap_or("—".into()), "");
        theme::stat(&mut cols[1], "UTXO coins", &s.as_ref().map(|s| units(s.num_coins as u64)).unwrap_or("—".into()), "");
        theme::stat(&mut cols[2], "Open commitments", &s.as_ref().map(|s| units(s.num_commitments as u64)).unwrap_or("—".into()), "");
    });
    ui.columns(3, |cols| {
        theme::stat(&mut cols[0], "Safe depth", &s.as_ref().map(|s| units(s.safe_depth)).unwrap_or("—".into()), "");
        theme::stat(&mut cols[1], "Block reward", &app.node.as_ref().map(|n| units(n.block_reward)).unwrap_or("—".into()), "units");
        theme::stat(&mut cols[2], "Last block", &s.as_ref().map(|s| fmt_dt(s.timestamp)).unwrap_or("—".into()), "utc");
    });

    let peer_count = s.as_ref().map(|s| s.peer_count).unwrap_or(0);
    theme::heading(ui, &format!("Peers ({peer_count})"));
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        match &app.node {
            Some(n) if !n.peers.is_empty() => {
                egui::ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
                    for p in &n.peers {
                        ui.label(theme::mono(p).size(11.5).color(theme::muted()));
                    }
                });
            }
            _ => theme::hint(
                ui,
                "No peers connected. The node keeps dialing bootstrap peers; check your \
                 network if this persists.",
            ),
        }
    });

    if let Some(url) = app.node.as_ref().and_then(|n| n.rpc_url.clone()) {
        theme::heading(ui, "Explorer");
        theme::hint(
            ui,
            &format!("Served by your own node at {url} — no third parties involved."),
        );
        if ui.button("Open explorer in browser").clicked() {
            ui.ctx().open_url(egui::OpenUrl::new_tab(url));
        }
    }
}
