use crate::app::App;
use crate::theme::{self, fmt_dt, short_hex, units};
use eframe::egui::{self, RichText, Ui};

pub fn show(app: &mut App, ui: &mut Ui) {
    theme::heading(ui, "History");
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        if app.history.is_empty() {
            theme::hint(ui, "Nothing recorded yet. Received coins and completed sends appear here.");
            return;
        }
        egui::Grid::new("history").num_columns(5).spacing([22.0, 8.0]).striped(true).show(ui, |ui| {
            for c in ["When", "Type", "Amount", "Fee", "Coins"] {
                ui.label(RichText::new(c.to_uppercase()).font(egui::FontId::monospace(10.0)).color(theme::muted()));
            }
            ui.end_row();
            for h in &app.history {
                ui.label(theme::mono(fmt_dt(h.timestamp)).color(theme::muted()).size(12.0));
                let incoming = h.kind == "received" || h.kind == "coinbase";
                theme::badge(ui, &h.kind, if incoming { theme::green() } else { theme::muted() });
                ui.label(theme::mono(if h.amount > 0 { units(h.amount) } else { "—".into() }));
                ui.label(theme::mono(if h.fee > 0 { units(h.fee) } else { "—".into() }).color(theme::muted()));
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 1.0;
                    if !h.inputs.is_empty() {
                        let extra = if h.inputs.len() > 2 { format!(" +{}", h.inputs.len() - 2) } else { String::new() };
                        let ids: Vec<String> = h.inputs.iter().take(2).map(|x| short_hex(x, 5)).collect();
                        ui.label(theme::mono(format!("in: {}{}", ids.join(" "), extra)).size(10.5).color(theme::faint()));
                    }
                    if !h.outputs.is_empty() {
                        let extra = if h.outputs.len() > 2 { format!(" +{}", h.outputs.len() - 2) } else { String::new() };
                        let ids: Vec<String> = h.outputs.iter().take(2).map(|x| short_hex(x, 5)).collect();
                        ui.label(theme::mono(format!("out: {}{}", ids.join(" "), extra)).size(10.5).color(theme::faint()));
                    }
                });
                ui.end_row();
            }
        });
    });
    theme::hint(
        ui,
        "Amounts are best-effort: this chain stores bare coin ids, so value is shown where \
         the wallet still holds the matching coins.",
    );
}
