//! Activity: every payment in and out, newest first, grouped by day.

use super::{active_sends, card, gmds, quiet, send_card, who};
use crate::app::App;
use crate::theme;
use eframe::egui::{self, RichText, Ui};
use midstate_walletd::api::{HistoryView, MwbHistoryView};

/// One line of activity.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub what: String,
    /// "+2 gMDS", "−1 gMDS"; empty when the amount is not known.
    pub amount: String,
    pub incoming: bool,
    pub time: u64,
    /// Not final yet (an MDW payment waiting for a block).
    pub pending: bool,
}

/// An MDS history entry in plain words, or nothing for bookkeeping that
/// neither brought money in nor sent it anywhere (tidying coins together).
pub fn mds_row(h: &HistoryView) -> Option<Row> {
    let sent = |h: &HistoryView| {
        let what = match h.to.as_deref() {
            Some(to) if !to.is_empty() => format!("Sent to {}", who(to)),
            _ => "Sent".to_string(),
        };
        let amount = h.sent.map(|v| format!("−{}", gmds(v))).unwrap_or_default();
        Row { what, amount, incoming: false, time: h.timestamp, pending: false }
    };
    let came = |what: &str, h: &HistoryView| Row {
        what: what.into(),
        amount: format!("+{}", gmds(h.amount)),
        incoming: true,
        time: h.timestamp,
        pending: false,
    };
    match h.kind.as_str() {
        "received" => Some(came("Received", h)),
        "coinbase" => Some(came("Mined", h)),
        "sent" => Some(sent(h)),
        "mixed" if h.sent.is_some() => Some(sent(h)),
        "mixed" if h.amount > 0 => Some(came("Received", h)),
        _ => None,
    }
}

/// An MDW history entry in plain words; failed ones are left out.
pub fn mdw_row(h: &MwbHistoryView) -> Option<Row> {
    if h.status == "failed" {
        return None;
    }
    let m = crate::views::mdw::mdw(h.amount);
    let what = match (h.incoming, h.kind.as_str()) {
        (true, "mined" | "coinbase") => "Mined".to_string(),
        (true, _) => "Received".to_string(),
        (false, _) => match h.to.as_deref() {
            Some(to) => format!("Sent to {}", who(to)),
            None => "Sent".to_string(),
        },
    };
    Some(Row {
        what,
        amount: if h.incoming { format!("+{m}") } else { format!("−{m}") },
        incoming: h.incoming,
        // One still waiting sorts as now.
        time: if h.status == "pending" { theme::now_secs() } else { h.time },
        pending: h.status == "pending",
    })
}

/// Both histories in plain words, newest first.
pub fn rows(app: &App) -> Vec<Row> {
    let mut v: Vec<Row> = app.history.iter().filter_map(mds_row).collect();
    if super::has_mdw(app) {
        v.extend(app.mwb_history.iter().filter_map(mdw_row));
    }
    v.sort_by(|a, b| b.time.cmp(&a.time));
    v
}

/// Draw rows; with `days`, under a heading for each day.
pub fn list(ui: &mut Ui, rows: &[Row], days: bool) {
    let now = theme::now_secs();
    let mut last_day = String::new();
    for (i, r) in rows.iter().enumerate() {
        if days {
            let day = if r.time > 0 { theme::day_label(r.time, now) } else { "Earlier".into() };
            if day != last_day {
                ui.add_space(if i == 0 { 0.0 } else { 12.0 });
                ui.label(RichText::new(&day).size(12.0).color(theme::muted()));
                ui.add_space(4.0);
                last_day = day;
            }
        } else if i > 0 {
            ui.separator();
        }
        ui.horizontal(|ui| {
            ui.set_min_height(28.0);
            ui.label(RichText::new(&r.what).size(15.0).color(theme::ink()));
            if r.pending {
                ui.label(RichText::new("waiting").size(12.0).color(theme::muted()));
            }
            theme::right_aligned(ui, |ui| {
                let when = if r.time == 0 {
                    String::new()
                } else if days {
                    theme::clock(r.time)
                } else {
                    theme::ago(r.time)
                };
                ui.add_sized([70.0, 20.0], egui::Label::new(RichText::new(when).size(12.5).color(theme::muted())));
                let colour = if r.incoming { theme::ink() } else { theme::bright() };
                ui.label(RichText::new(&r.amount).font(theme::font_mono_medium(15.0)).color(colour));
            });
        });
    }
}

pub fn show(app: &mut App, ui: &mut Ui) {
    super::title(ui, "Activity");
    let sends = active_sends(app);
    for s in &sends {
        send_card(app, ui, s);
    }
    if !sends.is_empty() {
        ui.add_space(8.0);
    }
    let rows = rows(app);
    if rows.is_empty() {
        card(ui, |ui| {
            ui.add_space(6.0);
            ui.label(RichText::new("Nothing here yet").size(15.0).color(theme::ink()));
            quiet(ui, "Payments in and out will show up here.");
            ui.add_space(6.0);
        });
        return;
    }
    card(ui, |ui| list(ui, &rows, true));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(kind: &str, amount: u64, sent: Option<u64>, to: Option<&str>) -> HistoryView {
        HistoryView {
            kind: kind.into(),
            fee: 61,
            timestamp: 1_795_422_060,
            inputs: vec![],
            outputs: vec![],
            amount,
            sent,
            to: to.map(String::from),
            change: 0,
            n_in: 1,
            n_out: 2,
            ours_out: 1,
        }
    }

    #[test]
    fn history_reads_in_plain_words() {
        let g = 1u64 << 30;
        let r = mds_row(&h("received", 2 * g, None, None)).unwrap();
        assert_eq!((r.what.as_str(), r.amount.as_str(), r.incoming), ("Received", "+2 gMDS", true));
        let r = mds_row(&h("coinbase", g / 2, None, None)).unwrap();
        assert_eq!((r.what.as_str(), r.amount.as_str()), ("Mined", "+0.5 gMDS"));
        let r = mds_row(&h("sent", 0, Some(g), Some("bob.mds"))).unwrap();
        assert_eq!((r.what.as_str(), r.amount.as_str(), r.incoming), ("Sent to bob.mds", "−1 gMDS", false));
        let r = mds_row(&h("sent", 0, Some(3 * g), Some(&"7f3a".repeat(18)))).unwrap();
        assert_eq!(r.what, "Sent to 7f3a…7f3a");
        // A send from before amounts were recorded: no amount, not a zero.
        let r = mds_row(&h("sent", 0, None, None)).unwrap();
        assert_eq!((r.what.as_str(), r.amount.as_str()), ("Sent", ""));
        assert_eq!(mds_row(&h("mixed", 0, Some(g), None)).unwrap().what, "Sent");
        assert_eq!(mds_row(&h("mixed", g, None, None)).unwrap().what, "Received");
        // Tidying coins together moves nothing anywhere.
        assert_eq!(mds_row(&h("consolidate", 0, None, None)), None);
        assert_eq!(mds_row(&h("mixed", 0, None, None)), None);
    }
}
