//! MDW, midwimble's coin, as every tab shows it: amounts, the account (or
//! the form that adds it), whether it can spend yet, and links into the two
//! block explorers this machine serves: midstate's on its node's RPC, and
//! midwimble's beside the embedded midwimble node.

use crate::app::App;
use crate::bridge::Action;
use crate::theme::{self, short_hex};
use eframe::egui::{self, FontId, RichText, TextEdit, Ui};

pub const COIN: u64 = 100_000_000;

/// "1.5 MDW". Midwimble amounts have eight decimals.
pub fn mdw(v: u64) -> String {
    format!("{} MDW", plain(v))
}

/// "1.5": an amount as the MDW fields take it.
pub fn plain(v: u64) -> String {
    let (whole, frac) = (v / COIN, v % COIN);
    if frac == 0 {
        return theme::units(whole);
    }
    let f = format!("{frac:08}");
    format!("{}.{}", theme::units(whole), f.trim_end_matches('0'))
}

/// "0.25" → 25,000,000 units; exact, or refused. Thousands separators, as
/// [`plain`] prints them, are accepted.
pub fn parse_mdw(text: &str) -> Result<u64, String> {
    let t: String = text.trim().chars().filter(|c| *c != ',').collect();
    if t.is_empty() {
        return Err(String::new());
    }
    let (i, f) = t.split_once('.').unwrap_or((&t, ""));
    if (i.is_empty() && f.is_empty()) || !i.chars().all(|c| c.is_ascii_digit()) || !f.chars().all(|c| c.is_ascii_digit()) {
        return Err("not a number".into());
    }
    if f.len() > 8 {
        return Err("MDW has eight decimal places".into());
    }
    let whole: u64 = if i.is_empty() { 0 } else { i.parse().map_err(|_| "amount is too large".to_string())? };
    let frac: u64 = if f.is_empty() { 0 } else { format!("{f:0<8}").parse().map_err(|_| "not a number".to_string())? };
    let v = whole.checked_mul(COIN).and_then(|w| w.checked_add(frac)).ok_or("amount is too large")?;
    if v == 0 {
        return Err("amount must be positive".into());
    }
    Ok(v)
}

/// Opens `url` in the system's browser.
pub fn open_url(url: &str) {
    #[cfg(target_os = "linux")]
    let cmd = "xdg-open";
    #[cfg(target_os = "macos")]
    let cmd = "open";
    #[cfg(target_os = "windows")]
    let cmd = "explorer";
    if let Err(e) = std::process::Command::new(cmd).arg(url).spawn() {
        tracing::warn!("could not open {url}: {e}");
    }
}

/// Midwimble's explorer, as the embedded node serves it.
pub fn mwb_explorer(app: &App) -> Option<String> {
    app.mwb_account
        .as_ref()
        .and_then(|a| a.node.as_ref())
        .and_then(|n| n.explorer.clone())
        .or_else(|| app.mwb_node.as_ref().and_then(|n| n.explorer.clone()))
}

/// Midstate's explorer: the page its node's RPC serves.
pub fn mds_explorer(app: &App) -> Option<String> {
    app.node.as_ref().and_then(|n| n.rpc_url.clone())
}

/// A page of midwimble's explorer: `kind` is "block", "output" or
/// "kernel" (by excess).
pub fn mwb_page(base: &str, kind: &str, id: &str) -> String {
    format!("{}/#/{kind}/{id}", base.trim_end_matches('/'))
}

/// Midstate's explorer, searching for `q` (a height, coin id or hash).
pub fn mds_page(base: &str, q: &str) -> String {
    format!("{}/?q={q}", base.trim_end_matches('/'))
}

/// A small button that opens `url`; nothing at all without one.
pub fn explorer_button(ui: &mut Ui, label: &str, url: Option<String>, hover: &str) {
    let Some(url) = url else { return };
    if ui.add(egui::Button::new(RichText::new(label).size(10.5))).on_hover_text(hover).clicked() {
        open_url(&url);
    }
}

/// Why MDW cannot be spent right now, if it cannot.
pub fn not_ready(app: &App) -> Option<&'static str> {
    let Some(a) = app.mwb_account.as_ref() else { return Some("Reading the account…") };
    let Some(n) = a.node.as_ref() else { return Some("midwimble is not running") };
    if n.syncing {
        Some("Waiting for midwimble sync…")
    } else if n.peers == 0 {
        Some("Waiting for midwimble peers…")
    } else if a.scanned < n.height {
        Some("Finding your coins…")
    } else {
        None
    }
}

/// The midwimble node in one line, and the account's scan when it lags.
pub fn node_line(app: &App, ui: &mut Ui) {
    let Some(a) = app.mwb_account.as_ref() else { return };
    let Some(n) = a.node.as_ref() else { return };
    ui.horizontal(|ui| {
        theme::dot(ui, n.peers > 0);
        ui.label(RichText::new("midwimble").font(FontId::monospace(9.5)).color(theme::muted()));
        ui.label(theme::mono(format!("height {}", theme::units(n.height))).size(11.0));
        ui.label(theme::mono(format!("{} peer(s)", n.peers)).size(11.0).color(theme::muted()));
        if n.syncing {
            theme::badge(ui, "syncing", theme::muted());
        }
    });
    if a.enabled && a.scanned < n.height {
        ui.label(
            RichText::new(format!(
                "Finding your coins: midwimble block {} of {}.",
                theme::units(a.scanned),
                theme::units(n.height)
            ))
            .size(11.5)
            .color(theme::muted()),
        );
    }
}

/// Whether this wallet's MDW account is there to use. When it is not, says
/// why: the midwimble node did not start, or the wallet predates the
/// account and needs its recovery phrase once to add it.
pub fn account_gate(app: &mut App, ui: &mut Ui, ctx: &egui::Context) -> bool {
    let Some(a) = app.mwb_account.clone() else {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            theme::hint(ui, "Reading your MDW account…");
        });
        return false;
    };
    if a.node.is_none() {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("The midwimble node is not running.").font(theme::font_medium(14.0)).color(theme::ink()));
            theme::hint(ui, "It failed to start, so MDW is unavailable this session; the log says why. Restart the app to try again.");
        });
        return false;
    }
    if a.enabled {
        return true;
    }
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        node_line(app, ui);
        ui.add_space(4.0);
        ui.label(RichText::new("This wallet has no MDW account yet.").font(theme::font_medium(14.0)).color(theme::ink()));
        theme::hint(
            ui,
            "It was made before the wallet held MDW. The account comes from your recovery phrase, \
             derived as midwimble's own wallets derive it, so the same words open it there too. \
             Enter the phrase once to add it: it is checked against this wallet and not kept.",
        );
        ui.add(
            TextEdit::multiline(&mut app.mwb_phrase)
                .hint_text("recovery phrase")
                .desired_rows(2)
                .desired_width(f32::INFINITY),
        );
        theme::right_aligned(ui, |ui| {
            let can = !app.busy && app.mwb_phrase.split_whitespace().count() >= 12;
            if ui.add_enabled(can, egui::Button::new("Add MDW account")).clicked() {
                app.busy = true;
                app.error.clear();
                let phrase = std::mem::take(&mut app.mwb_phrase);
                app.go(ctx, Action::MwbEnable { phrase });
            }
        });
        if !app.error.is_empty() {
            ui.label(RichText::new(&app.error).size(12.0).color(theme::red()));
        }
    });
    false
}

/// What an MDW history entry is, in a word.
pub fn kind_label(kind: &str) -> &str {
    match kind {
        "swap" => "bought",
        "swap lock" => "into swap",
        other => other,
    }
}

/// Where an MDW history entry stands.
pub fn status_text(h: &midstate_walletd::api::MwbHistoryView) -> String {
    match (h.status.as_str(), h.height) {
        ("pending", _) => "waiting for a block".into(),
        ("failed", Some(b)) => format!("failed · block {}", theme::units(b)),
        ("maturing", Some(b)) => format!("maturing · block {}", theme::units(b)),
        (_, Some(b)) => format!("block {}", theme::units(b)),
        (s, None) => s.to_string(),
    }
}

/// How settled an MDW entry in a block is: of its confirmations, those that
/// counted in full in fork choice ("3 at full weight"). Only they take bonds
/// to undo.
pub fn settled_text(h: &midstate_walletd::api::MwbHistoryView) -> Option<String> {
    Some(match h.full_weight_confirmations? {
        0 => "none at full weight yet".into(),
        n => format!("{} at full weight", theme::units(n)),
    })
}

/// [`status_text`] and [`settled_text`] on one line.
pub fn status_line(h: &midstate_walletd::api::MwbHistoryView) -> String {
    match settled_text(h) {
        Some(s) => format!("{} · {s}", status_text(h)),
        None => status_text(h),
    }
}

/// What [`settled_text`] means, for its hover.
pub const FULL_WEIGHT_HINT: &str = "Confirmations at full weight: blocks from this one to the tip that \
    counted in full in midwimble's fork choice. Only they take bonds to undo. A block over its \
    signer's cap adds next to nothing, and one full-weight block replaces a run of them, so an \
    entry with none at full weight can still be reorganised away.";

/// The explorer page for an MDW history entry: its kernel, else its coin,
/// else its block; a payment still waiting, the mempool it waits in.
pub fn history_page(base: &str, h: &midstate_walletd::api::MwbHistoryView) -> Option<String> {
    if h.status == "pending" {
        return Some(format!("{}/#/mempool", base.trim_end_matches('/')));
    }
    if let Some(k) = &h.kernel {
        if h.status == "confirmed" {
            return Some(mwb_page(base, "kernel", k));
        }
    }
    if let Some(o) = &h.output {
        return Some(mwb_page(base, "output", o));
    }
    h.height.map(|b| mwb_page(base, "block", &b.to_string()))
}

/// A short form of a midwimble address.
pub fn short_addr(a: &str) -> String {
    short_hex(a, 12)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mdw_amounts_parse_exactly_and_print_back() {
        assert_eq!(parse_mdw("1"), Ok(100_000_000));
        assert_eq!(parse_mdw("0.25"), Ok(25_000_000));
        assert_eq!(parse_mdw(".00000001"), Ok(1));
        assert_eq!(parse_mdw("1,234.5"), Ok(123_450_000_000), "as plain() prints it");
        assert!(parse_mdw("0.000000001").is_err(), "nine decimals");
        assert!(parse_mdw("0").is_err());
        assert!(parse_mdw("1.2.3").is_err());
        assert_eq!(mdw(25_000_000), "0.25 MDW");
        assert_eq!(mdw(1_200_000_000), "12 MDW");
        assert_eq!(parse_mdw(&plain(123_456_789_012)), Ok(123_456_789_012));
    }

    #[test]
    fn explorer_pages() {
        assert_eq!(mwb_page("http://127.0.0.1:9435/", "kernel", "ab"), "http://127.0.0.1:9435/#/kernel/ab");
        assert_eq!(mds_page("http://127.0.0.1:8545", "42"), "http://127.0.0.1:8545/?q=42");
    }
}
