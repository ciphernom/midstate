//! The simple view: Home, Send, Receive, Activity, Messages, Names and
//! Settings, in plain words, with one unit for every amount.
//!
//! New wallets start here. The full app is one switch away (Settings → Show
//! all features) and is untouched by this view: every page of it still works
//! exactly as before, and "Show all features" brings all of it back.

pub mod activity;
pub mod home;
pub mod names;
pub mod receive;
pub mod send;
pub mod settings;

use crate::app::{App, Tab};
use crate::theme;
use eframe::egui::{self, Color32, RichText, Stroke, Ui};
use midstate_walletd::api::{SendProgress, SendStage};

/// The unit the simple view shows and takes every MDS amount in. 1 gMDS is
/// 2^30 base units: the block reward the chain started with.
pub const UNIT: &str = "gMDS";
const ONE: u64 = 1 << 30;

/// The pages of the simple view, in rail order.
pub const TABS: [Tab; 7] = [Tab::Dashboard, Tab::Send, Tab::Receive, Tab::History, Tab::Chat, Tab::Names, Tab::Settings];

/// What the simple view calls a page.
pub fn tab_name(t: Tab) -> &'static str {
    match t {
        Tab::Dashboard => "Home",
        Tab::History => "Activity",
        other => other.name(),
    }
}

// ── The choice of view ──────────────────────────────────────────────────────

fn view_pref_path() -> std::path::PathBuf {
    crate::app::profile().base.join("view")
}

/// Which view to open: what this person chose last, or — the first time —
/// the simple one, unless a wallet was already here before there was a
/// choice. Someone who has been using the full app keeps it.
pub fn starts_simple(saved: Option<&str>, had_wallet: bool) -> bool {
    match saved.map(str::trim) {
        Some("full") => false,
        Some("simple") => true,
        _ => !had_wallet,
    }
}

/// Read the choice, settling and saving it the first time.
pub fn load_view_pref() -> bool {
    let saved = std::fs::read_to_string(view_pref_path()).ok();
    let had_wallet = crate::app::profile().base.join("wallets").join("wallet.dat").exists();
    let simple = starts_simple(saved.as_deref(), had_wallet);
    if saved.is_none() {
        save_view_pref(simple);
    }
    simple
}

pub fn save_view_pref(simple: bool) {
    let p = view_pref_path();
    if let Some(dir) = p.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(p, if simple { "simple" } else { "full" });
}

/// Switch views, keeping the choice for next time.
pub fn set_simple(app: &mut App, simple: bool) {
    app.simple = simple;
    save_view_pref(simple);
    app.error.clear();
    if simple && !TABS.contains(&app.tab) {
        app.tab = Tab::Dashboard;
    }
}

// ── Amounts ─────────────────────────────────────────────────────────────────

/// An amount in gMDS for people: up to four decimals, and more only when a
/// small amount would otherwise read as nothing at all.
pub fn amount(v: u64) -> String {
    let s = decimals(v, 4, true);
    if s == "0" && v > 0 {
        return decimals(v, 9, true);
    }
    s
}

/// An amount as it goes back into a field: every decimal it has (up to
/// nine), no thousands separators, so it reads back as itself.
pub fn amount_plain(v: u64) -> String {
    decimals(v, 9, false)
}

/// `amount` with the unit.
pub fn gmds(v: u64) -> String {
    format!("{} {UNIT}", amount(v))
}

fn decimals(v: u64, places: u32, grouped: bool) -> String {
    let scale = 10u128.pow(places);
    let one = ONE as u128;
    let mut whole = (v / ONE) as u128;
    let mut frac = ((v % ONE) as u128 * scale + one / 2) / one;
    if frac >= scale {
        whole += 1;
        frac -= scale;
    }
    let mut out = if grouped { theme::units(whole as u64) } else { whole.to_string() };
    if frac > 0 {
        let f = format!("{frac:0width$}", width = places as usize);
        out.push('.');
        out.push_str(f.trim_end_matches('0'));
    }
    out
}

/// gMDS typed by a person, in base units. Decimals round to the nearest base
/// unit, about a billionth of a gMDS, so "0.1" means what it says. Spaces are
/// ignored. Commas: beside a dot, or more than one, they group thousands
/// (1,000.5 and 1,000,000); a single one is a decimal point (2,5) — unless
/// three digits follow it, when 1,234 could be either and is refused rather
/// than guessed.
pub fn parse_amount(text: &str) -> Result<u64, &'static str> {
    let mut t: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    if t.is_empty() {
        return Err("");
    }
    let commas = t.matches(',').count();
    if commas > 0 {
        let dot = t.find('.');
        if dot.is_some_and(|d| t.rfind(',').is_some_and(|c| c > d)) {
            return Err("That isn't a number");
        }
        if dot.is_some() || commas > 1 {
            t.retain(|c| c != ',');
        } else if t.split_once(',').is_some_and(|(_, after)| after.len() == 3) {
            return Err("Use a dot for decimals");
        } else {
            t = t.replace(',', ".");
        }
    }
    let (int_s, frac_s) = t.split_once('.').unwrap_or((&t, ""));
    if (int_s.is_empty() && frac_s.is_empty())
        || !int_s.chars().all(|c| c.is_ascii_digit())
        || !frac_s.chars().all(|c| c.is_ascii_digit())
    {
        return Err("That isn't a number");
    }
    if frac_s.len() > 9 {
        return Err("Use at most 9 decimals");
    }
    let whole: u128 = if int_s.is_empty() { 0 } else { int_s.parse().map_err(|_| "That is too much")? };
    let frac: u128 = if frac_s.is_empty() { 0 } else { frac_s.parse().map_err(|_| "That isn't a number")? };
    let scale = 10u128.pow(frac_s.len() as u32);
    let one = ONE as u128;
    let units = whole
        .checked_mul(one)
        .and_then(|w| w.checked_add((frac * one + scale / 2) / scale))
        .ok_or("That is too much")?;
    if units == 0 {
        return Err("Enter an amount");
    }
    u64::try_from(units).map_err(|_| "That is too much")
}

// ── Words ───────────────────────────────────────────────────────────────────

/// Who an amount went to, in as few characters as say it: a name as it is,
/// an address by its ends.
pub fn who(to: &str) -> String {
    let t = to.trim();
    if midstate_walletd::names::is_name(t) || t.chars().count() <= 14 {
        t.to_string()
    } else {
        theme::short_hex(t, 4)
    }
}

/// A send's stage in plain words, with how far along it is.
pub fn send_step(stage: SendStage) -> (&'static str, f32) {
    match stage {
        SendStage::Committing => ("Preparing", 0.15),
        SendStage::CommitPending => ("Waiting for the network", 0.4),
        SendStage::WaitingReveal => ("Waiting for the network", 0.6),
        SendStage::RevealPending => ("Finishing", 0.85),
        SendStage::Confirmed => ("Sent", 1.0),
        SendStage::Stalled => ("Taking longer than usual", 0.4),
        SendStage::Failed => ("Didn't go through", 0.0),
    }
}

/// Sends still under way, or stuck and waiting for a decision.
pub fn active_sends(app: &App) -> Vec<SendProgress> {
    app.sends
        .iter()
        .filter(|s| !matches!(s.stage, SendStage::Confirmed | SendStage::Failed))
        .cloned()
        .collect()
}

/// The line at the foot of the rail: whether this wallet can reach the
/// network, and how far it has to catch up.
pub fn connection(app: &App) -> (bool, String) {
    match &app.sync {
        None => (false, "Starting…".into()),
        Some(s) if s.peer_count == 0 => (false, "Offline".into()),
        Some(s) if s.is_syncing => {
            let pct = app.sync_fraction().map(|f| (f * 100.0).floor() as u32).unwrap_or(0);
            (true, format!("Catching up · {pct}%"))
        }
        Some(s) if s.restore_pass > 0 => (true, "Finding your coins…".into()),
        Some(_) => (true, "Connected".into()),
    }
}

/// The MDW account, when this wallet has one turned on.
pub fn has_mdw(app: &App) -> bool {
    app.mwb_account.as_ref().is_some_and(|a| a.enabled)
}

// ── Widgets ─────────────────────────────────────────────────────────────────

pub fn title(ui: &mut Ui, text: &str) {
    ui.add_space(10.0);
    ui.label(RichText::new(text).font(theme::font_medium(24.0)).color(theme::ink()));
    ui.add_space(14.0);
}

/// A section label above a group.
pub fn label(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).font(theme::font_medium(14.0)).color(theme::ink()));
    ui.add_space(4.0);
}

/// Text that says something quietly.
pub fn quiet(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).size(13.0).color(theme::muted()));
}

/// A large button: ink-filled for the thing to do, outlined otherwise.
pub fn big_button(ui: &mut Ui, text: &str, primary: bool, enabled: bool) -> egui::Response {
    let ink = if primary { theme::on_ink() } else { theme::ink() };
    let b = egui::Button::new(RichText::new(text).font(theme::font_medium(15.0)).color(ink))
        .fill(if primary { theme::ink() } else { Color32::TRANSPARENT })
        .stroke(Stroke::new(1.0, if primary { theme::ink() } else { theme::border2() }))
        .min_size(egui::vec2(160.0, 46.0));
    // In a row, a button's text sits in its middle; in a column it would
    // follow the column's alignment, so give it a row of its own.
    if ui.layout().is_horizontal() {
        ui.add_enabled(enabled, b)
    } else {
        ui.horizontal(|ui| ui.add_enabled(enabled, b)).inner
    }
}

/// A smaller outlined button.
pub fn button(ui: &mut Ui, text: &str, enabled: bool) -> egui::Response {
    let b = egui::Button::new(RichText::new(text).size(13.0).color(theme::ink()))
        .fill(Color32::TRANSPARENT)
        .stroke(Stroke::new(1.0, theme::border2()))
        .min_size(egui::vec2(0.0, 30.0));
    ui.add_enabled(enabled, b)
}

/// A text field at the simple view's size.
pub fn field(ui: &mut Ui, text: &mut String, hint: &str, width: f32) -> egui::Response {
    ui.add(
        egui::TextEdit::singleline(text)
            .hint_text(hint)
            .font(egui::FontId::proportional(16.0))
            .margin(egui::Margin::symmetric(10, 9))
            .desired_width(width),
    )
}

/// One page in the rail: full width, text on the left, filled when open.
pub fn rail_item(ui: &mut Ui, selected: bool, text: &str) -> egui::Response {
    let (rect, r) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 36.0), egui::Sense::click());
    let fill = if selected {
        theme::highlight()
    } else if r.hovered() {
        theme::panel()
    } else {
        Color32::TRANSPARENT
    };
    ui.painter().rect_filled(rect, egui::CornerRadius::ZERO, fill);
    let colour = if selected { theme::ink() } else { theme::bright() };
    let font = if selected { theme::font_medium(15.0) } else { egui::FontId::proportional(15.0) };
    ui.painter().text(rect.left_center() + egui::vec2(12.0, 0.0), egui::Align2::LEFT_CENTER, text, font, colour);
    r.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// A plain card.
pub fn card<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    theme::panel_frame()
        .inner_margin(egui::Margin::symmetric(18, 16))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui)
        })
        .inner
}

/// A send under way, as one card: what, to whom, how far, and the two
/// choices a stuck one needs.
pub fn send_card(app: &mut App, ui: &mut Ui, s: &SendProgress) {
    let ctx = ui.ctx().clone();
    let (words, frac) = send_step(s.stage);
    card(ui, |ui| {
        ui.horizontal(|ui| {
            let what = if s.to.is_empty() {
                format!("Sending {}", gmds(s.amount))
            } else {
                format!("Sending {} to {}", gmds(s.amount), who(&s.to))
            };
            ui.label(RichText::new(what).size(15.0).color(theme::ink()));
            theme::right_aligned(ui, |ui| {
                ui.label(RichText::new(words).size(13.0).color(theme::muted()));
            });
        });
        ui.add_space(8.0);
        theme::progress_bar(ui, frac);
        if s.stage == SendStage::Stalled {
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if button(ui, "Try again", true).clicked() {
                    app.go(&ctx, crate::bridge::Action::RetrySend { id: s.id.clone() });
                }
                if button(ui, "Cancel", true).on_hover_text("Stop this send and keep the money").clicked() {
                    app.go(&ctx, crate::bridge::Action::AbandonSend { id: s.id.clone() });
                }
            });
        }
    });
    ui.add_space(8.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amounts_read_in_gmds() {
        assert_eq!(amount(0), "0");
        assert_eq!(amount(ONE), "1");
        assert_eq!(amount(37 * ONE + ONE / 2), "37.5");
        assert_eq!(amount(ONE / 4), "0.25");
        assert_eq!(amount(1_234 * ONE), "1,234");
        // Four decimals, rounded: 1/3 gMDS.
        assert_eq!(amount(ONE / 3), "0.3333");
        // Rounding up into the next whole number.
        assert_eq!(amount(2 * ONE - 1), "2");
        // A fee of 61 units is not nothing.
        assert_eq!(amount(61), "0.000000057");
        assert_eq!(gmds(ONE / 2), "0.5 gMDS");
    }

    #[test]
    fn typed_amounts_become_units() {
        assert_eq!(parse_amount("1"), Ok(ONE));
        assert_eq!(parse_amount(" 2.5 "), Ok(2 * ONE + ONE / 2));
        assert_eq!(parse_amount(".5"), Ok(ONE / 2));
        assert_eq!(parse_amount("2,5"), Ok(2 * ONE + ONE / 2), "a decimal comma");
        assert_eq!(parse_amount("0,25"), Ok(ONE / 4));
        assert_eq!(parse_amount("1,000.5"), Ok(1_000 * ONE + ONE / 2), "a thousands comma");
        assert_eq!(parse_amount("1,000,000"), Ok(1_000_000 * ONE));
        assert_eq!(parse_amount("1,234"), Err("Use a dot for decimals"), "a thousand, or one and a bit?");
        assert_eq!(parse_amount("1.5,0"), Err("That isn't a number"));
        assert_eq!(parse_amount("1 000"), Ok(1_000 * ONE));
        // 0.1 gMDS is not a whole number of units: it rounds to the nearest.
        assert_eq!(parse_amount("0.1"), Ok(107_374_182));
        assert_eq!(parse_amount("0.000000001"), Ok(1));
        assert_eq!(parse_amount(""), Err(""));
        assert_eq!(parse_amount("0"), Err("Enter an amount"));
        assert_eq!(parse_amount("0.0000000001"), Err("Use at most 9 decimals"));
        assert_eq!(parse_amount("1.2.3"), Err("That isn't a number"));
        assert_eq!(parse_amount("abc"), Err("That isn't a number"));
        assert_eq!(parse_amount("-1"), Err("That isn't a number"));
        assert_eq!(parse_amount("99999999999999999999"), Err("That is too much"));
        assert_eq!(parse_amount("17179869184"), Err("That is too much"), "2^34 gMDS overflows u64");
        // What is typed is what is shown back, and Max's figure reads back
        // as itself.
        for t in ["1", "2.5", "0.25", "37.5", "1,234.5"] {
            assert_eq!(amount(parse_amount(t).unwrap()), t);
        }
        for v in [1, 61, ONE / 3, 37 * ONE + ONE / 2, 40_265_318_339, u64::MAX] {
            let back = parse_amount(&amount_plain(v)).unwrap();
            assert!(back.abs_diff(v) <= 1, "{v} came back as {back}");
        }
    }

    #[test]
    fn the_first_view_depends_on_who_is_opening_it() {
        assert!(starts_simple(None, false), "a new install starts simple");
        assert!(!starts_simple(None, true), "a wallet from before keeps the full app");
        assert!(starts_simple(Some("simple\n"), true));
        assert!(!starts_simple(Some("full"), false));
        assert!(starts_simple(Some("something else"), false));
    }

    #[test]
    fn recipients_read_short() {
        assert_eq!(who("bob.mds"), "bob.mds");
        assert_eq!(who(&"7f3a".repeat(18)), "7f3a…7f3a");
        assert_eq!(who(" carol.mds "), "carol.mds");
    }

    #[test]
    fn send_stages_in_plain_words() {
        assert_eq!(send_step(SendStage::Confirmed), ("Sent", 1.0));
        assert_eq!(send_step(SendStage::Failed).0, "Didn't go through");
        let order = [SendStage::Committing, SendStage::CommitPending, SendStage::WaitingReveal, SendStage::RevealPending, SendStage::Confirmed];
        let fr: Vec<f32> = order.iter().map(|s| send_step(*s).1).collect();
        assert!(fr.windows(2).all(|w| w[0] < w[1]), "progress only moves forward: {fr:?}");
    }
}
