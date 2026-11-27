//! Send: who, how much, a look before it goes, then how it is going.

use super::{big_button, card, field, gmds, label, quiet, send_step, who};
use crate::app::App;
use crate::bridge::Action;
use crate::theme;
use crate::views::mdw::{self as m};
use eframe::egui::{self, RichText, Ui};
use midstate_walletd::api::{SendProgress, SendStage};
use midstate_walletd::mwb_account::{max_send, send_fee};

/// Where a send is: being filled in, being looked over, or handed over and
/// followed.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Flow {
    #[default]
    Form,
    Review { to: String, amount: u64, mdw: bool },
    Sent { to: String, amount: u64, mdw: bool, at: u64 },
}

pub fn show(app: &mut App, ui: &mut Ui) {
    super::title(ui, "Send");
    match app.simple_flow.clone() {
        Flow::Form => form(app, ui),
        Flow::Review { to, amount, mdw } => review(app, ui, &to, amount, mdw),
        Flow::Sent { to, amount, mdw, at } => sent(app, ui, &to, amount, mdw, at),
    }
}

fn asset_switch(app: &mut App, ui: &mut Ui) {
    if super::has_mdw(app) {
        if theme::segmented(ui, &[super::UNIT, "MDW"], &mut app.simple_asset) {
            app.error.clear();
        }
        ui.add_space(14.0);
    } else {
        app.simple_asset = 0;
    }
}

fn form(app: &mut App, ui: &mut Ui) {
    asset_switch(app, ui);
    if app.simple_asset == 1 {
        return form_mdw(app, ui);
    }
    let ctx = ui.ctx().clone();
    let width = ui.available_width().min(560.0);

    label(ui, "To");
    let r = field(ui, &mut app.send_to, "name.mds or address", width);
    if r.changed() {
        let t = app.send_to.trim().to_string();
        app.addr_resolved = None;
        if t.len() == 72 || midstate_walletd::names::is_name(&t) || midstate_walletd::sendplan::is_address_list(&t) {
            app.addr_ok = None;
            app.go(&ctx, Action::ValidateAddress { addr: t });
        } else {
            app.addr_ok = if t.is_empty() { None } else { Some(false) };
            app.addr_reason = None;
        }
    }
    let to = app.send_to.trim().to_string();
    ui.add_space(2.0);
    match app.addr_ok {
        _ if to.is_empty() => {
            ui.label(" ");
        }
        _ if to.starts_with("mw1") => {
            quiet(ui, "That is an MDW address.");
        }
        None => {
            quiet(ui, "Checking…");
        }
        Some(true) => {
            let named = midstate_walletd::names::is_name(&to);
            let fresh = named
                && app
                    .payee_backing
                    .as_ref()
                    .is_some_and(|(n, b)| *n == to && b.as_ref().is_none_or(|b| b.tier.is_empty()));
            ui.horizontal(|ui| {
                ui.label(RichText::new(if named { format!("✓ {to}") } else { "✓ Valid address".into() }).size(13.0).color(theme::ink()));
                if fresh {
                    ui.label(RichText::new("· a new name: make sure it is the right one").size(13.0).color(theme::muted()));
                }
            });
        }
        Some(false) => {
            let why = if to.len() < 72 && !midstate_walletd::names::is_name(&to) {
                "Not an address or a name".to_string()
            } else {
                app.addr_reason.clone().unwrap_or_else(|| "That address has a typo".into())
            };
            ui.label(RichText::new(why).size(13.0).color(theme::muted()));
        }
    }

    ui.add_space(12.0);
    let spendable = app.balance.as_ref().map(|b| b.confirmed).unwrap_or(0);
    let fee = crate::views::send::estimated_fee(app);
    let most = spendable.saturating_sub(fee);
    ui.horizontal(|ui| {
        label(ui, "Amount");
        ui.add_space(8.0);
        quiet(ui, &format!("You have {}", gmds(spendable)));
        if most > 0 && ui.link(RichText::new("Max").size(13.0)).clicked() {
            app.simple_amount = super::amount_plain(most);
        }
    });
    ui.horizontal(|ui| {
        field(ui, &mut app.simple_amount, "0", 220.0);
        ui.label(RichText::new(super::UNIT).font(theme::font_medium(16.0)).color(theme::muted()));
    });
    let amount = super::parse_amount(&app.simple_amount);
    let too_much = amount.as_ref().is_ok_and(|v| v.saturating_add(fee) > spendable);
    match &amount {
        Err(e) if !e.is_empty() => quiet(ui, e),
        Ok(_) if too_much => quiet(ui, "More than you have"),
        _ => {
            ui.label(" ");
        }
    }

    ui.add_space(16.0);
    let syncing = app.sync.as_ref().map(|s| s.is_syncing).unwrap_or(true);
    let restoring = app.sync.as_ref().is_some_and(|s| s.restore_pass > 0);
    let ready = app.addr_ok == Some(true) && amount.is_ok() && !too_much && !syncing && !restoring;
    let text = if syncing {
        "Catching up…"
    } else if restoring {
        "Finding your coins…"
    } else {
        "Continue"
    };
    if big_button(ui, text, true, ready).clicked() {
        app.error.clear();
        app.simple_flow = Flow::Review { to, amount: amount.unwrap_or(0), mdw: false };
    }
}

fn form_mdw(app: &mut App, ui: &mut Ui) {
    let width = ui.available_width().min(560.0);
    let a = app.mwb_account.clone().unwrap_or_default();
    let values = a.spendable_coins.clone();
    let most = max_send(&values);

    label(ui, "To");
    field(ui, &mut app.mwb_send_to, "name.mds or mw1… address", width);
    let to = app.mwb_send_to.trim().to_string();
    let to_ok = !to.is_empty() && (midstate_walletd::mwb::check_address(&to).is_ok() || midstate_walletd::names::is_name(&to));
    ui.add_space(2.0);
    if to.is_empty() || to_ok {
        ui.label(" ");
    } else {
        quiet(ui, "Not an MDW address or a name");
    }

    ui.add_space(12.0);
    ui.horizontal(|ui| {
        label(ui, "Amount");
        ui.add_space(8.0);
        quiet(ui, &format!("You have {}", m::mdw(a.spendable)));
        if most > 0 && ui.link(RichText::new("Max").size(13.0)).clicked() {
            app.mwb_send_amount = m::plain(most).replace(',', "");
        }
    });
    ui.horizontal(|ui| {
        field(ui, &mut app.mwb_send_amount, "0", 220.0);
        ui.label(RichText::new("MDW").font(theme::font_medium(16.0)).color(theme::muted()));
    });
    let amount = m::parse_mdw(&app.mwb_send_amount);
    let fee = amount.as_ref().ok().map(|v| send_fee(&values, *v));
    match (&amount, &fee) {
        (Ok(_), Some(Err(_))) => quiet(ui, "More than you have"),
        (Err(e), _) if !e.is_empty() => quiet(ui, e),
        _ => {
            ui.label(" ");
        }
    }

    ui.add_space(16.0);
    let waiting = m::not_ready(app);
    let ready = to_ok && matches!(fee, Some(Ok(_))) && waiting.is_none();
    if big_button(ui, waiting.unwrap_or("Continue"), true, ready).clicked() {
        app.error.clear();
        app.simple_flow = Flow::Review { to, amount: amount.unwrap_or(0), mdw: true };
    }
}

fn review(app: &mut App, ui: &mut Ui, to: &str, amount: u64, mdw: bool) {
    let ctx = ui.ctx().clone();
    let what = if mdw { m::mdw(amount) } else { gmds(amount) };
    card(ui, |ui| {
        ui.add_space(4.0);
        ui.label(RichText::new(format!("Send {what}")).font(theme::font_medium(24.0)).color(theme::ink()));
        ui.add_space(4.0);
        if midstate_walletd::names::is_name(to) {
            ui.label(RichText::new(format!("to {to}")).size(16.0).color(theme::ink()));
            if let Some(a) = app.addr_resolved.as_ref().filter(|_| !mdw) {
                ui.add_space(2.0);
                ui.label(RichText::new(theme::grouped_hash(a, 24)).monospace().size(11.5).color(theme::muted()));
            }
        } else {
            ui.label(RichText::new("to").size(16.0).color(theme::ink()));
            ui.add(egui::Label::new(RichText::new(theme::grouped_hash(to, 24)).monospace().size(13.0).color(theme::ink())).wrap());
        }
        ui.add_space(14.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 10.0;
            let label = if app.busy { "Sending…" } else { "Send now" };
            if big_button(ui, label, true, !app.busy).clicked() {
                app.busy = true;
                app.error.clear();
                app.mwb_notice.clear();
                if mdw {
                    app.go(&ctx, Action::MwbSend { to: to.to_string(), amount });
                } else {
                    app.go(&ctx, Action::Send { to: to.to_string(), amount, private: false });
                }
                app.simple_flow = Flow::Sent { to: to.to_string(), amount, mdw, at: theme::now_secs() };
            }
            if big_button(ui, "Back", false, !app.busy).clicked() {
                app.simple_flow = Flow::Form;
            }
        });
    });
}

/// The send this flow handed over, once walletd reports it.
pub fn find_send<'a>(sends: &'a [SendProgress], to: &str, amount: u64, at: u64) -> Option<&'a SendProgress> {
    sends
        .iter()
        .filter(|s| s.to == to && s.amount == amount && s.updated_at + 5 >= at)
        .max_by_key(|s| s.updated_at)
}

fn sent(app: &mut App, ui: &mut Ui, to: &str, amount: u64, mdw: bool, at: u64) {
    let what = if mdw { m::mdw(amount) } else { gmds(amount) };
    let failed = !app.busy && !app.error.is_empty();
    let progress = if mdw { None } else { find_send(&app.sends, to, amount, at).cloned() };
    let (words, frac, done) = if failed {
        ("Didn't go through", 0.0, false)
    } else if mdw {
        if app.busy { ("Sending", 0.3, false) } else { ("Sent", 1.0, true) }
    } else {
        match &progress {
            Some(p) => {
                let (w, f) = send_step(p.stage);
                (w, f, p.stage == SendStage::Confirmed)
            }
            None => ("Preparing", 0.1, false),
        }
    };
    card(ui, |ui| {
        ui.add_space(4.0);
        let head = if done { format!("Sent {what}") } else if failed { format!("{what} not sent") } else { format!("Sending {what}") };
        ui.label(RichText::new(head).font(theme::font_medium(24.0)).color(theme::ink()));
        ui.label(RichText::new(format!("to {}", who(to))).size(16.0).color(theme::ink()));
        ui.add_space(14.0);
        if !failed {
            theme::progress_bar(ui, frac);
            ui.add_space(8.0);
        }
        if failed {
            let e = app.error.trim_start_matches("send MDW: ").trim_start_matches("send: ");
            let mut c = e.chars();
            let e = c.next().map(|f| f.to_uppercase().chain(c).collect::<String>()).unwrap_or_default();
            quiet(ui, &format!("Your money is still in your wallet. {e}"));
        } else {
            ui.horizontal(|ui| {
                ui.label(RichText::new(if done { "✓ Done" } else { words }).size(14.0).color(theme::ink()));
                if !done && !mdw {
                    ui.label(RichText::new("· about two minutes").size(14.0).color(theme::muted()));
                }
            });
        }
        if progress.as_ref().is_some_and(|p| p.stage == SendStage::Stalled) {
            ui.add_space(8.0);
            let p = progress.clone().unwrap();
            super::send_card(app, ui, &p);
        }
        ui.add_space(14.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 10.0;
            if failed {
                if big_button(ui, "Try again", true, true).clicked() {
                    app.error.clear();
                    app.simple_flow = Flow::Review { to: to.to_string(), amount, mdw };
                }
                if big_button(ui, "Back", false, true).clicked() {
                    app.error.clear();
                    app.simple_flow = Flow::Form;
                }
            } else if big_button(ui, if done { "Done" } else { "Send something else" }, done, true).clicked() {
                app.simple_flow = Flow::Form;
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(to: &str, amount: u64, at: u64, stage: SendStage) -> SendProgress {
        SendProgress { id: format!("{to}{at}"), stage, detail: String::new(), amount, fee: 61, to: to.into(), updated_at: at }
    }

    #[test]
    fn the_flow_follows_its_own_send() {
        let sends = vec![
            p("bob.mds", 5, 100, SendStage::Confirmed),
            p("bob.mds", 7, 200, SendStage::CommitPending),
            p("carol.mds", 7, 210, SendStage::Committing),
        ];
        assert_eq!(find_send(&sends, "bob.mds", 7, 198).map(|s| s.updated_at), Some(200));
        // An older send of the same amount to the same name is not this one.
        assert!(find_send(&sends, "bob.mds", 5, 198).is_none());
        assert!(find_send(&sends, "dave.mds", 7, 198).is_none());
        // Clocks a little apart still match.
        assert_eq!(find_send(&sends, "carol.mds", 7, 213).map(|s| s.updated_at), Some(210));
    }
}
