use crate::app::App;
use crate::bridge::Action;
use crate::theme::{self, ago, send_timeline, short_hex, units};
use crate::views::mdw::{self as m, mdw};
use eframe::egui::{self, RichText, TextEdit, Ui};
use midstate_walletd::api::SendStage;
use midstate_walletd::mwb_account::{max_send, send_fee};

/// The coins this tab sends, in the order of its switch.
pub const ASSETS: [&str; 2] = ["MDS · midstate", "MDW · midwimble"];

pub fn show(app: &mut App, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    theme::heading(ui, "Send");
    if theme::segmented(ui, &ASSETS, &mut app.send_asset) {
        app.error.clear();
    }
    ui.add_space(6.0);
    if app.send_asset == 1 {
        send_mdw(app, ui, &ctx);
        return;
    }

    let syncing = app.sync.as_ref().map(|s| s.is_syncing).unwrap_or(true);
    // A restored wallet signs nothing until its restore scan has found every
    // key the phrase used before (walletd refuses too; this just says so).
    let restoring = app.sync.as_ref().map(|s| s.restore_pass > 0).unwrap_or(false);

    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());

        theme::hint(ui, "Recipient address");
        let r = ui.add(
            TextEdit::singleline(&mut app.send_to)
                .hint_text("72-character address")
                .font(egui::TextStyle::Monospace)
                .desired_width(f32::INFINITY),
        );
        if r.changed() {
            let t = app.send_to.trim().to_string();
            if t.len() == 72 {
                app.addr_ok = None;
                app.go(&ctx, Action::ValidateAddress { addr: t });
            } else {
                app.addr_ok = if t.is_empty() { None } else { Some(false) };
            }
        }
        if app.send_to.trim().starts_with("mw1") {
            // A midwimble address pasted here: the right form is one click away.
            ui.horizontal(|ui| {
                ui.label(RichText::new("That is a midwimble address: MDW goes there, not MDS.").size(12.0).color(theme::bright()));
                if ui.button(RichText::new("Send MDW to it").size(11.0)).clicked() {
                    app.mwb_send_to = std::mem::take(&mut app.send_to);
                    app.addr_ok = None;
                    app.send_asset = 1;
                }
            });
        } else if app.addr_ok == Some(false) && !app.send_to.trim().is_empty() {
            // Could be a typo, or a one-time key that has already signed —
            // walletd says which, and they call for different responses.
            ui.label(
                RichText::new(
                    app.addr_reason
                        .as_deref()
                        .unwrap_or("Address checksum does not match — check for typos."),
                )
                .color(theme::red())
                .size(12.0),
            );
        }

        // Ask a peer for a fresh destination over the chat bus. Their reply is
        // signed against the identity key typed here, so an onlooker cannot
        // answer in their place.
        egui::CollapsingHeader::new("Ask a peer for an address")
            .id_salt("ask_addr")
            .default_open(app.addr_ok == Some(false) && app.addr_reason.is_some())
            .show(ui, |ui| {
                ui.add(
                    TextEdit::singleline(&mut app.ask_peer)
                        .hint_text("their identity key (64 hex characters)")
                        .font(egui::TextStyle::Monospace)
                        .desired_width(f32::INFINITY),
                );
                ui.horizontal(|ui| {
                    let ok = app.ask_peer.trim().len() == 64
                        && app.ask_peer.trim().chars().all(|c| c.is_ascii_hexdigit());
                    if ui
                        .add_enabled(!app.busy && ok, egui::Button::new("Request address"))
                        .clicked()
                    {
                        app.busy = true;
                        app.error.clear();
                        app.go(&ctx, Action::RequestAddress { peer: app.ask_peer.trim().to_string() });
                    }
                    if app.ask_pending {
                        ui.spinner();
                        theme::hint(ui, "waiting for their reply…");
                    }
                });
                theme::hint(
                    ui,
                    "They mint a brand-new one-time address and sign it. The signature is \
                     checked against the key you typed, so nobody else watching the network \
                     can answer in their place. It fills in above automatically.",
                );
            });

        ui.add_space(6.0);
        let spendable = app.balance.as_ref().map(|b| b.confirmed).unwrap_or(0);
        ui.horizontal(|ui| {
            theme::hint(ui, "Amount");
            theme::right_aligned(ui, |ui| {
                // "Send all" leaves the fee behind rather than failing at the
                // planner: the fee comes out of the amount, not on top of it.
                if ui
                    .add_enabled(spendable > 0, egui::Button::new(RichText::new("send all").size(11.0)))
                    .on_hover_text("Sends your whole spendable balance, with the network fee taken out of it.")
                    .clicked()
                {
                    app.send_unit = 0;
                    app.send_amount = spendable.saturating_sub(estimated_fee(app)).to_string();
                }
                ui.label(
                    RichText::new(format!("{} available", units(spendable)))
                        .size(11.5)
                        .color(theme::muted()),
                );
            });
        });
        // Denomination picker belongs beside the field it scales, not further
        // down the form.
        ui.horizontal(|ui| {
            let r = ui.add(
                TextEdit::singleline(&mut app.send_amount)
                    .hint_text("0")
                    .font(egui::TextStyle::Monospace)
                    .desired_width(180.0),
            );
            if r.changed() {
                app.send_amount.retain(|c| c.is_ascii_digit() || c == '.');
            }
            theme::unit_selector(ui, &mut app.send_unit);
        });
        let parsed = theme::parse_in_unit(&app.send_amount, app.send_unit);
        match &parsed {
            Ok(raw) => {
                let compact = theme::compact_units(*raw);
                let tail = if app.send_unit == 0 && !compact.ends_with("MDS") {
                    format!(" · {compact}")
                } else {
                    String::new()
                };
                ui.label(
                    RichText::new(format!("= {} MDS{tail}", units(*raw)))
                        .size(12.0)
                        .color(theme::bright()),
                );
            }
            Err(e) if !e.is_empty() => {
                ui.label(RichText::new(e).size(12.0).color(theme::muted()));
            }
            _ => {}
        }
        theme::hint(
            ui,
            "Amounts split into power-of-two denominations on-chain; the network fee is \
             calculated from transaction size and added automatically.",
        );

        ui.add_space(6.0);
        ui.checkbox(&mut app.send_private, "Privacy delay");
        theme::hint(
            ui,
            "Waits a randomized period between the commit and the reveal so the two are \
             harder to link. The send takes noticeably longer.",
        );

        ui.add_space(10.0);
        let amount: Option<u64> = parsed.ok().filter(|v| *v > 0);
        let can = app.addr_ok == Some(true) && amount.is_some() && !app.busy && !syncing && !restoring;
        theme::right_aligned(ui, |ui| {
            let label = if app.busy {
                "Committing…"
            } else if syncing {
                "Waiting for sync…"
            } else if restoring {
                "Finding earlier coins…"
            } else {
                "Send"
            };
            if ui.add_enabled(can, egui::Button::new(RichText::new(label).strong())).clicked() {
                app.busy = true;
                app.error.clear();
                app.go(
                    &ctx,
                    Action::Send {
                        to: app.send_to.trim().to_string(),
                        amount: amount.unwrap(),
                        private: app.send_private,
                    },
                );
            }
        });
        if !app.error.is_empty() {
            ui.label(RichText::new(&app.error).color(theme::red()));
        }

        ui.add_space(6.0);
        theme::hint(
            ui,
            "Sends happen in two on-chain steps: a sealed commit, then a reveal once the \
             commit is mined. The app finishes both automatically — you can close it after \
             the commit is mined and it resumes on next unlock. Keys here are one-time-use: \
             change always returns to fresh addresses.",
        );
    });

    if !app.sends.is_empty() {
        theme::heading(ui, "Sends");
        let mut retry: Option<String> = None;
        let mut abandon: Option<String> = None;
        for s in &app.sends {
            theme::panel_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(theme::mono(format!("{} units", units(s.amount))));
                    if s.fee > 0 {
                        ui.label(theme::mono(format!("· fee {}", units(s.fee))).color(theme::muted()));
                    }
                    if !s.to.is_empty() {
                        ui.label(theme::mono(format!("· to {}", short_hex(&s.to, 8))).color(theme::muted()));
                    }
                    theme::right_aligned(ui, |ui| {
                        ui.label(theme::mono(short_hex(&s.id, 8)).color(theme::muted()));
                    });
                });
                send_timeline(ui, s);
                ui.horizontal(|ui| {
                    theme::hint(ui, &s.detail);
                    if s.stage == SendStage::Stalled {
                        theme::right_aligned(ui, |ui| {
                            if ui.button("Retry").clicked() {
                                retry = Some(s.id.clone());
                            }
                            if ui.button("Abandon").clicked() {
                                abandon = Some(s.id.clone());
                            }
                        });
                    }
                });
            });
            ui.add_space(4.0);
        }
        if let Some(id) = retry {
            app.go(&ctx, Action::RetrySend { id });
        }
        if let Some(id) = abandon {
            app.go(&ctx, Action::AbandonSend { id });
        }
    }
}

/// A deliberately generous fee estimate for "send all". Overshooting leaves a
/// little dust behind; undershooting makes the planner reject the send, which
/// is the more annoying failure.
fn estimated_fee(app: &App) -> u64 {
    let inputs = app.coins.iter().filter(|c| c.live && !c.wots_signed && !c.in_flight).count();
    // Matches the walletd fee model: ~1636 bytes per input, ~100 per output,
    // at 10 units per KiB, with headroom.
    let bytes = 100 + 1636 * inputs.max(1) as u64 + 100 * 8;
    bytes * 10 / 1024 + 40
}

// ── MDW ─────────────────────────────────────────────────────────────────

/// Pay MDW to a midwimble address. One transaction, priced before it is
/// built, then followed until a block takes it.
fn send_mdw(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    if !m::account_gate(app, ui, ctx) {
        return;
    }
    let a = app.mwb_account.clone().unwrap_or_default();
    let values = a.spendable_coins.clone();
    let most = max_send(&values);

    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());

        theme::hint(ui, "Recipient address");
        ui.add(
            TextEdit::singleline(&mut app.mwb_send_to)
                .hint_text("mw1…")
                .font(egui::TextStyle::Monospace)
                .desired_width(f32::INFINITY),
        );
        let to = app.mwb_send_to.trim().to_string();
        let to_ok = !to.is_empty() && midstate_walletd::mwb::check_address(&to).is_ok();
        if !to.is_empty() && !to_ok {
            if to.len() == 72 && to.chars().all(|c| c.is_ascii_hexdigit()) {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("That is a midstate address: MDS goes there, not MDW.").size(12.0).color(theme::bright()));
                    if ui.button(RichText::new("Send MDS to it").size(11.0)).clicked() {
                        app.send_to = std::mem::take(&mut app.mwb_send_to);
                        app.addr_ok = None;
                        app.go(ctx, Action::ValidateAddress { addr: app.send_to.clone() });
                        app.send_asset = 0;
                    }
                });
            } else {
                ui.label(
                    RichText::new("Not a midwimble address: they start mw1 and carry a checksum, so check for a typo.")
                        .size(12.0)
                        .color(theme::red()),
                );
            }
        }

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            theme::hint(ui, "Amount");
            theme::right_aligned(ui, |ui| {
                if ui
                    .add_enabled(most > 0, egui::Button::new(RichText::new("send all").size(11.0)))
                    .on_hover_text("Your whole spendable MDW, less the fee.")
                    .clicked()
                {
                    app.mwb_send_amount = m::plain(most);
                }
                ui.label(RichText::new(format!("{} available", mdw(a.spendable))).size(11.5).color(theme::muted()));
            });
        });
        ui.horizontal(|ui| {
            let r = ui.add(
                TextEdit::singleline(&mut app.mwb_send_amount)
                    .hint_text("0.0")
                    .font(egui::TextStyle::Monospace)
                    .desired_width(180.0),
            );
            if r.changed() {
                app.mwb_send_amount.retain(|c| c.is_ascii_digit() || c == '.' || c == ',');
            }
            ui.label(theme::mono("MDW").color(theme::muted()));
        });
        let amount = m::parse_mdw(&app.mwb_send_amount);
        let fee = amount.as_ref().ok().map(|v| send_fee(&values, *v));
        match (&amount, &fee) {
            (Ok(v), Some(Ok(f))) => {
                ui.label(
                    RichText::new(format!("= {} · fee {} · {} leaves the account", mdw(*v), mdw(*f), mdw(v + f)))
                        .size(12.0)
                        .color(theme::bright()),
                );
            }
            (Ok(_), Some(Err(_))) => {
                ui.label(
                    RichText::new(format!("More than this account can pay: {} at most, fee included.", mdw(most)))
                        .size(12.0)
                        .color(theme::amber()),
                );
            }
            (Err(e), _) if !e.is_empty() => {
                ui.label(RichText::new(e).size(12.0).color(theme::amber()));
            }
            _ => {}
        }
        theme::hint(
            ui,
            "One transaction: it is final once a block takes it, about a minute. The fee is \
             midwimble's relay minimum for the coins it spends. The address is a stealth address, \
             so the payment lands on a one-time output only its owner can find.",
        );

        ui.add_space(10.0);
        let waiting = m::not_ready(app);
        let can = to_ok && matches!(fee, Some(Ok(_))) && !app.busy && waiting.is_none();
        theme::right_aligned(ui, |ui| {
            let label = if app.busy { "Sending…" } else { waiting.unwrap_or("Send MDW") };
            if ui.add_enabled(can, egui::Button::new(RichText::new(label).strong())).clicked() {
                app.busy = true;
                app.error.clear();
                app.mwb_notice.clear();
                app.go(ctx, Action::MwbSend { to: to.clone(), amount: amount.clone().unwrap_or(0) });
            }
        });
        if !app.error.is_empty() {
            ui.label(RichText::new(&app.error).color(theme::red()));
        }
        if !app.mwb_notice.is_empty() {
            ui.label(RichText::new(&app.mwb_notice).size(12.0).color(theme::bright()));
        }
    });

    let payments: Vec<_> = app.mwb_history.iter().filter(|h| !h.incoming).take(12).cloned().collect();
    if payments.is_empty() {
        return;
    }
    theme::heading(ui, "MDW payments");
    let explorer = m::mwb_explorer(app);
    for p in &payments {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(theme::mono(mdw(p.amount)));
                if p.fee > 0 {
                    ui.label(theme::mono(format!("· fee {}", mdw(p.fee))).color(theme::muted()));
                }
                match &p.to {
                    Some(to) => {
                        ui.label(theme::mono(format!("· to {}", m::short_addr(to))).color(theme::muted()));
                    }
                    None => {
                        ui.label(theme::mono("· into a swap").color(theme::muted()));
                    }
                }
                theme::right_aligned(ui, |ui| {
                    m::explorer_button(
                        ui,
                        "explorer",
                        explorer.as_deref().and_then(|e| m::history_page(e, p)),
                        "Open it in midwimble's explorer, served by your own node",
                    );
                    if p.time > 0 {
                        ui.label(RichText::new(ago(p.time)).size(11.0).color(theme::faint()));
                    }
                });
            });
            ui.horizontal(|ui| {
                let at = p.height.map(|b| format!("block {}", units(b))).unwrap_or_default();
                let (badge, colour, note) = match p.status.as_str() {
                    "pending" => ("waiting for a block", theme::bright(), String::new()),
                    "failed" => ("failed", theme::red(), format!("its coins went to another transaction, {at}")),
                    _ => ("in a block", theme::ink(), at),
                };
                theme::badge(ui, badge, colour);
                if !note.is_empty() {
                    ui.label(RichText::new(note).size(11.0).color(theme::muted()));
                }
            });
        });
        ui.add_space(4.0);
    }
}
