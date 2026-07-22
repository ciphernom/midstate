use crate::app::App;
use crate::bridge::Action;
use crate::theme::{self, send_timeline, short_hex, units};
use eframe::egui::{self, RichText, TextEdit, Ui};
use midstate_walletd::api::SendStage;

pub fn show(app: &mut App, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    theme::heading(ui, "Send");

    let syncing = app.sync.as_ref().map(|s| s.is_syncing).unwrap_or(true);

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
        if app.addr_ok == Some(false) && !app.send_to.trim().is_empty() {
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
        theme::hint(ui, "Amount");
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
        let can = app.addr_ok == Some(true) && amount.is_some() && !app.busy && !syncing;
        theme::right_aligned(ui, |ui| {
            let label = if app.busy {
                "Committing…"
            } else if syncing {
                "Waiting for sync…"
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
