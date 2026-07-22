//! Q-Bolt channels. Unidirectional (Spilman) payment channels: the sender
//! locks coins behind a covenant, hands over successively better signed
//! states, and the receiver settles the best one before expiry. Capacity
//! flows one way only — spending it is spending it.

use crate::app::App;
use crate::bridge::Action;
use crate::theme::{self, short_hex, units};
use eframe::egui::{self, FontId, RichText, TextEdit, Ui};
use midstate_walletd::api::HubView;

fn fmt_blocks(left: i64) -> String {
    if left <= 0 {
        return "expired".into();
    }
    let l = left as u64;
    if l >= 1440 {
        format!("{l} blocks (~{:.1} days)", l as f64 / 1440.0)
    } else if l >= 60 {
        format!("{l} blocks (~{} h)", l / 60)
    } else {
        format!("{l} blocks (~{l} min)")
    }
}

pub fn show(app: &mut App, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    theme::heading(ui, "Channels");
    theme::hint(
        ui,
        "Lock coins toward one peer, then pay them instantly off-chain. Capacity runs \
         one way: what you spend toward a peer is gone until you close and re-fund. \
         Channel traffic rides the node's chat network.",
    );
    ui.add_space(4.0);

    let syncing = app.sync.as_ref().map(|s| s.is_syncing).unwrap_or(true);
    let tip = app.sync.as_ref().map(|s| s.height).unwrap_or(0);

    identity_panel(app, ui, &ctx);
    invoice_panel(app, ui, &ctx, syncing);
    hub_panel(app, ui, &ctx);
    open_panel(app, ui, &ctx, syncing);

    if syncing {
        theme::hint(ui, "Channels unlock when the node reaches the chain tip.");
    }
    if app.busy {
        ui.horizontal(|ui| {
            ui.spinner();
            theme::hint(ui, "working…");
        });
    }
    if !app.error.is_empty() {
        ui.label(RichText::new(&app.error).color(theme::muted()).size(12.0));
    }

    channel_list(app, ui, &ctx, tip);
}

// ── Identity ────────────────────────────────────────────────────────────

fn identity_panel(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Your channel identity").font(theme::font_medium(14.0)));
        match app.chan_identity.clone() {
            Some(id) => {
                ui.horizontal(|ui| {
                    ui.label(theme::mono(&id.pk).size(11.5).color(theme::bright()));
                    if ui.button(RichText::new("copy").size(11.0)).clicked() {
                        ui.ctx().copy_text(id.pk.clone());
                    }
                });
                let low = id.remaining_sigs < 64;
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("{} signatures left", id.remaining_sigs))
                            .size(12.0)
                            .color(if low { theme::ink() } else { theme::muted() }),
                    );
                    if ui
                        .button(RichText::new("New identity").size(11.0))
                        .on_hover_text(
                            "Start a fresh key. Existing channels keep settling with the old one.",
                        )
                        .clicked()
                    {
                        app.busy = true;
                        app.go(ctx, Action::RotateIdentity);
                    }
                });
                theme::hint(
                    ui,
                    "Share this key so others can open channels to you. Every off-chain state \
                     costs one one-time signature from this key — when it runs low, rotate it \
                     (8 are always reserved so open channels can still be closed).",
                );
            }
            None => theme::hint(ui, "Preparing your channel identity key…"),
        }
    });
    ui.add_space(6.0);
}

// ── Invoices ────────────────────────────────────────────────────────────

fn invoice_panel(app: &mut App, ui: &mut Ui, ctx: &egui::Context, syncing: bool) {
    egui::CollapsingHeader::new("Get paid / pay")
        .id_salt("qb_invoices")
        .default_open(true)
        .show(ui, |ui| {
            theme::panel_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());

                // Request money
                ui.label(RichText::new("Create an invoice").font(theme::font_medium(13.0)));
                ui.horizontal(|ui| {
                    let r = ui.add(
                        TextEdit::singleline(&mut app.inv_amount)
                            .hint_text("amount")
                            .font(egui::TextStyle::Monospace)
                            .desired_width(120.0),
                    );
                    if r.changed() {
                        app.inv_amount.retain(|c| c.is_ascii_digit());
                    }
                    let amt = app.inv_amount.parse::<u64>().unwrap_or(0);
                    if ui
                        .add_enabled(!app.busy && !syncing && amt > 0, egui::Button::new("Create"))
                        .clicked()
                    {
                        app.busy = true;
                        app.error.clear();
                        app.go(ctx, Action::CreateInvoice { amount: amt });
                    }
                });
                theme::hint(
                    ui,
                    "Anyone holding a channel toward you — or toward a hub that has one — can pay it.",
                );
                if let Some(inv) = app.last_invoice.clone() {
                    ui.horizontal(|ui| {
                        ui.label(theme::mono(short_hex(&inv.text, 22)).size(11.0).color(theme::bright()));
                        if ui.button(RichText::new("copy invoice").size(11.0)).clicked() {
                            ui.ctx().copy_text(inv.text.clone());
                        }
                    });
                }

                ui.add_space(10.0);
                // Pay
                ui.label(RichText::new("Pay an invoice").font(theme::font_medium(13.0)));
                ui.add(
                    TextEdit::singleline(&mut app.pay_invoice_text)
                        .hint_text("l2inv1:…")
                        .font(egui::TextStyle::Monospace)
                        .desired_width(f32::INFINITY),
                );
                theme::right_aligned(ui, |ui| {
                    let ok = app.pay_invoice_text.trim().starts_with("l2inv");
                    if ui.add_enabled(!app.busy && !syncing && ok, egui::Button::new("Pay")).clicked() {
                        app.busy = true;
                        app.error.clear();
                        app.go(ctx, Action::PayInvoice { text: app.pay_invoice_text.trim().to_string() });
                    }
                });
                theme::hint(
                    ui,
                    "The payment is hash-locked end to end: it either completes and returns a \
                     preimage as your receipt, or it fails and the amount comes back.",
                );

                ui.add_space(8.0);
                ui.label(RichText::new("…or ask a peer to invoice you").font(theme::font_medium(13.0)));
                ui.add(
                    TextEdit::singleline(&mut app.req_payee)
                        .hint_text("their identity key (64 hex characters)")
                        .font(egui::TextStyle::Monospace)
                        .desired_width(f32::INFINITY),
                );
                ui.horizontal(|ui| {
                    let r = ui.add(
                        TextEdit::singleline(&mut app.req_amount)
                            .hint_text("amount")
                            .font(egui::TextStyle::Monospace)
                            .desired_width(120.0),
                    );
                    if r.changed() {
                        app.req_amount.retain(|c| c.is_ascii_digit());
                    }
                    let amt = app.req_amount.parse::<u64>().unwrap_or(0);
                    let pk_ok = app.req_payee.trim().len() == 64
                        && app.req_payee.trim().chars().all(|c| c.is_ascii_hexdigit());
                    if ui
                        .add_enabled(!app.busy && !syncing && pk_ok && amt > 0, egui::Button::new("Request"))
                        .clicked()
                    {
                        app.busy = true;
                        app.error.clear();
                        app.go(
                            ctx,
                            Action::RequestInvoice {
                                payee: app.req_payee.trim().to_string(),
                                amount: amt,
                            },
                        );
                    }
                });
                theme::hint(
                    ui,
                    "They mint and sign an invoice over the bus and it is paid automatically. \
                     The signature is checked against the key you typed, so a forged reply from \
                     anyone else watching the bus is rejected.",
                );

                // Outstanding
                let outstanding: Vec<_> =
                    app.invoices.iter().filter(|i| i.paid.is_none()).cloned().collect();
                let paid: Vec<_> = app.invoices.iter().filter(|i| i.paid.is_some()).cloned().collect();
                if !outstanding.is_empty() || !paid.is_empty() {
                    ui.add_space(8.0);
                    ui.label(RichText::new("Your invoices").font(theme::font_medium(13.0)));
                    for i in outstanding.iter().take(8) {
                        ui.horizontal(|ui| {
                            theme::badge(ui, "awaiting", theme::muted());
                            ui.label(theme::mono(units(i.amount)));
                            ui.label(
                                RichText::new(short_hex(&i.hash, 8)).font(FontId::monospace(10.0)).color(theme::faint()),
                            );
                            if ui.button(RichText::new("copy").size(10.0)).clicked() {
                                ui.ctx().copy_text(i.text.clone());
                            }
                        });
                    }
                    for i in paid.iter().take(5) {
                        ui.horizontal(|ui| {
                            theme::badge(ui, "paid", theme::ink());
                            ui.label(theme::mono(units(i.paid.unwrap_or(i.amount))));
                            ui.label(
                                RichText::new(short_hex(&i.hash, 8)).font(FontId::monospace(10.0)).color(theme::faint()),
                            );
                        });
                    }
                }
            });
        });
    ui.add_space(6.0);
}

// ── Hub ─────────────────────────────────────────────────────────────────

fn hub_panel(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    let Some(cfg) = app.hub.clone() else { return };
    let running = cfg.forward;
    egui::CollapsingHeader::new(if running { "Routing hub — on" } else { "Routing hub — off" })
        .id_salt("qb_hub")
        .show(ui, |ui| {
            theme::panel_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                let mut next = cfg.clone();
                ui.checkbox(&mut next.auto_accept, "Accept incoming channels");
                theme::hint(ui, "Costs you nothing — the other side funds it.");
                ui.checkbox(&mut next.forward, "Forward payments for others");
                theme::hint(
                    ui,
                    "Relays hash-locked payments toward their destination, keeping a 50-unit fee \
                     per hop. Because channels are one-way, each forward permanently spends your \
                     outbound capacity toward that peer — the fee pays for the capacity you give \
                     up and the two signatures the hop costs.",
                );
                ui.checkbox(&mut next.jit_open, "Open channels on demand to complete a route");
                theme::hint(
                    ui,
                    "When the last hop has no channel, fund one immediately. This spends real \
                     coins on-chain, so leave it off unless you intend to provide capacity.",
                );
                ui.horizontal(|ui| {
                    theme::hint(ui, "on-demand size");
                    let mut cap = next.jit_capacity.to_string();
                    if ui
                        .add(TextEdit::singleline(&mut cap).font(egui::TextStyle::Monospace).desired_width(110.0))
                        .changed()
                    {
                        cap.retain(|c| c.is_ascii_digit());
                        next.jit_capacity = cap.parse().unwrap_or(next.jit_capacity);
                    }
                    theme::hint(ui, "keep at least");
                    let mut leaves = next.min_leaves.to_string();
                    if ui
                        .add(TextEdit::singleline(&mut leaves).font(egui::TextStyle::Monospace).desired_width(70.0))
                        .changed()
                    {
                        leaves.retain(|c| c.is_ascii_digit());
                        next.min_leaves = leaves.parse().unwrap_or(next.min_leaves);
                    }
                    theme::hint(ui, "signatures in reserve");
                });

                let remaining = app.chan_identity.as_ref().map(|i| i.remaining_sigs);
                if let Some(rem) = remaining {
                    let out: u64 = app
                        .channels
                        .iter()
                        .filter(|c| c.role == "sender" && c.status == "active")
                        .map(|c| c.my_balance)
                        .sum();
                    ui.label(
                        RichText::new(format!(
                            "Routable outbound capacity {} · {} signatures left",
                            units(out),
                            rem
                        ))
                        .size(12.0)
                        .color(theme::muted()),
                    );
                }

                if changed(&cfg, &next) {
                    theme::right_aligned(ui, |ui| {
                        if ui.add_enabled(!app.busy, egui::Button::new("Save hub settings")).clicked() {
                            app.busy = true;
                            app.go(ctx, Action::SetHub { cfg: next.clone() });
                        }
                    });
                }
                app.hub = Some(next);
            });
        });
    ui.add_space(6.0);
}

fn changed(a: &HubView, b: &HubView) -> bool {
    a.auto_accept != b.auto_accept
        || a.forward != b.forward
        || a.jit_open != b.jit_open
        || a.jit_capacity != b.jit_capacity
        || a.min_leaves != b.min_leaves
}

// ── Open ────────────────────────────────────────────────────────────────

fn open_panel(app: &mut App, ui: &mut Ui, ctx: &egui::Context, syncing: bool) {
    egui::CollapsingHeader::new("Open a channel")
        .id_salt("qb_open")
        .show(ui, |ui| {
            theme::panel_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.add(
                    TextEdit::singleline(&mut app.chan_peer)
                        .hint_text("peer identity key (64 hex characters)")
                        .font(egui::TextStyle::Monospace)
                        .desired_width(f32::INFINITY),
                );
                ui.horizontal(|ui| {
                    theme::hint(ui, "amount");
                    let r = ui.add(
                        TextEdit::singleline(&mut app.chan_amount)
                            .hint_text("min 4096")
                            .font(egui::TextStyle::Monospace)
                            .desired_width(110.0),
                    );
                    if r.changed() {
                        app.chan_amount.retain(|c| c.is_ascii_digit());
                    }
                    theme::hint(ui, "lifetime");
                    let r2 = ui.add(
                        TextEdit::singleline(&mut app.chan_life)
                            .font(egui::TextStyle::Monospace)
                            .desired_width(80.0),
                    );
                    if r2.changed() {
                        app.chan_life.retain(|c| c.is_ascii_digit());
                    }
                    theme::hint(ui, "blocks (4320 ≈ 3 days)");
                });
                let peer_ok = app.chan_peer.trim().len() == 64
                    && app.chan_peer.trim().chars().all(|c| c.is_ascii_hexdigit());
                let amt = app.chan_amount.parse::<u64>().unwrap_or(0);
                theme::right_aligned(ui, |ui| {
                    let can = !app.busy && !syncing && peer_ok && amt >= 4096;
                    if ui.add_enabled(can, egui::Button::new("Open channel")).clicked() {
                        let lifetime = app.chan_life.parse::<u64>().unwrap_or(4320);
                        app.busy = true;
                        app.error.clear();
                        app.go(
                            ctx,
                            Action::ChannelOpen {
                                peer: app.chan_peer.trim().to_string(),
                                amount: amt,
                                lifetime,
                            },
                        );
                    }
                });
                if !app.chan_peer.trim().is_empty() && !peer_ok {
                    theme::hint(ui, "The peer key is 64 hex characters — not an address.");
                }
                theme::hint(
                    ui,
                    "Funding is a normal on-chain send into the channel covenant. The 2000-unit \
                     settlement fee comes out of channel value at close.",
                );
            });
        });
    ui.add_space(6.0);
}

// ── Channels ────────────────────────────────────────────────────────────

fn channel_list(app: &mut App, ui: &mut Ui, ctx: &egui::Context, tip: u64) {
    if app.channels.is_empty() {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            theme::hint(
                ui,
                "No channels yet. Open one above, or share your identity key — an incoming \
                 channel appears here once its funding confirms.",
            );
        });
        return;
    }

    let chans = app.channels.clone();
    let mut do_pay: Option<(String, u64)> = None;
    let mut do_close: Option<String> = None;
    let mut do_refund: Option<String> = None;

    egui::ScrollArea::vertical().id_salt("chan_list").show(ui, |ui| {
        for c in &chans {
            theme::panel_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    theme::badge(ui, if c.role == "sender" { "outbound" } else { "inbound" }, theme::ink());
                    ui.label(theme::mono(short_hex(&c.peer, 8)).color(theme::muted()).size(12.0));
                    ui.label(RichText::new(&c.status).color(theme::muted()).size(12.0));
                    if c.role == "sender" && c.status == "active" && !c.acked {
                        theme::badge(ui, "delivering", theme::muted());
                    }
                });
                let left = c.expiry as i64 - tip as i64;
                ui.horizontal_wrapped(|ui| {
                    ui.label(theme::mono(format!(
                        "capacity {}   yours {}   state {}",
                        units(c.capacity),
                        units(c.my_balance),
                        c.nonce
                    )));
                    let warn = left > 0 && left < 240;
                    ui.label(
                        RichText::new(format!("· expires in {}", fmt_blocks(left)))
                            .size(12.0)
                            .color(if warn || left <= 0 { theme::ink() } else { theme::faint() }),
                    );
                });

                // In-flight hash-locked payments.
                for h in &c.htlcs {
                    ui.horizontal(|ui| {
                        theme::badge(ui, if h.claiming { "claiming" } else { "in flight" }, theme::muted());
                        ui.label(theme::mono(units(h.amount)).size(12.0));
                        ui.label(
                            RichText::new(format!(
                                "locked · reclaimable at block {} ({})",
                                h.timeout,
                                fmt_blocks(h.timeout as i64 - tip as i64)
                            ))
                            .size(11.0)
                            .color(theme::faint()),
                        );
                    });
                }

                if c.status == "active" {
                    ui.add_space(2.0);
                    ui.horizontal(|ui| {
                        if c.role == "sender" {
                            let pay = app.chan_pay.entry(c.id.clone()).or_default();
                            let r = ui.add(
                                TextEdit::singleline(pay)
                                    .hint_text("amount")
                                    .font(egui::TextStyle::Monospace)
                                    .desired_width(110.0),
                            );
                            if r.changed() {
                                pay.retain(|ch| ch.is_ascii_digit());
                            }
                            let amt = pay.parse::<u64>().unwrap_or(0);
                            let payable = left > 90;
                            let can = !app.busy && payable && amt > 0 && amt <= c.my_balance;
                            if ui.add_enabled(can, egui::Button::new("Pay")).clicked() {
                                do_pay = Some((c.id.clone(), amt));
                            }
                            if !payable {
                                theme::hint(ui, "too close to expiry to pay — refund unlocks at expiry");
                            }
                            if left <= 0
                                && ui.add_enabled(!app.busy, egui::Button::new("Refund now")).clicked()
                            {
                                do_refund = Some(c.id.clone());
                            }
                        } else {
                            if ui.add_enabled(!app.busy, egui::Button::new("Close & settle")).clicked() {
                                do_close = Some(c.id.clone());
                            }
                            theme::hint(ui, "settles your balance; auto-closes 60 blocks before expiry");
                        }
                    });
                }
                ui.label(
                    RichText::new(format!("channel {}", short_hex(&c.id, 8)))
                        .font(FontId::monospace(10.0))
                        .color(theme::faint()),
                );
            });
            ui.add_space(6.0);
        }
    });

    if let Some((id, amount)) = do_pay {
        app.busy = true;
        app.error.clear();
        app.chan_pay.remove(&id);
        app.go(ctx, Action::ChannelPay { id, amount });
    }
    if let Some(id) = do_close {
        app.busy = true;
        app.error.clear();
        app.go(ctx, Action::ChannelClose { id });
    }
    if let Some(id) = do_refund {
        app.busy = true;
        app.error.clear();
        app.go(ctx, Action::ChannelRefund { id });
    }
}
