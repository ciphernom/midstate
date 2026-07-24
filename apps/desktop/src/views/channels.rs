//! Q-Bolt channels. Unidirectional (Spilman) payment channels: the sender
//! locks coins behind a covenant, hands over successively better signed
//! states, and the receiver settles the best one before expiry. Capacity
//! flows one way only — spending it is spending it.
//!
//! Layout note: this tab holds three distinct jobs (watching channels,
//! moving money, running a hub), so they live on sub-tabs rather than
//! stacked panels. Nothing here opens its own ScrollArea — the tab content
//! is already inside one, and nesting them starves the inner view of height.

use crate::app::{App, ChanTab};
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
    let syncing = app.sync.as_ref().map(|s| s.is_syncing).unwrap_or(true);
    let tip = app.sync.as_ref().map(|s| s.height).unwrap_or(0);

    theme::heading(ui, "Channels");
    identity_strip(app, ui);
    ui.add_space(8.0);

    // ── Sub-navigation ──────────────────────────────────────────────────
    let open_n = app
        .channels
        .iter()
        .filter(|c| c.status == "active" || c.status.starts_with("opening"))
        .count();
    let owed_n = app.invoices.iter().filter(|i| i.paid.is_none()).count();
    let l0 = if open_n > 0 { format!("Channels ({open_n})") } else { "Channels".to_string() };
    let l1 = if owed_n > 0 { format!("Pay ({owed_n})") } else { "Pay".to_string() };
    let l2 = if app.hub.as_ref().map(|h| h.forward).unwrap_or(false) {
        "Hub · on".to_string()
    } else {
        "Hub".to_string()
    };
    let labels = [l0.as_str(), l1.as_str(), l2.as_str()];
    let mut ix = match app.chan_tab {
        ChanTab::List => 0,
        ChanTab::Pay => 1,
        ChanTab::Hub => 2,
    };
    if theme::segmented(ui, &labels, &mut ix) {
        app.chan_tab = [ChanTab::List, ChanTab::Pay, ChanTab::Hub][ix];
        app.error.clear();
    }
    ui.add_space(10.0);

    if syncing {
        theme::hint(ui, "Channels unlock when the node reaches the chain tip.");
        ui.add_space(4.0);
    }

    match app.chan_tab {
        ChanTab::List => {
            channel_list(app, ui, &ctx, tip);
            ui.add_space(10.0);
            hub_directory(app, ui, &ctx, tip);
    open_panel(app, ui, &ctx, syncing);
        }
        ChanTab::Pay => invoice_panel(app, ui, &ctx, syncing),
        ChanTab::Hub => hub_panel(app, ui, &ctx),
    }

    if app.busy {
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.spinner();
            theme::hint(ui, "working…");
        });
    }
    if !app.error.is_empty() {
        ui.add_space(4.0);
        ui.label(RichText::new(&app.error).color(theme::muted()).size(12.0));
    }
}

// ── Identity ────────────────────────────────────────────────────────────

/// One compact line: the key others need to open a channel to you.
fn identity_strip(app: &mut App, ui: &mut Ui) {
    let Some(id) = app.chan_identity.clone() else {
        theme::hint(ui, "Preparing your channel identity key…");
        return;
    };
    ui.horizontal(|ui| {
        ui.label(RichText::new("your key").font(FontId::monospace(10.0)).color(theme::muted()));
        ui.label(theme::mono(short_hex(&id.pk, 12)).size(11.5).color(theme::bright()));
        if ui.button(RichText::new("copy").size(11.0)).clicked() {
            ui.ctx().copy_text(id.pk.clone());
        }
        if id.remaining_sigs < 64 {
            ui.label(
                RichText::new(format!("· {} signatures left", id.remaining_sigs))
                    .size(11.5)
                    .color(theme::ink()),
            );
        }
    });
}

// ── Channels ────────────────────────────────────────────────────────────

fn channel_list(app: &mut App, ui: &mut Ui, ctx: &egui::Context, tip: u64) {
    if app.channels.is_empty() {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            theme::hint(
                ui,
                "No channels yet. Open one below to pay someone instantly, or share the key \
                 above — a channel opened to you appears here once its funding confirms.",
            );
        });
        return;
    }

    let chans = app.channels.clone();
    let mut do_pay: Option<(String, u64)> = None;
    let mut do_close: Option<String> = None;
    let mut do_refund: Option<String> = None;

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

fn open_panel(app: &mut App, ui: &mut Ui, ctx: &egui::Context, syncing: bool) {
    egui::CollapsingHeader::new("Open a channel")
        .id_salt("qb_open")
        .default_open(app.channels.is_empty())
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
}

// ── Pay ─────────────────────────────────────────────────────────────────

fn invoice_panel(app: &mut App, ui: &mut Ui, ctx: &egui::Context, syncing: bool) {
    let outbound = app
        .channels
        .iter()
        .filter(|c| c.role == "sender" && c.status == "active")
        .map(|c| c.my_balance)
        .sum::<u64>();

    // Receive
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Create an invoice").font(theme::font_medium(14.0)));
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
    });

    // Send
    ui.add_space(6.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Pay an invoice").font(theme::font_medium(14.0)));
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
            "They mint and sign an invoice over the bus and it is paid automatically. The \
             signature is checked against the key you typed, so a forged reply from anyone \
             else watching the bus is rejected.",
        );

        if outbound == 0 {
            theme::hint(
                ui,
                "You have no outbound channel balance yet — open a channel before paying.",
            );
        } else {
            theme::hint(ui, &format!("Spendable across your channels: {}", units(outbound)));
        }
    });

    // Ledger
    let outstanding: Vec<_> = app.invoices.iter().filter(|i| i.paid.is_none()).cloned().collect();
    let paid: Vec<_> = app.invoices.iter().filter(|i| i.paid.is_some()).cloned().collect();
    if outstanding.is_empty() && paid.is_empty() {
        return;
    }
    ui.add_space(6.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Your invoices").font(theme::font_medium(14.0)));
        for i in outstanding.iter().take(12) {
            ui.horizontal(|ui| {
                theme::badge(ui, "awaiting", theme::muted());
                ui.label(theme::mono(units(i.amount)));
                ui.label(
                    RichText::new(short_hex(&i.hash, 8))
                        .font(FontId::monospace(10.0))
                        .color(theme::faint()),
                );
                if ui.button(RichText::new("copy").size(10.0)).clicked() {
                    ui.ctx().copy_text(i.text.clone());
                }
            });
        }
        for i in paid.iter().take(8) {
            ui.horizontal(|ui| {
                theme::badge(ui, "paid", theme::ink());
                ui.label(theme::mono(units(i.paid.unwrap_or(i.amount))));
                ui.label(
                    RichText::new(short_hex(&i.hash, 8))
                        .font(FontId::monospace(10.0))
                        .color(theme::faint()),
                );
            });
        }
    });
}

// ── Hub ─────────────────────────────────────────────────────────────────

fn hub_panel(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    // `saved` is what the daemon is actually enforcing; `next` is the edit
    // buffer. Keeping them in separate fields is what allows the Save button to
    // persist across frames — see `App::hub_draft`.
    let Some(saved) = app.hub.clone() else {
        theme::hint(ui, "Loading hub settings…");
        return;
    };
    let mut next = app.hub_draft.clone().unwrap_or_else(|| saved.clone());

    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.checkbox(&mut next.auto_accept, "Accept incoming channels");
        theme::hint(ui, "Costs you nothing — the other side funds it.");

        ui.add_space(6.0);
        ui.checkbox(&mut next.forward, "Forward payments for others");
        theme::hint(
            ui,
            "Relays hash-locked payments toward their destination, keeping a 50-unit fee per \
             hop. Because channels are one-way, each forward permanently spends your outbound \
             capacity toward that peer — the fee pays for the capacity you give up and the two \
             signatures the hop costs.",
        );

        ui.add_space(6.0);
        ui.checkbox(&mut next.auto_open_on_request, "Open a channel when someone asks");
        theme::hint(
            ui,
            "A channel only carries value one way, so a buyer cannot open one to pay you — \
             they can only ask you to open one to them. Turning this on is what lets someone \
             trade with you instantly without arranging anything first.",
        );
        ui.horizontal(|ui| {
            theme::hint(ui, "at most");
            let mut cap = next.max_auto_capacity.to_string();
            if ui
                .add(TextEdit::singleline(&mut cap).font(egui::TextStyle::Monospace).desired_width(110.0))
                .changed()
            {
                cap.retain(|c| c.is_ascii_digit());
                next.max_auto_capacity = cap.parse().unwrap_or(next.max_auto_capacity);
            }
            theme::hint(ui, "per channel, and");
            let mut bud = next.auto_capacity_budget.to_string();
            if ui
                .add(TextEdit::singleline(&mut bud).font(egui::TextStyle::Monospace).desired_width(130.0))
                .changed()
            {
                bud.retain(|c| c.is_ascii_digit());
                next.auto_capacity_budget = bud.parse().unwrap_or(next.auto_capacity_budget);
            }
            theme::hint(ui, "in total");
        });

        ui.add_space(6.0);
        ui.checkbox(&mut next.jit_open, "Open channels on demand to complete a route");
        theme::hint(
            ui,
            "When the last hop has no channel, fund one immediately. This spends real coins \
             on-chain, so leave it off unless you intend to provide capacity.",
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

        // Nothing on this panel takes effect until it is sent to the daemon,
        // so an unsaved edit has to look unsaved. A checkbox that stays ticked
        // while the daemon still declines the request is worse than no
        // checkbox at all.
        let dirty = changed(&saved, &next);
        if dirty {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("unsaved — these settings are not in force yet")
                        .size(11.0)
                        .color(theme::ink()),
                );
                theme::right_aligned(ui, |ui| {
                    if ui.add_enabled(!app.busy, egui::Button::new("Save hub settings")).clicked() {
                        app.busy = true;
                        app.go(ctx, Action::SetHub { cfg: next.clone() });
                    }
                    if ui
                        .add_enabled(!app.busy, egui::Button::new(RichText::new("Discard").size(11.0)))
                        .clicked()
                    {
                        next = saved.clone();
                    }
                });
            });
        }
    });

    // Capacity + key budget: the two things that actually limit a hub.
    ui.add_space(6.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Capacity").font(theme::font_medium(14.0)));
        let out: u64 = app
            .channels
            .iter()
            .filter(|c| c.role == "sender" && c.status == "active")
            .map(|c| c.my_balance)
            .sum();
        let inb: u64 = app
            .channels
            .iter()
            .filter(|c| c.role == "receiver" && c.status == "active")
            .map(|c| c.sender_amt)
            .sum();
        ui.label(theme::mono(format!("routable outbound  {}", units(out))));
        ui.label(theme::mono(format!("inbound available  {}", units(inb))));
        theme::hint(
            ui,
            "Outbound is what you can forward. It drains as you route and only comes back by \
             closing and re-funding — a hub is a capacity vendor, not a toll booth.",
        );

        if let Some(id) = app.chan_identity.clone() {
            ui.add_space(8.0);
            ui.label(RichText::new("Identity key").font(theme::font_medium(14.0)));
            ui.horizontal(|ui| {
                ui.label(theme::mono(&id.pk).size(11.0).color(theme::bright()));
                if ui.button(RichText::new("copy").size(11.0)).clicked() {
                    ui.ctx().copy_text(id.pk.clone());
                }
            });
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!("{} one-time signatures left", id.remaining_sigs))
                        .size(12.0)
                        .color(if id.remaining_sigs < 64 { theme::ink() } else { theme::muted() }),
                );
                if ui
                    .button(RichText::new("New identity").size(11.0))
                    .on_hover_text("Existing channels keep settling with the old key.")
                    .clicked()
                {
                    app.busy = true;
                    app.go(ctx, Action::RotateIdentity);
                }
            });
            theme::hint(
                ui,
                "Every off-chain state costs one signature from this key, so a busy hub burns \
                 through them. Rotate before it runs dry — 8 are always reserved so open \
                 channels can still be closed.",
            );
        }
    });

    // Park the edits in the draft, never in `app.hub`. Once they match the
    // saved copy again — because they were saved, or discarded — drop the
    // draft so the periodic reload can resume.
    app.hub_draft = if changed(&saved, &next) { Some(next) } else { None };
}

fn changed(a: &HubView, b: &HubView) -> bool {
    a.auto_accept != b.auto_accept
        || a.forward != b.forward
        || a.jit_open != b.jit_open
        || a.jit_capacity != b.jit_capacity
        || a.min_leaves != b.min_leaves
        || a.auto_open_on_request != b.auto_open_on_request
        || a.max_auto_capacity != b.max_auto_capacity
        || a.auto_capacity_budget != b.auto_capacity_budget
}

// ── Hub directory ───────────────────────────────────────────────────────

/// Hubs that have advertised themselves over the chat bus.
///
/// Discovery has to live somewhere, and the bus already carries channel
/// negotiation and charges proof-of-work per message — which makes it both the
/// natural place to look and a poor place to spam.
fn hub_directory(app: &mut App, ui: &mut Ui, ctx: &egui::Context, tip: u64) {
    if app.hubs.is_empty() {
        return;
    }
    egui::CollapsingHeader::new(format!("Hubs on the network ({})", app.hubs.len()))
        .id_salt("hub_dir")
        .show(ui, |ui| {
            theme::panel_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                theme::hint(
                    ui,
                    "Routing hubs that have announced themselves. Everything below is their own \
                     claim — nothing here is verified until you actually open a channel. A hub \
                     lets you pay people you have no direct channel with.",
                );
                ui.add_space(4.0);
                let mut ask: Option<String> = None;
                for h in &app.hubs {
                    ui.horizontal_wrapped(|ui| {
                        if h.connected {
                            theme::badge(ui, "connected", theme::ink());
                        }
                        ui.label(theme::mono(short_hex(&h.pk, 10)).size(11.0).color(theme::muted()));
                        ui.label(
                            RichText::new(format!(
                                "claims {} routable · {} fee per hop",
                                units(h.outbound),
                                units(h.hop_fee)
                            ))
                            .size(11.0)
                            .color(theme::muted()),
                        );
                        let stale = tip > 0 && tip.saturating_sub(h.heard) > 2_000;
                        if stale {
                            ui.label(
                                RichText::new("quiet lately").size(10.0).color(theme::faint()),
                            )
                            .on_hover_text("Not heard from recently — it may be offline.");
                        }
                        if !h.connected
                            && ui
                                .add_enabled(!app.busy, egui::Button::new(RichText::new("ask for a channel").size(10.5)))
                                .on_hover_text(
                                    "Asks this hub to open a lane to you, so it can route \
                                     payments in your direction.",
                                )
                                .clicked()
                        {
                            ask = Some(h.pk.clone());
                        }
                        if ui.button(RichText::new("copy key").size(10.0)).clicked() {
                            ui.ctx().copy_text(h.pk.clone());
                        }
                    });
                }
                if let Some(pk) = ask {
                    app.busy = true;
                    app.error.clear();
                    app.go(ctx, Action::RequestChannel { peer: pk, capacity: 65_536 });
                }
            });
        });
    ui.add_space(6.0);
}
