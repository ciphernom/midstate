//! Network chat. Messages are up to ten words from a fixed dictionary,
//! each message proof-of-work-mined by your own node before broadcast.
//! This same channel (with attachments) is the transport qbolt payment
//! channels will ride on.

use crate::app::App;
use crate::bridge::Action;
use crate::theme::{self, ago, short_hex};
use eframe::egui::{self, FontId, RichText, TextEdit, Ui};

/// Sealed messages to and from names: only the two wallets read them.
fn sealed(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    theme::heading(ui, "Sealed messages");
    theme::hint(
        ui,
        "To a name, sealed (post-quantum: ML-KEM) so only its wallet can read it. Nobody watching the \
         network sees who it is for. Your first message to a name is signed as the name you hold, if \
         its identity is this wallet's; after that the session's key does the work. A name takes \
         them when its records carry a kem key.",
    );
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.add(TextEdit::singleline(&mut app.sealed_to).hint_text("to: name.mds").desired_width(160.0));
            ui.add(TextEdit::singleline(&mut app.sealed_text).hint_text("message").desired_width(ui.available_width() - 90.0));
            let ok = !app.sealed_to.trim().is_empty() && !app.sealed_text.trim().is_empty() && app.sealed_text.len() <= 750;
            if ui.add_enabled(ok && !app.sealed_busy, egui::Button::new(if app.sealed_busy { "Sealing…" } else { "Send" })).clicked() {
                app.sealed_busy = true;
                app.go(ctx, Action::SealedSend { to: app.sealed_to.trim().to_string(), text: app.sealed_text.trim().to_string() });
            }
        });
        if app.sealed_text.len() > 750 {
            theme::hint(ui, "At most 750 bytes a message.");
        }
        inbox_bar(app, ui, ctx);
        let requests = app.conversations.iter().filter(|c| c.request).count();
        if requests > 0 {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut app.show_requests, false, "Inbox");
                ui.selectable_value(&mut app.show_requests, true, format!("Requests ({requests})"));
            });
        } else {
            app.show_requests = false;
        }
        let shown: Vec<_> = app.conversations.iter().filter(|c| c.request == app.show_requests).cloned().collect();
        if shown.is_empty() {
            theme::hint(ui, if app.show_requests { "No message requests." } else { "No sealed messages yet." });
        }
        for c in shown {
            ui.separator();
            ui.horizontal(|ui| {
                let who = if c.peer.is_empty() { "anonymous".to_string() } else { c.peer.clone() };
                ui.label(RichText::new(&who).strong());
                if c.verified {
                    theme::badge(ui, "verified", theme::gold());
                } else if !c.peer.is_empty() {
                    theme::badge(ui, "unverified", theme::muted());
                }
                if let Some(t) = midstate_walletd::burns::tier(c.backing) {
                    theme::badge(ui, t, theme::gold());
                } else if c.verified && c.backing > 0.0 {
                    theme::hint(ui, &midstate_walletd::burns::blocks_text(c.backing));
                }
                if !c.peer.is_empty() && ui.small_button("Reply").clicked() {
                    app.sealed_to = c.peer.clone();
                }
            });
            for l in c.lines.iter().rev().take(6).collect::<Vec<_>>().into_iter().rev() {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new(if l.mine { "you" } else { "them" }).font(FontId::monospace(10.5)).color(theme::faint()));
                    ui.label(RichText::new(&l.text).color(theme::ink()));
                    ui.label(RichText::new(ago(l.at)).size(11.0).color(theme::faint()));
                });
            }
        }
    });
}

/// Who reaches the inbox: everyone, or names backed at least so much. The
/// rest wait under Requests, unread by default but never dropped.
fn inbox_bar(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    let bar = app.status.as_ref().and_then(|s| s.inbox_bar);
    ui.horizontal(|ui| {
        theme::hint(ui, "Inbox:");
        let choices: [(Option<f64>, &str); 4] =
            [(None, "everyone"), (Some(0.0), "verified names"), (Some(60.0), "names backed an hour+"), (Some(1_440.0), "names backed a day+")];
        for (v, label) in choices {
            if ui.selectable_label(bar == v, label).on_hover_text("Backing is MDS burned for a name (the Burns page), counted in blocks of issuance").clicked() && bar != v {
                app.go(ctx, Action::SetInboxBar { bar: v });
                app.go(ctx, Action::LoadStatus);
            }
        }
    });
}

pub fn show(app: &mut App, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    sealed(app, ui, &ctx);
    ui.add_space(10.0);
    theme::heading(ui, "Chat");
    theme::hint(
        ui,
        "Public, relayed to every peer, and permanent for as long as nodes keep it. \
         Each message costs proof-of-work, mined by your node. Your name here is \
         your node's peer id.",
    );
    ui.add_space(4.0);

    // ── History ─────────────────────────────────────────────────────────
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        // Fixed viewport: the room keeps the same footprint whether it holds
        // no messages or a thousand, and scrolls within it.
        ui.set_min_height(320.0);
        if app.chat.is_empty() {
            theme::hint(ui, "Nothing yet — either the room is quiet or history is still arriving from peers.");
        } else {
            egui::ScrollArea::vertical()
                .max_height(320.0)
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for m in &app.chat {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(
                                RichText::new(short_hex(&m.sender, 6))
                                    .font(FontId::monospace(11.0))
                                    .color(theme::faint()),
                            );
                            ui.label(RichText::new(&m.text).color(theme::ink()));
                            if m.sealed {
                                ui.label(
                                    RichText::new("[sealed message]")
                                        .font(FontId::monospace(10.0))
                                        .color(theme::muted()),
                                );
                            } else if m.attachments > 0 {
                                ui.label(
                                    RichText::new(format!("[{} attachment(s)]", m.attachments))
                                        .font(FontId::monospace(10.0))
                                        .color(theme::muted()),
                                );
                            }
                            ui.label(RichText::new(ago(m.timestamp)).size(11.0).color(theme::faint()));
                        });
                    }
                });
        }
    });

    // ── Composer ────────────────────────────────────────────────────────
    ui.add_space(6.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());

        let mut send_now = false;
        ui.horizontal(|ui| {
            let r = ui.add(
                TextEdit::singleline(&mut app.chat_input)
                    .hint_text("up to ten dictionary words")
                    .font(egui::TextStyle::Monospace)
                    .desired_width(ui.available_width() - 90.0),
            );
            if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                send_now = true;
            }
            let ok = composer_ok(app);
            if ui
                .add_enabled(ok && !app.chat_busy, egui::Button::new(if app.chat_busy { "Mining…" } else { "Send" }))
                .clicked()
            {
                send_now = true;
            }
        });

        // Live dictionary validation + suggestions for the word being typed.
        let words: Vec<String> = app.chat_input.split_whitespace().map(str::to_lowercase).collect();
        let bad: Vec<&String> = words
            .iter()
            .filter(|w| !app.chat_dict.iter().any(|d| d.eq_ignore_ascii_case(w)))
            .collect();
        if words.len() > 10 {
            ui.label(RichText::new(format!("Ten words maximum ({} typed).", words.len())).color(theme::muted()).size(12.0));
        }
        if !bad.is_empty() && !app.chat_dict.is_empty() {
            ui.label(
                RichText::new(format!(
                    "Not in the dictionary: {}",
                    bad.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                ))
                .color(theme::muted())
                .size(12.0),
            );
            // Prefix suggestions for the last word.
            if let Some(last) = words.last() {
                let sugg: Vec<&String> = app
                    .chat_dict
                    .iter()
                    .filter(|d| d.starts_with(last.as_str()) && !d.eq_ignore_ascii_case(last))
                    .take(6)
                    .collect();
                if !sugg.is_empty() {
                    ui.horizontal_wrapped(|ui| {
                        theme::hint(ui, "try:");
                        let mut replace_with: Option<String> = None;
                        for s in sugg {
                            if ui.button(RichText::new(s).font(FontId::monospace(11.5))).clicked() {
                                replace_with = Some(s.clone());
                            }
                        }
                        if let Some(rep) = replace_with {
                            let mut parts: Vec<String> = app
                                .chat_input
                                .split_whitespace()
                                .map(str::to_string)
                                .collect();
                            if let Some(l) = parts.last_mut() {
                                *l = rep;
                            }
                            app.chat_input = parts.join(" ") + " ";
                        }
                    });
                }
            }
        }
        theme::hint(
            ui,
            "The chat vocabulary is fixed — every node knows the same word list, which \
             keeps messages tiny and spam expensive.",
        );

        // The whole vocabulary, browsable and click-to-insert.
        egui::CollapsingHeader::new(format!("Dictionary ({} words)", app.chat_dict.len()))
            .id_salt("chat_dict")
            .show(ui, |ui| {
                ui.add(
                    TextEdit::singleline(&mut app.dict_filter)
                        .hint_text("filter")
                        .desired_width(180.0),
                );
                let filter = app.dict_filter.trim().to_lowercase();
                let mut insert: Option<String> = None;
                egui::ScrollArea::vertical()
                    .max_height(160.0)
                    .id_salt("dict_scroll")
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            for w in app
                                .chat_dict
                                .iter()
                                .filter(|w| filter.is_empty() || w.to_lowercase().contains(&filter))
                            {
                                if ui
                                    .button(RichText::new(w).font(FontId::monospace(11.0)))
                                    .on_hover_text("add to your message")
                                    .clicked()
                                {
                                    insert = Some(w.clone());
                                }
                            }
                        });
                    });
                if let Some(w) = insert {
                    if !app.chat_input.is_empty() && !app.chat_input.ends_with(' ') {
                        app.chat_input.push(' ');
                    }
                    app.chat_input.push_str(&w);
                }
            });

        if send_now && composer_ok(app) && !app.chat_busy {
            app.chat_busy = true;
            app.error.clear();
            app.go(&ctx, Action::ChatSend { text: app.chat_input.trim().to_string() });
        }
        if !app.error.is_empty() {
            ui.label(RichText::new(&app.error).color(theme::muted()).size(12.0));
        }
    });
}

fn composer_ok(app: &App) -> bool {
    let words: Vec<&str> = app.chat_input.split_whitespace().collect();
    !words.is_empty()
        && words.len() <= 10
        && (app.chat_dict.is_empty()
            || words
                .iter()
                .all(|w| app.chat_dict.iter().any(|d| d.eq_ignore_ascii_case(w))))
}
