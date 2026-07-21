//! Network chat. Messages are up to ten words from a fixed dictionary,
//! each message proof-of-work-mined by your own node before broadcast.
//! This same channel (with attachments) is the transport qbolt payment
//! channels will ride on.

use crate::app::App;
use crate::bridge::Action;
use crate::theme::{self, ago, short_hex};
use eframe::egui::{self, FontId, RichText, TextEdit, Ui};

pub fn show(app: &mut App, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
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
        if app.chat.is_empty() {
            theme::hint(ui, "Nothing yet — either the room is quiet or history is still arriving from peers.");
        } else {
            egui::ScrollArea::vertical()
                .max_height(340.0)
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
                            if m.attachments > 0 {
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
