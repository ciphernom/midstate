use crate::app::App;
use crate::bridge::Action;
use crate::theme::{self, units};
use midstate_walletd::api::AddressInfo;
use eframe::egui::{self, Color32, ColorImage, RichText, TextEdit, TextureOptions, Ui};
use qrcode::QrCode;

pub fn show(app: &mut App, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    theme::heading(ui, "Receive");

    ui.columns(2, |cols| {
        // Left: generator
        theme::panel_frame().show(&mut cols[0], |ui| {
            ui.set_width(ui.available_width());
            theme::hint(ui, "Label (optional, stored locally)");
            ui.add(
                TextEdit::singleline(&mut app.recv_label)
                    .hint_text("e.g. mining payouts")
                    .desired_width(f32::INFINITY),
            );
            ui.add_space(6.0);
            ui.checkbox(&mut app.recv_mss, "Reusable address");
            theme::hint(
                ui,
                &format!(
                    "Good for {} incoming payments. Untick for a strictly one-time \
                     address — safest, but never share it twice.",
                    units(1024)
                ),
            );
            ui.add_space(10.0);
            theme::right_aligned(ui, |ui| {
                if ui
                    .add_enabled(!app.busy, egui::Button::new(RichText::new("New address").strong()))
                    .clicked()
                {
                    app.busy = true;
                    app.error.clear();
                    let label = {
                        let l = app.recv_label.trim();
                        if l.is_empty() { None } else { Some(l.to_string()) }
                    };
                    app.go(&ctx, Action::NewAddress { mss: app.recv_mss, label });
                }
            });
            if !app.error.is_empty() {
                ui.label(RichText::new(&app.error).color(theme::red()));
            }
        });

        // Right: current address + QR
        theme::panel_frame().show(&mut cols[1], |ui| {
            ui.set_width(ui.available_width());
            match app.current_addr.clone() {
                None => {
                    ui.add_space(50.0);
                    ui.vertical_centered(|ui| {
                        theme::hint(ui, "Generate an address to show it here with a QR code.");
                    });
                    ui.add_space(50.0);
                }
                Some(a) => {
                    ensure_qr(app, &ctx, &a.address);
                    ui.vertical_centered(|ui| {
                        if let Some((_, tex)) = &app.qr {
                            ui.image(egui::load::SizedTexture::new(tex.id(), egui::vec2(168.0, 168.0)));
                        }
                        ui.add_space(6.0);
                        ui.label(
                            RichText::new(theme::grouped_hash(&a.address, 24))
                                .monospace()
                                .size(11.0),
                        );
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.add_space(ui.available_width() / 2.0 - 90.0);
                            theme::badge(
                                ui,
                                if a.kind == "mss" { "reusable" } else { "one-time" },
                                if a.kind == "mss" { theme::gold() } else { theme::muted() },
                            );
                            let now = ui.input(|i| i.time);
                            let copied = app.copied_at.map(|t| now - t < 1.5).unwrap_or(false);
                            if ui.button(if copied { "Copied" } else { "Copy address" }).clicked() {
                                ui.ctx().copy_text(a.address.clone());
                                app.copied_at = Some(now);
                            }
                        });
                    });
                }
            }
        });
    });

    theme::heading(ui, "Your addresses");
    theme::hint(
        ui,
        "Select any reusable address that still has signatures, or any one-time address whose \
         key has not signed yet, to show it again below. A one-time key stays safe until it \
         signs — but every coin at that address must then be spent in a single transaction, \
         so receiving there repeatedly builds up one large mandatory spend.",
    );
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        if app.addresses.is_empty() {
            theme::hint(ui, "No addresses yet.");
            return;
        }
        // Collected outside the closure so we can mutate app state afterward.
        let mut reselect: Option<AddressInfo> = None;
        let current = app.current_addr.as_ref().map(|a| a.address.clone());
        // Live coins per address. A one-time key stays safe until it signs, but
        // every coin at it must be spent in ONE transaction (the co-spend rule),
        // and a normal send caps out at 256 inputs — so the count matters.
        let mut coins_at: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for c in app.coins.iter().filter(|c| c.live) {
            *coins_at.entry(c.address.clone()).or_insert(0) += 1;
        }

        egui::Grid::new("addresses").num_columns(5).spacing([20.0, 8.0]).striped(true).show(ui, |ui| {
            header(ui, &["Address", "Type", "Label", "Status", ""]);
            for a in &app.addresses {
                ui.label(theme::mono(theme::short_hex(&a.address, 10)).color(theme::muted()).size(12.0));
                theme::badge(
                    ui,
                    if a.kind == "mss" { "reusable" } else { "one-time" },
                    if a.kind == "mss" { theme::gold() } else { theme::muted() },
                );
                ui.label(RichText::new(a.label.as_deref().unwrap_or("—")).color(theme::muted()));

                // Eligibility: MSS keeps working while it has signatures; a
                // one-time key is safe only until it has been used.
                let reusable = match (a.kind.as_str(), a.used, a.remaining_sigs) {
                    ("mss", _, Some(r)) => {
                        let low = r == 0;
                        ui.label(
                            theme::mono(format!("{} sigs left", units(r)))
                                .color(if low { theme::amber() } else { theme::muted() })
                                .size(12.0),
                        );
                        r > 0
                    }
                    (_, true, _) => {
                        // The key has signed. A second signature would leak it,
                        // so the wallet refuses to import anything more here.
                        ui.label(
                            RichText::new("spent — coins sent here cannot be recovered")
                                .color(theme::amber())
                                .size(12.0),
                        );
                        false
                    }
                    _ => {
                        let n = coins_at.get(&a.address).copied().unwrap_or(0);
                        if n == 0 {
                            ui.label(RichText::new("unused").color(theme::green()).size(12.0));
                        } else {
                            // Key unsigned, so still safe to receive to — but the
                            // siblings pile up into one mandatory spend.
                            let tight = n >= 200;
                            ui.label(
                                RichText::new(format!("unsigned key · {n} coin(s) to co-spend"))
                                    .color(if tight { theme::amber() } else { theme::muted() })
                                    .size(12.0),
                            );
                        }
                        true
                    }
                };

                let is_current = current.as_deref() == Some(a.address.as_str());
                if is_current {
                    ui.label(RichText::new("showing").color(theme::muted()).size(11.0));
                } else if reusable {
                    if ui.button(RichText::new("Select").size(11.0)).clicked() {
                        reselect = Some(a.clone());
                    }
                } else {
                    ui.label("");
                }
                ui.end_row();
            }
        });

        if let Some(a) = reselect {
            // Local-only: current_addr already holds an AddressInfo and the QR
            // panel re-renders from it. No wallet round-trip needed.
            app.copied_at = None;
            app.current_addr = Some(a);
        }
    });
}

fn header(ui: &mut Ui, cols: &[&str]) {
    for c in cols {
        ui.label(
            RichText::new(c.to_uppercase())
                .font(egui::FontId::monospace(10.0))
                .color(theme::muted()),
        );
    }
    ui.end_row();
}

/// Build (and cache) the QR texture for the current address.
fn ensure_qr(app: &mut App, ctx: &egui::Context, addr: &str) {
    if app.qr.as_ref().map(|(a, _)| a == addr).unwrap_or(false) {
        return;
    }
    let Ok(code) = QrCode::new(addr.as_bytes()) else { return };
    let w = code.width();
    let colors = code.to_colors();
    let margin = 4usize;
    let n = w + margin * 2;
    let mut pixels = vec![Color32::WHITE; n * n];
    for y in 0..w {
        for x in 0..w {
            if colors[y * w + x] == qrcode::Color::Dark {
                pixels[(y + margin) * n + (x + margin)] = Color32::BLACK;
            }
        }
    }
    let img = ColorImage { size: [n, n], pixels };
    let tex = ctx.load_texture(format!("qr:{addr}"), img, TextureOptions::NEAREST);
    app.qr = Some((addr.to_string(), tex));
}
