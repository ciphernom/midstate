//! Burns (MSB-1): back a name, a Base account, a pool or a hub by burning
//! MDS, or notarize a document; and see what the chain burns against what it
//! issues.

use crate::app::App;
use crate::bridge::Action;
use crate::theme;
use crate::views::names::gmds;
use eframe::egui::{self, RichText, TextEdit, Ui};
use midstate_walletd::api::{BackingView, BurnRequest};
use midstate_walletd::burns::{self, Purpose};

/// What can be backed, as the form offers it: purpose, label, hint.
const PURPOSES: [(&str, &str, &str); 5] = [
    ("name", "A name", "nick or nick.mds"),
    ("evm", "A Base account", "0x and 40 hex characters"),
    ("pool", "A mining pool", "the pool's fee address"),
    ("hub", "A channel hub", "the hub's key, 64 hex characters"),
    ("key", "Any key or address", "64 hex characters, or an address"),
];

/// Burn sizes, in blocks of issuance.
const SIZES: [(&str, u64); 3] = [("an hour", 60), ("a day", 1_440), ("a week", 10_080)];

/// The burn being drafted.
#[derive(Clone, Debug, Default)]
pub struct BurnForm {
    /// Index into [`PURPOSES`].
    pub purpose: usize,
    pub subject: String,
    /// Index into [`SIZES`], or `SIZES.len()` for `custom`.
    pub size: usize,
    /// gMDS, when the size is custom.
    pub custom: String,
    pub label: String,
    pub confirm: bool,
}

impl BurnForm {
    /// A form backing `subject` as `purpose`.
    pub fn backing(purpose: &str, subject: &str) -> Self {
        BurnForm { purpose: PURPOSES.iter().position(|(p, _, _)| *p == purpose).unwrap_or(0), subject: subject.into(), ..Self::default() }
    }
}

/// A document picked to notarize: where it is, and its hashes.
#[derive(Clone, Debug)]
pub struct Doc {
    pub path: String,
    pub size: u64,
    pub blake3: String,
    pub sha256: String,
}

fn pct(x: f64) -> String {
    if x < 0.001 && x > 0.0 { "under 0.1%".into() } else { format!("{:.1}%", x * 100.0) }
}

pub fn show(app: &mut App, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    theme::heading(ui, "Burns");
    theme::hint(
        ui,
        "Burning MDS backs a name, an account, a pool or a hub: public proof that it cost something, which \
         wallets show and rank by. A burn weighs what it cost in block rewards, and its weight halves every \
         year, so backing keeps a steady price as rewards halve, and is kept up rather than bought once.",
    );
    chain(app, ui);
    back(app, ui, &ctx);
    mine(app, ui);
    notarize(app, ui, &ctx);
    leaders(app, ui);
    recent(app, ui);
}

/// What the chain burns against what it issues.
fn chain(app: &App, ui: &mut Ui) {
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        let Some(b) = &app.burns else {
            theme::hint(ui, "Reading the node's burn index…");
            return;
        };
        if !b.note.is_empty() {
            theme::hint(ui, &b.note);
        }
        let share = if b.issued > 0 { b.burned as f64 / b.issued as f64 } else { 0.0 };
        ui.horizontal(|ui| {
            ui.label(RichText::new("The last 365 days").strong());
            theme::hint(ui, &format!("issued {}, burned {} — {} of issuance", gmds(b.issued), gmds(b.burned), pct(share)));
        });
        theme::progress_bar(ui, share.min(1.0) as f32);
        theme::hint(
            ui,
            &format!("Names {} · backing {} · other {}", gmds(b.names_burned), gmds(b.backing_burned), gmds(b.other_burned)),
        );
        if b.year > 0 {
            let y = if b.year_issued > 0 { b.year_burned as f64 / b.year_issued as f64 } else { 0.0 };
            theme::hint(ui, &format!("Halving year {} so far: issued {}, burned {} ({}).", b.year, gmds(b.year_issued), gmds(b.year_burned), pct(y)));
        }
        theme::hint(
            ui,
            &format!(
                "Backing held, everyone's: {}. Keeping it up burns {} of each year's issuance; the chain aims to burn a little less than it issues.",
                burns::blocks_text(b.held),
                pct(b.upkeep_share)
            ),
        );
        theme::hint(ui, &format!("A block of issuance is {} now.", gmds(b.reward)));
    });
}

/// Units the form would burn now, if it can say.
fn units(app: &App) -> Option<u64> {
    let f = &app.burn_form;
    match SIZES.get(f.size) {
        Some((_, blocks)) => Some(burns::cost(*blocks, app.burns.as_ref()?.tip)),
        None => {
            let x: f64 = f.custom.trim().parse().ok()?;
            (x > 0.0 && x < 1e9).then(|| (x * (1u64 << 30) as f64).round() as u64)
        }
    }
}

/// The form: what to back, by how much.
fn back(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    theme::heading(ui, "Back something");
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal_wrapped(|ui| {
            for (i, (_, label, _)) in PURPOSES.iter().enumerate() {
                if ui.selectable_label(app.burn_form.purpose == i, *label).clicked() && app.burn_form.purpose != i {
                    app.burn_form = BurnForm { purpose: i, ..BurnForm::default() };
                }
            }
        });
        let (word, _, hint) = PURPOSES[app.burn_form.purpose];
        let purpose = Purpose::parse(word).unwrap_or(Purpose::NAME);
        ui.add(TextEdit::singleline(&mut app.burn_form.subject).hint_text(hint).desired_width(f32::INFINITY));
        // This wallet's own, one click away.
        let own: Vec<BackingView> = app.burns.as_ref().map(|b| b.mine.iter().filter(|m| m.purpose == word).cloned().collect()).unwrap_or_default();
        if !own.is_empty() {
            ui.horizontal_wrapped(|ui| {
                theme::hint(ui, "Yours:");
                for m in own {
                    if ui.small_button(&m.subject).clicked() {
                        app.burn_form.subject = m.subject.clone();
                    }
                }
            });
        }
        let subject = burns::parse_subject(purpose, &app.burn_form.subject);
        if let (Err(e), false) = (&subject, app.burn_form.subject.trim().is_empty()) {
            ui.label(RichText::new(e.to_string()).color(theme::red()).size(12.0));
        }
        ui.horizontal(|ui| {
            theme::hint(ui, "Burn");
            for (i, (label, _)) in SIZES.iter().enumerate() {
                ui.radio_value(&mut app.burn_form.size, i, *label);
            }
            ui.radio_value(&mut app.burn_form.size, SIZES.len(), "gMDS:");
            if app.burn_form.size == SIZES.len() {
                ui.add(TextEdit::singleline(&mut app.burn_form.custom).hint_text("1.5").desired_width(70.0));
            } else {
                theme::hint(ui, "of issuance");
            }
        });
        ui.add(TextEdit::singleline(&mut app.burn_form.label).hint_text("a note kept with it (optional, 43 bytes)").char_limit(burns::LABEL_MAX).desired_width(f32::INFINITY));
        let tip = app.burns.as_ref().map_or(0, |b| b.tip);
        let Some(amount) = units(app) else { return };
        let weight = burns::weight(amount, tip);
        theme::hint(
            ui,
            &format!("Burns {}: weighs {} now, {} a year from now.", gmds(amount), burns::blocks_text(weight), burns::blocks_text(weight / 2.0)),
        );
        let Ok(subject) = subject else { return };
        let shown = burns::subject_text(purpose, &subject);
        if !app.burn_form.confirm {
            if ui.add_enabled(!app.busy, egui::Button::new(RichText::new("Burn…").strong())).clicked() {
                app.burn_form.confirm = true;
            }
            return;
        }
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.label(
                RichText::new(format!(
                    "Burns {} for {} {shown}, plus a network fee of a few hundred units. Nobody can spend what is burned, \
                     you included: it backs {shown} for good, at half the weight each year.",
                    gmds(amount),
                    word
                ))
                .size(12.5),
            );
            ui.horizontal(|ui| {
                if ui.add_enabled(!app.busy, egui::Button::new(RichText::new("Confirm").strong())).clicked() {
                    app.busy = true;
                    app.error.clear();
                    let req = BurnRequest { purpose: word.into(), subject: app.burn_form.subject.trim().into(), label: app.burn_form.label.clone(), units: amount };
                    app.go(ctx, Action::Burn { req });
                }
                if ui.button("Cancel").clicked() {
                    app.burn_form.confirm = false;
                }
            });
        });
        if !app.error.is_empty() {
            ui.label(RichText::new(&app.error).color(theme::red()));
        }
    });
}

fn backing_line(ui: &mut Ui, b: &BackingView) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(&b.subject).monospace().size(12.0));
        if !b.tier.is_empty() {
            theme::badge(ui, &b.tier, theme::gold());
        }
        let burns = if b.count == 1 { "1 burn".to_string() } else { format!("{} burns", b.count) };
        theme::hint(ui, &if b.count > 0 { format!("{} · {burns}", b.text) } else { b.text.clone() });
        if !b.label.is_empty() {
            theme::hint(ui, &format!("“{}”", b.label));
        }
    });
}

/// This wallet's names and account, and what backs them.
fn mine(app: &mut App, ui: &mut Ui) {
    let Some(b) = app.burns.clone() else { return };
    if b.mine.is_empty() {
        return;
    }
    theme::heading(ui, "Your backing");
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        theme::hint(ui, "Wallets show a name's backing to everyone who pays it or reads its messages: an hour of issuance makes it backed, a day well backed, a week strongly backed.");
        for m in &b.mine {
            ui.horizontal(|ui| {
                backing_line(ui, m);
                if ui.small_button("Back…").clicked() {
                    app.burn_form = BurnForm::backing(&m.purpose, &m.subject);
                }
            });
        }
    });
}

/// Timestamp a document: its hash on chain, for anyone to check later.
fn notarize(app: &mut App, ui: &mut Ui, ctx: &egui::Context) {
    theme::heading(ui, "Notarize a document");
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        theme::hint(
            ui,
            "Puts a document's hash on chain, so anyone can later check it existed by then and has not changed. \
             The document stays here: only its hash leaves. Drop a file on this window, or give its path.",
        );
        ui.horizontal(|ui| {
            ui.add(TextEdit::singleline(&mut app.doc_path).hint_text("path to a file, or a hash (64 hex) to look up").desired_width(ui.available_width() - 90.0));
            if ui.button("Check").clicked() {
                let t = app.doc_path.trim().to_string();
                if t.len() == 64 && hex::decode(&t).is_ok() {
                    app.doc = None;
                    app.notary = None;
                    app.go(ctx, Action::LookupNotary { hash: t.to_lowercase() });
                } else if !t.is_empty() {
                    app.go(ctx, Action::HashDocument { path: t });
                }
            }
        });
        if let Some(d) = app.doc.clone() {
            ui.label(RichText::new(format!("{} — {} bytes", d.path, d.size)).size(12.0));
            ui.label(RichText::new(format!("BLAKE3   {}", d.blake3)).monospace().size(10.5).color(theme::muted()));
            ui.label(RichText::new(format!("SHA-256  {}", d.sha256)).monospace().size(10.5).color(theme::muted()));
        }
        if let Some(n) = app.notary.clone() {
            if n.marks.is_empty() {
                theme::hint(ui, "Not notarized on this chain yet.");
            } else {
                let first = &n.marks[0];
                ui.label(
                    RichText::new(format!("Notarized at block {} ({}): it existed by then.", first.height, theme::fmt_dt(first.time))).strong(),
                );
                for m in &n.marks {
                    let burned = if m.value > 0 { format!(", burning {}", gmds(m.value)) } else { String::new() };
                    theme::hint(ui, &format!("block {} · {} · {}{burned} {}", m.height, theme::fmt_date(m.time), m.purpose, m.label));
                }
            }
        }
        let Some(d) = app.doc.clone() else { return };
        ui.horizontal(|ui| {
            ui.add(TextEdit::singleline(&mut app.burn_form.label).hint_text("what it is (optional, 43 bytes)").char_limit(burns::LABEL_MAX).desired_width(260.0));
            for (word, hash, what) in [("notary", &d.blake3, "Notarize (BLAKE3)"), ("notary-sha256", &d.sha256, "Notarize (SHA-256)")] {
                if ui.add_enabled(!app.busy, egui::Button::new(what)).on_hover_text("Costs the network fee and nothing more").clicked() {
                    app.busy = true;
                    app.error.clear();
                    let req = BurnRequest { purpose: word.into(), subject: hash.clone(), label: app.burn_form.label.clone(), units: 0 };
                    app.go(ctx, Action::Burn { req });
                }
            }
        });
    });
}

/// The most-backed names, pools and hubs.
fn leaders(app: &mut App, ui: &mut Ui) {
    let Some(b) = app.burns.clone() else { return };
    for (title, list, purpose) in [("Most-backed names", &b.top_names, "name"), ("Most-backed pools", &b.top_pools, "pool"), ("Most-backed hubs", &b.top_hubs, "hub")] {
        if list.is_empty() {
            continue;
        }
        theme::heading(ui, title);
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            for (i, x) in list.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{:>2}.", i + 1)).monospace().color(theme::muted()));
                    backing_line(ui, x);
                    if ui.small_button("Back…").clicked() {
                        app.burn_form = BurnForm::backing(purpose, &x.subject);
                    }
                });
            }
        });
    }
}

/// The newest MSB-1 burns on the chain.
fn recent(app: &App, ui: &mut Ui) {
    let Some(b) = &app.burns else { return };
    if b.recent.is_empty() {
        return;
    }
    theme::heading(ui, "Recent burns");
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        for r in &b.recent {
            let subject = if r.purpose.starts_with("notary") { theme::short_hex(&r.subject, 12) } else { r.subject.clone() };
            let what = if r.purpose.starts_with("notary") { "notarized".to_string() } else { format!("{:.1} blocks for", r.blocks) };
            theme::hint(ui, &format!("block {} · {} {} {} {}", r.height, what, r.purpose, subject, r.label));
        }
    });
}
