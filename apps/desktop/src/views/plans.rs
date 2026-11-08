//! Inheritance plans and payment streams (walletd `plans`,
//! `docs/design/inheritance-and-streams.md`).
//!
//! Everything here runs between wallets: no server, no operator. People swap
//! codes (a payee code to be named or paid, a plan or stream code to follow
//! one), and the chain's covenants do the rest. Every action that moves coins
//! is two presses: the first shows what it binds and what can go wrong.

use crate::app::App;
use crate::bridge::Action;
use crate::theme::{self, compact_units, short_hex, UNIT_NAMES};
use eframe::egui::{self, RichText, TextEdit, Ui};
use midstate_walletd::api::PlansView;

/// Blocks a day at one a minute.
const DAY: u64 = 1_440;
/// Units an amount is typed in: kMDS, mMDS, gMDS.
const UNITS: [usize; 3] = [1, 2, 3];

pub fn show(app: &mut App, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    let blocked = app.handle.is_none();
    ui.label(RichText::new("Plans").font(theme::font_medium(20.0)));
    theme::hint(
        ui,
        "Inheritance plans and payment streams, enforced by midstate's scripts between wallets: nobody holds \
         your coins, and there is no server to trust or to go away.",
    );
    ui.add_space(8.0);
    let tabs = ["Inheritance", "Streams", "Codes"];
    if theme::segmented(ui, &tabs, &mut app.plans_tab) {
        app.plans_confirm = None;
    }
    ui.add_space(8.0);
    let Some(v) = app.plans.clone() else {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.spinner();
                theme::hint(ui, "Loading…");
            });
        });
        return;
    };
    if !v.blocked.is_empty() {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Not available yet").font(theme::font_medium(14.0)));
            theme::hint(ui, &v.blocked);
        });
        return;
    }
    if v.payouts_waiting > 0 || v.payouts_sweeping > 0 {
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(theme::mono(compact_units(v.payouts_waiting + v.payouts_sweeping)).color(theme::ink()));
                theme::hint(
                    ui,
                    if v.payouts_sweeping > 0 {
                        "paid to you, on its way into your balance"
                    } else {
                        "paid to you, waiting to be moved into your balance (the wallet does it within hours)"
                    },
                );
            });
            if v.payouts_waiting > 0 {
                theme::right_aligned(ui, |ui| {
                    if ui.add_enabled(!app.busy && !blocked, egui::Button::new("Move it now")).clicked() {
                        act(app, &ctx, Action::SweepPayouts);
                    }
                });
            }
        });
        ui.add_space(8.0);
    }
    match app.plans_tab {
        0 => inheritance(app, ui, &ctx, &v, blocked),
        1 => streams(app, ui, &ctx, &v, blocked),
        _ => codes(app, ui, &ctx, blocked),
    }
    if !app.plans_notice.is_empty() {
        ui.add_space(6.0);
        ui.label(RichText::new(&app.plans_notice).size(12.0).color(theme::bright()));
    }
    if app.busy {
        ui.add_space(6.0);
        ui.spinner();
    }
    if !app.error.is_empty() {
        ui.add_space(6.0);
        ui.label(RichText::new(&app.error).size(12.0).color(theme::bright()));
    }
}

fn act(app: &mut App, ctx: &egui::Context, action: Action) {
    app.busy = true;
    app.error.clear();
    app.plans_notice.clear();
    app.plans_confirm = None;
    app.go(ctx, action);
}

/// A two-press button: the first press shows `warning`, the second acts.
#[allow(clippy::too_many_arguments)]
fn confirm(
    app: &mut App,
    ui: &mut Ui,
    ctx: &egui::Context,
    key: String,
    label: &str,
    confirm_label: &str,
    warning: &[String],
    enabled: bool,
    action: impl FnOnce() -> Action,
) {
    let armed = app.plans_confirm.as_deref() == Some(key.as_str());
    if armed {
        for w in warning {
            theme::hint(ui, w);
        }
    }
    theme::right_aligned(ui, |ui| {
        if armed {
            if ui.add_enabled(enabled, egui::Button::new(confirm_label)).clicked() {
                act(app, ctx, action());
            }
            if ui.small_button("Back").clicked() {
                app.plans_confirm = None;
            }
        } else if ui.add_enabled(enabled, egui::Button::new(label)).clicked() {
            app.plans_confirm = Some(key);
            app.plans_notice.clear();
        }
    });
}

fn amount_row(ui: &mut Ui, label: &str, text: &mut String, unit: &mut usize) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).size(12.0).color(theme::muted()));
        changed |= ui.add(TextEdit::singleline(text).font(egui::TextStyle::Monospace).desired_width(120.0)).changed();
        let labels: Vec<&str> = UNITS.iter().map(|u| UNIT_NAMES[*u]).collect();
        let mut pick = UNITS.iter().position(|u| u == unit).unwrap_or(2);
        if theme::segmented(ui, &labels, &mut pick) {
            *unit = UNITS[pick];
            changed = true;
        }
    });
    changed
}

fn number_row(ui: &mut Ui, label: &str, text: &mut String, after: &str) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).size(12.0).color(theme::muted()));
        if ui.add(TextEdit::singleline(text).font(egui::TextStyle::Monospace).desired_width(70.0)).changed() {
            text.retain(|c| c.is_ascii_digit() || c == '.');
            changed = true;
        }
        theme::hint(ui, after);
    });
    changed
}

fn copy_row(ui: &mut Ui, what: &str, code: &str) {
    ui.horizontal(|ui| {
        theme::hint(ui, what);
        ui.label(theme::mono(short_hex(code, 18)).size(11.0).color(theme::faint()));
        if ui.small_button("Copy").clicked() {
            ui.ctx().copy_text(code.to_string());
        }
    });
}

fn pct(bps: u32) -> String {
    let s = format!("{}.{:02}", bps / 100, bps % 100);
    let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    format!("{s}%")
}

// ── Inheritance ─────────────────────────────────────────────────────────

fn inheritance(app: &mut App, ui: &mut Ui, ctx: &egui::Context, v: &PlansView, blocked: bool) {
    plan_form(app, ui, ctx, blocked);
    if !v.owned.is_empty() {
        ui.add_space(10.0);
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Your plans").font(theme::font_medium(14.0)));
            for p in &v.owned {
                ui.add_space(6.0);
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&p.label).size(13.0).color(theme::ink()));
                    theme::badge(ui, &p.status, theme::muted());
                    if p.overdue {
                        theme::badge(ui, "past its deadline", theme::bright());
                    } else if p.due_soon {
                        theme::badge(ui, "check in soon", theme::bright());
                    }
                    ui.label(theme::mono(compact_units(p.balance)).size(12.0));
                });
                let heirs: Vec<String> = p.heirs.iter().map(|(n, b)| format!("{} {}", if n.is_empty() { "(unnamed)" } else { n }, pct(*b))).collect();
                theme::hint(ui, &format!("Heirs: {}. Check in every {} days.", heirs.join(", "), p.period_days));
                if !p.deadline_text.is_empty() {
                    theme::hint(
                        ui,
                        &if p.overdue {
                            format!(
                                "The deadline passed ({}). Anyone may pay your heirs now; check in to stop that, if \
                                 nobody has yet.",
                                p.deadline_text
                            )
                        } else {
                            format!("Your heirs can claim {} unless you check in first.", p.deadline_text)
                        },
                    );
                }
                if !p.note.is_empty() {
                    theme::hint(ui, &p.note);
                }
                copy_row(ui, "Plan code (give to each heir; keep a copy):", &p.code);
                let can = !app.busy && !blocked;
                if p.can_check_in {
                    confirm(
                        app,
                        ui,
                        ctx,
                        format!("checkin:{}", p.id),
                        "Check in",
                        "Confirm: check in",
                        &[format!(
                            "Moves the plan's coins to the next deadline, at least {} days from now, paying a small \
                             fee from them. One signature of your plan key ({} left).",
                            p.period_days, p.owner_signatures_left
                        )],
                        can,
                        || Action::CheckIn { id: p.id.clone() },
                    );
                }
                if p.can_close {
                    confirm(
                        app,
                        ui,
                        ctx,
                        format!("close:{}", p.id),
                        "Close plan",
                        "Confirm: close it",
                        &[format!(
                            "Takes all {} back into your balance, less a small fee. Your heirs get nothing from this \
                             plan afterwards.",
                            compact_units(p.balance)
                        )],
                        can,
                        || Action::ClosePlan { id: p.id.clone() },
                    );
                }
            }
        });
    }
    if !v.held.is_empty() {
        ui.add_space(10.0);
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Plans naming you").font(theme::font_medium(14.0)));
            for p in &v.held {
                ui.add_space(6.0);
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&p.label).size(13.0).color(theme::ink()));
                    if p.claimable {
                        theme::badge(ui, "claimable", theme::bright());
                    }
                    ui.label(theme::mono(compact_units(p.balance)).size(12.0));
                    theme::hint(ui, &format!("your share {}", pct(p.my_share_bps)));
                });
                let others: Vec<String> = p.heirs.iter().map(|(n, me)| if *me { "you".into() } else { n.clone() }).collect();
                theme::hint(ui, &format!("Heirs: {}.", others.join(", ")));
                if p.balance == 0 {
                    theme::hint(ui, "Nothing is in the plan now: it was claimed, closed, or not yet funded.");
                } else if p.claimable {
                    theme::hint(ui, &format!("The owner has not checked in: the deadline passed {}.", p.deadline_text));
                } else {
                    theme::hint(ui, &format!("Claimable {} if the owner does not check in.", p.deadline_text));
                }
                if p.claimable {
                    confirm(
                        app,
                        ui,
                        ctx,
                        format!("claim:{}", p.id),
                        "Pay the heirs",
                        "Confirm: pay every heir",
                        &["Pays every heir its share from the plan, you included; 1/65,536 of it goes to the fee. \
                           The owner can still spend the coins until this confirms. Your share then comes into \
                           your balance by itself."
                            .to_string()],
                        !app.busy && !blocked,
                        || Action::ClaimPlan { id: p.id.clone() },
                    );
                }
            }
        });
    }
}

fn plan_form(app: &mut App, ui: &mut Ui, ctx: &egui::Context, blocked: bool) {
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Make an inheritance plan").font(theme::font_medium(14.0)));
        theme::hint(
            ui,
            "Your coins stay yours: you spend or take them back whenever you like. Check in once a period (one \
             transaction) and nothing else happens. Miss a deadline and anyone may move the coins, but only by paying \
             each heir its share. Heirs give you their payee codes (the Codes tab).",
        );
        ui.add_space(6.0);
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label(RichText::new("Name").size(12.0).color(theme::muted()));
            changed |= ui.add(TextEdit::singleline(&mut app.plan_label).hint_text("For the children").desired_width(220.0)).changed();
        });
        changed |= amount_row(ui, "Amount", &mut app.plan_amount, &mut app.plan_unit);
        changed |= number_row(ui, "Check in every", &mut app.plan_days, "days");
        ui.add_space(4.0);
        let mut remove = None;
        for (i, (code, share)) in app.plan_heirs.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("Heir {}", i + 1)).size(12.0).color(theme::muted()));
                changed |= ui
                    .add(TextEdit::singleline(code).hint_text("mdpayee1…").font(egui::TextStyle::Monospace).desired_width(280.0))
                    .changed();
                changed |= ui.add(TextEdit::singleline(share).hint_text("%").desired_width(50.0)).changed();
                theme::hint(ui, "%");
                if ui.small_button("✕").clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            app.plan_heirs.remove(i);
            changed = true;
        }
        if app.plan_heirs.len() < 8 && ui.small_button("Add an heir").clicked() {
            app.plan_heirs.push((String::new(), String::new()));
            changed = true;
        }
        if changed {
            app.plans_confirm = None;
        }

        let amount = theme::parse_in_unit(&app.plan_amount, app.plan_unit).ok().filter(|a| *a > 0);
        let days: Option<u64> = app.plan_days.parse().ok().filter(|d| (1..=3_660).contains(d));
        let bps: Vec<Option<u32>> = app
            .plan_heirs
            .iter()
            .map(|(_, s)| s.trim().parse::<f64>().ok().filter(|p| *p > 0.0 && *p <= 100.0).map(|p| (p * 100.0).round() as u32))
            .collect();
        let mut problems = Vec::new();
        if amount.is_none() {
            problems.push("enter an amount".to_string());
        }
        if days.is_none() {
            problems.push("a check-in period is between a day and ten years".to_string());
        }
        if app.plan_heirs.is_empty() || app.plan_heirs.iter().any(|(c, _)| !c.trim().starts_with("mdpayee1")) {
            problems.push("each heir needs a payee code (mdpayee1…)".to_string());
        }
        if bps.iter().any(|b| b.is_none()) || bps.iter().flatten().sum::<u32>() != 10_000 {
            problems.push("the shares must add up to 100%".to_string());
        }
        let in_plan = amount.map(midstate_walletd::plans::plan_denominations).unwrap_or_default();
        if amount.is_some() && in_plan.is_empty() {
            problems.push(format!("a plan holds at least {}", compact_units(midstate_walletd::plans::MIN_COIN)));
        }
        let put: u64 = in_plan.iter().sum();
        if let (Some(a), true) = (amount, put > 0) {
            if put < a {
                theme::hint(
                    ui,
                    &format!(
                        "A plan holds at most eight coins: {} goes in, and {} stays in your balance.",
                        compact_units(put),
                        compact_units(a - put)
                    ),
                );
            }
        }
        for p in &problems {
            theme::hint(ui, &format!("Can't make it yet: {p}."));
        }
        let warning = vec![
            format!(
                "Check in at least every {} days. The wallet reminds you when a deadline is near, but only while it is \
                 open: a missed deadline lets anyone pay your heirs, and you cannot undo that once it confirms.",
                days.unwrap_or(0)
            ),
            "After it is funded, give each heir the plan's code, and keep a copy yourself: with your recovery phrase \
             it restores the plan in any wallet."
                .to_string(),
            "The heirs' shares and their keys are fixed once made. To change them, close the plan and make another."
                .to_string(),
        ];
        let ready = problems.is_empty() && !app.busy && !blocked;
        let heirs: Vec<(String, u32)> = app
            .plan_heirs
            .iter()
            .zip(&bps)
            .map(|((c, _), b)| (c.trim().to_string(), b.unwrap_or(0)))
            .collect();
        let label = if app.plan_label.trim().is_empty() { "Inheritance".to_string() } else { app.plan_label.trim().to_string() };
        let (period_days, amount) = (days.unwrap_or(0), put);
        confirm(
            app,
            ui,
            ctx,
            "plan".into(),
            "Make plan",
            &format!("Confirm: lock {} in the plan", compact_units(put)),
            &warning,
            ready,
            || Action::CreatePlan { label, heirs, period_days, amount },
        );
    });
}

// ── Streams ─────────────────────────────────────────────────────────────

fn streams(app: &mut App, ui: &mut Ui, ctx: &egui::Context, v: &PlansView, blocked: bool) {
    stream_form(app, ui, ctx, blocked);
    if !app.stream_code_out.is_empty() {
        ui.add_space(8.0);
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Give the recipient this code").font(theme::font_medium(14.0)));
            theme::hint(ui, "With it their wallet finds the payments and collects each one as it falls due.");
            copy_row(ui, "Stream code:", &app.stream_code_out.clone());
        });
    }
    if !v.sent.is_empty() {
        ui.add_space(10.0);
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Streams you pay").font(theme::font_medium(14.0)));
            for s in &v.sent {
                ui.add_space(6.0);
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&s.label).size(13.0).color(theme::ink()));
                    theme::badge(ui, &s.status, theme::muted());
                    theme::badge(ui, if s.cancellable { "cancellable" } else { "fixed" }, theme::muted());
                    if !s.recipient.is_empty() {
                        theme::hint(ui, &format!("to {}", s.recipient));
                    }
                });
                theme::hint(
                    ui,
                    &format!(
                        "{} every {:.1} days, {} payments; {} still on chain ({} not yet due). Next {}; last {}.",
                        compact_units(s.amount),
                        s.period_days,
                        s.count,
                        s.waiting,
                        s.not_yet_due,
                        if s.next_due_text.is_empty() { "—" } else { &s.next_due_text },
                        s.last_due_text
                    ),
                );
                if !s.note.is_empty() {
                    theme::hint(ui, &s.note);
                }
                copy_row(ui, "Stream code:", &s.code);
                if s.can_cancel {
                    confirm(
                        app,
                        ui,
                        ctx,
                        format!("cancel:{}", s.id),
                        "Stop future payments",
                        "Confirm: take them back",
                        &[format!(
                            "Takes back the {} payments not yet due into your balance, less a small fee. Payments \
                             already due stay the recipient's to collect.",
                            s.not_yet_due
                        )],
                        !app.busy && !blocked,
                        || Action::CancelStream { id: s.id.clone() },
                    );
                }
            }
        });
    }
    if !v.received.is_empty() {
        ui.add_space(10.0);
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Streams paying you").font(theme::font_medium(14.0)));
            for s in &v.received {
                ui.add_space(6.0);
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&s.label).size(13.0).color(theme::ink()));
                    theme::badge(ui, if s.cancellable { "payer can stop it" } else { "fixed" }, theme::muted());
                    if s.due_now > 0 {
                        theme::badge(ui, &format!("{} due", s.due_now), theme::bright());
                    }
                });
                theme::hint(
                    ui,
                    &format!(
                        "{} every {:.1} days, {} payments; {} still to come or collect. Next {}; last {}.",
                        compact_units(s.amount),
                        s.period_days,
                        s.count,
                        s.waiting,
                        if s.next_due_text.is_empty() { "—" } else { &s.next_due_text },
                        s.last_due_text
                    ),
                );
                if s.cancellable {
                    theme::hint(
                        ui,
                        "The payer can take back any payment that has not been collected, so this wallet collects \
                         each one as it falls due — while it is open.",
                    );
                }
                ui.horizontal(|ui| {
                    let mut on = s.auto_collect;
                    if ui.checkbox(&mut on, "Collect payments as they fall due").changed() {
                        act(app, ctx, Action::AutoCollect { id: s.id.clone(), on });
                    }
                });
                if s.due_now > 0 {
                    theme::right_aligned(ui, |ui| {
                        if ui.add_enabled(!app.busy && !blocked, egui::Button::new("Collect now")).clicked() {
                            act(app, ctx, Action::CollectStream { id: s.id.clone() });
                        }
                    });
                }
            }
        });
    }
}

fn stream_form(app: &mut App, ui: &mut Ui, ctx: &egui::Context, blocked: bool) {
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Start a payment stream").font(theme::font_medium(14.0)));
        theme::hint(
            ui,
            "Lock a run of payments now; each becomes the recipient's as it falls due: an allowance, a salary, rent. \
             The recipient gives you a payee code (the Codes tab).",
        );
        ui.add_space(6.0);
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label(RichText::new("Name").size(12.0).color(theme::muted()));
            changed |= ui.add(TextEdit::singleline(&mut app.stream_label).hint_text("Rent").desired_width(220.0)).changed();
        });
        ui.horizontal(|ui| {
            ui.label(RichText::new("Recipient").size(12.0).color(theme::muted()));
            changed |= ui
                .add(TextEdit::singleline(&mut app.stream_payee).hint_text("mdpayee1…").font(egui::TextStyle::Monospace).desired_width(f32::INFINITY))
                .changed();
        });
        changed |= amount_row(ui, "Each payment", &mut app.stream_amount, &mut app.stream_unit);
        changed |= number_row(ui, "Every", &mut app.stream_every, "days");
        changed |= number_row(ui, "Payments", &mut app.stream_count, "in all");
        changed |= number_row(ui, "First one in", &mut app.stream_delay, "days (0 for now)");
        if ui.checkbox(&mut app.stream_cancellable, "I can stop payments that are not yet due").changed() {
            changed = true;
        }
        if changed {
            app.plans_confirm = None;
        }
        let amount = theme::parse_in_unit(&app.stream_amount, app.stream_unit).ok().filter(|a| *a > 0);
        let every: Option<f64> = app.stream_every.parse().ok().filter(|d: &f64| *d > 0.0 && *d <= 366.0);
        let count: Option<u32> = app.stream_count.parse().ok().filter(|c: &u32| (1..=120).contains(c));
        let delay: Option<f64> = if app.stream_delay.is_empty() { Some(0.0) } else { app.stream_delay.parse().ok().filter(|d: &f64| *d >= 0.0) };
        let mut problems = Vec::new();
        if !app.stream_payee.trim().starts_with("mdpayee1") {
            problems.push("paste the recipient's payee code".to_string());
        }
        if amount.is_none() {
            problems.push("enter an amount per payment".to_string());
        }
        if every.is_none() {
            problems.push("payments are between a block and a year apart".to_string());
        }
        if count.is_none() {
            problems.push("one to 120 payments".to_string());
        }
        if delay.is_none() {
            problems.push("the first payment's delay is a number of days".to_string());
        }
        if let Some(a) = amount {
            let coins = a.count_ones();
            let min = midstate_walletd::plans::MIN_COIN;
            if a & (min - 1) != 0 || coins > 4 {
                problems.push(format!(
                    "each payment must be at most four coins of {} or more: try a rounder amount",
                    compact_units(min)
                ));
            }
        }
        if let (Some(a), Some(c), Some(e)) = (amount, count, every) {
            theme::hint(
                ui,
                &format!(
                    "Locks {} now: {} payments of {}, one every {e} days, the last about {:.0} days from the first.",
                    compact_units(a.saturating_mul(c as u64)),
                    c,
                    compact_units(a),
                    e * (c.saturating_sub(1)) as f64
                ),
            );
        }
        for p in &problems {
            theme::hint(ui, &format!("Can't start it yet: {p}."));
        }
        let mut warning = vec![
            "Each payment is the recipient's once it falls due: their wallet, or anyone, collects it for them; 1/65,536 \
             of it pays the fee."
                .to_string(),
            "Give the recipient the stream's code after you start it.".to_string(),
        ];
        warning.push(if app.stream_cancellable {
            "You can take back payments that are not yet due. A payment that is due but not yet collected could also \
             be taken back on chain; this wallet never does that."
                .to_string()
        } else {
            "Once funded, nothing can take these payments back, you included.".to_string()
        });
        let ready = problems.is_empty() && !app.busy && !blocked;
        let label = if app.stream_label.trim().is_empty() { "Stream".to_string() } else { app.stream_label.trim().to_string() };
        let payee = app.stream_payee.trim().to_string();
        let (amount_v, count_v) = (amount.unwrap_or(0), count.unwrap_or(0));
        let period_blocks = (every.unwrap_or(0.0) * DAY as f64).round().max(1.0) as u64;
        let delay_blocks = (delay.unwrap_or(0.0) * DAY as f64).round() as u64;
        let cancellable = app.stream_cancellable;
        confirm(
            app,
            ui,
            ctx,
            "stream".into(),
            "Start stream",
            &format!("Confirm: lock {}", compact_units(amount_v.saturating_mul(count_v as u64))),
            &warning,
            ready,
            || Action::CreateStream { label, payee, amount: amount_v, period_blocks, count: count_v, delay_blocks, cancellable },
        );
    });
}

// ── Codes ───────────────────────────────────────────────────────────────

fn codes(app: &mut App, ui: &mut Ui, ctx: &egui::Context, blocked: bool) {
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Your payee code").font(theme::font_medium(14.0)));
        theme::hint(
            ui,
            "Give this to someone who wants to name you as an heir or pay you a stream. Each code is a new key of \
             this wallet's, made from your recovery phrase; it lets them pay you, never spend for you.",
        );
        ui.horizontal(|ui| {
            ui.label(RichText::new("Your name for them").size(12.0).color(theme::muted()));
            ui.add(TextEdit::singleline(&mut app.payee_name).hint_text("Ann").desired_width(160.0));
            if ui.add_enabled(!app.busy && !blocked, egui::Button::new("Make code")).clicked() {
                app.busy = true;
                app.error.clear();
                app.go(ctx, Action::MakePayeeCode { name: app.payee_name.trim().to_string() });
            }
        });
        if !app.payee_code_out.is_empty() {
            copy_row(ui, "Payee code:", &app.payee_code_out.clone());
        }
    });
    ui.add_space(10.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Follow a plan or stream").font(theme::font_medium(14.0)));
        theme::hint(
            ui,
            "Paste a plan code an owner gave you, or a stream code from a payer. It must name one of this wallet's \
             payee codes. Owners and payers can paste their own codes too, to restore a plan or stream after \
             restoring this wallet.",
        );
        ui.add(
            TextEdit::multiline(&mut app.plan_import_code)
                .hint_text("mdplan1… or mdstream1…")
                .font(egui::TextStyle::Monospace)
                .desired_rows(2)
                .desired_width(f32::INFINITY),
        );
        ui.horizontal(|ui| {
            ui.label(RichText::new("Name it").size(12.0).color(theme::muted()));
            ui.add(TextEdit::singleline(&mut app.plan_import_label).hint_text("From Dad").desired_width(200.0));
        });
        theme::right_aligned(ui, |ui| {
            let ok = !app.plan_import_code.trim().is_empty() && !app.busy && !blocked;
            if ui.add_enabled(ok, egui::Button::new("Follow it")).clicked() {
                act(
                    app,
                    ctx,
                    Action::ImportPlanCode { code: app.plan_import_code.trim().to_string(), label: app.plan_import_label.trim().to_string() },
                );
            }
        });
    });
}
