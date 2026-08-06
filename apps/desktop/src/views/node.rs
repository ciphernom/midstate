//! The node panel. This app runs a full node in-process, so everything here
//! is read from local consensus state rather than asked of a third party —
//! which is the point, and worth showing rather than just asserting.

use crate::app::App;
use crate::bridge::Action;
use crate::theme::{self, ago, fmt_dt, short_hex, units};
use eframe::egui::{self, FontId, RichText, Ui};

pub fn show(app: &mut App, ui: &mut Ui) {
    theme::heading(ui, "Node");

    let Some(n) = app.node.clone() else {
        theme::hint(ui, "Reading node state…");
        return;
    };
    let syncing = app.sync.as_ref().map(|s| s.is_syncing).unwrap_or(false);

    // Ambient midstate — the chain's whole state compressed to 32 bytes.
    ui.label(
        RichText::new(theme::grouped_hash(&n.midstate, 8))
            .font(FontId::monospace(26.0))
            .color(theme::ambient()),
    );
    ui.add_space(2.0);

    // ── Chain tip ───────────────────────────────────────────────────────
    ui.columns(3, |c| {
        theme::stat(&mut c[0], "Height", &units(n.height), "blocks");
        theme::stat(&mut c[1], "UTXO coins", &units(n.utxo_count as u64), "");
        theme::stat(&mut c[2], "Block reward", &units(n.block_reward), "units");
    });
    ui.columns(3, |c| {
        theme::stat(&mut c[0], "Difficulty", &format!("{}", n.difficulty_bits), "leading zeros");
        theme::stat(&mut c[1], "Open commitments", &units(n.commitment_count as u64), "");
        theme::stat(&mut c[2], "Retired keys", &units(n.burned_count as u64), "one-time");
    });
    ui.add_space(6.0);

    // ── Tip detail ──────────────────────────────────────────────────────
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Chain tip").font(theme::font_medium(14.0)));

        let age = ago(n.tip_timestamp);
        row(ui, "last block", &format!("{}  ·  {}", fmt_dt(n.tip_timestamp), age), None);
        row_copy(ui, "header hash", &n.header_hash);
        row_copy(ui, "midstate", &n.midstate);
        row(ui, "cumulative work", &n.depth, None);
        row(
            ui,
            "confirmation depth",
            &format!("{} blocks", n.safe_depth),
            Some(
                "How deep a transaction must be before this node treats it as settled. It is \
                 estimated from recent chain behaviour, not fixed.",
            ),
        );
        row(
            ui,
            "mempool",
            &format!("{} transaction(s) waiting", n.mempool),
            None,
        );
        if syncing {
            theme::hint(
                ui,
                "Still syncing — these figures describe the chain as far as this node has \
                 verified it, not the network tip.",
            );
        }
    });

    // ── Peers ───────────────────────────────────────────────────────────
    ui.add_space(6.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.label(RichText::new("Peers").font(theme::font_medium(14.0)));
            theme::badge(
                ui,
                &format!("{}", n.peers.len()),
                if n.peers.is_empty() { theme::muted() } else { theme::ink() },
            );
        });
        if n.peers.is_empty() {
            theme::hint(
                ui,
                "No peers connected. The node keeps dialing the bootstrap list — if this \
                 persists, check whether outbound connections on port 9333 are blocked.",
            );
            return;
        }
        theme::hint(ui, "Node identities this wallet is currently connected to.");
        egui::ScrollArea::vertical()
            .max_height(180.0)
            .auto_shrink([false, false])
            .id_salt("peers")
            .show(ui, |ui| {
                for p in &n.peers {
                    ui.horizontal(|ui| {
                        ui.label(theme::mono(short_hex(p, 14)).size(11.0).color(theme::muted()));
                        if ui.button(RichText::new("copy").size(10.0)).clicked() {
                            ui.ctx().copy_text(p.clone());
                        }
                    });
                }
            });
    });

    // ── Local ───────────────────────────────────────────────────────────
    ui.add_space(6.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("This machine").font(theme::font_medium(14.0)));
        row_copy(ui, "data directory", &n.data_dir);
        if let Some(url) = &n.rpc_url {
            row_copy(ui, "rpc", url);
            theme::hint(
                ui,
                "Your own node answers this, so the block explorer below involves no third \
                 party. Other wallets on this machine can point at the same address.",
            );
            if ui.button("Open explorer in browser").clicked() {
                let _ = open_url(url);
            }
        } else {
            theme::hint(ui, "The RPC listener is not enabled for this session.");
        }
    });

    ui.add_space(6.0);
    mining_panel(app, ui, syncing);
}

/// Mining controls.
///
/// The app does not hash in-process: it supervises a hasher process, and
/// optionally a loopback Stratum pool, both the `midstate` binary. That is what
/// lets rewards be paid to a wallet address instead of the node's own internally
/// generated mining seed, which the wallet could never spend.
///
/// # Reasoning (UI)
/// Startup is genuinely slow — a local pool has to bind and answer before the
/// hasher can be pointed at it — so every phase is shown explicitly rather than
/// toggling a single button. A user who presses Start and sees nothing change
/// assumes it is broken.
///
/// Thread count uses preset buttons rather than a `Slider` deliberately: this
/// app's theme sets `widgets.inactive.bg_fill` to transparent, and egui draws a
/// slider's rail with exactly that, so a slider renders as a bare handle with no
/// visible track.
fn mining_panel(app: &mut App, ui: &mut Ui, syncing: bool) {
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Mining").font(theme::font_medium(14.0)));

        let Some(m) = app.mining.clone() else {
            theme::hint(ui, "Reading mining status…");
            return;
        };

        if !m.available {
            theme::hint(
                ui,
                "The `midstate` binary could not be found next to this app or on PATH. \
                 Mining runs it as a separate process, so it needs to be installed alongside.",
            );
            return;
        }

        let phase = m.phase.as_str();

        // ── Status line: always says what is happening, in words ──────────
        ui.horizontal(|ui| {
            let (dot, text, colour) = match phase {
                "running"  => ("●", "Mining".to_string(), theme::ink()),
                "starting" => ("◐", "Starting…".to_string(), theme::bright()),
                "stopping" => ("◐", "Stopping…".to_string(), theme::bright()),
                "error"    => ("✕", "Stopped".to_string(), theme::red()),
                _          => ("○", "Not mining".to_string(), theme::muted()),
            };
            ui.label(RichText::new(dot).color(colour).size(13.0));
            ui.label(RichText::new(text).color(colour).size(12.5));
            if phase == "running" {
                ui.label(
                    RichText::new(format!("· {} · {} threads",
                        fmt_secs(m.uptime_secs),
                        if m.threads == 0 { "all".to_string() } else { m.threads.to_string() }))
                        .color(theme::muted()).size(11.5),
                );
            }
        });

        // Progress detail while a transition is in flight, plus a spinner so
        // the window visibly repaints rather than looking hung.
        if let Some(msg) = &m.message {
            ui.horizontal(|ui| {
                ui.add(egui::Spinner::new().size(12.0));
                ui.label(RichText::new(msg).color(theme::muted()).size(11.5));
            });
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
        }

        if let Some(err) = &m.last_error {
            ui.label(RichText::new(err).color(theme::red()).size(11.5));
        }

        if phase == "running" {
            ui.add_space(6.0);

            if m.local_pool {
                // Headline rates. Network share is the number that actually
                // answers "is this worth leaving on?".
                ui.columns(3, |c| {
                    theme::stat(&mut c[0], "Your hashrate", &rate(m.hashrate), "H/s est");
                    theme::stat(&mut c[1], "Network", &rate(m.network_hashrate), "H/s");
                    theme::stat(
                        &mut c[2],
                        "Your share",
                        &format!("{:.3}", m.network_share * 100.0),
                        "% of network",
                    );
                });

                ui.add_space(4.0);
                ui.columns(3, |c| {
                    theme::stat(&mut c[0], "Shares", &units(m.shares), "accepted");
                    theme::stat(
                        &mut c[1],
                        "Rejected",
                        &units(m.rejected),
                        &efficiency(m.shares, m.rejected),
                    );
                    theme::stat(&mut c[2], "Blocks found", &units(m.blocks_found), "by you");
                });

                // Expected time to a block, straight from share of network and
                // block time. The single most useful number for a solo miner,
                // and the one no other screen can give them.
                if m.network_share > 0.0 {
                    let secs = 60.0 / m.network_share;
                    row(
                        ui,
                        "expected block",
                        &format!("about every {}", fmt_secs(secs as u64)),
                        Some("Statistical average — actual intervals vary enormously."),
                    );
                    let per_day = 86_400.0 * m.network_share * m.block_reward as f64 / 60.0;
                    row(ui, "expected/day", &format!("{:.2} MDS", per_day), None);
                }

                if !m.workers.is_empty() {
                    ui.add_space(4.0);
                    ui.label(theme::mono("rigs").size(11.5));
                    for (name, score) in &m.workers {
                        row(ui, name, &units(*score), None);
                    }
                }

                ui.add_space(4.0);
                row(ui, "pool", &m.pool_url, None);
                row(
                    ui,
                    "round",
                    &format!("{} of {} points · {} miners", units(m.score), units(m.total_score), m.active_miners),
                    Some("Your score against the pool's, which sets your split of the next block."),
                );
                if m.network_height > 0 {
                    row(ui, "height", &units(m.network_height), None);
                }

                theme::hint(
                    ui,
                    "Hashrate is estimated from accepted shares against the pool's share \
                     target, so it settles over the first few minutes. No shares yet is \
                     normal on a fresh start.",
                );
                theme::hint(ui, MINING_AUDIT_NOTE);
            } else {
                ui.columns(2, |c| {
                    theme::stat(&mut c[0], "Uptime", &fmt_secs(m.uptime_secs), "");
                    theme::stat(
                        &mut c[1],
                        "Threads",
                        &if m.threads == 0 { "all".into() } else { m.threads.to_string() },
                        "",
                    );
                });
                row(ui, "pool", &m.pool_url, None);
                theme::hint(
                    ui,
                    "Connected to a remote pool — shares, payouts and hashrate are tracked \
                     on that pool's own dashboard, not here.",
                );
            }
            ui.add_space(4.0);
        }

        // ── Payout address ────────────────────────────────────────────────
        if !m.payout_address.is_empty() {
            row_copy(ui, "paying to", &m.payout_address);
            theme::hint(
                ui,
                "A reusable multi-signature address in this wallet. Rewards arrive here \
                 directly and are spendable like any other coin.",
            );
        }

        ui.add_space(6.0);

        // ── Controls ──────────────────────────────────────────────────────
        if phase == "running" || phase == "stopping" {
            let stopping = phase == "stopping";
            if ui
                .add_enabled(!stopping, egui::Button::new("Stop mining"))
                .clicked()
            {
                app.go(ui.ctx(), Action::StopMining);
            }
            return;
        }

        if phase == "starting" {
            // Cancelling mid-start is a stop: the supervisor unwinds whatever
            // it has already launched.
            if ui.button("Cancel").clicked() {
                app.go(ui.ctx(), Action::StopMining);
            }
            return;
        }

        // Idle / error: full controls.
        let max = m.max_threads.max(1);

        ui.horizontal(|ui| {
            ui.label(theme::mono("threads").size(11.5));
            for (label, value) in [
                ("1", 1usize),
                ("half", (max / 2).max(1)),
                ("all", 0usize),
            ] {
                let selected = app.mining_threads == value;
                let text = if selected {
                    RichText::new(label).color(theme::ink())
                } else {
                    RichText::new(label).color(theme::muted())
                };
                if ui.selectable_label(selected, text).clicked() {
                    app.mining_threads = value;
                }
            }
            // Explicit count, for anything the presets do not cover.
            let mut n = app.mining_threads;
            if ui
                .add(egui::DragValue::new(&mut n).range(0..=max).speed(0.25))
                .changed()
            {
                app.mining_threads = n.min(max);
            }
        });
        theme::hint(
            ui,
            if app.mining_threads == 0 {
                "Using every core. Leave one or two free if you want the machine responsive."
            } else {
                "Fewer threads means less heat and a more usable machine."
            },
        );

        ui.add_space(4.0);
        ui.label(theme::mono("pool").size(11.5));
        let resp = ui.add(
            egui::TextEdit::singleline(&mut app.mining_pool)
                .hint_text(&m.default_pool_url)
                .desired_width(f32::INFINITY),
        );
        if resp.changed() {
            app.mining_pool_touched = true;
        }
        ui.horizontal(|ui| {
            if ui.small_button("use this machine").clicked() {
                app.mining_pool = m.default_pool_url.clone();
                app.mining_pool_touched = true;
            }
            theme::hint(
                ui,
                "Leave as the local address to run your own pool, or enter any \
                 stratum+tcp:// URL to mine on someone else's.",
            );
        });

        ui.add_space(6.0);
        let ready = !syncing;
        if ui
            .add_enabled(ready, egui::Button::new("Start mining"))
            .clicked()
        {
            app.go(
                ui.ctx(),
                Action::StartMining {
                    threads: app.mining_threads,
                    pool_url: app.mining_pool.clone(),
                },
            );
        }
        if syncing {
            theme::hint(
                ui,
                "Waiting for the chain to finish syncing — mining on a stale tip only \
                 produces orphans.",
            );
        }
    });
}

/// Note shown when this machine is running the pool.
const MINING_AUDIT_NOTE: &str =
    "The local pool's audit page is reachable from your network on port 8081.";

/// Formats a hash rate with SI-ish units.
fn rate(h: f64) -> String {
    const UNITS: [&str; 5] = ["", "K", "M", "G", "T"];
    let mut v = h;
    let mut i = 0;
    while v >= 1000.0 && i < UNITS.len() - 1 {
        v /= 1000.0;
        i += 1;
    }
    if i == 0 { format!("{v:.0}") } else { format!("{v:.2}{}", UNITS[i]) }
}

/// Accepted/rejected as a share-quality percentage.
fn efficiency(accepted: u64, rejected: u64) -> String {
    let total = accepted + rejected;
    if total == 0 {
        return "no shares yet".into();
    }
    format!("{:.1}% good", accepted as f64 * 100.0 / total as f64)
}

/// Renders a duration as a compact human string.
fn fmt_secs(s: u64) -> String {
    match s {
        0..=59 => format!("{s}s"),
        60..=3599 => format!("{}m {}s", s / 60, s % 60),
        _ => format!("{}h {}m", s / 3600, (s % 3600) / 60),
    }
}

fn row(ui: &mut Ui, label: &str, value: &str, help: Option<&str>) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(label.to_uppercase())
                .font(FontId::monospace(9.5))
                .color(theme::muted()),
        );
        let l = ui.label(theme::mono(value).size(11.5));
        if let Some(h) = help {
            l.on_hover_text(h);
        }
    });
}

fn row_copy(ui: &mut Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(label.to_uppercase())
                .font(FontId::monospace(9.5))
                .color(theme::muted()),
        );
        ui.label(theme::mono(short_hex(value, 18)).size(11.0).color(theme::bright()));
        if ui.button(RichText::new("copy").size(10.0)).clicked() {
            ui.ctx().copy_text(value.to_string());
        }
    });
}

fn open_url(url: &str) -> std::io::Result<()> {
    #[cfg(target_os = "linux")]
    let cmd = "xdg-open";
    #[cfg(target_os = "macos")]
    let cmd = "open";
    #[cfg(target_os = "windows")]
    let cmd = "explorer";
    std::process::Command::new(cmd).arg(url).spawn().map(|_| ())
}
