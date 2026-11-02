//! The node panel. This app runs two full nodes in-process, midstate's and
//! midwimble's, so everything here is read from local consensus state rather
//! than asked of a third party — which is the point, and worth showing
//! rather than just asserting.

use crate::app::App;
use crate::bridge::Action;
use crate::theme::{self, ago, fmt_dt, short_hex, units};
use crate::views::mdw::{self as m, mdw};
use eframe::egui::{self, FontId, RichText, Ui};

pub fn show(app: &mut App, ui: &mut Ui) {
    theme::heading(ui, "Node");
    overview(app, ui);
    ui.add_space(8.0);
    theme::segmented(ui, &["midstate", "midwimble"], &mut app.node_chain);
    ui.add_space(6.0);
    if app.node_chain == 1 {
        midwimble(app, ui);
    } else {
        midstate(app, ui);
    }
}

/// Both nodes side by side: how each is doing, and its explorer.
fn overview(app: &mut App, ui: &mut Ui) {
    let mds = app.sync.clone();
    let mds_node = app.node.clone();
    let mwb = app.mwb_account.as_ref().and_then(|a| a.node.clone());
    let mwb_info = app.mwb_node.clone();
    let (mds_explorer, mwb_explorer) = (m::mds_explorer(app), m::mwb_explorer(app));
    ui.columns(2, |c| {
        theme::panel_frame().show(&mut c[0], |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                theme::dot(ui, mds.as_ref().is_some_and(|s| s.peer_count > 0));
                ui.label(RichText::new("midstate").font(theme::font_medium(14.0)));
                theme::right_aligned(ui, |ui| {
                    m::explorer_button(ui, "explorer", mds_explorer.clone(), "midstate's block explorer, served by this node");
                });
            });
            let Some(s) = mds else {
                theme::hint(ui, "starting…");
                return;
            };
            row(ui, "height", &units(s.height), None);
            row(ui, "peers", &s.peer_count.to_string(), None);
            row(ui, "state", if s.is_syncing { "syncing" } else { "in sync" }, None);
            if let Some(n) = &mds_node {
                row(ui, "last block", &ago(n.tip_timestamp), None);
            }
            row(ui, "mempool", &s.mempool.to_string(), None);
        });
        theme::panel_frame().show(&mut c[1], |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                theme::dot(ui, mwb.as_ref().is_some_and(|n| n.peers > 0));
                ui.label(RichText::new("midwimble").font(theme::font_medium(14.0)));
                theme::right_aligned(ui, |ui| {
                    m::explorer_button(ui, "explorer", mwb_explorer.clone(), "midwimble's block explorer, served beside its node");
                });
            });
            let Some(n) = mwb else {
                theme::hint(ui, if app.mwb_account.is_some() { "not running this session" } else { "starting…" });
                return;
            };
            row(ui, "height", &units(n.height), None);
            row(ui, "peers", &n.peers.to_string(), None);
            row(ui, "state", if n.syncing { "syncing" } else if n.peers == 0 { "looking for peers" } else { "in sync" }, None);
            if let Some(i) = &mwb_info {
                row(ui, "last block", &ago(i.tip_timestamp), None);
                row(ui, "mempool", &i.mempool.to_string(), None);
            }
        });
    });
}

/// Midwimble's node: the same picture as midstate's, from its own chain.
fn midwimble(app: &mut App, ui: &mut Ui) {
    let Some(n) = app.mwb_node.clone() else {
        let off = app.mwb_account.as_ref().is_some_and(|a| a.node.is_none());
        theme::hint(
            ui,
            if off {
                "The midwimble node did not start this session, so MDW is unavailable; the log says why."
            } else {
                "Reading node state…"
            },
        );
        return;
    };
    ui.label(
        RichText::new(theme::grouped_hash(&n.midstate, 8))
            .font(FontId::monospace(26.0))
            .color(theme::ambient()),
    );
    ui.add_space(2.0);
    ui.columns(3, |c| {
        theme::stat(&mut c[0], "Height", &units(n.height), "blocks");
        theme::stat(&mut c[1], "Unspent outputs", &units(n.utxo_count as u64), "");
        theme::stat(&mut c[2], "Block reward", &m::plain(n.block_reward), "MDW");
    });
    ui.columns(3, |c| {
        theme::stat(&mut c[0], "Difficulty", &n.difficulty_bits.to_string(), "leading zeros");
        theme::stat(&mut c[1], "Supply", &m::plain(n.supply), "MDW mined");
        theme::stat(&mut c[2], "Mining bonds", &units(n.bonds as u64), "registered");
    });
    ui.add_space(6.0);

    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("Chain tip").font(theme::font_medium(14.0)));
        row(ui, "last block", &format!("{}  ·  {}", fmt_dt(n.tip_timestamp), ago(n.tip_timestamp)), None);
        row_copy(ui, "header hash", &n.header_hash);
        row_copy(ui, "midstate", &n.midstate);
        row(ui, "cumulative work", &n.depth, None);
        row(ui, "mempool", &format!("{} transaction(s) waiting", n.mempool), None);
        if n.syncing {
            theme::hint(
                ui,
                "Downloading a heavier chain: these figures describe the chain as far as this node \
                 has verified it, and its tip is about to move.",
            );
        }
    });

    ui.add_space(6.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.label(RichText::new("Peers").font(theme::font_medium(14.0)));
            theme::badge(ui, &n.peers.len().to_string(), if n.peers.is_empty() { theme::muted() } else { theme::ink() });
        });
        if n.peers.is_empty() {
            theme::hint(
                ui,
                &format!(
                    "No peers connected. The node keeps dialing midwimble's seed; if this persists, check \
                     whether outbound connections on port {} are blocked.",
                    if n.p2p_port == 0 { 9433 } else { n.p2p_port }
                ),
            );
            return;
        }
        theme::hint(ui, "Midwimble nodes this one is connected to, and who dialled whom.");
        egui::ScrollArea::vertical()
            .max_height(180.0)
            .auto_shrink([false, false])
            .id_salt("mwb_peers")
            .show(ui, |ui| {
                for p in &n.peers {
                    ui.horizontal(|ui| {
                        theme::badge(ui, if p.outbound { "out" } else { "in" }, theme::muted());
                        ui.label(theme::mono(short_hex(&p.id, 12)).size(11.0).color(theme::muted()));
                        ui.label(theme::mono(&p.addr).size(10.5).color(theme::faint()));
                        if ui.button(RichText::new("copy").size(10.0)).clicked() {
                            ui.ctx().copy_text(p.id.clone());
                        }
                    });
                }
            });
    });

    ui.add_space(6.0);
    theme::panel_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new("This machine").font(theme::font_medium(14.0)));
        row_copy(ui, "data directory", &n.data_dir);
        row(ui, "p2p port", &n.p2p_port.to_string(), None);
        row_copy(ui, "node id", &n.peer_id);
        match &n.explorer {
            Some(url) => {
                row_copy(ui, "explorer", url);
                theme::hint(
                    ui,
                    "Midwimble's block explorer, served by this app from the chain this node verified, \
                     on this machine only. Every MDW payment in History opens there.",
                );
                if ui.button("Open explorer in browser").clicked() {
                    m::open_url(url);
                }
            }
            None => theme::hint(ui, "The explorer is not served this session (MIDWIMBLE_EXPLORER_PORT=none, or no port was free)."),
        }
    });
    if let Some(a) = app.mwb_account.as_ref().filter(|a| a.enabled) {
        ui.add_space(6.0);
        theme::panel_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Your MDW account").font(theme::font_medium(14.0)));
            row(ui, "spendable", &mdw(a.spendable), None);
            row(ui, "scanned to", &format!("block {}", units(a.scanned.saturating_sub(1))), None);
            theme::hint(ui, "Settings can scan midwimble again from any height, if a payment seems missing.");
        });
    }
}

/// Midstate's node.
fn midstate(app: &mut App, ui: &mut Ui) {
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
                m::open_url(url);
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

        // ── MDW: the same hashing, on midwimble ───────────────────────────
        // The hasher names this wallet's MDW address to whichever pool it
        // mines on; the local pool merge-mines too while a bond is ready.
        if !m.mdw_address.is_empty() {
            row_copy(ui, "mdw to", &m.mdw_address);
        }
        if m.mdw_merge && !m.mdw_bond.is_empty() {
            row(
                ui,
                "signed by",
                &format!("bond {}", short_hex(&m.mdw_bond, 16)),
                Some(
                    "The bond whose key signs the midwimble blocks this pool finds. \
                     Its card on the Bonds tab says where it stands.",
                ),
            );
        }
        if !m.mdw_note.is_empty() {
            theme::hint(ui, &m.mdw_note);
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
            if !m.merge_pool.is_empty()
                && ui
                    .small_button("pool that pays MDW")
                    .on_hover_text(format!(
                        "{}: a public pool that merge-mines midwimble. It pays your MDW address its \
                         share of the midwimble blocks it finds, with no bond of your own; its page \
                         at rpc.cypherpunk.gold/pool lists them.",
                        m.merge_pool
                    ))
                    .clicked()
            {
                app.mining_pool = m.merge_pool.clone();
                app.mining_pool_touched = true;
            }
        });
        theme::hint(
            ui,
            "Leave as the local address to run your own pool, or enter any \
             stratum+tcp:// URL to mine on someone else's.",
        );

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

