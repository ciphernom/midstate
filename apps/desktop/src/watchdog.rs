//! Notes in the log when the window stops drawing.
//!
//! A window that stops repainting goes white on Windows, and all a user can
//! report is that it froze. The UI thread cannot report its own stall, so a
//! thread of its own reads the marks each frame leaves: whether the UI
//! thread is stuck inside a frame, and at what (handling messages, drawing a
//! tab), or between frames (putting the frame on the screen, or the window
//! system), and for how long. One line when a stall passes a few seconds,
//! one when it ends, and nothing at all while frames come.
//!
//! The marks are a few atomic stores a frame; the watcher wakes four times a
//! second.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Stuck this long inside one frame is worth a line.
const IN_FRAME: Duration = Duration::from_secs(2);

/// No frame for this long while the window is open is worth a line: the app
/// asks for a frame every second, and Windows shows a window that has not
/// answered for five as not responding.
const BETWEEN_FRAMES: Duration = Duration::from_secs(5);

/// When the current frame began (ms since `epoch`, plus one); 0 between
/// frames.
static BEGAN: AtomicU64 = AtomicU64::new(0);
/// When the last frame ended (ms since `epoch`).
static ENDED: AtomicU64 = AtomicU64::new(0);
/// Whether the window was minimized at the last frame: a minimized window
/// draws nothing, and is not stuck for it.
static HIDDEN: AtomicBool = AtomicBool::new(true);
/// What the UI thread is doing in this frame, in two parts ("drawing the
/// tab", "Trade").
static DOING: Mutex<(&str, &str)> = Mutex::new(("starting", ""));

fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

fn now_ms() -> u64 {
    epoch().elapsed().as_millis() as u64
}

/// Marks one frame, from `begin` until it is dropped, so that every return
/// out of `update` ends the frame.
pub struct Frame;

impl Frame {
    pub fn begin(minimized: bool) -> Frame {
        HIDDEN.store(minimized, Relaxed);
        BEGAN.store(now_ms() + 1, Relaxed);
        Frame
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        ENDED.store(now_ms(), Relaxed);
        BEGAN.store(0, Relaxed);
    }
}

/// What the UI thread turns to next in this frame.
pub fn doing(what: &'static str, on: &'static str) {
    if let Ok(mut d) = DOING.lock() {
        *d = (what, on);
    }
}

/// Start watching (once).
pub fn start() {
    static STARTED: AtomicBool = AtomicBool::new(false);
    if STARTED.swap(true, Relaxed) {
        return;
    }
    ENDED.store(now_ms(), Relaxed);
    if let Err(e) = std::thread::Builder::new().name("ui-watchdog".into()).spawn(watch) {
        tracing::warn!("UI watchdog not started: {e}");
    }
}

fn watch() {
    // When the stall reported last began, until it ends.
    let mut stall: Option<u64> = None;
    loop {
        std::thread::sleep(Duration::from_millis(250));
        let now = now_ms();
        let began = BEGAN.load(Relaxed);
        let in_frame = began != 0;
        let since = if in_frame { began - 1 } else { ENDED.load(Relaxed) };
        let stuck = if in_frame {
            now.saturating_sub(since) >= IN_FRAME.as_millis() as u64
        } else {
            !HIDDEN.load(Relaxed) && now.saturating_sub(since) >= BETWEEN_FRAMES.as_millis() as u64
        };
        match (stuck, stall) {
            (true, None) => {
                let secs = now.saturating_sub(since) as f64 / 1000.0;
                if in_frame {
                    let (what, on) = DOING.lock().map(|d| *d).unwrap_or(("?", ""));
                    tracing::warn!("UI thread stuck for {secs:.1}s in one frame, {what} {on}");
                } else {
                    tracing::warn!(
                        "no frame for {secs:.1}s though the app asks for one a second: putting frames \
                         on the screen is stuck (the graphics driver, or a GPU busy with other work), \
                         unless the window was just minimized"
                    );
                }
                stall = Some(since);
            }
            (false, Some(from)) => {
                tracing::info!("UI thread drawing again, after {:.1}s", now.saturating_sub(from) as f64 / 1000.0);
                stall = None;
            }
            _ => {}
        }
    }
}
