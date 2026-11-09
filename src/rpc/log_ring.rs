//! The last few hundred log lines, kept in memory for the miner's dashboard
//! (its Logs tab), as a `tracing` layer beside the one that prints them.
//!
//! Only midstate's own lines at INFO and above are kept, plus warnings and
//! errors from anything else. The ring is bounded, so it costs a few hundred
//! kilobytes at most, whatever runs.

use serde::Serialize;
use std::collections::VecDeque;
use std::fmt::Write as _;
use std::sync::Mutex;
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::layer::{Context, Layer};

/// Lines kept.
pub const KEEP: usize = 500;

/// One log line.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct LogLine {
    /// Counts up from 1, so a reader asks only for what it has not seen.
    pub seq: u64,
    /// Unix milliseconds.
    pub ms: u64,
    pub level: &'static str,
    pub target: String,
    pub message: String,
}

struct Ring {
    lines: VecDeque<LogLine>,
    next: u64,
}

static RING: Mutex<Ring> = Mutex::new(Ring { lines: VecDeque::new(), next: 1 });

/// The lines after `seq`, at most `max`, oldest first.
pub fn after(seq: u64, max: usize) -> Vec<LogLine> {
    let ring = RING.lock().unwrap_or_else(|e| e.into_inner());
    let newer: Vec<&LogLine> = ring.lines.iter().filter(|l| l.seq > seq).collect();
    let skip = newer.len().saturating_sub(max);
    newer.into_iter().skip(skip).cloned().collect()
}

fn push(level: &'static str, target: &str, message: String) {
    let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |t| t.as_millis() as u64);
    let mut ring = RING.lock().unwrap_or_else(|e| e.into_inner());
    let seq = ring.next;
    ring.next += 1;
    ring.lines.push_back(LogLine { seq, ms, level, target: target.to_string(), message });
    while ring.lines.len() > KEEP {
        ring.lines.pop_front();
    }
}

/// The `tracing` layer that fills the ring.
pub struct LogLayer;

impl<S: Subscriber> Layer<S> for LogLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let meta = event.metadata();
        let level = *meta.level();
        let ours = meta.target().starts_with("midstate");
        if level > Level::INFO || (!ours && level > Level::WARN) {
            return;
        }
        let mut text = Text::default();
        event.record(&mut text);
        let name = match level {
            Level::ERROR => "error",
            Level::WARN => "warn",
            _ => "info",
        };
        push(name, meta.target(), text.0);
    }
}

/// An event's message, then its other fields as `key=value`.
#[derive(Default)]
struct Text(String);

impl Visit for Text {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let rest = std::mem::take(&mut self.0);
            let _ = write!(self.0, "{value:?}");
            self.0.push_str(&rest);
        } else {
            let _ = write!(self.0, " {}={value:?}", field.name());
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.0.insert_str(0, value);
        } else {
            let _ = write!(self.0, " {}={value}", field.name());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::layer::SubscriberExt;

    #[test]
    fn it_keeps_midstates_lines_and_a_reader_gets_only_new_ones() {
        let subscriber = tracing_subscriber::registry().with(LogLayer);
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(target: "midstate::mining", "share accepted");
            tracing::debug!(target: "midstate::mining", "too chatty to keep");
            tracing::info!(target: "hyper::client", "someone else's info");
            tracing::warn!(target: "hyper::client", "someone else's warning");
            tracing::warn!(target: "midstate::mining", job = 7, "stale job");
        });
        let all = after(0, KEEP);
        let mine: Vec<&LogLine> = all.iter().filter(|l| l.message.contains("share accepted") || l.message.contains("stale job") || l.target == "hyper::client").collect();
        let texts: Vec<&str> = mine.iter().map(|l| l.message.as_str()).collect();
        assert!(texts.contains(&"share accepted"));
        assert!(texts.contains(&"stale job job=7"), "{texts:?}");
        assert!(texts.contains(&"someone else's warning"));
        assert!(!texts.contains(&"someone else's info"));
        assert!(!all.iter().any(|l| l.message == "too chatty to keep"));
        let last = all.last().unwrap().seq;
        assert!(after(last, KEEP).is_empty(), "nothing new after the last one");
        assert_eq!(after(0, 1).len(), 1, "at most what is asked for, newest kept");
    }
}
