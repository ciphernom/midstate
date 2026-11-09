//! The settings `midstate miner` mines with, and the file the dashboard
//! saves them in.
//!
//! The command line gives the first settings. Once the dashboard saves some,
//! they are kept in [`SETTINGS_FILE`] beside the miner and replace the command
//! line's whole, then and at every start after, until the dashboard is told to
//! use the command line's again (which removes the file). The miner says at
//! start which it uses.
//!
//! Every setting is checked before it is used or kept: an address that is not
//! one would otherwise only show as a refused login, or (the payout address)
//! stop the miner at its first audit.

use crate::mining::{MinerControl, PoolSettings};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Where the dashboard keeps the settings it saves, beside the miner.
pub const SETTINGS_FILE: &str = "miner-settings.toml";
/// The heat limits the dashboard may set, in degrees Celsius.
pub const HEAT_LIMIT_RANGE: std::ops::RangeInclusive<f32> = 50.0..=85.0;

fn default_worker() -> String {
    "default".into()
}

/// What `midstate miner` mines with: the pool, who is paid, and how hard it
/// works.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MinerSettings {
    pub pool_url: String,
    /// Mined on while `pool_url` cannot be reached.
    #[serde(default)]
    pub fallback_url: Option<String>,
    pub payout_address: String,
    #[serde(default = "default_worker")]
    pub worker: String,
    #[serde(default)]
    pub midwimble_address: Option<String>,
    /// CPU threads (0: every core).
    #[serde(default)]
    pub threads: usize,
    /// Degrees Celsius above which threads are shed (`None`: no limit).
    #[serde(default)]
    pub heat_limit_c: Option<f32>,
}

/// A pool's stratum address, trimmed: `host:port`, `stratum+tcp://` in front
/// or not. The error is the text to show.
pub fn check_pool_url(what: &str, url: &str) -> Result<String, String> {
    let url = url.trim();
    let rest = url.strip_prefix("stratum+tcp://").unwrap_or(url);
    let example = "like stratum+tcp://pool.example:3333";
    let Some((host, port)) = rest.rsplit_once(':') else {
        return Err(format!("{what}: give a host and a port, {example}"));
    };
    if !port.parse::<u16>().is_ok_and(|p| p > 0) {
        return Err(format!("{what}: \"{port}\" is not a port, {example}"));
    }
    let host_ok = match host.strip_prefix('[').and_then(|h| h.strip_suffix(']')) {
        Some(v6) => v6.parse::<std::net::Ipv6Addr>().is_ok(),
        None => {
            !host.is_empty()
                && host.len() <= 253
                && host.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
        }
    };
    if !host_ok {
        return Err(format!("{what}: \"{host}\" is not a host name, {example}"));
    }
    Ok(url.to_string())
}

impl MinerSettings {
    /// These settings checked and tidied (trimmed, a blank optional field
    /// taken as none), or the first thing wrong, as the text to show.
    pub fn check(&self) -> Result<MinerSettings, String> {
        let pool_url = check_pool_url("Pool", &self.pool_url)?;
        let fallback_url = match self.fallback_url.as_deref().map(str::trim).filter(|u| !u.is_empty()) {
            Some(u) => {
                let u = check_pool_url("Fallback pool", u)?;
                if u.trim_start_matches("stratum+tcp://") == pool_url.trim_start_matches("stratum+tcp://") {
                    return Err("Fallback pool: it is the same pool as the main one".into());
                }
                Some(u)
            }
            None => None,
        };
        let payout_address = self.payout_address.trim().to_string();
        crate::core::types::parse_address_flexible(&payout_address)
            .map_err(|e| format!("Payout address: {e}"))?;
        let worker = match self.worker.trim() {
            "" => default_worker(),
            w if w.chars().count() > 64 => return Err("Worker name: 64 characters at most".into()),
            w if w.chars().any(char::is_control) => return Err("Worker name: no control characters".into()),
            w => w.to_string(),
        };
        let midwimble_address = match self.midwimble_address.as_deref().map(str::trim).filter(|a| !a.is_empty()) {
            Some(a) => Some(crate::pool::parse_midwimble_address(a).map_err(|e| format!("Midwimble address: {e}"))?),
            None => None,
        };
        let cores = crate::mining::cores();
        if self.threads > cores {
            return Err(format!("Threads: this machine has {cores} cores"));
        }
        if let Some(limit) = self.heat_limit_c {
            if !HEAT_LIMIT_RANGE.contains(&limit) {
                return Err(format!(
                    "Heat limit: between {:.0} and {:.0} °C",
                    HEAT_LIMIT_RANGE.start(),
                    HEAT_LIMIT_RANGE.end()
                ));
            }
        }
        Ok(MinerSettings {
            pool_url,
            fallback_url,
            payout_address,
            worker,
            midwimble_address,
            threads: self.threads,
            heat_limit_c: self.heat_limit_c,
        })
    }

    /// The pool, and who it pays.
    pub fn pool(&self) -> PoolSettings {
        PoolSettings {
            pool_url: self.pool_url.clone(),
            fallback_url: self.fallback_url.clone(),
            payout_address: self.payout_address.clone(),
            worker: self.worker.clone(),
            midwimble_address: self.midwimble_address.clone(),
        }
    }

    /// The settings the miner runs with now.
    pub fn current(control: &MinerControl) -> MinerSettings {
        let p = control.pool();
        MinerSettings {
            pool_url: p.pool_url,
            fallback_url: p.fallback_url,
            payout_address: p.payout_address,
            worker: p.worker,
            midwimble_address: p.midwimble_address,
            threads: control.threads().want,
            heat_limit_c: control.heat_limit(),
        }
    }

    /// Runs the miner with these from now on.
    pub fn apply(&self, control: &MinerControl) {
        control.set_heat_limit(self.heat_limit_c);
        control.set_threads(self.threads);
        control.set_pool(self.pool());
    }

    /// The settings kept at `path`, `None` when there are none.
    pub fn load(path: &Path) -> Result<Option<MinerSettings>, String> {
        match std::fs::read_to_string(path) {
            Ok(text) => toml::from_str(&text).map(Some).map_err(|e| format!("{}: {}", path.display(), e)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("{}: {}", path.display(), e)),
        }
    }

    /// Keeps these at `path`, whole or not at all.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let text = toml::to_string(self).map_err(|e| e.to_string())?;
        let header = "# Saved by the midstate miner's dashboard. These replace the command line's\n\
                      # settings; remove this file (or choose \"Use the command line's\") to go back.\n\n";
        let tmp = path.with_extension("toml.new");
        std::fs::write(&tmp, format!("{header}{text}")).map_err(|e| format!("{}: {}", tmp.display(), e))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("{}: {}", path.display(), e))
    }

    /// Removes what is kept at `path`.
    pub fn forget(path: &Path) -> Result<(), String> {
        match std::fs::remove_file(path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(format!("{}: {}", path.display(), e)),
            _ => Ok(()),
        }
    }

    /// What to start with: the settings kept at `path` if there are any and
    /// they check out, else the command line's. With a line for the log.
    pub fn at_start(command_line: &MinerSettings, path: &Path) -> (MinerSettings, Option<String>) {
        match MinerSettings::load(path) {
            Ok(Some(saved)) => match saved.check() {
                Ok(saved) if saved == *command_line => (saved, None),
                Ok(saved) => (
                    saved,
                    Some(format!(
                        "using the settings saved from the dashboard ({}), not the command line's",
                        path.display()
                    )),
                ),
                Err(e) => (
                    command_line.clone(),
                    Some(format!("ignoring {}: {}. using the command line's settings", path.display(), e)),
                ),
            },
            Ok(None) => (command_line.clone(), None),
            Err(e) => (command_line.clone(), Some(format!("ignoring {e}. using the command line's settings"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A midwimble address (the pool's tests' ALICE).
    pub(crate) const MW: &str = "mw1q960ls6efz2kjcfp6t62djd408kjp2ff48ljdmtjumsgpdnk0xez4axmugrxsgp7tr6pdc7jg4fdgmung293gcyqql4pud3c0qat8usvfx0mmkexn3vqntw2rsmcxjpzkgysqv7q7jgnqzzg0qxwf60ncqkqvj9ses";

    /// Settings that check out: a payout address in the 64-hex form.
    pub(crate) fn valid() -> MinerSettings {
        MinerSettings {
            pool_url: "stratum+tcp://pool.example:3333".into(),
            fallback_url: None,
            payout_address: "07".repeat(32),
            worker: "rig".into(),
            midwimble_address: Some(MW.into()),
            threads: 0,
            heat_limit_c: None,
        }
    }

    #[test]
    fn settings_are_checked_and_tidied() {
        let ok = valid().check().expect("valid");
        assert_eq!(ok, valid());

        let mut s = valid();
        s.pool_url = "  pool.example:3333 ".into();
        s.fallback_url = Some("   ".into());
        s.midwimble_address = Some(String::new());
        s.worker = " ".into();
        let tidy = s.check().unwrap();
        assert_eq!(tidy.pool_url, "pool.example:3333");
        assert_eq!((tidy.fallback_url, tidy.midwimble_address), (None, None), "blank means none");
        assert_eq!(tidy.worker, "default");

        let wrong = |f: fn(&mut MinerSettings)| {
            let mut s = valid();
            f(&mut s);
            s.check().unwrap_err()
        };
        assert!(wrong(|s| s.pool_url = "pool.example".into()).starts_with("Pool: give a host and a port"));
        assert!(wrong(|s| s.pool_url = "pool.example:99999".into()).contains("not a port"));
        assert!(wrong(|s| s.pool_url = "pool example:3333".into()).contains("not a host name"));
        assert!(wrong(|s| s.pool_url = "http://pool.example:3333".into()).contains("not a host name"));
        assert!(wrong(|s| s.fallback_url = Some("pool.example:3333".into())).contains("same pool"));
        assert!(wrong(|s| s.fallback_url = Some("x:0".into())).starts_with("Fallback pool:"));
        assert!(wrong(|s| s.payout_address = "abc".into()).starts_with("Payout address:"));
        assert!(wrong(|s| s.midwimble_address = Some("mw1nope".into())).starts_with("Midwimble address:"));
        assert!(wrong(|s| s.worker = "a\nb".into()).starts_with("Worker name:"));
        assert!(wrong(|s| s.threads = 100_000).starts_with("Threads:"));
        assert!(wrong(|s| s.heat_limit_c = Some(95.0)).starts_with("Heat limit:"));

        assert!(check_pool_url("Pool", "[::1]:3333").is_ok());
        assert!(check_pool_url("Pool", "192.168.1.5:3334").is_ok());
    }

    #[test]
    fn saved_settings_replace_the_command_lines_until_forgotten() {
        let dir = std::env::temp_dir().join(format!("midstate-settings-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(SETTINGS_FILE);
        let cli = valid();

        let (s, note) = MinerSettings::at_start(&cli, &path);
        assert_eq!((s, note), (cli.clone(), None), "nothing saved");

        let mut mine = valid();
        mine.pool_url = "stratum+tcp://other.example:3334".into();
        mine.fallback_url = Some("stratum+tcp://pool.example:3333".into());
        mine.threads = 1;
        mine.heat_limit_c = Some(75.0);
        mine.save(&path).unwrap();
        assert_eq!(MinerSettings::load(&path).unwrap(), Some(mine.clone()));
        let (s, note) = MinerSettings::at_start(&cli, &path);
        assert_eq!(s, mine);
        assert!(note.unwrap().contains("saved from the dashboard"));

        // A file that no longer checks out is ignored, and says so.
        let mut bad = mine.clone();
        bad.payout_address = "nope".into();
        bad.save(&path).unwrap();
        let (s, note) = MinerSettings::at_start(&cli, &path);
        assert_eq!(s, cli);
        assert!(note.unwrap().contains("Payout address"));

        std::fs::write(&path, "pool_url = 3").unwrap();
        assert_eq!(MinerSettings::at_start(&cli, &path).0, cli, "an unreadable file too");

        MinerSettings::forget(&path).unwrap();
        MinerSettings::forget(&path).unwrap();
        assert_eq!(MinerSettings::load(&path).unwrap(), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn settings_reach_the_running_miner() {
        let control = MinerControl::new(valid().pool(), 0, None);
        let mut next = valid();
        next.pool_url = "stratum+tcp://other.example:3334".into();
        next.threads = 2;
        next.heat_limit_c = Some(70.0);
        next.apply(&control);
        assert_eq!(MinerSettings::current(&control), next);
        assert_eq!(control.threads().want, 2);
        assert_eq!(control.heat_limit(), Some(70.0));
    }
}
