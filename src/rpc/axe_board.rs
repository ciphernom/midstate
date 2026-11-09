//! The Raspberry Pi's own settings for `midstate miner`: its clock, its fan
//! curve and a reboot. They live in the Pi's boot config, which only root may
//! write, and the miner does not run as root. So a root-owned copy of the
//! midstate binary does them, at [`HELPER_PATH`], and nothing else:
//!
//! - `sudo midstate board-setup` installs it once: the copy, owned by root,
//!   and one sudoers line letting the miner's user run that copy, and only
//!   that, without a password. `--remove` takes both away again.
//! - The copy knows it is the helper by its own name and then never reaches
//!   the rest of midstate, whatever it is given ([`run_if_helper`]). It
//!   accepts three things: `status`, `apply` with a clock and a fan curve,
//!   each checked against the board's limits (`rpc::axe`) here, as root, not
//!   only in the dashboard; and `reboot`.
//! - The miner calls it with `sudo -n`, so a missing setup shows as a message
//!   on the dashboard rather than a prompt nobody sees.
//!
//! # The fan
//!
//! A Raspberry Pi 5 drives its active cooler from four temperature levels
//! (`dtparam=fan_temp0`…`fan_temp3`, each with a hysteresis and a speed of
//! 0–255). The defaults turn the fan on at 50 °C and to full at 75 °C. A
//! curve set here is a marked block at the end of `config.txt`, after the
//! overclock's, so it wins over any earlier lines; removing it leaves those
//! (or the defaults) in force. It takes effect at the next boot.
//!
//! # The boot config
//!
//! Each change keeps the file as it was in `config.txt.midstate-bak`, then
//! writes the new one beside it and renames it into place, so a power cut
//! leaves the old file or the new one, never half of one.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The root-owned copy.
pub const HELPER_PATH: &str = "/usr/local/sbin/midstate-axe-board";
/// Its file name, by which it knows it is the helper.
pub const HELPER_NAME: &str = "midstate-axe-board";
/// The sudoers line that lets the miner's user run it.
pub const SUDOERS_PATH: &str = "/etc/sudoers.d/midstate-axe-board";
/// Bumped when the helper's commands change, so a miner can tell an old copy.
pub const PROTOCOL: u32 = 1;

/// Opens the block of `config.txt` lines that set the fan.
pub const FAN_START: &str = "# --- MidstateAxe fan (set from the miner dashboard) ---";
/// Closes it.
pub const FAN_END: &str = "# --- end MidstateAxe fan ---";

/// A fan curve for a Raspberry Pi 5's active cooler.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", tag = "mode")]
pub enum Fan {
    /// The Pi's own curve (or whatever else `config.txt` says).
    Default,
    /// Earlier and harder: from 40 °C, full at 68 °C.
    Cool,
    /// Full speed all the time.
    Full,
    /// This speed (percent) all the time, and full above 75 °C whatever it is.
    Fixed { percent: u8 },
}

/// The slowest fixed speed: below it some coolers stall.
pub const FAN_MIN_PERCENT: u8 = 20;
/// Degrees, in millidegrees, a level holds on by as the board cools.
const FAN_HYST: u32 = 5000;

impl Fan {
    /// `default`, `cool`, `full` or `fixed:<percent>`.
    pub fn parse(s: &str) -> Result<Fan, String> {
        match s {
            "default" => Ok(Fan::Default),
            "cool" => Ok(Fan::Cool),
            "full" => Ok(Fan::Full),
            _ => {
                let p = s
                    .strip_prefix("fixed:")
                    .and_then(|p| p.parse::<u8>().ok())
                    .ok_or_else(|| format!("unknown fan setting \"{s}\": default, cool, full or fixed:<percent>"))?;
                if !(FAN_MIN_PERCENT..=100).contains(&p) {
                    return Err(format!("a fixed fan speed is {FAN_MIN_PERCENT}–100%"));
                }
                Ok(Fan::Fixed { percent: p })
            }
        }
    }

    /// As [`Fan::parse`] reads it.
    pub fn arg(&self) -> String {
        match self {
            Fan::Default => "default".into(),
            Fan::Cool => "cool".into(),
            Fan::Full => "full".into(),
            Fan::Fixed { percent } => format!("fixed:{percent}"),
        }
    }

    /// The four levels: (from millidegrees Celsius, speed 0–255). `None`
    /// for the Pi's own.
    pub fn levels(&self) -> Option<[(u32, u32); 4]> {
        match *self {
            Fan::Default => None,
            Fan::Cool => Some([(40_000, 100), (50_000, 150), (60_000, 200), (68_000, 255)]),
            Fan::Full => Some([(25_000, 255), (30_000, 255), (35_000, 255), (40_000, 255)]),
            Fan::Fixed { percent } => {
                let s = (u32::from(percent) * 255 + 50) / 100;
                Some([(30_000, s), (40_000, s), (50_000, s), (75_000, 255)])
            }
        }
    }

    /// In words, for the dashboard and the log.
    pub fn describe(&self) -> String {
        match self {
            Fan::Default => "the Pi's own curve".into(),
            Fan::Cool => "cooler: on from 40 °C, full from 68 °C".into(),
            Fan::Full => "full speed".into(),
            Fan::Fixed { percent } => format!("{percent}%, full above 75 °C"),
        }
    }
}

/// Lines of `config` outside the closed block `start`…`end` (a start with no
/// end, an edit gone wrong, keeps the rest of the file), with blank lines
/// at the end dropped.
fn without_block<'a>(config: &'a str, start: &str, end: &str) -> Vec<&'a str> {
    let closed = config.lines().any(|l| l.trim() == end);
    let mut out = Vec::new();
    let mut inside = false;
    for line in config.lines() {
        let t = line.trim();
        if t == start {
            inside = closed;
            continue;
        }
        if t == end {
            inside = false;
            continue;
        }
        if !inside {
            out.push(line);
        }
    }
    while out.last().is_some_and(|l| l.trim().is_empty()) {
        out.pop();
    }
    out
}

/// `config` with the fan block taken out and, unless `fan` is the Pi's own
/// curve, a new one at the end, for every model.
pub fn rewrite_fan(config: &str, fan: Fan) -> String {
    let mut text = without_block(config, FAN_START, FAN_END).join("\n");
    if !text.is_empty() {
        text.push('\n');
    }
    let Some(levels) = fan.levels() else { return text };
    if !text.is_empty() {
        text.push('\n');
    }
    text.push_str(FAN_START);
    text.push_str("\n[all]\n");
    for (i, (temp, speed)) in levels.iter().enumerate() {
        text.push_str(&format!("dtparam=fan_temp{i}={temp},fan_temp{i}_hyst={FAN_HYST},fan_temp{i}_speed={speed}\n"));
    }
    text.push_str(FAN_END);
    text.push('\n');
    text
}

/// The fan curve `config`'s fan block sets: [`Fan::Default`] with none.
pub fn configured_fan(config: &str) -> Fan {
    let mut inside = false;
    let mut levels = [(0u32, 0u32); 4];
    let mut seen = 0;
    for line in config.lines() {
        let t = line.trim();
        if t == FAN_START {
            inside = true;
        } else if t == FAN_END {
            inside = false;
        } else if inside {
            for part in t.strip_prefix("dtparam=").unwrap_or("").split(',') {
                let Some((key, value)) = part.split_once('=') else { continue };
                let Ok(value) = value.trim().parse::<u32>() else { continue };
                let Some(rest) = key.strip_prefix("fan_temp") else { continue };
                let (i, field) = rest.split_at(1.min(rest.len()));
                let Ok(i) = i.parse::<usize>() else { continue };
                if i >= 4 {
                    continue;
                }
                match field {
                    "" => levels[i].0 = value,
                    "_speed" => {
                        levels[i].1 = value;
                        seen += 1;
                    }
                    _ => {}
                }
            }
        }
    }
    if seen == 0 {
        return Fan::Default;
    }
    for fan in [Fan::Cool, Fan::Full] {
        if fan.levels() == Some(levels) {
            return fan;
        }
    }
    let s = levels[0].1;
    let percent = ((s * 100 + 127) / 255).clamp(u32::from(FAN_MIN_PERCENT), 100) as u8;
    Fan::Fixed { percent }
}

/// The Pi's boot config: `/boot/firmware/config.txt` on Raspberry Pi OS
/// from Bookworm on, `/boot/config.txt` on older images.
pub fn boot_config_path() -> Option<&'static str> {
    ["/boot/firmware/config.txt", "/boot/config.txt"].into_iter().find(|p| Path::new(p).exists())
}

/// The board's device-tree model ("Raspberry Pi 5 Model B Rev 1.0").
pub fn board_model() -> Option<String> {
    std::fs::read_to_string("/sys/firmware/devicetree/base/model")
        .ok()
        .map(|m| m.trim_end_matches('\0').trim().to_string())
        .filter(|m| !m.is_empty())
}

/// What `apply` changes: the clock (MHz and the extra voltage, in the
/// board's unit) and the fan curve. `None` leaves it as it is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Change {
    pub clock: Option<(u32, u32)>,
    pub fan: Option<Fan>,
}

impl Change {
    /// The helper's arguments for this change.
    pub fn args(&self) -> Vec<String> {
        let mut args = vec!["apply".to_string()];
        if let Some((mhz, voltage)) = self.clock {
            args.extend(["--clock".into(), format!("{mhz}:{voltage}")]);
        }
        if let Some(fan) = self.fan {
            args.extend(["--fan".into(), fan.arg()]);
        }
        args
    }

    /// Checks the change against the board (its model string) and returns
    /// `config` with it made. The error is the text to show.
    pub fn rewrite(&self, model: &str, config: &str) -> Result<String, String> {
        let profile = crate::rpc::axe::profile(model);
        let mut next = config.to_string();
        if let Some((mhz, voltage)) = self.clock {
            let profile = profile.as_ref().ok_or_else(|| format!("No clock settings are known for this board ({model})."))?;
            crate::rpc::axe::check(profile, mhz, voltage)?;
            next = crate::rpc::axe::rewrite_config(&next, profile, mhz, voltage);
        }
        if let Some(fan) = self.fan {
            if profile.as_ref().map(|p| p.family) != Some(crate::rpc::axe::Family::Pi5) {
                return Err("Fan curves are for a Raspberry Pi 5's active cooler.".into());
            }
            next = rewrite_fan(&next, fan);
        }
        Ok(next)
    }
}

/// Reads the helper's `apply` arguments.
fn parse_apply(args: &[String]) -> Result<Change, String> {
    let mut change = Change::default();
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--clock" if change.clock.is_none() => {
                let (mhz, voltage) = value.split_once(':').ok_or("--clock takes MHZ:VOLTAGE")?;
                let mhz = mhz.parse().map_err(|_| "--clock: the MHz is not a number")?;
                let voltage = voltage.parse().map_err(|_| "--clock: the voltage is not a number")?;
                change.clock = Some((mhz, voltage));
            }
            "--fan" if change.fan.is_none() => change.fan = Some(Fan::parse(value)?),
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    if change == Change::default() {
        return Err("nothing to apply: give --clock, --fan or both".into());
    }
    Ok(change)
}

/// Writes `text` to the boot config at `path`, keeping the old file beside
/// it, through a new file renamed into place.
fn write_boot_config(path: &Path, old: &str, text: &str) -> Result<(), String> {
    use std::io::Write;
    let backup = PathBuf::from(format!("{}.midstate-bak", path.display()));
    std::fs::write(&backup, old).map_err(|e| format!("{}: {}", backup.display(), e))?;
    let fresh = PathBuf::from(format!("{}.midstate-new", path.display()));
    let mut f = std::fs::File::create(&fresh).map_err(|e| format!("{}: {}", fresh.display(), e))?;
    f.write_all(text.as_bytes()).and_then(|_| f.sync_all()).map_err(|e| format!("{}: {}", fresh.display(), e))?;
    drop(f);
    std::fs::rename(&fresh, path).map_err(|e| format!("{}: {}", path.display(), e))?;
    if let Some(dir) = path.parent().and_then(|d| std::fs::File::open(d).ok()) {
        let _ = dir.sync_all();
    }
    Ok(())
}

/// What the helper prints: one line of JSON.
fn answer(result: Result<serde_json::Value, String>) -> i32 {
    let (line, code) = match result {
        Ok(mut v) => {
            if let Some(o) = v.as_object_mut() {
                o.insert("ok".into(), true.into());
            }
            (v, 0)
        }
        Err(e) => (serde_json::json!({ "ok": false, "error": e }), 1),
    };
    println!("{line}");
    code
}

/// Whether this process is the root-owned copy: by the file it runs from
/// (which a replaced copy still names, " (deleted)" after it) or the name it
/// was started by.
fn is_helper() -> bool {
    let named = |p: &Path| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with(HELPER_NAME));
    std::env::current_exe().is_ok_and(|p| named(&p))
        || std::env::args_os().next().is_some_and(|a| Path::new(&a).file_name().is_some_and(|n| n == HELPER_NAME))
}

/// When this process is the helper, runs it and returns its exit code; the
/// rest of midstate is never reached. `None` otherwise.
pub fn run_if_helper() -> Option<i32> {
    if !is_helper() {
        return None;
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    Some(answer(helper(&args)))
}

fn helper(args: &[String]) -> Result<serde_json::Value, String> {
    match args.first().map(String::as_str) {
        Some("status") if args.len() == 1 => Ok(serde_json::json!({
            "protocol": PROTOCOL,
            "version": env!("CARGO_PKG_VERSION"),
            "root": effective_uid() == Some(0),
        })),
        Some("apply") => {
            let change = parse_apply(&args[1..])?;
            let model = board_model().ok_or("This is not a Raspberry Pi: it has no device-tree model.")?;
            let path = boot_config_path().ok_or("The Pi's boot config was not found.")?;
            let old = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
            let next = change.rewrite(&model, &old)?;
            if next == old {
                return Ok(serde_json::json!({ "message": "The boot config already says this.", "changed": false }));
            }
            write_boot_config(Path::new(path), &old, &next)?;
            Ok(serde_json::json!({ "message": format!("Written to {path}. It takes effect at the next boot."), "changed": true }))
        }
        Some("reboot") if args.len() == 1 => {
            let status = std::process::Command::new("systemctl")
                .arg("reboot")
                .status()
                .or_else(|_| std::process::Command::new("reboot").status())
                .map_err(|e| format!("could not reboot: {e}"))?;
            if !status.success() {
                return Err(format!("reboot failed ({status})"));
            }
            Ok(serde_json::json!({ "message": "Restarting." }))
        }
        _ => Err(format!("{HELPER_NAME} takes: status | apply [--clock MHZ:VOLTAGE] [--fan SETTING] | reboot")),
    }
}

/// This process's effective user id, from `/proc`.
fn effective_uid() -> Option<u32> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("Uid:"))?;
    line.split_whitespace().nth(2)?.parse().ok()
}

/// Whether `name` can be a user's name in a sudoers line as it stands.
fn plain_user_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase() || c == '_')
        && name.len() <= 32
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
}

/// The sudoers line for `user`.
pub fn sudoers_text(user: &str) -> String {
    format!(
        "# Lets the midstate miner's dashboard set this Pi's clock and fan curve and\n\
         # reboot it, through a root-owned copy of midstate that does nothing else.\n\
         # Written by `midstate board-setup`; `midstate board-setup --remove` removes it.\n\
         {user} ALL=(root) NOPASSWD: {HELPER_PATH}\n"
    )
}

/// `midstate board-setup`, run with sudo: installs the helper for `user`
/// (the one sudo was run by, unless given), or with `remove` takes it away.
/// What was done, as text, or why not.
#[cfg(unix)]
pub fn setup(user: Option<String>, remove: bool) -> Result<String, String> {
    use std::os::unix::fs::PermissionsExt;
    if effective_uid() != Some(0) {
        let me = std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_else(|_| "midstate".into());
        return Err(format!("This needs root: sudo {me} board-setup{}", if remove { " --remove" } else { "" }));
    }
    if remove {
        for path in [SUDOERS_PATH, HELPER_PATH] {
            match std::fs::remove_file(path) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(format!("{path}: {e}")),
                _ => {}
            }
        }
        return Ok(format!("Removed {SUDOERS_PATH} and {HELPER_PATH}. The dashboard can no longer set the clock or fan, or reboot."));
    }
    let user = user
        .or_else(|| std::env::var("SUDO_USER").ok())
        .filter(|u| !u.is_empty() && u != "root")
        .ok_or("Name the user the miner runs as: sudo midstate board-setup --user <name>")?;
    if !plain_user_name(&user) {
        return Err(format!("\"{user}\" is not a user name this can write into sudoers."));
    }
    let passwd = std::fs::read_to_string("/etc/passwd").unwrap_or_default();
    if !passwd.lines().any(|l| l.split(':').next() == Some(user.as_str())) {
        return Err(format!("There is no user \"{user}\" on this machine."));
    }
    if board_model().as_deref().and_then(crate::rpc::axe::profile).is_none() {
        return Err("This is not a Raspberry Pi the dashboard knows: there is nothing to set up.".into());
    }

    // The copy: root's, written beside its place and renamed in.
    let exe = std::env::current_exe().map_err(|e| format!("could not find this program: {e}"))?;
    let dir = Path::new(HELPER_PATH).parent().unwrap_or(Path::new("/usr/local/sbin"));
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {}", dir.display(), e))?;
    let fresh = dir.join(format!(".{HELPER_NAME}.new"));
    std::fs::copy(&exe, &fresh).map_err(|e| format!("copying {} to {}: {}", exe.display(), fresh.display(), e))?;
    std::fs::set_permissions(&fresh, std::fs::Permissions::from_mode(0o755)).map_err(|e| e.to_string())?;
    std::fs::rename(&fresh, HELPER_PATH).map_err(|e| format!("{HELPER_PATH}: {e}"))?;

    // The sudoers line, checked by visudo before sudo reads it. sudo skips
    // files with a dot in their name, so the draft is never read.
    let draft = format!("/etc/sudoers.d/.{HELPER_NAME}.new");
    std::fs::write(&draft, sudoers_text(&user)).map_err(|e| format!("{draft}: {e}"))?;
    std::fs::set_permissions(&draft, std::fs::Permissions::from_mode(0o440)).map_err(|e| e.to_string())?;
    let checked = std::process::Command::new("visudo").args(["-cf", &draft]).output();
    match checked {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            let _ = std::fs::remove_file(&draft);
            return Err(format!("visudo refused the line: {}", String::from_utf8_lossy(&out.stderr).trim()));
        }
        Err(e) => {
            let _ = std::fs::remove_file(&draft);
            return Err(format!("could not run visudo to check the line: {e}"));
        }
    }
    std::fs::rename(&draft, SUDOERS_PATH).map_err(|e| format!("{SUDOERS_PATH}: {e}"))?;
    Ok(format!(
        "Installed {HELPER_PATH} (root's) and {SUDOERS_PATH}: {user} can now set the clock and fan and reboot from the miner's dashboard.\n\
         After installing a new midstate, run this again to update the copy. To undo: sudo {} board-setup --remove",
        exe.display()
    ))
}

#[cfg(not(unix))]
pub fn setup(_user: Option<String>, _remove: bool) -> Result<String, String> {
    Err("board-setup is for a Raspberry Pi.".into())
}

// ── The miner's side ────────────────────────────────────────────────────────

/// Whether the helper can be used from this process, and if not, why.
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct HelperStatus {
    pub ready: bool,
    /// What stands in the way, as the text to show.
    pub problem: Option<String>,
    /// The helper's midstate version, when it answers.
    pub version: Option<String>,
}

/// Runs the helper through `sudo -n` (never a password prompt) with `args`,
/// and reads its answer.
async fn call_helper(args: &[String], wait: std::time::Duration) -> Result<serde_json::Value, String> {
    if !Path::new(HELPER_PATH).exists() {
        return Err("Board settings need a one-time setup on the Pi.".into());
    }
    let run = tokio::process::Command::new("sudo").arg("-n").arg(HELPER_PATH).args(args).output();
    let out = tokio::time::timeout(wait, run)
        .await
        .map_err(|_| "The board helper did not answer.".to_string())?
        .map_err(|e| format!("could not run sudo: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    match stdout.lines().rev().find_map(|l| serde_json::from_str::<serde_json::Value>(l).ok()) {
        Some(v) if v["ok"] == true => Ok(v),
        Some(v) => Err(v["error"].as_str().unwrap_or("the board helper failed").to_string()),
        // No answer of its own: sudo stopped it (no sudoers line, or the
        // service runs with NoNewPrivileges).
        None => {
            let why = String::from_utf8_lossy(&out.stderr).trim().to_string();
            Err(if why.is_empty() { format!("the board helper failed ({})", out.status) } else { why })
        }
    }
}

/// Whether the helper is installed and this user may run it.
pub async fn helper_status() -> HelperStatus {
    match call_helper(&["status".to_string()], std::time::Duration::from_secs(10)).await {
        Ok(v) if v["protocol"].as_u64() == Some(u64::from(PROTOCOL)) => HelperStatus {
            ready: true,
            problem: None,
            version: v["version"].as_str().map(str::to_string),
        },
        Ok(v) => HelperStatus {
            ready: false,
            problem: Some("The board helper is from another version of midstate: run the setup again.".into()),
            version: v["version"].as_str().map(str::to_string),
        },
        Err(problem) => HelperStatus { ready: false, problem: Some(problem), version: None },
    }
}

/// Has the helper make `change` in the boot config. Its message, or why not.
pub async fn helper_apply(change: Change) -> Result<String, String> {
    let v = call_helper(&change.args(), std::time::Duration::from_secs(20)).await?;
    Ok(v["message"].as_str().unwrap_or("Done.").to_string())
}

/// Has the helper reboot the Pi.
pub async fn helper_reboot() -> Result<(), String> {
    call_helper(&["reboot".to_string()], std::time::Duration::from_secs(20)).await.map(|_| ())
}

/// The command that sets the helper up, for the program running now.
pub fn setup_command() -> String {
    let exe = std::env::current_exe()
        .map(|p| p.display().to_string().trim_end_matches(" (deleted)").to_string())
        .unwrap_or_else(|_| "./midstate".into());
    format!("sudo {exe} board-setup")
}

#[cfg(test)]
mod tests {
    use super::*;

    const PI5: &str = "Raspberry Pi 5 Model B Rev 1.0";
    const PI_OS: &str = "dtparam=audio=on\narm_64bit=1\ndtparam=fan_temp0=55000\n\n[cm5]\ndtoverlay=dwc2,dr_mode=host\n";

    #[test]
    fn fan_settings_read_and_write_as_the_helper_takes_them() {
        for fan in [Fan::Default, Fan::Cool, Fan::Full, Fan::Fixed { percent: 40 }, Fan::Fixed { percent: 100 }] {
            assert_eq!(Fan::parse(&fan.arg()), Ok(fan));
        }
        assert!(Fan::parse("fixed:10").unwrap_err().contains("20–100%"));
        assert!(Fan::parse("fixed:x").is_err());
        assert!(Fan::parse("loud").is_err());
        // Each curve rises, and every one but the Pi's own reaches full speed.
        for fan in [Fan::Cool, Fan::Full, Fan::Fixed { percent: 20 }, Fan::Fixed { percent: 60 }] {
            let l = fan.levels().unwrap();
            assert!(l.windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 <= w[1].1), "{fan:?}: {l:?}");
            assert_eq!(l[3].1, 255, "{fan:?} reaches full speed");
        }
        assert_eq!(Fan::Fixed { percent: 40 }.levels().unwrap()[0].1, 102);
    }

    #[test]
    fn the_fan_curve_is_one_block_at_the_end_for_every_model() {
        let cool = rewrite_fan(PI_OS, Fan::Cool);
        assert!(cool.starts_with(PI_OS), "the rest stays as it was: {cool}");
        let block = cool.split(FAN_START).nth(1).unwrap();
        assert!(block.starts_with("\n[all]\ndtparam=fan_temp0=40000,fan_temp0_hyst=5000,fan_temp0_speed=100\n"), "{block}");
        assert_eq!(block.matches("dtparam=fan_temp").count(), 4);
        assert_eq!(configured_fan(&cool), Fan::Cool);

        let fixed = rewrite_fan(&cool, Fan::Fixed { percent: 55 });
        assert_eq!(fixed.matches(FAN_START).count(), 1, "replaced, not added");
        assert_eq!(configured_fan(&fixed), Fan::Fixed { percent: 55 });
        assert_eq!(configured_fan(&rewrite_fan(PI_OS, Fan::Full)), Fan::Full);

        let back = rewrite_fan(&fixed, Fan::Default);
        assert_eq!(back, PI_OS, "the Pi's own curve: the block goes, the file's own line stays");
        assert_eq!(configured_fan(PI_OS), Fan::Default, "lines outside the block are not ours");

        let broken = format!("{FAN_START}\n[all]\ndtparam=audio=on\n");
        assert_eq!(rewrite_fan(&broken, Fan::Default), "[all]\ndtparam=audio=on\n", "an unclosed block keeps the rest of the file");
    }

    #[test]
    fn the_clock_and_fan_blocks_live_side_by_side() {
        let change = Change { clock: Some((2800, 0)), fan: Some(Fan::Cool) };
        let both = change.rewrite(PI5, PI_OS).unwrap();
        assert!(both.contains(crate::rpc::axe::BLOCK_START) && both.contains(FAN_START), "{both}");
        assert_eq!(crate::rpc::axe::configured(&both, crate::rpc::axe::profile(PI5).as_ref()), (Some(2800), None));
        // Changing one leaves the other.
        let stock = Change { clock: Some((2400, 0)), fan: None }.rewrite(PI5, &both).unwrap();
        assert!(!stock.contains("arm_freq=") && configured_fan(&stock) == Fan::Cool, "{stock}");
        let quiet = Change { clock: None, fan: Some(Fan::Default) }.rewrite(PI5, &both).unwrap();
        assert!(quiet.contains("arm_freq=2800") && !quiet.contains(FAN_START), "{quiet}");
    }

    #[test]
    fn the_helper_checks_what_it_is_given() {
        let err = |model: &str, c: Change| c.rewrite(model, PI_OS).unwrap_err();
        assert!(err(PI5, Change { clock: Some((3500, 0)), fan: None }).contains("outside this board's range"));
        assert!(err(PI5, Change { clock: Some((3000, 200_000)), fan: None }).contains("at most"));
        assert!(err("Raspberry Pi 4 Model B Rev 1.5", Change { clock: None, fan: Some(Fan::Full) }).contains("Raspberry Pi 5"));
        assert!(err("Radxa ROCK 5B", Change { clock: Some((2000, 0)), fan: None }).contains("No clock settings"));

        let args = |a: &[&str]| parse_apply(&a.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        assert_eq!(args(&["--clock", "2800:25000", "--fan", "cool"]), Ok(Change { clock: Some((2800, 25_000)), fan: Some(Fan::Cool) }));
        assert!(args(&[]).is_err(), "nothing to do");
        assert!(args(&["--clock", "2800"]).is_err());
        assert!(args(&["--clock", "2800:0", "--clock", "2400:0"]).is_err(), "once each");
        assert!(args(&["--path", "/etc/shadow"]).is_err());
        assert!(args(&["--fan"]).is_err());
        let change = Change { clock: Some((3000, 25_000)), fan: Some(Fan::Fixed { percent: 50 }) };
        assert_eq!(parse_apply(&change.args()[1..]), Ok(change), "what the miner sends is what the helper reads");

        assert!(helper(&["status".into(), "extra".into()]).is_err());
        assert!(helper(&["node".into()]).is_err(), "the helper runs nothing else");
        assert!(helper(&[]).is_err());
        assert_eq!(helper(&["status".into()]).unwrap()["protocol"], PROTOCOL);
    }

    #[test]
    fn sudoers_names_one_user_and_one_program() {
        assert!(plain_user_name("midstate") && plain_user_name("_axe") && plain_user_name("pi-5"));
        for bad in ["", "Root", "a b", "a,b", "ALL", "x\nroot ALL=(ALL) ALL", "%sudo", "1abc"] {
            assert!(!plain_user_name(bad), "{bad:?}");
        }
        let text = sudoers_text("midstate");
        assert!(text.ends_with("midstate ALL=(root) NOPASSWD: /usr/local/sbin/midstate-axe-board\n"), "{text}");
        assert_eq!(text.lines().filter(|l| !l.starts_with('#')).count(), 1);
    }
}
