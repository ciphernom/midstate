//! The MidstateAxe's board: which Raspberry Pi it is, how far the dashboard
//! may move its clock, the `config.txt` lines that do it, and what
//! `vcgencmd get_throttled` says about its power and temperature. The `/axe`
//! handlers (`handlers.rs`) use this; it is plain logic, tested off the
//! device.
//!
//! # The board
//!
//! From block 420,000 midstate mines Midstate Walk, which wants AES
//! instructions, and the Raspberry Pi 5 is the first Pi that has them: it is
//! the Axe's board. Older Pis still run a full node and can still be set
//! here, within limits right for them, but they mine the Walk at a few
//! percent of a Pi 5.
//!
//! # Safety
//!
//! - Limits are per family, so no board can be set to another family's
//!   clock. (A single 800–1400 MHz range, the Zero 2 W's, used to apply to
//!   every board, which would have run a Pi 5 at well under half speed.)
//! - Nothing writes `force_turbo`. A miner keeps every core busy, so the
//!   firmware runs at `arm_freq` anyway, and on a Pi 4 or older
//!   `force_turbo` together with any `over_voltage` permanently sets the
//!   board's warranty bit.
//! - `over_voltage` stops at 6, above which those boards also set it.
//! - The settings go in a marked `[all]` block at the end of `config.txt`.
//!   Lines merely appended there land in whatever section the file ends in,
//!   `[cm5]` say, and apply to that model only.

use serde::Serialize;

/// A Raspberry Pi family the dashboard can set clocks on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Family {
    /// Raspberry Pi 5, 500 and Compute Module 5 (BCM2712: Cortex-A76, AES).
    Pi5,
    /// Raspberry Pi 4, 400 and Compute Module 4 (BCM2711: no AES).
    Pi4,
    /// Raspberry Pi 3, Zero 2 W and Compute Module 3 (no AES).
    Pi3,
}

/// How a family's extra voltage is set in `config.txt`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Voltage {
    /// `over_voltage_delta`: microvolts added to the firmware's own voltage
    /// (Pi 5).
    DeltaMicrovolts,
    /// `over_voltage`: 25 mV steps (Pi 4 and older).
    Steps,
}

/// A suggested clock, as the dashboard offers it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Preset {
    pub mhz: u32,
    /// In the family's [`Voltage`] unit.
    pub voltage: u32,
    pub label: &'static str,
}

/// What the dashboard may set on a board.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ClockProfile {
    pub family: Family,
    /// The board's own clock. Choosing it, with no extra voltage, removes
    /// the overclock rather than writing it, so the firmware's default
    /// stands (which on later Pi 4s is above this figure).
    pub stock_mhz: u32,
    pub min_mhz: u32,
    pub max_mhz: u32,
    pub voltage: Voltage,
    pub voltage_max: u32,
    pub voltage_step: u32,
    pub presets: Vec<Preset>,
}

/// The profile of a board by its device-tree model string
/// (`/sys/firmware/devicetree/base/model`), if it is a Pi the dashboard
/// knows.
pub fn profile(model: &str) -> Option<ClockProfile> {
    let m = model.to_ascii_lowercase();
    if m.contains("raspberry pi 5") || m.contains("compute module 5") {
        // "Raspberry Pi 500" contains "raspberry pi 5". Reaching 3.0 GHz and
        // over usually takes some extra voltage and always an active cooler;
        // past that it is the silicon lottery, so the presets are starting
        // points to test (`/axe/bench`), not promises.
        return Some(ClockProfile {
            family: Family::Pi5,
            stock_mhz: 2400,
            min_mhz: 1500,
            max_mhz: 3200,
            voltage: Voltage::DeltaMicrovolts,
            voltage_max: 100_000,
            voltage_step: 25_000,
            presets: vec![
                Preset { mhz: 2400, voltage: 0, label: "Stock" },
                Preset { mhz: 2600, voltage: 0, label: "Mild" },
                Preset { mhz: 2800, voltage: 0, label: "Fast (active cooler)" },
                Preset { mhz: 3000, voltage: 25_000, label: "Faster (active cooler)" },
                Preset { mhz: 3100, voltage: 50_000, label: "Max (active cooler, test it)" },
            ],
        });
    }
    if m.contains("raspberry pi 4") || m.contains("compute module 4") {
        return Some(ClockProfile {
            family: Family::Pi4,
            stock_mhz: 1500,
            min_mhz: 1000,
            max_mhz: 2147,
            voltage: Voltage::Steps,
            voltage_max: 6,
            voltage_step: 1,
            presets: vec![
                Preset { mhz: 1500, voltage: 0, label: "Stock (firmware default)" },
                Preset { mhz: 1800, voltage: 2, label: "Mild" },
                Preset { mhz: 2000, voltage: 6, label: "Fast (active cooler)" },
            ],
        });
    }
    if m.contains("raspberry pi 3") || m.contains("zero 2") || m.contains("compute module 3") {
        let stock = if m.contains("zero 2") {
            1000
        } else if m.contains("plus") {
            1400
        } else {
            1200
        };
        let mut presets = vec![Preset { mhz: stock, voltage: 0, label: "Stock" }];
        for (mhz, voltage, label) in [(1100, 2, "Light"), (1200, 4, "Turbo"), (1300, 6, "Aggressive"), (1400, 6, "Max (active cooler)")] {
            if mhz > stock {
                presets.push(Preset { mhz, voltage, label });
            }
        }
        return Some(ClockProfile {
            family: Family::Pi3,
            stock_mhz: stock,
            min_mhz: 800,
            max_mhz: 1400,
            voltage: Voltage::Steps,
            voltage_max: 6,
            voltage_step: 1,
            presets,
        });
    }
    None
}

/// Checks a requested clock against the board's limits. The error is the
/// text to show.
pub fn check(profile: &ClockProfile, mhz: u32, voltage: u32) -> Result<(), String> {
    if mhz < profile.min_mhz || mhz > profile.max_mhz {
        return Err(format!(
            "{} MHz is outside this board's range ({}–{} MHz)",
            mhz, profile.min_mhz, profile.max_mhz
        ));
    }
    if voltage > profile.voltage_max {
        return Err(match profile.voltage {
            Voltage::DeltaMicrovolts => format!(
                "+{} mV is more than this board takes (+{} mV at most)",
                voltage / 1000,
                profile.voltage_max / 1000
            ),
            Voltage::Steps => format!("over_voltage {} is more than this board takes ({} at most)", voltage, profile.voltage_max),
        });
    }
    Ok(())
}

/// Opens the block of `config.txt` lines the dashboard owns.
pub const BLOCK_START: &str = "# --- MidstateAxe overclock (set from the /axe dashboard) ---";
/// Closes it.
pub const BLOCK_END: &str = "# --- end MidstateAxe overclock ---";

/// Lines that set the CPU clock or voltage, wherever they are, which the
/// dashboard's block replaces. (`arm_freq_min=` and the like are left
/// alone: none of these prefixes matches them.)
const CLOCK_KEYS: [&str; 4] = ["arm_freq=", "over_voltage=", "over_voltage_delta=", "force_turbo="];

fn is_clock_line(line: &str) -> bool {
    let t = line.trim_start();
    CLOCK_KEYS.iter().any(|k| t.starts_with(k))
}

/// `config` (a Raspberry Pi `config.txt`) with any earlier overclock taken
/// out and, unless `mhz` and `voltage` are the board's stock settings, a
/// block at the end that sets them for every model.
pub fn rewrite_config(config: &str, profile: &ClockProfile, mhz: u32, voltage: u32) -> String {
    let mut out: Vec<&str> = Vec::new();
    let mut in_block = false;
    // A start marker with no end (an edit gone wrong) must not swallow the
    // rest of the file: the block is only skipped when it is closed.
    let closed = config.lines().any(|l| l.trim() == BLOCK_END);
    for line in config.lines() {
        let t = line.trim();
        if t == BLOCK_START {
            in_block = closed;
            continue;
        }
        if t == BLOCK_END {
            in_block = false;
            continue;
        }
        if in_block || is_clock_line(line) {
            continue;
        }
        out.push(line);
    }
    while out.last().is_some_and(|l| l.trim().is_empty()) {
        out.pop();
    }
    let mut text = out.join("\n");
    if !text.is_empty() {
        text.push('\n');
    }
    if mhz == profile.stock_mhz && voltage == 0 {
        return text;
    }
    if !text.is_empty() {
        text.push('\n');
    }
    text.push_str(BLOCK_START);
    text.push_str("\n[all]\n");
    text.push_str(&format!("arm_freq={}\n", mhz));
    if voltage > 0 {
        match profile.voltage {
            Voltage::DeltaMicrovolts => text.push_str(&format!("over_voltage_delta={}\n", voltage)),
            Voltage::Steps => text.push_str(&format!("over_voltage={}\n", voltage)),
        }
    }
    text.push_str(BLOCK_END);
    text.push('\n');
    text
}

/// The clock a `config.txt` sets: (MHz, extra voltage in the profile's
/// unit). The last line for each wins, as it does for the firmware. Either
/// is `None` when not set.
pub fn configured(config: &str, profile: Option<&ClockProfile>) -> (Option<u32>, Option<u32>) {
    let mut mhz = None;
    let mut steps = None;
    let mut delta = None;
    for line in config.lines() {
        let t = line.trim();
        if let Some(v) = t.strip_prefix("arm_freq=") {
            mhz = v.trim().parse().ok().or(mhz);
        } else if let Some(v) = t.strip_prefix("over_voltage_delta=") {
            delta = v.trim().parse().ok().or(delta);
        } else if let Some(v) = t.strip_prefix("over_voltage=") {
            steps = v.trim().parse().ok().or(steps);
        }
    }
    let voltage = match profile.map(|p| p.voltage) {
        Some(Voltage::DeltaMicrovolts) => delta,
        Some(Voltage::Steps) => steps,
        None => steps.or(delta),
    };
    (mhz, voltage)
}

/// The flags in `vcgencmd get_throttled`'s answer (`throttled=0x50005`).
pub fn parse_throttled(output: &str) -> Option<u32> {
    let hex = output.trim().strip_prefix("throttled=")?;
    u32::from_str_radix(hex.trim_start_matches("0x"), 16).ok()
}

/// What `vcgencmd get_throttled`'s flags say, in words: first what holds
/// now, then what has happened since boot.
pub fn throttle_words(flags: u32) -> Vec<&'static str> {
    const NOW: [(u32, &str); 4] = [
        (1 << 0, "under-voltage now"),
        (1 << 1, "clock capped now"),
        (1 << 2, "throttled now"),
        (1 << 3, "soft temperature limit now"),
    ];
    const SINCE_BOOT: [(u32, &str); 4] = [
        (1 << 16, "under-voltage since boot"),
        (1 << 17, "clock capped since boot"),
        (1 << 18, "throttled since boot"),
        (1 << 19, "soft temperature limit since boot"),
    ];
    NOW.iter().chain(SINCE_BOOT.iter()).filter(|(bit, _)| flags & bit != 0).map(|(_, w)| *w).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PI5: &str = "Raspberry Pi 5 Model B Rev 1.0\0";

    #[test]
    fn each_board_gets_its_own_family() {
        let family = |m: &str| profile(m).map(|p| p.family);
        assert_eq!(family(PI5), Some(Family::Pi5));
        assert_eq!(family("Raspberry Pi 500 Rev 1.0"), Some(Family::Pi5));
        assert_eq!(family("Raspberry Pi Compute Module 5 Rev 1.0"), Some(Family::Pi5));
        assert_eq!(family("Raspberry Pi 4 Model B Rev 1.5"), Some(Family::Pi4));
        assert_eq!(family("Raspberry Pi 400 Rev 1.1"), Some(Family::Pi4));
        assert_eq!(family("Raspberry Pi Compute Module 4 Rev 1.0"), Some(Family::Pi4));
        assert_eq!(family("Raspberry Pi Zero 2 W Rev 1.0"), Some(Family::Pi3));
        assert_eq!(family("Raspberry Pi 3 Model B Plus Rev 1.3"), Some(Family::Pi3));
        assert_eq!(family("Radxa ROCK 5B"), None);
        assert_eq!(profile("Raspberry Pi Zero 2 W Rev 1.0").unwrap().stock_mhz, 1000);
        assert_eq!(profile("Raspberry Pi 3 Model B Plus Rev 1.3").unwrap().stock_mhz, 1400);
        assert_eq!(profile("Raspberry Pi 3 Model B Rev 1.2").unwrap().stock_mhz, 1200);
    }

    #[test]
    fn a_pi_5_cannot_be_given_a_zero_2_w_clock() {
        let pi5 = profile(PI5).unwrap();
        assert!(check(&pi5, 1400, 0).is_err(), "well under half a Pi 5's speed");
        assert!(check(&pi5, 2800, 0).is_ok());
        assert!(check(&pi5, 3100, 50_000).is_ok());
        assert!(check(&pi5, 3300, 0).is_err());
        assert!(check(&pi5, 3000, 125_000).unwrap_err().contains("+100 mV at most"));
        let zero = profile("Raspberry Pi Zero 2 W Rev 1.0").unwrap();
        assert!(check(&zero, 2400, 0).is_err());
        assert!(check(&zero, 1200, 4).is_ok());
        assert!(check(&zero, 1200, 8).is_err(), "above 6 sets the warranty bit");
        for p in [pi5, zero, profile("Raspberry Pi 4 Model B Rev 1.5").unwrap()] {
            for preset in &p.presets {
                assert!(check(&p, preset.mhz, preset.voltage).is_ok(), "{:?} preset {:?}", p.family, preset);
            }
            assert_eq!((p.presets[0].mhz, p.presets[0].voltage), (p.stock_mhz, 0), "the first preset is stock");
        }
    }

    /// Raspberry Pi OS's own config.txt ends in model sections, so lines
    /// merely appended would apply to a Compute Module 5 only.
    const PI_OS_CONFIG: &str = "\
dtparam=audio=on
camera_auto_detect=1
arm_64bit=1
arm_freq=2600
force_turbo=1

[cm4]
otg_mode=1

[cm5]
dtoverlay=dwc2,dr_mode=host
";

    #[test]
    fn the_overclock_is_one_block_for_every_model() {
        let pi5 = profile(PI5).unwrap();
        let once = rewrite_config(PI_OS_CONFIG, &pi5, 3100, 50_000);
        assert!(!once.contains("arm_freq=2600") && !once.contains("force_turbo"), "{once}");
        assert!(once.contains("[cm5]\ndtoverlay=dwc2,dr_mode=host\n"), "{once}");
        let block = once.split(BLOCK_START).nth(1).expect("a block");
        assert_eq!(block, "\n[all]\narm_freq=3100\nover_voltage_delta=50000\n# --- end MidstateAxe overclock ---\n");
        assert_eq!(configured(&once, Some(&pi5)), (Some(3100), Some(50_000)));

        // Setting it again replaces the block rather than adding another.
        let twice = rewrite_config(&once, &pi5, 2800, 0);
        assert_eq!(twice.matches(BLOCK_START).count(), 1);
        assert!(twice.contains("arm_freq=2800\n") && !twice.contains("over_voltage"), "{twice}");

        // Stock takes the overclock out and leaves the rest as it was.
        let stock = rewrite_config(&twice, &pi5, 2400, 0);
        assert!(!stock.contains(BLOCK_START) && !stock.contains("arm_freq="), "{stock}");
        assert!(stock.starts_with("dtparam=audio=on\n") && stock.ends_with("dtoverlay=dwc2,dr_mode=host\n"), "{stock:?}");

        // Other clock settings are left alone.
        let keep = rewrite_config("arm_freq_min=1500\nover_voltage_min=2\n", &pi5, 3000, 25_000);
        assert!(keep.starts_with("arm_freq_min=1500\nover_voltage_min=2\n"), "{keep}");

        // A block missing its end marker doesn't take the rest of the file.
        let broken = format!("{BLOCK_START}\narm_freq=3000\ndtparam=audio=on\n");
        let mended = rewrite_config(&broken, &pi5, 2400, 0);
        assert_eq!(mended, "dtparam=audio=on\n");
    }

    #[test]
    fn older_boards_take_over_voltage_steps() {
        let pi4 = profile("Raspberry Pi 4 Model B Rev 1.5").unwrap();
        let cfg = rewrite_config("", &pi4, 2000, 6);
        assert_eq!(cfg, format!("{BLOCK_START}\n[all]\narm_freq=2000\nover_voltage=6\n{BLOCK_END}\n"));
        assert_eq!(configured(&cfg, Some(&pi4)), (Some(2000), Some(6)));
    }

    #[test]
    fn throttling_reads_as_words() {
        assert_eq!(parse_throttled("throttled=0x0\n"), Some(0));
        assert_eq!(parse_throttled("throttled=0x50005"), Some(0x50005));
        assert_eq!(parse_throttled("error"), None);
        assert_eq!(throttle_words(0), Vec::<&str>::new());
        assert_eq!(
            throttle_words(0x50005),
            vec!["under-voltage now", "throttled now", "under-voltage since boot", "throttled since boot"]
        );
    }
}
