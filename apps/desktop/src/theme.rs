//! Design tokens + small widgets, matched to midstate.css:
//! "ABSOLUTE HYPER-MEGA-MINIMAL. Pure ink on paper. Zero hue. Zero shadow.
//!  Zero radius. Zero motion. Zero decorative fills."
//! State is carried by sign, weight, labels and structural borders — never
//! by color. Inter for UI, JetBrains Mono for data (both SIL OFL, embedded).

use eframe::egui::{
    self, Align, Color32, Context, CornerRadius, FontFamily, FontId, Frame, Margin, RichText,
    Stroke, Ui,
};
use midstate_walletd::api::{SendProgress, SendStage};

// ── Ink-on-paper palette (midstate.css tokens, both themes) ───────────────
// midstate.css ships `:root/[data-theme="dark"]` and `[data-theme="light"]`
// token sets. These accessors resolve to whichever is active, so the whole
// UI re-tones from one switch. Semantic names survive; hues never appear.

use std::sync::atomic::{AtomicU8, Ordering};

/// 0 = follow the OS, 1 = force dark, 2 = force light.
static MODE: AtomicU8 = AtomicU8::new(0);
/// What the OS most recently told us (eframe reports this per frame).
static OS_DARK: AtomicU8 = AtomicU8::new(1);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ThemeMode {
    System,
    Dark,
    Light,
}

pub fn set_mode(m: ThemeMode) {
    MODE.store(match m { ThemeMode::System => 0, ThemeMode::Dark => 1, ThemeMode::Light => 2 }, Ordering::Relaxed);
}
pub fn mode() -> ThemeMode {
    match MODE.load(Ordering::Relaxed) {
        1 => ThemeMode::Dark,
        2 => ThemeMode::Light,
        _ => ThemeMode::System,
    }
}
/// Called each frame with what the OS reports, so `System` tracks it live.
pub fn set_os_dark(dark: bool) {
    OS_DARK.store(if dark { 1 } else { 0 }, Ordering::Relaxed);
}
/// Where the theme preference lives (same dir family as the wallet data).
fn pref_path() -> std::path::PathBuf {
    let base = std::env::var("XDG_DATA_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
                .join(".local/share")
        });
    base.join("midstate-desktop").join("theme")
}

pub fn load_pref() {
    if let Ok(s) = std::fs::read_to_string(pref_path()) {
        set_mode(match s.trim() {
            "dark" => ThemeMode::Dark,
            "light" => ThemeMode::Light,
            _ => ThemeMode::System,
        });
    }
}

pub fn save_pref() {
    let p = pref_path();
    if let Some(dir) = p.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(
        p,
        match mode() {
            ThemeMode::Dark => "dark",
            ThemeMode::Light => "light",
            ThemeMode::System => "system",
        },
    );
}

pub fn is_dark() -> bool {
    match MODE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => OS_DARK.load(Ordering::Relaxed) == 1,
    }
}

#[inline]
fn pick(dark: (u8, u8, u8), light: (u8, u8, u8)) -> Color32 {
    let (r, g, b) = if is_dark() { dark } else { light };
    Color32::from_rgb(r, g, b)
}

pub fn bg() -> Color32 { pick((0x0a, 0x0a, 0x0a), (0xff, 0xff, 0xff)) }
pub fn panel() -> Color32 { pick((0x16, 0x16, 0x16), (0xfa, 0xfa, 0xfa)) }
pub fn panel2() -> Color32 { pick((0x14, 0x14, 0x14), (0xff, 0xff, 0xff)) }
pub fn highlight() -> Color32 { pick((0x1f, 0x1f, 0x1f), (0xef, 0xef, 0xef)) }
pub fn border() -> Color32 { pick((0x26, 0x26, 0x26), (0xe3, 0xe3, 0xe3)) }
pub fn border2() -> Color32 { pick((0x36, 0x36, 0x36), (0xcf, 0xcf, 0xcf)) }
pub fn ink() -> Color32 { pick((0xf5, 0xf5, 0xf5), (0x11, 0x11, 0x11)) }
pub fn bright() -> Color32 { pick((0xcf, 0xcf, 0xcf), (0x38, 0x38, 0x38)) }
pub fn muted() -> Color32 { pick((0x9a, 0x9a, 0x9a), (0x56, 0x56, 0x56)) }
pub fn faint() -> Color32 { pick((0x5a, 0x5a, 0x5a), (0xa2, 0xa2, 0xa2)) }
/// Ambient text (the giant midstate hash) — just off the page tone.
pub fn ambient() -> Color32 { pick((0x19, 0x19, 0x19), (0xf1, 0xf1, 0xf1)) }
/// Text drawn ON an ink fill.
pub fn on_ink() -> Color32 { pick((0x0a, 0x0a, 0x0a), (0xff, 0xff, 0xff)) }

// Semantic tokens collapse to ink/grey, exactly as midstate.css does.
pub fn gold() -> Color32 { ink() }
pub fn green() -> Color32 { ink() }
pub fn red() -> Color32 { muted() }
pub fn amber() -> Color32 { muted() }

pub fn apply(ctx: &Context) {
    // Brand fonts, embedded (see assets/fonts/OFL-NOTICE.txt).
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "Inter".into(),
        egui::FontData::from_static(include_bytes!("../assets/fonts/Inter-Regular.ttf")).into(),
    );
    fonts.font_data.insert(
        "Inter-Medium".into(),
        egui::FontData::from_static(include_bytes!("../assets/fonts/Inter-Medium.ttf")).into(),
    );
    fonts.font_data.insert(
        "JetBrainsMono".into(),
        egui::FontData::from_static(include_bytes!("../assets/fonts/JetBrainsMono-Regular.ttf"))
            .into(),
    );
    fonts.font_data.insert(
        "JetBrainsMono-Medium".into(),
        egui::FontData::from_static(include_bytes!("../assets/fonts/JetBrainsMono-Medium.ttf"))
            .into(),
    );
    if let Some(fam) = fonts.families.get_mut(&FontFamily::Proportional) {
        fam.insert(0, "Inter".into());
    }
    if let Some(fam) = fonts.families.get_mut(&FontFamily::Monospace) {
        fam.insert(0, "JetBrainsMono".into());
    }
    fonts.families.insert(
        FontFamily::Name("medium".into()),
        vec!["Inter-Medium".into(), "Inter".into()],
    );
    fonts.families.insert(
        FontFamily::Name("mono-medium".into()),
        vec!["JetBrainsMono-Medium".into(), "JetBrainsMono".into()],
    );
    ctx.set_fonts(fonts);

    let mut v = egui::Visuals::dark();
    v.panel_fill = bg();
    v.window_fill = panel();
    v.extreme_bg_color = Color32::from_rgb(0x0f, 0x0f, 0x0f); // --msg-bg (inputs)
    v.faint_bg_color = panel2(); // striped rows
    v.override_text_color = Some(ink());
    v.hyperlink_color = ink();
    v.selection.bg_fill = border2();
    v.selection.stroke = Stroke::new(1.0, ink()); // focus ring: 1px accent(=ink)
    v.window_stroke = Stroke::new(1.0, border());
    v.window_shadow = egui::Shadow::NONE; // zero shadow
    v.popup_shadow = egui::Shadow::NONE;
    v.window_corner_radius = CornerRadius::ZERO; // zero radius
    v.menu_corner_radius = CornerRadius::ZERO;

    // Zero radius, no decorative fills, hover carried by border only.
    for w in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        w.corner_radius = CornerRadius::ZERO;
        w.expansion = 0.0;
    }
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, border());
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, muted());
    v.widgets.inactive.bg_fill = Color32::TRANSPARENT;
    v.widgets.inactive.weak_bg_fill = Color32::TRANSPARENT;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, border2());
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, ink());
    v.widgets.hovered.bg_fill = Color32::TRANSPARENT;
    v.widgets.hovered.weak_bg_fill = Color32::TRANSPARENT;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, ink()); // .btn:hover
    v.widgets.hovered.fg_stroke = Stroke::new(1.0, ink());
    v.widgets.active.bg_fill = Color32::TRANSPARENT;
    v.widgets.active.weak_bg_fill = highlight();
    v.widgets.active.bg_stroke = Stroke::new(1.0, ink());
    v.widgets.active.fg_stroke = Stroke::new(1.0, ink());
    v.widgets.open.weak_bg_fill = highlight();
    ctx.set_visuals(v);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(10.0, 8.0);
    style.spacing.button_padding = egui::vec2(14.0, 7.0);
    ctx.set_style(style);
}

pub fn font_medium(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name("medium".into()))
}
pub fn font_mono_medium(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name("mono-medium".into()))
}

pub fn panel_frame() -> Frame {
    Frame::default()
        .fill(panel())
        .stroke(Stroke::new(1.0, border()))
        .corner_radius(CornerRadius::ZERO)
        .inner_margin(Margin::symmetric(16, 14))
}

// ── Formatting ──────────────────────────────────────────────────────────────

/// Integer units with thousands separators (the chain has no decimal subunit).
pub fn units(v: u64) -> String {
    let s = v.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

pub fn short_hex(h: &str, n: usize) -> String {
    if h.len() <= n * 2 + 1 {
        h.to_string()
    } else {
        format!("{}…{}", &h[..n], &h[h.len() - n..])
    }
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn ago(ts: u64) -> String {
    if ts == 0 {
        return "—".into();
    }
    let s = now().saturating_sub(ts);
    match s {
        0..=59 => format!("{s}s ago"),
        60..=3599 => format!("{}m ago", s / 60),
        3600..=86_399 => format!("{}h ago", s / 3600),
        _ => format!("{}d ago", s / 86_400),
    }
}

pub fn fmt_duration(secs: u64) -> String {
    match secs {
        0..=89 => format!("{secs}s"),
        90..=5399 => format!("{}m", (secs + 30) / 60),
        5400..=172_799 => format!("{}h {}m", secs / 3600, (secs % 3600) / 60),
        _ => format!("{}d {}h", secs / 86_400, (secs % 86_400) / 3600),
    }
}

/// Unix seconds → "YYYY-MM-DD HH:MM" (UTC), no chrono dependency.
/// Civil-from-days per Howard Hinnant's algorithm.
pub fn fmt_dt(ts: u64) -> String {
    if ts == 0 {
        return "—".into();
    }
    let days = (ts / 86_400) as i64;
    let secs = ts % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02} {:02}:{:02}", y, m, d, secs / 3600, (secs % 3600) / 60)
}

// ── Small widgets ───────────────────────────────────────────────────────────

pub fn mono(t: impl ToString) -> RichText {
    RichText::new(t.to_string()).monospace()
}

pub fn heading(ui: &mut Ui, t: &str) {
    ui.add_space(14.0);
    ui.label(RichText::new(t).font(font_medium(16.0)).color(ink()));
    ui.add_space(6.0);
}

pub fn hint(ui: &mut Ui, t: &str) {
    ui.label(RichText::new(t).size(12.0).color(muted()));
}

/// Connection dot: filled ink when live, outlined when not.
/// (Font-independent — painted, so no missing-glyph boxes.)
pub fn dot(ui: &mut Ui, live: bool) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(9.0, 12.0), egui::Sense::hover());
    let c = egui::pos2(rect.center().x, rect.center().y + 1.0);
    if live {
        ui.painter().circle_filled(c, 3.0, ink());
    } else {
        ui.painter().circle_stroke(c, 3.0, Stroke::new(1.0, muted()));
    }
}

/// Zero-radius progress bar: ink fill on a structural trough.
pub fn progress_bar(ui: &mut Ui, frac: f32) {
    let w = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(w, 6.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::ZERO, highlight());
    let mut fill = rect;
    fill.set_width(w * frac.clamp(0.0, 1.0));
    ui.painter().rect_filled(fill, CornerRadius::ZERO, ink());
}

pub fn badge(ui: &mut Ui, text: &str, color: Color32) {
    Frame::default()
        .stroke(Stroke::new(1.0, if color == ink() { border2() } else { border() } ))
        .corner_radius(CornerRadius::ZERO)
        .inner_margin(Margin::symmetric(7, 1))
        .show(ui, |ui| {
            ui.label(
                RichText::new(text.to_uppercase())
                    .font(FontId::monospace(9.5))
                    .color(color),
            );
        });
}

pub fn stat(ui: &mut Ui, label: &str, value: &str, suffix: &str) {
    panel_frame().show(ui, |ui| {
        ui.set_min_width(150.0);
        ui.label(RichText::new(label.to_uppercase()).font(FontId::monospace(10.0)).color(muted()));
        ui.horizontal(|ui| {
            ui.label(RichText::new(value).font(font_mono_medium(21.0)).color(ink()));
            if !suffix.is_empty() {
                ui.label(RichText::new(suffix).size(11.0).color(muted()));
            }
        });
    });
}

/// The signature element: a send rendered as a hash-chain. Each stage is a
/// state carrying a fragment of the commitment hash. Monochrome per the
/// stylesheet: progress is weight and border, never hue. No motion.
pub fn send_timeline(ui: &mut Ui, p: &SendProgress) {
    const STEPS: [&str; 5] = ["commit", "mined", "delay", "reveal", "final"];
    let (pos, broken) = match p.stage {
        SendStage::Committing => (0, false),
        SendStage::CommitPending => (1, false),
        SendStage::WaitingReveal => (2, false),
        SendStage::RevealPending => (3, false),
        SendStage::Confirmed => (4, false),
        SendStage::Stalled | SendStage::Failed => (1, true),
    };
    ui.horizontal(|ui| {
        for (i, step) in STEPS.iter().enumerate() {
            if i > 0 {
                let (rect, _) = ui.allocate_exact_size(egui::vec2(18.0, 24.0), egui::Sense::hover());
                let col = if i <= pos && !broken { ink() } else { border2() };
                ui.painter().hline(
                    egui::Rangef::new(rect.min.x, rect.max.x),
                    rect.center().y,
                    Stroke::new(1.0, col),
                );
            }
            let frag: String = p.id.chars().skip(i * 4).take(4).collect();
            let frag = if frag.is_empty() { "····".into() } else { frag };
            let done = i < pos || (i == pos && p.stage == SendStage::Confirmed);
            let active = i == pos && !done && !broken;
            let halted = broken && i == pos;
            let (fg, stroke, fill) = if halted {
                (muted(), border2(), bg())
            } else if active {
                (ink(), ink(), highlight())
            } else if done {
                (bright(), border2(), bg())
            } else {
                (faint(), border(), bg())
            };
            Frame::default()
                .fill(fill)
                .stroke(Stroke::new(1.0, stroke))
                .corner_radius(CornerRadius::ZERO)
                .inner_margin(Margin::symmetric(9, 4))
                .show(ui, |ui| {
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = 0.0;
                        let mut label = RichText::new(*step).color(fg);
                        label = if active {
                            label.font(font_mono_medium(11.0))
                        } else {
                            label.font(FontId::monospace(11.0))
                        };
                        if halted {
                            label = label.strikethrough();
                        }
                        ui.label(label);
                        ui.label(RichText::new(frag).font(FontId::monospace(9.0)).color(fg));
                    });
                });
        }
    });
}

/// Word-wrappable rendering for a 64-char hash: grouped so egui can break it.
pub fn grouped_hash(h: &str, group: usize) -> String {
    h.as_bytes()
        .chunks(group)
        .map(|c| std::str::from_utf8(c).unwrap_or(""))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn right_aligned(ui: &mut Ui, add: impl FnOnce(&mut Ui)) {
    ui.with_layout(egui::Layout::right_to_left(Align::Center), add);
}

/// Denomination prefixes, matching the web wallet's MDS_UNITS exactly.
pub const UNIT_NAMES: [&str; 4] = ["MDS", "kMDS", "mMDS", "gMDS"];
pub const UNIT_MULS: [u64; 4] = [1, 1024, 1_048_576, 1_073_741_824];

/// Render a raw unit count in the largest prefix that condenses it.
pub fn compact_units(v: u64) -> String {
    for i in (1..4).rev() {
        if v >= UNIT_MULS[i] {
            let whole = v / UNIT_MULS[i];
            let frac = (v % UNIT_MULS[i]) as f64 / UNIT_MULS[i] as f64;
            let s = format!("{:.4}", whole as f64 + frac);
            let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
            return format!("{s} {}", UNIT_NAMES[i]);
        }
    }
    format!("{} MDS", units(v))
}

/// Parse an amount typed in `unit_ix` into raw units, rejecting fractions
/// that would not land on a whole unit.
pub fn parse_in_unit(text: &str, unit_ix: usize) -> Result<u64, String> {
    let mul = UNIT_MULS[unit_ix.min(3)];
    let t = text.trim();
    if t.is_empty() {
        return Err(String::new());
    }
    let v: f64 = t.parse().map_err(|_| "not a number".to_string())?;
    if v <= 0.0 {
        return Err("amount must be positive".into());
    }
    let raw = v * mul as f64;
    if raw.fract().abs() > 1e-6 {
        return Err(format!("{t} {} is not a whole number of MDS", UNIT_NAMES[unit_ix.min(3)]));
    }
    Ok(raw.round() as u64)
}

/// A segmented unit picker. Returns true when the selection changed.
pub fn unit_selector(ui: &mut Ui, current: &mut usize) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        for (i, name) in UNIT_NAMES.iter().enumerate() {
            let active = *current == i;
            let txt = RichText::new(*name)
                .font(FontId::monospace(11.5))
                .color(if active { on_ink() } else { muted() });
            let btn = egui::Button::new(txt)
                .fill(if active { ink() } else { Color32::TRANSPARENT })
                .stroke(Stroke::new(1.0, border()));
            if ui.add(btn).clicked() && !active {
                *current = i;
                changed = true;
            }
        }
    });
    changed
}

// ── Logo ──────────────────────────────────────────────────────────────────
// Pre-rasterised from logo.svg. Two forms, deliberately:
//   • an ALPHA MASK for use inside the app — the artwork's shapes only, tinted
//     to the current ink tone, so the mark obeys the zero-hue rule in both
//     light and dark;
//   • full-colour RGBA for the OS window/taskbar icon, where brand colour is
//     appropriate because it sits outside this design system.
// Stored as raw bytes rather than PNG so no image decoder is needed.
pub const LOGO_MASK: &[u8] = include_bytes!("../assets/logo-mask-256.gray");
pub const LOGO_MASK_DIM: usize = 256;
pub const LOGO_ICON: &[u8] = include_bytes!("../assets/logo-icon-128.rgba");
pub const LOGO_ICON_DIM: u32 = 128;

/// Upload the logo mask once. White pixels carrying the shape in alpha, so a
/// tint at draw time recolours it without re-uploading on theme change.
pub fn load_logo(ctx: &Context) -> egui::TextureHandle {
    let mut rgba = Vec::with_capacity(LOGO_MASK.len() * 4);
    for &a in LOGO_MASK {
        rgba.extend_from_slice(&[255, 255, 255, a]);
    }
    let img = egui::ColorImage::from_rgba_unmultiplied([LOGO_MASK_DIM, LOGO_MASK_DIM], &rgba);
    ctx.load_texture("midstate-logo", img, egui::TextureOptions::LINEAR)
}

/// Draw the mark at `px` square, inked in the current theme tone.
pub fn logo(ui: &mut Ui, tex: &egui::TextureHandle, px: f32, color: Color32) {
    let sized = egui::load::SizedTexture::new(tex.id(), egui::vec2(px, px));
    ui.add(egui::Image::from_texture(sized).tint(color));
}
