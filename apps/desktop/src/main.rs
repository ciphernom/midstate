//! Midstate Desktop — native egui frontend. One process, one language:
//! the node, walletd, and the UI all live in this binary.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod bridge;
mod theme;
mod views;

use std::io::Write;

/// Log sink: the file if we could open one, and stdout regardless.
///
/// Release builds on Windows are `windows_subsystem = "windows"`, so the
/// process gets no console and every stdout write is dropped on the floor.
/// That makes the file the only channel through which a user can tell us what
/// happened. Write errors are swallowed on purpose — logging must never be
/// the thing that takes the app down.
struct Tee(Option<std::sync::Arc<std::fs::File>>);

impl Write for Tee {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Some(f) = &self.0 {
            let mut w: &std::fs::File = f;
            let _ = w.write_all(buf);
        }
        let _ = std::io::stdout().write_all(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        if let Some(f) = &self.0 {
            let mut w: &std::fs::File = f;
            let _ = w.flush();
        }
        let _ = std::io::stdout().flush();
        Ok(())
    }
}

/// Put a message in front of the user when there is no window to put it in.
///
/// `run_native` can fail before anything is on screen — most often because a
/// GL context could not be created — and `fn main() -> eframe::Result` prints
/// that error to a stderr that does not exist on a GUI-subsystem build. The
/// user sees nothing at all, which is indistinguishable from the app not
/// having launched. A message box is the only surface left.
#[cfg(windows)]
fn fatal_dialog(title: &str, body: &str) {
    // A modal box with nobody to click it never returns. Under CI, headless
    // sessions, or any automation, that turns a clean failure into a process
    // that hangs forever — worse than the silence it was added to replace.
    if std::env::var_os("MIDSTATE_NO_DIALOG").is_some() {
        return;
    }

    #[link(name = "user32")]
    extern "system" {
        fn MessageBoxW(
            hwnd: *mut core::ffi::c_void,
            text: *const u16,
            caption: *const u16,
            utype: u32,
        ) -> i32;
    }
    const MB_ICONERROR: u32 = 0x0000_0010;
    // Without these the box can open behind whatever currently has focus,
    // which looks exactly like the silent failure we are trying to fix.
    const MB_SETFOREGROUND: u32 = 0x0001_0000;
    const MB_TOPMOST: u32 = 0x0004_0000;

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }
    let text = wide(body);
    let caption = wide(title);
    unsafe {
        MessageBoxW(
            std::ptr::null_mut(),
            text.as_ptr(),
            caption.as_ptr(),
            MB_ICONERROR | MB_SETFOREGROUND | MB_TOPMOST,
        );
    }
}

#[cfg(not(windows))]
fn fatal_dialog(_title: &str, _body: &str) {
    // Other platforms keep a usable stderr, so the returned Err is visible.
}

fn main() -> eframe::Result {
    // Same profile rules as the wallet data, so the log lands beside it and
    // MIDSTATE_PROFILE / MIDSTATE_DATA_DIR move both together.
    let log_path = {
        let base = app::profile().base;
        let _ = std::fs::create_dir_all(&base);
        base.join("midstate-desktop.log")
    };
    // Truncating: one file per run, so what a user sends is the run they are
    // describing rather than a pile of old sessions.
    let log_file = std::fs::File::create(&log_path).ok().map(std::sync::Arc::new);

    tracing_subscriber::fmt()
        // Escape codes are noise in a file that gets pasted into a chat window.
        .with_ansi(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,midstate=info".into()),
        )
        .with_writer(move || Tee(log_file.clone()))
        .init();

    // First line out, before anything can fail. Its presence or absence in the
    // log splits "died before main" from "died during startup".
    tracing::info!(
        "midstate-desktop {} starting on {} ({})",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
    );
    tracing::info!("log file: {}", log_path.display());

    // Brand-coloured badge for the window, taskbar and dock.
    let icon = eframe::egui::IconData {
        rgba: theme::LOGO_ICON.to_vec(),
        width: theme::LOGO_ICON_DIM,
        height: theme::LOGO_ICON_DIM,
    };

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size(eframe::egui::vec2(1180.0, 780.0))
            .with_min_inner_size(eframe::egui::vec2(940.0, 620.0))
            .with_icon(icon),
        ..Default::default()
    };

    let result = eframe::run_native(
        "Midstate Desktop",
        options,
        Box::new(|cc| Ok(Box::new(app::App::new(cc)))),
    );

    if let Err(e) = &result {
        tracing::error!("startup failed: {e}");
        fatal_dialog(
            "Midstate Desktop",
            &format!(
                "Midstate Desktop could not start.\n\n\
                 {e}\n\n\
                 This is almost always a graphics problem. The app needs \
                 OpenGL 2.0, which every working display driver provides — so \
                 this usually means no graphics driver is installed at all \
                 (Windows falls back to \"Microsoft Basic Display Adapter\"), \
                 or you are running over Remote Desktop or in a virtual \
                 machine without 3D support.\n\n\
                 Installing your graphics driver normally fixes it.\n\n\
                 A log was written to:\n{}",
                log_path.display()
            ),
        );
    }
    result
}
