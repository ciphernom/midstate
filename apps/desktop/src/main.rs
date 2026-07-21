//! Midstate Desktop — native egui frontend. One process, one language:
//! the node, walletd, and the UI all live in this binary.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod bridge;
mod theme;
mod views;

fn main() -> eframe::Result {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,midstate=info".into()),
        )
        .init();

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

    eframe::run_native(
        "Midstate Desktop",
        options,
        Box::new(|cc| Ok(Box::new(app::App::new(cc)))),
    )
}
