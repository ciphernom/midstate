//! Gives midstate-desktop.exe the Midstate logo as its icon on Windows.
//!
//! The window sets its own icon once it opens (`main.rs`), but Explorer,
//! shortcuts, the Start menu and a pinned taskbar entry read the icon
//! embedded in the executable. Without one they show Windows' generic
//! program icon. `embed-resource` compiles `assets/windows/midstate-desktop.rc`
//! with the target's resource compiler (rc.exe from the Windows SDK for
//! MSVC, windres for MinGW) and links it into the binary. For any other
//! target it does nothing.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/windows/midstate-desktop.rc");
    println!("cargo:rerun-if-changed=assets/windows/midstate-desktop.ico");

    // The include directory lets every resource compiler find the icon the
    // script names, whatever directory it resolves names against.
    let result = embed_resource::compile(
        "assets/windows/midstate-desktop.rc",
        embed_resource::ParamsIncludeDirs(&["assets/windows"]),
    );
    if let embed_resource::CompilationResult::NotAttempted(why) = &result {
        // Still a working program, only with the generic icon. Release
        // builds fail on this instead (.github/workflows/build.yml).
        println!("cargo:warning=midstate-desktop.exe is being built without its icon: {why}");
    }
    result.manifest_optional().unwrap();
}
