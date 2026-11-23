//! Automatic updates.
//!
//! Every few hours the app asks GitHub for the latest release of
//! `ciphernom/midstate`. A newer one is installed only if:
//!
//! * its `midstate-release.txt` carries an Ed25519 signature, in
//!   `midstate-release.txt.sig`, by one of the keys in [`RELEASE_KEYS`] —
//!   compiled into this app, so a release put up by anyone else (or an asset
//!   swapped afterwards) is refused;
//! * that signed file names the same version as the release, and the version
//!   is newer than this one, so an old release cannot be passed off as new;
//! * the archive for this computer hashes to the SHA-256 the signed file
//!   gives for it.
//!
//! The two programs in the archive (the app and `midstate`, which mines) are
//! unpacked into the profile folder and put in place the next time the app
//! starts — or at once, from "Restart now". The old ones are moved aside
//! first and put back if anything fails. A folder this user cannot write to
//! (an install for every user of the computer) gets a download link instead.
//!
//! Builds from source never update themselves: only release builds know
//! their version (CI stamps it in as `MIDSTATE_RELEASE`).

use anyhow::{anyhow, bail, ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Where releases come from.
pub const REPO: &str = "ciphernom/midstate";

/// Ed25519 public keys (hex) that sign releases. A release is installed only
/// when one of them signed its `midstate-release.txt`. Empty: updates are
/// off. Make the key with `tools/release/keygen.sh`.
pub const RELEASE_KEYS: &[&str] = &["45ff33ab5555622d948be7fe9d0791cae36bcf2642a94ddd24c93368b4d19cc3"];

/// This build's release tag, stamped in by CI. `None` for a build from
/// source, which never updates itself.
pub const THIS_RELEASE: Option<&str> = option_env!("MIDSTATE_RELEASE");

/// The signed list of a release's files, and its signature.
pub const MANIFEST: &str = "midstate-release.txt";
pub const SIGNATURE: &str = "midstate-release.txt.sig";

const API: &str = "https://api.github.com";
/// How often to look for a release.
const EVERY: Duration = Duration::from_secs(6 * 3600);
/// The first look waits for the app to settle.
const FIRST_LOOK: Duration = Duration::from_secs(45);
/// Caps on what is downloaded and unpacked.
const MAX_SMALL: u64 = 64 << 10;
const MAX_ARCHIVE: u64 = 512 << 20;
const MAX_BINARY: u64 = 512 << 20;

pub type Version = (u64, u64, u64);

/// `v6.5.2` (or `6.5.2`) as numbers. Anything else — a pre-release, a
/// fourth number — is not a version this app installs.
pub fn parse_version(tag: &str) -> Option<Version> {
    let t = tag.trim().strip_prefix('v').unwrap_or(tag.trim());
    let mut parts = t.split('.');
    let mut n = || -> Option<u64> {
        let p = parts.next()?;
        if p.is_empty() || !p.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        p.parse().ok()
    };
    let v = (n()?, n()?, n()?);
    parts.next().is_none().then_some(v)
}

/// This app's version, for people.
pub fn this_version() -> String {
    match THIS_RELEASE {
        Some(v) => v.trim_start_matches('v').to_string(),
        None => "(built from source)".to_string(),
    }
}

/// The release archive for a computer, if releases are built for it.
pub fn platform_asset(os: &str, arch: &str) -> Option<&'static str> {
    match (os, arch) {
        ("linux", "x86_64") => Some("midstate-linux-amd64.tar.gz"),
        ("windows", "x86_64") => Some("midstate-windows-amd64.zip"),
        ("macos", "aarch64") => Some("midstate-macos-arm64.tar.gz"),
        ("macos", "x86_64") => Some("midstate-macos-amd64.tar.gz"),
        _ => None,
    }
}

/// The programs an update replaces: the app first, then the miner beside it.
pub fn binaries(os: &str) -> [String; 2] {
    let ext = if os == "windows" { ".exe" } else { "" };
    [format!("midstate-desktop{ext}"), format!("midstate{ext}")]
}

// ── The signed manifest ─────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct Manifest {
    pub version: String,
    pub files: BTreeMap<String, [u8; 32]>,
}

fn hex32(s: &str) -> Option<[u8; 32]> {
    let v = hex::decode(s).ok()?;
    v.try_into().ok()
}

/// `midstate-release 1`, then `version <tag>` and a `sha256 <hex> <file>`
/// line per file. Lines of any other kind are skipped, so later versions of
/// the format can add some.
pub fn parse_manifest(text: &str) -> Result<Manifest> {
    let mut lines = text.lines();
    ensure!(lines.next().map(str::trim) == Some("midstate-release 1"), "not a midstate release manifest");
    let mut version = None;
    let mut files = BTreeMap::new();
    for line in lines {
        let mut w = line.split_whitespace();
        match w.next() {
            Some("version") => {
                let v = w.next().ok_or_else(|| anyhow!("a version line with no version"))?;
                ensure!(version.is_none(), "two versions in one manifest");
                ensure!(parse_version(v).is_some(), "not a version: {v}");
                version = Some(v.to_string());
            }
            Some("sha256") => {
                let (Some(h), Some(name), None) = (w.next(), w.next(), w.next()) else { bail!("a malformed sha256 line") };
                let h = hex32(h).ok_or_else(|| anyhow!("a malformed hash for {name}"))?;
                ensure!(files.insert(name.to_string(), h).is_none(), "{name} is listed twice");
            }
            _ => {}
        }
    }
    Ok(Manifest { version: version.ok_or_else(|| anyhow!("the manifest names no version"))?, files })
}

/// A signature as uploaded: the 64 raw bytes, or them in hex.
fn signature_bytes(sig: &[u8]) -> Option<[u8; 64]> {
    if sig.len() == 64 {
        return sig.try_into().ok();
    }
    let t = std::str::from_utf8(sig).ok()?.trim();
    hex::decode(t).ok()?.try_into().ok()
}

/// The manifest, if one of `keys` signed exactly these bytes.
pub fn verify(manifest: &[u8], sig: &[u8], keys: &[&str]) -> Result<Manifest> {
    use ed25519_dalek::{Signature, VerifyingKey};
    let sig = Signature::from_bytes(&signature_bytes(sig).ok_or_else(|| anyhow!("the signature is malformed"))?);
    let signed = keys.iter().any(|k| {
        hex32(k)
            .and_then(|k| VerifyingKey::from_bytes(&k).ok())
            .is_some_and(|k| k.verify_strict(manifest, &sig).is_ok())
    });
    ensure!(signed, "the release is not signed by a Midstate release key");
    parse_manifest(std::str::from_utf8(manifest).context("the manifest is not text")?)
}

// ── Unpacking ───────────────────────────────────────────────────────────────

/// Which wanted program an archive entry is, if it is one: a plain relative
/// path, at most one folder deep, with nothing like `..` in it.
fn wanted<'a>(path: &Path, names: &'a [String]) -> Option<&'a String> {
    use std::path::Component;
    let parts: Vec<_> = path.components().collect();
    if parts.is_empty() || parts.len() > 2 || !parts.iter().all(|c| matches!(c, Component::Normal(_))) {
        return None;
    }
    let file = path.file_name()?.to_str()?;
    names.iter().find(|n| n.as_str() == file)
}

fn copy_capped(mut r: impl Read, to: &Path) -> Result<()> {
    let mut f = std::fs::File::create(to).with_context(|| format!("could not write {}", to.display()))?;
    let n = std::io::copy(&mut (&mut r).take(MAX_BINARY + 1), &mut f)?;
    ensure!(n <= MAX_BINARY, "a program in the archive is too large");
    f.sync_all()?;
    Ok(())
}

/// Unpack `names` from a .tar.gz or .zip into `out`. Everything else in the
/// archive is skipped; each name must be there exactly once.
pub fn extract(archive: &Path, names: &[String], out: &Path) -> Result<()> {
    let mut found: Vec<&String> = Vec::new();
    let name = archive.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    if name.ends_with(".zip") {
        let mut z = zip::ZipArchive::new(std::fs::File::open(archive)?).context("not a zip archive")?;
        for i in 0..z.len() {
            let e = z.by_index(i)?;
            if !e.is_file() {
                continue;
            }
            let Some(p) = e.enclosed_name() else { continue };
            if let Some(n) = wanted(&p, names) {
                ensure!(!found.contains(&n), "{n} is in the archive twice");
                copy_capped(e, &out.join(n))?;
                found.push(n);
            }
        }
    } else {
        let gz = flate2::read::GzDecoder::new(std::fs::File::open(archive)?);
        let mut t = tar::Archive::new(gz);
        for e in t.entries().context("not a tar.gz archive")? {
            let e = e?;
            if e.header().entry_type() != tar::EntryType::Regular {
                continue;
            }
            let p = e.path()?.into_owned();
            if let Some(n) = wanted(&p, names) {
                ensure!(!found.contains(&n), "{n} is in the archive twice");
                copy_capped(e, &out.join(n))?;
                found.push(n);
            }
        }
    }
    for n in names {
        ensure!(found.contains(&n), "the archive has no {n}");
    }
    Ok(())
}

fn sha256_file(p: &Path) -> Result<[u8; 32]> {
    let mut f = std::fs::File::open(p)?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h)?;
    Ok(h.finalize().into())
}

// ── Staging and putting in place ────────────────────────────────────────────

/// An update unpacked and waiting for the next start.
#[derive(Clone, Debug, PartialEq)]
pub struct Staged {
    pub version: String,
    pub dir: PathBuf,
    pub files: Vec<(String, [u8; 32])>,
}

pub fn updates_dir(base: &Path) -> PathBuf {
    base.join("updates")
}

/// The marker is written last: a folder without one was never finished.
const READY: &str = "ready";

fn write_marker(s: &Staged) -> Result<()> {
    let mut text = format!("version {}\n", s.version);
    for (n, h) in &s.files {
        text.push_str(&format!("sha256 {} {n}\n", hex::encode(h)));
    }
    let tmp = s.dir.join("ready.tmp");
    std::fs::write(&tmp, text)?;
    std::fs::rename(tmp, s.dir.join(READY))?;
    Ok(())
}

/// The update waiting in `base`, if there is a finished one.
pub fn read_staged(base: &Path) -> Option<Staged> {
    let root = updates_dir(base);
    for e in std::fs::read_dir(&root).ok()?.flatten() {
        let dir = e.path();
        let Ok(text) = std::fs::read_to_string(dir.join(READY)) else { continue };
        let mut version = None;
        let mut files = Vec::new();
        for line in text.lines() {
            let w: Vec<&str> = line.split_whitespace().collect();
            match w.as_slice() {
                ["version", v] => version = Some(v.to_string()),
                ["sha256", h, n] => files.push((n.to_string(), hex32(h)?)),
                _ => {}
            }
        }
        if let Some(version) = version {
            return Some(Staged { version, dir, files });
        }
    }
    None
}

/// Forget every update waiting in `base`.
pub fn discard(base: &Path) {
    let _ = std::fs::remove_dir_all(updates_dir(base).join("staging"));
    if let Ok(rd) = std::fs::read_dir(updates_dir(base)) {
        for e in rd.flatten() {
            if e.path().is_dir() {
                let _ = std::fs::remove_dir_all(e.path());
            }
        }
    }
}

/// Whether this user can put files in `dir`.
pub fn writable(dir: &Path) -> bool {
    let probe = dir.join(format!(".midstate-write-test-{}", std::process::id()));
    let ok = std::fs::File::create(&probe).is_ok();
    let _ = std::fs::remove_file(&probe);
    ok
}

fn old_name(p: &Path) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(".old");
    PathBuf::from(s)
}

/// Remove what an earlier update moved aside. On Windows a running program
/// cannot be deleted, only renamed, so this waits for the next start.
pub fn cleanup_old(install_dir: &Path) {
    for n in binaries(std::env::consts::OS) {
        let _ = std::fs::remove_file(old_name(&install_dir.join(n)));
    }
}

/// Put the waiting update in place beside the running app, if it is newer
/// than `current`. Returns the version put in place. Nothing is touched
/// unless every staged file still hashes as it did when it was checked, and
/// a failure part-way puts the old programs back.
pub fn apply(base: &Path, install_dir: &Path, current: Version) -> Result<Option<String>> {
    let Some(s) = read_staged(base) else { return Ok(None) };
    if parse_version(&s.version).is_none_or(|v| v <= current) {
        discard(base);
        return Ok(None);
    }
    for (n, h) in &s.files {
        if sha256_file(&s.dir.join(n)).ok().as_ref() != Some(h) {
            discard(base);
            bail!("the downloaded update was changed after it was checked; it will be fetched again");
        }
    }
    ensure!(writable(install_dir), "cannot write to {}", install_dir.display());

    // The miner may not be there (a lone copy of the app): only what is there
    // is replaced. The app always is.
    let todo: Vec<&(String, [u8; 32])> = s
        .files
        .iter()
        .enumerate()
        .filter(|(i, (n, _))| *i == 0 || install_dir.join(n).exists())
        .map(|(_, f)| f)
        .collect();
    // First copy every new file next to its target, so the swap is renames.
    let mut news = Vec::new();
    for (n, _) in &todo {
        let new = install_dir.join(format!(".{n}.new"));
        std::fs::copy(s.dir.join(n), &new).with_context(|| format!("could not copy {n} into place"))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&new, std::fs::Permissions::from_mode(0o755))?;
        }
        news.push((new, install_dir.join(n)));
    }
    let mut done: Vec<(PathBuf, bool)> = Vec::new();
    let swap = |new: &Path, target: &Path, done: &mut Vec<(PathBuf, bool)>| -> Result<()> {
        let had = target.exists();
        if had {
            std::fs::rename(target, old_name(target))?;
        }
        done.push((target.to_path_buf(), had));
        std::fs::rename(new, target)?;
        Ok(())
    };
    for (new, target) in &news {
        if let Err(e) = swap(new, target, &mut done) {
            for (target, had) in done.iter().rev() {
                if *had {
                    let _ = std::fs::rename(old_name(target), target);
                } else {
                    let _ = std::fs::remove_file(target);
                }
            }
            for (new, _) in &news {
                let _ = std::fs::remove_file(new);
            }
            return Err(e).context("could not put the update in place; the old version is back");
        }
    }
    if !cfg!(windows) {
        cleanup_old(install_dir);
    }
    discard(base);
    Ok(Some(s.version))
}

// ── Talking to GitHub ───────────────────────────────────────────────────────

#[derive(Clone, Debug, serde::Deserialize)]
pub struct Release {
    pub tag_name: String,
    #[serde(default)]
    pub html_url: String,
    #[serde(default)]
    pub assets: Vec<Asset>,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct Asset {
    pub name: String,
    pub browser_download_url: String,
    #[serde(default)]
    pub size: u64,
}

impl Release {
    fn asset(&self, name: &str) -> Option<&Asset> {
        self.assets.iter().find(|a| a.name == name)
    }
}

pub fn client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .user_agent(format!("midstate-desktop/{}", this_version()))
        .connect_timeout(Duration::from_secs(20))
        .timeout(Duration::from_secs(30 * 60))
        .build()?)
}

async fn latest(client: &reqwest::Client, api: &str) -> Result<Release> {
    let url = format!("{api}/repos/{REPO}/releases/latest");
    let r = client
        .get(url)
        .header("Accept", "application/vnd.github+json")
        .timeout(Duration::from_secs(30))
        .send()
        .await?
        .error_for_status()?;
    Ok(r.json().await?)
}

async fn fetch(client: &reqwest::Client, url: &str, max: u64, dest: Option<&Path>, mut progress: impl FnMut(u64, u64)) -> Result<(Vec<u8>, [u8; 32])> {
    let mut r = client.get(url).send().await?.error_for_status()?;
    let total = r.content_length().unwrap_or(0);
    ensure!(total <= max, "{url} is larger than expected");
    let mut file = match dest {
        Some(p) => Some(std::fs::File::create(p)?),
        None => None,
    };
    let mut kept = Vec::new();
    let mut h = Sha256::new();
    let mut done = 0u64;
    while let Some(chunk) = r.chunk().await? {
        done += chunk.len() as u64;
        ensure!(done <= max, "{url} is larger than expected");
        h.update(&chunk);
        match file.as_mut() {
            Some(f) => f.write_all(&chunk)?,
            None => kept.extend_from_slice(&chunk),
        }
        progress(done, total.max(done));
    }
    if let Some(f) = file {
        f.sync_all()?;
    }
    Ok((kept, h.finalize().into()))
}

// ── What the app shows ──────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    /// This build does not update itself, and why.
    Off(&'static str),
    Waiting,
    Checking,
    UpToDate,
    /// A newer version, not downloaded: updates are set to ask first.
    Available { version: String },
    Downloading { version: String, done: u64, total: u64 },
    /// Downloaded and checked: in place at the next start.
    Ready { version: String },
    /// A newer version this copy cannot install itself.
    Manual { version: String, url: String },
    Failed(String),
}

/// The updater as the app holds it.
pub struct Updater {
    pub status: Arc<Mutex<Status>>,
    wake: Arc<tokio::sync::Notify>,
    install: Arc<AtomicBool>,
    auto: Arc<AtomicBool>,
    base: PathBuf,
}

impl Updater {
    pub fn status(&self) -> Status {
        self.status.lock().map(|s| s.clone()).unwrap_or(Status::Waiting)
    }
    /// Look now (and, with `install`, download whatever is found).
    pub fn check_now(&self, install: bool) {
        if install {
            self.install.store(true, Ordering::SeqCst);
        }
        self.wake.notify_one();
    }
    pub fn auto(&self) -> bool {
        self.auto.load(Ordering::SeqCst)
    }
    pub fn set_auto(&self, on: bool) {
        self.auto.store(on, Ordering::SeqCst);
        set_auto(&self.base, on);
        if on {
            self.wake.notify_one();
        }
    }
}

/// Whether updates download by themselves (the default).
pub fn auto_on(base: &Path) -> bool {
    std::fs::read_to_string(updates_dir(base).join("auto")).map(|s| s.trim() != "off").unwrap_or(true)
}

pub fn set_auto(base: &Path, on: bool) {
    let d = updates_dir(base);
    let _ = std::fs::create_dir_all(&d);
    let _ = std::fs::write(d.join("auto"), if on { "on" } else { "off" });
}

/// Why this build cannot update itself, if it cannot.
pub fn off_reason(this: Option<&str>, keys: &[&str], os: &str, arch: &str) -> Option<&'static str> {
    if this.and_then(parse_version).is_none() {
        Some("This copy was built from source. Update it with git.")
    } else if keys.is_empty() {
        Some("Updates are not set up in this version.")
    } else if platform_asset(os, arch).is_none() {
        Some("There are no ready-made updates for this computer.")
    } else {
        None
    }
}

/// One look: what the latest release is, and — with `download` — fetch,
/// check and unpack it. `install_dir` is where the running app is.
pub async fn run_once(
    client: &reqwest::Client,
    api: &str,
    base: &Path,
    install_dir: &Path,
    current: Version,
    keys: &[&str],
    download: bool,
    set: &(dyn Fn(Status) + Send + Sync),
) -> Result<Status> {
    if let Some(s) = read_staged(base) {
        if parse_version(&s.version).is_some_and(|v| v > current) {
            return Ok(Status::Ready { version: s.version });
        }
    }
    set(Status::Checking);
    let rel = latest(client, api).await.context("could not reach GitHub")?;
    let Some(v) = parse_version(&rel.tag_name) else { return Ok(Status::UpToDate) };
    if v <= current {
        return Ok(Status::UpToDate);
    }
    let shown = rel.tag_name.trim_start_matches('v').to_string();
    let os = std::env::consts::OS;
    let Some(archive) = platform_asset(os, std::env::consts::ARCH) else {
        return Ok(Status::Manual { version: shown, url: rel.html_url });
    };
    // Until a release is signed it is not offered at all.
    let (Some(m), Some(sig), Some(a)) = (rel.asset(MANIFEST), rel.asset(SIGNATURE), rel.asset(archive)) else {
        tracing::info!("release {} is not signed (or has no {archive}) yet", rel.tag_name);
        return Ok(Status::UpToDate);
    };
    let (manifest, _) = fetch(client, &m.browser_download_url, MAX_SMALL, None, |_, _| {}).await?;
    let (sig, _) = fetch(client, &sig.browser_download_url, MAX_SMALL, None, |_, _| {}).await?;
    let manifest = verify(&manifest, &sig, keys)?;
    ensure!(manifest.version == rel.tag_name, "the release says {} but its signed manifest says {}", rel.tag_name, manifest.version);
    let want = *manifest.files.get(archive).ok_or_else(|| anyhow!("the signed manifest does not list {archive}"))?;
    if !writable(install_dir) {
        return Ok(Status::Manual { version: shown, url: rel.html_url });
    }
    if !download {
        return Ok(Status::Available { version: shown });
    }
    ensure!(a.size <= MAX_ARCHIVE, "{archive} is larger than expected");

    let staging = updates_dir(base).join("staging");
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging)?;
    let file = staging.join(archive);
    // Progress once per percent, not once per chunk.
    let mut last = u64::MAX;
    let (_, got) = fetch(client, &a.browser_download_url, MAX_ARCHIVE, Some(&file), |done, total| {
        let pct = done * 100 / total.max(1);
        if pct != last {
            last = pct;
            set(Status::Downloading { version: shown.clone(), done, total });
        }
    })
    .await?;
    ensure!(got == want, "{archive} does not match its signed hash");

    let names = binaries(os);
    let out = updates_dir(base).join(&rel.tag_name);
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out)?;
    let (file2, out2, names2) = (file.clone(), out.clone(), names.clone());
    tokio::task::spawn_blocking(move || extract(&file2, &names2, &out2)).await??;
    let mut files = Vec::new();
    for n in &names {
        files.push((n.clone(), sha256_file(&out.join(n))?));
    }
    write_marker(&Staged { version: rel.tag_name.clone(), dir: out, files })?;
    let _ = std::fs::remove_dir_all(&staging);
    Ok(Status::Ready { version: rel.tag_name })
}

/// Start looking for updates in the background.
pub fn spawn(rt: &tokio::runtime::Runtime, ctx: eframe::egui::Context, base: PathBuf) -> Updater {
    let status = Arc::new(Mutex::new(Status::Waiting));
    let wake = Arc::new(tokio::sync::Notify::new());
    let install = Arc::new(AtomicBool::new(false));
    let auto = Arc::new(AtomicBool::new(auto_on(&base)));
    let u = Updater { status: status.clone(), wake: wake.clone(), install: install.clone(), auto: auto.clone(), base: base.clone() };
    if let Some(why) = off_reason(THIS_RELEASE, RELEASE_KEYS, std::env::consts::OS, std::env::consts::ARCH) {
        *status.lock().unwrap() = Status::Off(why);
        return u;
    }
    let current = THIS_RELEASE.and_then(parse_version).unwrap_or_default();
    let install_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf));
    rt.spawn(async move {
        let set = {
            let (status, ctx) = (status.clone(), ctx.clone());
            move |s: Status| {
                if let Ok(mut g) = status.lock() {
                    *g = s;
                }
                ctx.request_repaint();
            }
        };
        let Some(install_dir) = install_dir else {
            set(Status::Off("Cannot tell where this app is installed."));
            return;
        };
        let client = match client() {
            Ok(c) => c,
            Err(e) => {
                set(Status::Failed(format!("{e:#}")));
                return;
            }
        };
        tokio::select! {
            _ = tokio::time::sleep(FIRST_LOOK) => {}
            _ = wake.notified() => {}
        }
        loop {
            let download = auto.load(Ordering::SeqCst) || install.swap(false, Ordering::SeqCst);
            let r = run_once(&client, API, &base, &install_dir, current, RELEASE_KEYS, download, &set).await;
            set(match r {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("update check: {e:#}");
                    Status::Failed(format!("{e:#}"))
                }
            });
            tokio::select! {
                _ = tokio::time::sleep(EVERY) => {}
                _ = wake.notified() => {}
            }
        }
    });
    u
}

// ── Restarting into the new version ─────────────────────────────────────────

static RESTART: AtomicBool = AtomicBool::new(false);

/// Ask for the app to start again once its window has closed.
pub fn request_restart() {
    RESTART.store(true, Ordering::SeqCst);
}

pub fn restart_requested() -> bool {
    RESTART.load(Ordering::SeqCst)
}

/// At start, before anything else runs: put a waiting update in place.
/// Returns true when the caller should start the new version and exit.
pub fn apply_at_start(base: &Path, exe: &Path) -> bool {
    let Some(dir) = exe.parent() else { return false };
    cleanup_old(dir);
    let Some(current) = THIS_RELEASE.and_then(parse_version) else { return false };
    match apply(base, dir, current) {
        Ok(Some(v)) => {
            tracing::info!("updated to {v}");
            true
        }
        Ok(None) => false,
        Err(e) => {
            tracing::warn!("could not update: {e:#}");
            false
        }
    }
}

/// Start `exe` again with this run's arguments, telling it it was restarted.
pub fn relaunch(exe: &Path) {
    let r = std::process::Command::new(exe).args(std::env::args_os().skip(1)).env("MIDSTATE_RELAUNCHED", "1").spawn();
    if let Err(e) = r {
        tracing::error!("could not start {} again: {e}", exe.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Made with openssl, the way tools/release/sign.sh signs:
    //   openssl genpkey -algorithm ed25519 -out test-key.pem
    //   openssl pkeyutl -sign -inkey test-key.pem -rawin -in m.txt -out m.sig
    const OPENSSL_PUB: &str = "1f8799c9bff53a68747e29769c982f254263c5e95c211f16ed7f89aad52e46e0";
    const OPENSSL_SEED: &str = "f7b7a512b2ef4e947b1ebe6737edcc325f8b5ffc01b37202e9a033f4fa72a6c5";
    const OPENSSL_SIG: &str = "9f0ed5d0900a2d8500e87803ba39be0e1953af9b662629335a21eed21fa5b1b6d6b12e8107ba138c5b6cdc4139f1764c23b19879fb946ade7c6c9f4419789a09";
    const OPENSSL_MANIFEST: &str = "midstate-release 1\nversion v6.6.0\nsha256 2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881 midstate-linux-amd64.tar.gz\n";

    fn sign(seed: &str, msg: &[u8]) -> Vec<u8> {
        use ed25519_dalek::Signer;
        let k = ed25519_dalek::SigningKey::from_bytes(&hex32(seed).unwrap());
        k.sign(msg).to_bytes().to_vec()
    }

    #[test]
    fn versions() {
        assert_eq!(parse_version("v6.5.2"), Some((6, 5, 2)));
        assert_eq!(parse_version("6.10.0"), Some((6, 10, 0)));
        assert!(parse_version("v6.5.10") > parse_version("v6.5.9"), "numbers, not text");
        for bad in ["", "v6.5", "v6.5.2.1", "v6.5.2-rc1", "v6..2", "vx.y.z", "v 6.5.2"] {
            assert_eq!(parse_version(bad), None, "{bad}");
        }
    }

    #[test]
    fn each_computer_gets_its_archive() {
        assert_eq!(platform_asset("linux", "x86_64"), Some("midstate-linux-amd64.tar.gz"));
        assert_eq!(platform_asset("windows", "x86_64"), Some("midstate-windows-amd64.zip"));
        assert_eq!(platform_asset("macos", "aarch64"), Some("midstate-macos-arm64.tar.gz"));
        assert_eq!(platform_asset("macos", "x86_64"), Some("midstate-macos-amd64.tar.gz"));
        // A Raspberry Pi runs the node only: there is no app to update there.
        assert_eq!(platform_asset("linux", "aarch64"), None);
        assert_eq!(binaries("windows"), ["midstate-desktop.exe".to_string(), "midstate.exe".into()]);
        assert_eq!(binaries("linux"), ["midstate-desktop".to_string(), "midstate".into()]);
    }

    #[test]
    fn only_release_builds_with_a_key_update() {
        assert!(off_reason(None, &[OPENSSL_PUB], "linux", "x86_64").is_some(), "from source");
        assert!(off_reason(Some("v6.6.0"), &[], "linux", "x86_64").is_some(), "no key");
        assert!(off_reason(Some("v6.6.0"), &[OPENSSL_PUB], "linux", "aarch64").is_some());
        assert_eq!(off_reason(Some("v6.6.0"), &[OPENSSL_PUB], "linux", "x86_64"), None);
    }

    #[test]
    fn an_openssl_signature_is_accepted() {
        // The seed really is that key's, so the other tests sign as openssl does.
        let k = ed25519_dalek::SigningKey::from_bytes(&hex32(OPENSSL_SEED).unwrap());
        assert_eq!(hex::encode(k.verifying_key().to_bytes()), OPENSSL_PUB);
        assert_eq!(hex::encode(sign(OPENSSL_SEED, OPENSSL_MANIFEST.as_bytes())), OPENSSL_SIG, "Ed25519 is deterministic");

        let raw = hex::decode(OPENSSL_SIG).unwrap();
        let m = verify(OPENSSL_MANIFEST.as_bytes(), &raw, &[OPENSSL_PUB]).unwrap();
        assert_eq!(m.version, "v6.6.0");
        assert_eq!(hex::encode(m.files["midstate-linux-amd64.tar.gz"]), "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881");
        // Hex in a text file works too, with a trailing newline.
        let as_hex = format!("{OPENSSL_SIG}\n");
        assert!(verify(OPENSSL_MANIFEST.as_bytes(), as_hex.as_bytes(), &[OPENSSL_PUB]).is_ok());
        // Any one of several keys.
        let other = "0".repeat(64);
        assert!(verify(OPENSSL_MANIFEST.as_bytes(), &raw, &[&other, OPENSSL_PUB]).is_ok());
    }

    #[test]
    fn anything_else_is_refused() {
        let raw = hex::decode(OPENSSL_SIG).unwrap();
        // One changed byte in what was signed.
        let changed = OPENSSL_MANIFEST.replace("2d71", "2d72");
        assert!(verify(changed.as_bytes(), &raw, &[OPENSSL_PUB]).is_err());
        // Someone else's key.
        let theirs = "8f".repeat(32);
        let sig = sign(&theirs, OPENSSL_MANIFEST.as_bytes());
        assert!(verify(OPENSSL_MANIFEST.as_bytes(), &sig, &[OPENSSL_PUB]).is_err());
        // No keys at all.
        assert!(verify(OPENSSL_MANIFEST.as_bytes(), &raw, &[]).is_err());
        // A bad signature file.
        assert!(verify(OPENSSL_MANIFEST.as_bytes(), b"not a signature", &[OPENSSL_PUB]).is_err());
        assert!(verify(OPENSSL_MANIFEST.as_bytes(), &raw[..63], &[OPENSSL_PUB]).is_err());
    }

    #[test]
    fn manifests() {
        let h = "ab".repeat(32);
        let m = parse_manifest(&format!("midstate-release 1\nversion v1.2.3\nsha256 {h} a.zip\nnote anything\n\nsha256 {h} b.tar.gz\n")).unwrap();
        assert_eq!(m.version, "v1.2.3");
        assert_eq!(m.files.len(), 2, "an unknown line is skipped");
        for bad in [
            format!("version v1.2.3\nsha256 {h} a.zip\n"),
            format!("midstate-release 1\nsha256 {h} a.zip\n"),
            format!("midstate-release 1\nversion v1.2.3\nversion v1.2.4\n"),
            format!("midstate-release 1\nversion v1.2\n"),
            "midstate-release 1\nversion v1.2.3\nsha256 abc a.zip\n".to_string(),
            format!("midstate-release 1\nversion v1.2.3\nsha256 {h} a.zip\nsha256 {h} a.zip\n"),
            format!("midstate-release 1\nversion v1.2.3\nsha256 {h} a b\n"),
            format!("midstate-release 2\nversion v1.2.3\n"),
        ] {
            assert!(parse_manifest(&bad).is_err(), "{bad:?}");
        }
    }

    fn tar_gz(path: &Path, entries: &[(&str, &[u8])]) {
        let f = std::fs::File::create(path).unwrap();
        let gz = flate2::write::GzEncoder::new(f, flate2::Compression::fast());
        let mut b = tar::Builder::new(gz);
        for (name, data) in entries {
            let mut h = tar::Header::new_gnu();
            h.set_size(data.len() as u64);
            h.set_mode(0o755);
            h.set_entry_type(tar::EntryType::Regular);
            // Written raw, so a hostile name like ../x goes in as is.
            let raw = h.as_old_mut().name.as_mut();
            raw[..name.len()].copy_from_slice(name.as_bytes());
            h.set_cksum();
            b.append(&h, *data).unwrap();
        }
        b.into_inner().unwrap().finish().unwrap();
    }

    #[test]
    fn unpacking_takes_only_the_two_programs() {
        let d = tempfile::tempdir().unwrap();
        let names = binaries("linux");
        let a = d.path().join("midstate-linux-amd64.tar.gz");
        tar_gz(
            &a,
            &[
                ("midstate-linux-amd64/README", b"hello"),
                ("midstate-linux-amd64/midstate-desktop", b"APP"),
                ("midstate-linux-amd64/midstate", b"MINER"),
            ],
        );
        let out = d.path().join("out");
        std::fs::create_dir(&out).unwrap();
        extract(&a, &names, &out).unwrap();
        assert_eq!(std::fs::read(out.join("midstate-desktop")).unwrap(), b"APP");
        assert_eq!(std::fs::read(out.join("midstate")).unwrap(), b"MINER");
        assert!(!out.join("README").exists());

        // ../ in a name: skipped, so the archive is missing a program.
        let evil = d.path().join("evil.tar.gz");
        tar_gz(&evil, &[("../midstate-desktop", b"EVIL"), ("x/midstate", b"MINER")]);
        let out2 = d.path().join("out2");
        std::fs::create_dir(&out2).unwrap();
        assert!(extract(&evil, &names, &out2).is_err());
        assert!(!d.path().join("midstate-desktop").exists(), "nothing written outside");
        // Too deep, or listed twice.
        let deep = d.path().join("deep.tar.gz");
        tar_gz(&deep, &[("a/b/midstate-desktop", b"APP"), ("midstate", b"M")]);
        assert!(extract(&deep, &names, &out2).is_err());
        let twice = d.path().join("twice.tar.gz");
        tar_gz(&twice, &[("midstate-desktop", b"A"), ("x/midstate-desktop", b"B"), ("midstate", b"M")]);
        assert!(extract(&twice, &names, &out2).is_err());
    }

    #[test]
    fn unpacking_a_zip() {
        let d = tempfile::tempdir().unwrap();
        let a = d.path().join("midstate-windows-amd64.zip");
        {
            let mut z = zip::ZipWriter::new(std::fs::File::create(&a).unwrap());
            let o = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
            z.start_file("midstate-desktop.exe", o).unwrap();
            z.write_all(&b"APP".repeat(1000)).unwrap();
            z.start_file("midstate.exe", o).unwrap();
            z.write_all(b"MINER").unwrap();
            z.start_file("../evil.exe", o).unwrap();
            z.write_all(b"EVIL").unwrap();
            z.finish().unwrap();
        }
        let out = d.path().join("out");
        std::fs::create_dir(&out).unwrap();
        extract(&a, &binaries("windows"), &out).unwrap();
        assert_eq!(std::fs::read(out.join("midstate-desktop.exe")).unwrap(), b"APP".repeat(1000));
        assert_eq!(std::fs::read(out.join("midstate.exe")).unwrap(), b"MINER");
        assert!(!d.path().join("evil.exe").exists());
    }

    fn stage(base: &Path, version: &str, files: &[(&str, &[u8])]) {
        let dir = updates_dir(base).join(version);
        std::fs::create_dir_all(&dir).unwrap();
        let mut fs = Vec::new();
        for (n, data) in files {
            std::fs::write(dir.join(n), data).unwrap();
            fs.push((n.to_string(), Sha256::digest(data).into()));
        }
        write_marker(&Staged { version: version.into(), dir, files: fs }).unwrap();
    }

    #[test]
    fn a_staged_update_goes_in_beside_the_app() {
        let d = tempfile::tempdir().unwrap();
        let (base, inst) = (d.path().join("base"), d.path().join("inst"));
        std::fs::create_dir_all(&inst).unwrap();
        std::fs::write(inst.join("midstate-desktop"), b"old app").unwrap();
        std::fs::write(inst.join("midstate"), b"old miner").unwrap();
        stage(&base, "v6.6.0", &[("midstate-desktop", b"new app"), ("midstate", b"new miner")]);
        assert_eq!(read_staged(&base).unwrap().version, "v6.6.0");

        assert_eq!(apply(&base, &inst, (6, 5, 2)).unwrap().as_deref(), Some("v6.6.0"));
        assert_eq!(std::fs::read(inst.join("midstate-desktop")).unwrap(), b"new app");
        assert_eq!(std::fs::read(inst.join("midstate")).unwrap(), b"new miner");
        assert!(!old_name(&inst.join("midstate")).exists(), "the old ones are gone");
        assert!(read_staged(&base).is_none(), "and the update is used up");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(inst.join("midstate-desktop")).unwrap().permissions().mode();
            assert_eq!(mode & 0o111, 0o111, "it can run");
        }
        // Nothing waiting: nothing happens.
        assert_eq!(apply(&base, &inst, (6, 6, 0)).unwrap(), None);
    }

    #[test]
    fn a_lone_app_gets_no_miner() {
        let d = tempfile::tempdir().unwrap();
        let (base, inst) = (d.path().join("base"), d.path().join("inst"));
        std::fs::create_dir_all(&inst).unwrap();
        std::fs::write(inst.join("midstate-desktop"), b"old app").unwrap();
        stage(&base, "v6.6.0", &[("midstate-desktop", b"new app"), ("midstate", b"new miner")]);
        apply(&base, &inst, (6, 5, 2)).unwrap();
        assert_eq!(std::fs::read(inst.join("midstate-desktop")).unwrap(), b"new app");
        assert!(!inst.join("midstate").exists());
    }

    #[test]
    fn old_or_tampered_updates_are_thrown_away() {
        let d = tempfile::tempdir().unwrap();
        let (base, inst) = (d.path().join("base"), d.path().join("inst"));
        std::fs::create_dir_all(&inst).unwrap();
        std::fs::write(inst.join("midstate-desktop"), b"app").unwrap();
        // Not newer than what runs.
        stage(&base, "v6.5.2", &[("midstate-desktop", b"x"), ("midstate", b"y")]);
        assert_eq!(apply(&base, &inst, (6, 5, 2)).unwrap(), None);
        assert!(read_staged(&base).is_none());
        assert_eq!(std::fs::read(inst.join("midstate-desktop")).unwrap(), b"app");
        // Changed after it was checked.
        stage(&base, "v6.6.0", &[("midstate-desktop", b"x"), ("midstate", b"y")]);
        std::fs::write(updates_dir(&base).join("v6.6.0").join("midstate-desktop"), b"swapped").unwrap();
        assert!(apply(&base, &inst, (6, 5, 2)).is_err());
        assert!(read_staged(&base).is_none());
        assert_eq!(std::fs::read(inst.join("midstate-desktop")).unwrap(), b"app", "untouched");
        // Unfinished (no marker): not an update.
        std::fs::create_dir_all(updates_dir(&base).join("v6.7.0")).unwrap();
        assert!(read_staged(&base).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn a_failed_swap_puts_the_old_version_back() {
        let d = tempfile::tempdir().unwrap();
        let (base, inst) = (d.path().join("base"), d.path().join("inst"));
        std::fs::create_dir_all(&inst).unwrap();
        std::fs::write(inst.join("midstate-desktop"), b"old app").unwrap();
        // A folder where the miner should be: the app's swap succeeds, the
        // miner's cannot (a file cannot replace a non-empty folder).
        std::fs::create_dir_all(inst.join("midstate").join("busy")).unwrap();
        std::fs::create_dir_all(old_name(&inst.join("midstate")).join("busy")).unwrap();
        stage(&base, "v6.6.0", &[("midstate-desktop", b"new app"), ("midstate", b"new miner")]);
        assert!(apply(&base, &inst, (6, 5, 2)).is_err());
        assert_eq!(std::fs::read(inst.join("midstate-desktop")).unwrap(), b"old app", "rolled back");
        assert!(!inst.join(".midstate-desktop.new").exists());
        assert!(!inst.join(".midstate.new").exists());
    }

    fn release_json(url: &str, tag: &str, assets: &[(&str, u64)]) -> Vec<u8> {
        let a: Vec<String> = assets
            .iter()
            .map(|(n, size)| format!(r#"{{"name":"{n}","browser_download_url":"{url}/dl/{n}","size":{size}}}"#))
            .collect();
        format!(r#"{{"tag_name":"{tag}","html_url":"{url}/release","assets":[{}]}}"#, a.join(",")).into_bytes()
    }

    #[tokio::test]
    async fn a_signed_release_is_fetched_checked_and_staged() {
        let d = tempfile::tempdir().unwrap();
        let (base, inst) = (d.path().join("base"), d.path().join("inst"));
        std::fs::create_dir_all(&inst).unwrap();
        let archive = platform_asset(std::env::consts::OS, std::env::consts::ARCH).unwrap_or("midstate-linux-amd64.tar.gz");
        // This test builds a tar.gz; it runs where the archive is one.
        if archive.ends_with(".zip") {
            return;
        }
        let names = binaries(std::env::consts::OS);
        let a = d.path().join(archive);
        tar_gz(&a, &[(&format!("pkg/{}", names[0]), b"NEW APP"), (&format!("pkg/{}", names[1]), b"NEW MINER")]);
        let bytes = std::fs::read(&a).unwrap();
        let hash = hex::encode(Sha256::digest(&bytes));
        let manifest = format!("midstate-release 1\nversion v6.6.0\nsha256 {hash} {archive}\n");
        let sig = sign(OPENSSL_SEED, manifest.as_bytes());

        let routes = |url: &str, tag: &str, manifest: &str, sig: &[u8]| {
            vec![
                (format!("/repos/{REPO}/releases/latest"), release_json(url, tag, &[(MANIFEST, 100), (SIGNATURE, 64), (archive, bytes.len() as u64)])),
                (format!("/dl/{MANIFEST}"), manifest.as_bytes().to_vec()),
                (format!("/dl/{SIGNATURE}"), sig.to_vec()),
                (format!("/dl/{archive}"), bytes.clone()),
            ]
        };
        // A stand-in for GitHub per case, whose links point back at itself.
        async fn served(make: impl Fn(&str) -> Vec<(String, Vec<u8>)>) -> String {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", l.local_addr().unwrap());
            let routes = make(&url);
            tokio::spawn(async move {
                loop {
                    let Ok((mut s, _)) = l.accept().await else { return };
                    let routes = routes.clone();
                    tokio::spawn(async move {
                        let mut buf = vec![0u8; 8192];
                        let mut got = Vec::new();
                        while !got.windows(4).any(|w| w == b"\r\n\r\n") {
                            let Ok(n) = s.read(&mut buf).await else { return };
                            if n == 0 {
                                return;
                            }
                            got.extend_from_slice(&buf[..n]);
                        }
                        let req = String::from_utf8_lossy(&got);
                        let path = req.split_whitespace().nth(1).unwrap_or("").to_string();
                        let (status, body) = match routes.iter().find(|(p, _)| *p == path) {
                            Some((_, b)) => ("200 OK", b.clone()),
                            None => ("404 Not Found", b"no".to_vec()),
                        };
                        let head = format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                        let _ = s.write_all(head.as_bytes()).await;
                        let _ = s.write_all(&body).await;
                    });
                }
            });
            url
        }

        let client = client().unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let set = {
            let seen = seen.clone();
            move |s: Status| seen.lock().unwrap().push(s)
        };

        // Up to date: nothing is fetched.
        let api = served(|u| routes(u, "v6.5.2", &manifest, &sig)).await;
        let s = run_once(&client, &api, &base, &inst, (6, 5, 2), &[OPENSSL_PUB], true, &set).await.unwrap();
        assert_eq!(s, Status::UpToDate);

        // Asked first: found, not downloaded.
        let api = served(|u| routes(u, "v6.6.0", &manifest, &sig)).await;
        let s = run_once(&client, &api, &base, &inst, (6, 5, 2), &[OPENSSL_PUB], false, &set).await.unwrap();
        assert_eq!(s, Status::Available { version: "6.6.0".into() });
        assert!(read_staged(&base).is_none());

        // Signed by someone else: refused.
        let forged = sign(&"8f".repeat(32), manifest.as_bytes());
        let api = served(|u| routes(u, "v6.6.0", &manifest, &forged)).await;
        assert!(run_once(&client, &api, &base, &inst, (6, 5, 2), &[OPENSSL_PUB], true, &set).await.is_err());
        assert!(read_staged(&base).is_none());

        // An old signed manifest under a new tag: refused.
        let old = format!("midstate-release 1\nversion v6.5.9\nsha256 {hash} {archive}\n");
        let old_sig = sign(OPENSSL_SEED, old.as_bytes());
        let api = served(|u| routes(u, "v6.6.0", &old, &old_sig)).await;
        let e = run_once(&client, &api, &base, &inst, (6, 5, 2), &[OPENSSL_PUB], true, &set).await.unwrap_err();
        assert!(format!("{e:#}").contains("signed manifest says v6.5.9"), "{e:#}");

        // The archive swapped for another: refused.
        let wrong = format!("midstate-release 1\nversion v6.6.0\nsha256 {} {archive}\n", "00".repeat(32));
        let wrong_sig = sign(OPENSSL_SEED, wrong.as_bytes());
        let api = served(|u| routes(u, "v6.6.0", &wrong, &wrong_sig)).await;
        let e = run_once(&client, &api, &base, &inst, (6, 5, 2), &[OPENSSL_PUB], true, &set).await.unwrap_err();
        assert!(format!("{e:#}").contains("does not match its signed hash"), "{e:#}");
        assert!(read_staged(&base).is_none());

        // Not signed yet: not offered.
        let api = served(|u| {
            let mut r = routes(u, "v6.6.0", &manifest, &sig);
            r[0].1 = release_json(u, "v6.6.0", &[(MANIFEST, 100), (archive, bytes.len() as u64)]);
            r
        })
        .await;
        let s = run_once(&client, &api, &base, &inst, (6, 5, 2), &[OPENSSL_PUB], true, &set).await.unwrap();
        assert_eq!(s, Status::UpToDate);

        // The real thing: downloaded, checked, unpacked, waiting.
        seen.lock().unwrap().clear();
        let api = served(|u| routes(u, "v6.6.0", &manifest, &sig)).await;
        let s = run_once(&client, &api, &base, &inst, (6, 5, 2), &[OPENSSL_PUB], true, &set).await.unwrap();
        assert_eq!(s, Status::Ready { version: "v6.6.0".into() });
        assert!(seen.lock().unwrap().iter().any(|s| matches!(s, Status::Downloading { .. })), "progress was shown");
        let st = read_staged(&base).unwrap();
        assert_eq!(std::fs::read(st.dir.join(&names[0])).unwrap(), b"NEW APP");
        // Asked again, it does not fetch a second time.
        let s = run_once(&client, "http://127.0.0.1:9", &base, &inst, (6, 5, 2), &[OPENSSL_PUB], true, &set).await.unwrap();
        assert_eq!(s, Status::Ready { version: "v6.6.0".into() });

        // And the next start puts it in place.
        std::fs::write(inst.join(&names[0]), b"OLD APP").unwrap();
        std::fs::write(inst.join(&names[1]), b"OLD MINER").unwrap();
        assert_eq!(apply(&base, &inst, (6, 5, 2)).unwrap().as_deref(), Some("v6.6.0"));
        assert_eq!(std::fs::read(inst.join(&names[0])).unwrap(), b"NEW APP");
        assert_eq!(std::fs::read(inst.join(&names[1])).unwrap(), b"NEW MINER");
    }

    #[test]
    fn the_automatic_setting_is_kept() {
        let d = tempfile::tempdir().unwrap();
        assert!(auto_on(d.path()), "on unless turned off");
        set_auto(d.path(), false);
        assert!(!auto_on(d.path()));
        set_auto(d.path(), true);
        assert!(auto_on(d.path()));
    }
}
