# Automatic updates (desktop app)

Status: built. Off until a release key is put in `apps/desktop/src/update.rs`
and a release is signed with it.

## What the app does

Every six hours (and from Settings → Updates → Check now) the desktop app asks
GitHub for the latest release of `ciphernom/midstate`. It installs a newer one
only when all of these hold:

1. The release has `midstate-release.txt` and `midstate-release.txt.sig`, and
   the signature is by one of the keys in `RELEASE_KEYS`, which are compiled
   into the app. A release made by anyone else, or an asset swapped after the
   fact, is refused.
2. The signed file names the same version as the release tag, and that
   version is newer than the running one. An old signed release cannot be
   passed off as a new one, and nothing is ever downgraded.
3. The archive for this computer (`midstate-linux-amd64.tar.gz`,
   `midstate-windows-amd64.zip`, `midstate-macos-arm64.tar.gz` or
   `midstate-macos-amd64.tar.gz`) hashes to the SHA-256 the signed file lists.

It then unpacks the two programs from the archive — the app and `midstate`,
which mines — into the profile folder (`updates/<tag>/`), skipping everything
else in the archive and any path that is not a plain file name at most one
folder deep. The next time the app starts, or at once from **Restart now**, the
running programs are moved aside, the new ones put in their place, and the app
starts again. If anything fails part-way, the old programs go back. The files
are hashed again just before they go in, so a change after the check is caught.

A copy installed where this user cannot write (an "all users" install under
Program Files) gets a **Download** button that opens the release page instead.
Builds from source never update themselves: only CI builds know their version.

Settings → Updates has "Download new versions automatically" (on by default).
Off, the app still looks, and offers **Install**.

## The release key

The key is Ed25519. CI never sees it: CI writes `midstate-release.txt`, and the
maintainer signs it afterwards from a trusted computer. Someone who takes over
the GitHub account or the CI can publish a release, but cannot make the app
install it.

Make it once:

```sh
tools/release/keygen.sh            # writes ~/midstate-release-key.pem, prints the public key
```

Put the public key it prints in `apps/desktop/src/update.rs`:

```rust
pub const RELEASE_KEYS: &[&str] = &["<64 hex characters>"];
```

Keep the `.pem` offline and backed up. If it is lost, releases can no longer be
signed for the copies people have, and they would install the next version by
hand. To change keys, ship a version that lists both the old and the new key,
signed with the old one; later versions can drop the old key.

## Releasing

1. Tag as before (`git tag v6.6.0 && git push origin v6.6.0`). CI builds,
   stamps the tag into the app (`MIDSTATE_RELEASE`), and publishes the release
   with `midstate-release.txt` among its assets.
2. When the release is up, sign it:

   ```sh
   tools/release/sign.sh v6.6.0
   ```

   This downloads every asset (it needs `gh` and `openssl`), checks each one
   against `midstate-release.txt`, signs, checks the signature, and uploads
   `midstate-release.txt.sig`. It refuses if an asset is missing or differs.
3. From then on, copies of the app find it within six hours.

The first version that has this code still has to be installed by hand: only
copies that already update themselves will pick up later versions.

## Limits

- Tested here on Linux, including the restart path end to end. The Windows
  path (renaming the running `.exe` aside, removing it at the next start) and
  the macOS path (replacing the programs inside the `.app`) follow the same
  code but were not run on those systems.
- On macOS the bundle's `Info.plist` keeps the version it was installed with;
  the app itself reports the right one.
- Raspberry Pi builds are the node only, with no app to update them.
