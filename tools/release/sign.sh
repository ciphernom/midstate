#!/bin/sh
# Sign a release so the desktop app installs it.
#
#   tools/release/sign.sh v6.6.0 [--key ~/midstate-release-key.pem]
#
# Downloads every asset of the release, checks each one against the
# midstate-release.txt that CI wrote, signs that file with the release key and
# uploads midstate-release.txt.sig. Until the .sig is there, the app does not
# offer the release at all.
#
# Also: --dir DIR uses assets already in DIR instead of downloading them, and
# --no-upload leaves the .sig in that folder.
set -eu

repo=ciphernom/midstate
tag=""
key="$HOME/midstate-release-key.pem"
dir=""
upload=1
while [ $# -gt 0 ]; do
    case "$1" in
        --key) key=$2; shift 2 ;;
        --dir) dir=$2; shift 2 ;;
        --no-upload) upload=0; shift ;;
        -*) echo "unknown option $1" >&2; exit 2 ;;
        *) tag=$1; shift ;;
    esac
done
[ -n "$tag" ] || { echo "usage: $0 vX.Y.Z [--key KEY.pem] [--dir DIR] [--no-upload]" >&2; exit 2; }
[ -f "$key" ] || { echo "no key at $key (make one with tools/release/keygen.sh)" >&2; exit 1; }
command -v openssl >/dev/null || { echo "openssl is needed" >&2; exit 1; }
if command -v sha256sum >/dev/null; then sha() { sha256sum "$1" | cut -d' ' -f1; }
else sha() { shasum -a 256 "$1" | cut -d' ' -f1; }; fi

if [ -z "$dir" ]; then
    command -v gh >/dev/null || { echo "the GitHub CLI (gh) is needed" >&2; exit 1; }
    dir=$(mktemp -d)
    echo "Downloading the assets of $tag into $dir…"
    gh release download "$tag" --repo "$repo" --dir "$dir"
fi
m="$dir/midstate-release.txt"
[ -f "$m" ] || { echo "$tag has no midstate-release.txt" >&2; exit 1; }

# What is signed must be what is there.
[ "$(head -n 1 "$m")" = "midstate-release 1" ] || { echo "not a release manifest" >&2; exit 1; }
grep -qx "version $tag" "$m" || { echo "the manifest is not for $tag" >&2; exit 1; }
n=0
while read -r kind hash file; do
    [ "$kind" = sha256 ] || continue
    [ -f "$dir/$file" ] || { echo "MISSING: $file" >&2; exit 1; }
    got=$(sha "$dir/$file")
    [ "$got" = "$hash" ] || { echo "CHANGED: $file is $got, the manifest says $hash" >&2; exit 1; }
    echo "ok  $file"
    n=$((n + 1))
done < "$m"
[ "$n" -gt 0 ] || { echo "the manifest lists no files" >&2; exit 1; }
for f in "$dir"/*; do
    b=$(basename "$f")
    case "$b" in midstate-release.txt|midstate-release.txt.sig) continue ;; esac
    grep -q " $b\$" "$m" || echo "note: $b is on the release but not in the manifest" >&2
done

pub=$(openssl pkey -in "$key" -pubout -outform DER | tail -c 32 | od -An -tx1 | tr -d ' \n')
here=$(dirname "$0")
if [ -f "$here/../../apps/desktop/src/update.rs" ] && ! grep -q "$pub" "$here/../../apps/desktop/src/update.rs"; then
    echo "warning: this key ($pub) is not in RELEASE_KEYS in apps/desktop/src/update.rs" >&2
fi

openssl pkeyutl -sign -inkey "$key" -rawin -in "$m" -out "$m.sig"
openssl pkey -in "$key" -pubout -out "$dir/.pub.pem"
openssl pkeyutl -verify -pubin -inkey "$dir/.pub.pem" -rawin -in "$m" -sigfile "$m.sig" >/dev/null
rm -f "$dir/.pub.pem"
echo "Signed $n files of $tag with key $pub"

if [ "$upload" = 1 ]; then
    gh release upload "$tag" "$m.sig" --repo "$repo" --clobber
    echo "Uploaded midstate-release.txt.sig: the desktop app will offer $tag now."
else
    echo "Signature: $m.sig (not uploaded)"
fi
