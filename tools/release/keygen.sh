#!/bin/sh
# Make the key that signs Midstate releases for the desktop app's automatic
# updates. Run it once, on a computer you trust.
#
#   tools/release/keygen.sh [where-to-put-the-key.pem]
#
# The private key (default ~/midstate-release-key.pem) never goes near CI or
# GitHub: keep it offline and back it up. Lose it and the app can no longer be
# updated automatically (people would have to install a version with a new
# key by hand); leak it and someone else could sign an update.
set -eu

out="${1:-$HOME/midstate-release-key.pem}"
if [ -e "$out" ]; then
    echo "$out already exists; not overwriting it." >&2
    exit 1
fi
command -v openssl >/dev/null || { echo "openssl is needed" >&2; exit 1; }

umask 077
openssl genpkey -algorithm ed25519 -out "$out"
pub=$(openssl pkey -in "$out" -pubout -outform DER | tail -c 32 | od -An -tx1 | tr -d ' \n')

echo "Private key: $out   (keep it secret; back it up)"
echo "Public key:  $pub"
echo
echo "Put the public key in apps/desktop/src/update.rs:"
echo
echo "    pub const RELEASE_KEYS: &[&str] = &[\"$pub\"];"
