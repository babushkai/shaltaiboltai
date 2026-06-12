#!/bin/sh
# shaltaiboltai installer.
#
#   curl -fsSL https://github.com/babushkai/shaltaiboltai/releases/latest/download/install.sh | sh
#
# Downloads the prebuilt binary for your platform from the latest GitHub
# release, verifies its checksum, and installs it. No Rust toolchain needed.
#
# Environment overrides:
#   SHALTAI_INSTALL_DIR   where to install (default: $HOME/.local/bin)
#   SHALTAI_VERSION       release tag to install (default: latest)
set -eu

REPO="babushkai/shaltaiboltai"
BIN="shaltaiboltai"
INSTALL_DIR="${SHALTAI_INSTALL_DIR:-$HOME/.local/bin}"
VERSION="${SHALTAI_VERSION:-latest}"

err() {
	printf 'error: %s\n' "$1" >&2
	exit 1
}

# --- detect platform --------------------------------------------------------
os="$(uname -s)"
arch="$(uname -m)"
case "$os" in
Darwin)
	case "$arch" in
	arm64 | aarch64) target="aarch64-apple-darwin" ;;
	x86_64) target="x86_64-apple-darwin" ;;
	*) err "unsupported macOS architecture: $arch" ;;
	esac
	;;
Linux)
	case "$arch" in
	x86_64 | amd64) target="x86_64-unknown-linux-gnu" ;;
	*) err "unsupported Linux architecture: $arch (build from source: cargo install --git https://github.com/$REPO)" ;;
	esac
	;;
*)
	err "unsupported OS: $os"
	;;
esac

# --- resolve download URLs --------------------------------------------------
if [ "$VERSION" = "latest" ]; then
	base="https://github.com/$REPO/releases/latest/download"
else
	base="https://github.com/$REPO/releases/download/$VERSION"
fi
archive="${BIN}-${target}.tar.gz"
url="$base/$archive"

command -v curl >/dev/null 2>&1 || err "curl is required"
command -v tar >/dev/null 2>&1 || err "tar is required"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

printf 'Downloading %s …\n' "$archive"
curl -fSL --proto '=https' --tlsv1.2 "$url" -o "$tmp/$archive" ||
	err "download failed: $url"

# --- verify checksum (best-effort: skip if the .sha256 isn't published) -----
if curl -fsSL "$url.sha256" -o "$tmp/$archive.sha256" 2>/dev/null; then
	expected="$(awk '{print $1}' "$tmp/$archive.sha256")"
	if command -v sha256sum >/dev/null 2>&1; then
		actual="$(sha256sum "$tmp/$archive" | awk '{print $1}')"
	else
		actual="$(shasum -a 256 "$tmp/$archive" | awk '{print $1}')"
	fi
	[ "$expected" = "$actual" ] || err "checksum mismatch (expected $expected, got $actual)"
	printf 'Checksum OK\n'
fi

# --- install ----------------------------------------------------------------
tar -xzf "$tmp/$archive" -C "$tmp"
mkdir -p "$INSTALL_DIR"
mv "$tmp/$BIN" "$INSTALL_DIR/$BIN"
chmod +x "$INSTALL_DIR/$BIN"

printf '\nInstalled %s to %s\n' "$BIN" "$INSTALL_DIR/$BIN"
case ":$PATH:" in
*":$INSTALL_DIR:"*) printf 'Run: %s\n' "$BIN" ;;
*)
	printf '\n%s is not on your PATH. Add this to your shell profile:\n' "$INSTALL_DIR"
	# $PATH is intentionally literal here — it's text the user pastes.
	# shellcheck disable=SC2016
	printf '    export PATH="%s:$PATH"\n' "$INSTALL_DIR"
	;;
esac
