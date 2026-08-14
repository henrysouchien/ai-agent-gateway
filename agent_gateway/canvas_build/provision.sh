#!/usr/bin/env bash
set -euo pipefail

PACKAGED_BUILD_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_RUNTIME="${CANVAS_BUILD_DIR:-$PACKAGED_BUILD_SOURCE}"
CANDIDATE_ROOT="${CANVAS_PROVISION_CANDIDATE_ROOT:-}"
NODE_VERSION="$(tr -d '[:space:]' < "$PACKAGED_BUILD_SOURCE/.node-version")"
INSTALL_PREFIX="${CANVAS_NODE_PREFIX:-/opt/hank/node-v$NODE_VERSION}"
EXPECTED_INSTALL_BASENAME="node-v$NODE_VERSION"
if [[ "$INSTALL_PREFIX" != /* || "$(basename "$INSTALL_PREFIX")" != "$EXPECTED_INSTALL_BASENAME" ]]; then
  echo "CANVAS_NODE_PREFIX must be an absolute path ending in $EXPECTED_INSTALL_BASENAME" >&2
  exit 1
fi
if [[ -n "$CANDIDATE_ROOT" ]]; then
  if [[ ! "$CANDIDATE_ROOT" =~ ^/opt/hank/\.agent-gateway-deploy-[0-9a-f]{64}$ ]] \
    || [[ -L "$CANDIDATE_ROOT" || ! -d "$CANDIDATE_ROOT" ]] \
    || [[ "$BUILD_RUNTIME" != "$CANDIDATE_ROOT/canvas-build" ]] \
    || [[ "$INSTALL_PREFIX" != "$CANDIDATE_ROOT/$EXPECTED_INSTALL_BASENAME" ]]; then
    echo "Canvas candidate paths must be exact children of the protected deploy root" >&2
    exit 1
  fi
elif [[ "$BUILD_RUNTIME" != "$PACKAGED_BUILD_SOURCE" ]] \
  && [[ "$BUILD_RUNTIME" != "/opt/hank/canvas-build" ]]; then
  echo "external CANVAS_BUILD_DIR must be /opt/hank/canvas-build" >&2
  exit 1
fi
SYSTEM_NAME="$(uname -s | tr '[:upper:]' '[:lower:]')"
MACHINE_NAME="$(uname -m)"
case "$SYSTEM_NAME-$MACHINE_NAME" in
  linux-x86_64) PLATFORM=linux-x64 ;;
  linux-aarch64|linux-arm64) PLATFORM=linux-arm64 ;;
  darwin-arm64) PLATFORM=darwin-arm64 ;;
  *) echo "Unsupported Canvas Node platform: $SYSTEM_NAME-$MACHINE_NAME" >&2; exit 1 ;;
esac

read -r ARCHIVE EXPECTED_SHA < <(
  python3 - "$PACKAGED_BUILD_SOURCE/node_checksums.json" "$PLATFORM" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
entry = doc["platforms"][sys.argv[2]]
print(entry["filename"], entry["sha256"])
PY
)
if [[ "$EXPECTED_SHA" == TODO* || ! "$EXPECTED_SHA" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Canvas Node checksum is unverified for $PLATFORM" >&2
  exit 1
fi

DOWNLOAD_DIR="$(mktemp -d)"
NODE_STAGE=""
RUNTIME_STAGE=""
cleanup_canvas_provision() {
  rm -rf -- "$DOWNLOAD_DIR"
  if [[ -n "${NODE_STAGE:-}" ]]; then
    rm -rf -- "$NODE_STAGE"
  fi
  if [[ -n "${RUNTIME_STAGE:-}" ]]; then
    rm -rf -- "$RUNTIME_STAGE"
  fi
}
trap cleanup_canvas_provision EXIT
curl \
  --fail \
  --location \
  --show-error \
  --connect-timeout 10 \
  --max-time 120 \
  --speed-limit 1024 \
  --speed-time 30 \
  --retry 3 \
  --retry-delay 2 \
  --retry-max-time 180 \
  --retry-connrefused \
  "https://nodejs.org/download/release/v$NODE_VERSION/$ARCHIVE" \
  --output "$DOWNLOAD_DIR/$ARCHIVE"
python3 - "$EXPECTED_SHA" "$DOWNLOAD_DIR/$ARCHIVE" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

expected = sys.argv[1]
digest = sha256()
with Path(sys.argv[2]).open("rb") as stream:
  while chunk := stream.read(1024 * 1024):
    digest.update(chunk)
actual = digest.hexdigest()
if actual != expected:
  raise SystemExit(f"Canvas Node archive checksum mismatch: expected {expected}, got {actual}")
PY
INSTALL_PARENT="$(dirname "$INSTALL_PREFIX")"
mkdir -p "$INSTALL_PARENT"
NODE_STAGE="$(mktemp -d "$INSTALL_PARENT/.$EXPECTED_INSTALL_BASENAME.new.XXXXXXXX")"
tar -xJf "$DOWNLOAD_DIR/$ARCHIVE" --strip-components=1 -C "$NODE_STAGE"
rm -rf -- "$INSTALL_PREFIX"
mv "$NODE_STAGE" "$INSTALL_PREFIX"
NODE_STAGE=""

if [[ "$BUILD_RUNTIME" == "$PACKAGED_BUILD_SOURCE" ]]; then
  PATH="$INSTALL_PREFIX/bin:$PATH" npm ci --prefix "$PACKAGED_BUILD_SOURCE"
else
  RUNTIME_PARENT="$(dirname "$BUILD_RUNTIME")"
  mkdir -p "$RUNTIME_PARENT"
  RUNTIME_STAGE="$(mktemp -d "$RUNTIME_PARENT/.canvas-build.new.XXXXXXXX")"
  for support_file in \
    .node-version \
    node_checksums.json \
    package.json \
    package-lock.json \
    build.mjs \
    policy.mjs \
    tsconfig.json; do
    install -m 0644 \
      "$PACKAGED_BUILD_SOURCE/$support_file" \
      "$RUNTIME_STAGE/$support_file"
  done
  PATH="$INSTALL_PREFIX/bin:$PATH" npm ci --prefix "$RUNTIME_STAGE"
  rm -rf -- "$BUILD_RUNTIME"
  mv "$RUNTIME_STAGE" "$BUILD_RUNTIME"
  RUNTIME_STAGE=""
fi
echo "Canvas build environment provisioned at $INSTALL_PREFIX; build files at $BUILD_RUNTIME"
