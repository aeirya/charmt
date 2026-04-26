#!/usr/bin/env bash
set -euo pipefail

FOLDER_ID="$1"
OUT="$2"

mkdir -p "$OUT"

source ".env"
RCLONE="$HOME/.local/bin/rclone"
REMOTE="${RCLONE_REMOTE:-gdrive}"

"$RCLONE" copy "${REMOTE}:" "$OUT" \
  --drive-root-folder-id "$FOLDER_ID" \
  --drive-client-id "$RCLONE_DRIVE_CLIENT_ID" \
  --drive-client-secret "$RCLONE_DRIVE_CLIENT_SECRET" \
  --progress \
  --ignore-existing \
  --transfers 2 \
  --checkers 4 \
  --tpslimit 4 \
  --bwlimit 100M \
  --retries 10 \
  --low-level-retries 20 \
  --drive-chunk-size 64M \
  --drive-acknowledge-abuse \
  --stats 10s