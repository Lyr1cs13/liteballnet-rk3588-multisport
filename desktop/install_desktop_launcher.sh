#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="$HOME/Desktop/RK3588_Board_Verification.desktop"

mkdir -p "$HOME/Desktop"
chmod +x "$ROOT/desktop/launch_board_demo.sh"
cat > "$LAUNCHER" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=RK3588 Board Verification
Comment=Replay board-generated multi-sport tracking results
Exec=$ROOT/desktop/launch_board_demo.sh
Terminal=false
Categories=Development;Science;
EOF
chmod +x "$LAUNCHER"
if command -v gio >/dev/null 2>&1; then
    gio set "$LAUNCHER" metadata::trusted true || true
fi
printf 'Desktop launcher created: %s\n' "$LAUNCHER"
