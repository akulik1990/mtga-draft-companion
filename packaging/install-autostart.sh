#!/usr/bin/env bash
# Install the companion as a systemd user service that starts with the graphical session (Linux).
#   packaging/install-autostart.sh            install and start
#   packaging/install-autostart.sh --remove   stop and remove
set -euo pipefail
root=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
unit="$HOME/.config/systemd/user/mtga-companion.service"
if [ "${1:-}" = "--remove" ]; then
    systemctl --user disable --now mtga-companion.service || true
    rm -f "$unit"
    systemctl --user daemon-reload
    echo "removed $unit"
    exit 0
fi
uv=$(command -v uv) || { echo "uv not found" >&2; exit 1; }
mkdir -p "$(dirname "$unit")"
sed -e "s|@ROOT@|$root|g" -e "s|@UV@|$uv|g" "$root/packaging/mtga-companion.service" > "$unit"
systemctl --user daemon-reload
systemctl --user enable --now mtga-companion.service
echo "installed $unit; status: systemctl --user status mtga-companion"
