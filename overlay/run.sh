#!/usr/bin/env bash
# Start the draft overlay. Uses quickshell from PATH, else the one Noctalia is running.
qs=$(command -v quickshell || pgrep -a -f '/bin/quickshell' | awk 'NR==1{print $2}')
[ -n "$qs" ] || { echo "quickshell not found" >&2; exit 1; }
exec "$qs" -p "$(dirname "$(readlink -f "$0")")/draft.qml" "$@"
