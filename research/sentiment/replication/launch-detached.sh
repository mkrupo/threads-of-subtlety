#!/usr/bin/env bash
# Start RUN_DIR's job script in its own session so it survives SSH disconnects,
# and record the job's process identity next to its log.
#
# Usage (on the GPU host): launch-detached.sh RUN_DIR JOB_SCRIPT
set -euo pipefail

run="$(cd "$1" && pwd)"
script="$run/$(basename "$2")"
[[ -f "$script" ]] || { echo "Missing $script" >&2; exit 2; }
[[ ! -e "$run/launch.pid" ]] || { echo "$run already has a launch record" >&2; exit 2; }
cd "$run"
setsid nohup bash "$script" "$run" > "$run/launch.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$run/launch.pid"
sha256sum "$script" > "$run/launch.script.sha256"
sleep 2
ps -o pid,sid,lstart,args -p "$pid" > "$run/launch.process"
cat "$run/launch.process"
