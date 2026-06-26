#!/bin/bash
# watchdog_bots.sh — Ensure run_bots.py is always running.
# Runs as a persistent loop (launched once by cron @reboot), checking every 5 seconds.

LOG=/tmp/run_bots.log
PIDFILE=/tmp/run_bots.pid
DIR=/Users/shikharsehgal/futureflow/worldcup
PYTHON=/usr/bin/python3

# Prevent multiple watchdog instances
WDOG_LOCK=/tmp/watchdog_bots.lock
if [ -f "$WDOG_LOCK" ]; then
    OLD_PID=$(cat "$WDOG_LOCK")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        exit 0  # watchdog already running
    fi
fi
echo $$ > "$WDOG_LOCK"
trap "rm -f $WDOG_LOCK" EXIT

while true; do
    if ! pgrep -f "run_bots.py" > /dev/null 2>&1; then
        echo "[watchdog $(date)] run_bots.py not found, restarting..." >> "$LOG"
        cd "$DIR"
        nohup "$PYTHON" -u run_bots.py >> "$LOG" 2>&1 &
        echo $! > "$PIDFILE"
        echo "[watchdog $(date)] launched PID $(cat $PIDFILE)" >> "$LOG"
    fi
    sleep 5
done
