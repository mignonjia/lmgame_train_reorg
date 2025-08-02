#!/usr/bin/env bash
# run_agent_test.sh
# Stream-tee python agent_test.py to ./test_logs/<timestamp>.log

set -euo pipefail

SCRIPT="agent_test.py"                # change if the test file has another name
LOG_DIR="test_logs"                   # fixed log folder
mkdir -p "$LOG_DIR"

TS=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/$(basename "$SCRIPT" .py)_$TS.log"

echo "📝 Logging to $LOG_FILE"
echo "⌛ $(date) — running $SCRIPT $*" | tee "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Stream both stdout and stderr to console *and* the log file
python "$SCRIPT" "$@" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "============================================================" | tee -a "$LOG_FILE"
echo "🏁 Finished with exit code $EXIT_CODE at $(date)" | tee -a "$LOG_FILE"
exit "$EXIT_CODE"
