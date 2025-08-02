#!/usr/bin/env bash
#
# Helper to run tests/gsm8k_tests/agent_test.py
# Logs full console output to logs/gsm8k_agent_test_<timestamp>.log
#

set -euo pipefail                    #  safer bash

# ── locations ────────────────────────────────────────────────────────────────
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEST_FILE="tests/gsm8k_tests/agent_test.py"   # adjust if you moved the test
LOG_DIR="$ROOT_DIR/test_logs"
mkdir -p "$LOG_DIR"

# ── logfile name with timestamp ──────────────────────────────────────────────
STAMP="$(date '+%Y%m%d_%H%M%S')"
LOGFILE="$LOG_DIR/gsm8k_agent_test_${STAMP}.log"

# ── run the test, teeing output ─────────────────────────────────────────────
echo "▶️  Running $TEST_FILE"
echo "   logging to $LOGFILE"
echo "===================================================================="

# The subshell ensures we capture the script’s real exit code even when using tee
(
    set +e                         # don’t exit inside subshell so we can grab rc
    python "$TEST_FILE"
) 2>&1 | tee "$LOGFILE"
RC=${PIPESTATUS[0]}                # exit-status of python, not tee

echo "===================================================================="
echo "📝 Finished – exit code $RC  (full log in $LOGFILE)"
exit $RC
