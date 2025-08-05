#!/usr/bin/env bash
# run_webshop_env_tests.sh
# Executes webshop_env_test.py and mirrors all output to ./test_logs/.

set -euo pipefail
IFS=$'\n\t'

# ── resolve repo root (directory containing this script) ────────────────
root_dir="$( cd -- "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$root_dir"

# ── ensure test_logs exists and use it consistently ─────────────────────
log_dir="$root_dir/test_logs"
mkdir -p "$log_dir"

# ── driver log file (timestamped) ───────────────────────────────────────
ts="$(date +%Y%m%d_%H%M%S)"
driver_log="$log_dir/webshop_env_driver_${ts}.log"

echo "📝 Driver log → $driver_log"
echo "🚀 Running WebShopEnv tests…"

# ── run the tests, capture both stdout & stderr into log_dir ────────────
python env.py 2>&1 | tee "$driver_log"

echo "✅ Finished at $(date)"
echo "📂 All logs are in: $log_dir"
