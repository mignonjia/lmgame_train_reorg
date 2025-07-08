#!/bin/bash

# LMGame Test Runner with Logging
# This script runs tests with comprehensive logging and error handling

echo "🧪 Starting LMGame test suite with logging..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please ensure Python is installed and in PATH."
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p test_logs

# Run the test runner with logging
echo "📁 Working directory: $SCRIPT_DIR"
echo "📝 Logs will be saved to: $SCRIPT_DIR/test_logs/"

# Set environment variables for better error reporting
export PYTHONPATH="$SCRIPT_DIR/../../..:$PYTHONPATH"
export HYDRA_FULL_ERROR=1

# Run the tests
python run_tests_with_logging.py

# Show the result
EXIT_CODE=$?

echo ""
echo "🔍 Test run completed. Check logs in test_logs/ directory for details."

# List recent log files
if [ -d "test_logs" ] && [ "$(ls -A test_logs)" ]; then
    echo ""
    echo "Recent log files:"
    ls -la test_logs/*.log 2>/dev/null | tail -5
fi

exit $EXIT_CODE 