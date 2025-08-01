#!/bin/bash

# Create cache directory if it doesn't exist
mkdir -p cache

# Create timestamped log file
LOG_FILE="cache/test_modules_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1

echo "📝 Logging test_modules output to: $LOG_FILE"
echo "🔍 Checking git submodule status for external/webshop-minimal..."

git submodule status external/webshop-minimal