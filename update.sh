#!/bin/bash
# OpenAlgo Update Script

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Updating OpenAlgo..."

# Backup .env
cp .env .env.backup

# Pull latest
git fetch origin
git pull origin main

# Check if .env needs updating (compare versions)
CURRENT_VERSION=$(grep ENV_CONFIG_VERSION .env | cut -d"'" -f2)
SAMPLE_VERSION=$(grep ENV_CONFIG_VERSION .sample.env | cut -d"'" -f2)

if [ "$CURRENT_VERSION" != "$SAMPLE_VERSION" ]; then
    echo ""
    echo "WARNING: .env config version mismatch!"
    echo "  Your version: $CURRENT_VERSION"
    echo "  New version:  $SAMPLE_VERSION"
    echo "  Check .sample.env for new variables"
    echo ""
fi

# Update Python dependencies
echo "Updating Python dependencies..."
uv sync

# Rebuild frontend if package.json changed
if git diff HEAD~1 --name-only | grep -q "frontend/package"; then
    echo "Frontend dependencies changed, rebuilding..."
    cd frontend && npm install && npm run build && cd ..
else
    echo "Rebuilding frontend..."
    cd frontend && npm run build && cd ..
fi

echo ""
echo "Update complete!"
echo "Restart the app to apply changes: uv run app.py"
