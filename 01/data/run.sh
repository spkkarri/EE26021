#!/bin/bash
# ============================================================
#  Solar Plant AI Planner — Run Script
#  Starts the Flask web server
# ============================================================

set -e

echo "============================================"
echo "  Solar Plant AI Planner — Starting Server"
echo "============================================"

# ── Navigate to project root ─────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")/WebScrapingAgent"

if [ ! -d "$PROJECT_DIR" ]; then
  echo "ERROR: Project directory not found at: $PROJECT_DIR"
  echo "Please run ./data/setup.sh first."
  exit 1
fi

cd "$PROJECT_DIR"

# ── Check API key is set ─────────────────────────────────────
if grep -q "paste_your_gsk_key_here" solar.py; then
  echo "ERROR: Groq API key not configured."
  echo "Run ./data/setup.sh or edit solar.py manually."
  exit 1
fi

# ── Check for required packages ──────────────────────────────
PYTHON=$(command -v python3 || command -v python)
$PYTHON -c "import flask, groq, requests, bs4" 2>/dev/null || {
  echo "ERROR: Missing Python packages. Run ./data/setup.sh first."
  exit 1
}

# ── Start the server ─────────────────────────────────────────
echo ""
echo " Server starting at: http://127.0.0.1:5000"
echo " Press Ctrl+C to stop."
echo ""

$PYTHON solar.py
