#!/bin/bash
# ============================================================
#  Solar Plant AI Planner — Cleanup Script
#  Removes auto-generated files and optionally resets feedback
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")/WebScrapingAgent"

if [ ! -d "$PROJECT_DIR" ]; then
  echo "ERROR: Project directory not found at: $PROJECT_DIR"
  exit 1
fi

cd "$PROJECT_DIR"

echo "============================================"
echo "  Solar Plant AI Planner — Cleanup"
echo "============================================"
echo ""

# ── Remove auto-saved results files ──────────────────────────
RESULTS_COUNT=$(ls results_*.txt 2>/dev/null | wc -l)
if [ "$RESULTS_COUNT" -gt 0 ]; then
  echo "Removing $RESULTS_COUNT results file(s)..."
  rm -f results_*.txt
  echo "Done."
else
  echo "No results files to remove."
fi

# ── Optionally reset feedback data ───────────────────────────
echo ""
if [ -f "feedback_data.json" ]; then
  read -p "Reset feedback_data.json (RLHF data will be lost)? [y/N]: " CONFIRM
  if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    rm -f feedback_data.json
    echo "Feedback data reset."
  else
    echo "Feedback data kept."
  fi
else
  echo "No feedback data file found."
fi

echo ""
echo "============================================"
echo " Cleanup complete."
echo "============================================"
