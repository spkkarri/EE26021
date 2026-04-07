#!/bin/bash
# ============================================================
#  Solar Plant AI Planner — Setup Script
#  Run this once to install dependencies and configure the app
# ============================================================

set -e  # Exit immediately on error

echo "============================================"
echo "  Solar Plant AI Planner — Setup"
echo "============================================"

# ── 1. Check Python version ──────────────────────────────────
echo ""
echo "[1/4] Checking Python version..."
PYTHON=$(command -v python3 || command -v python)

if [ -z "$PYTHON" ]; then
  echo "ERROR: Python 3.10+ is required but not found."
  exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "Found: Python $PY_VERSION"

# ── 2. Clone or update repository ───────────────────────────
echo ""
echo "[2/4] Checking repository..."

if [ -d "WebScrapingAgent" ]; then
  echo "Repo already exists. Pulling latest changes..."
  cd WebScrapingAgent && git pull && cd ..
else
  echo "Cloning repository..."
  # Replace the URL below with your actual repo URL
  git clone https://github.com/YOUR_USERNAME/WebScrapingAgent.git
fi

cd WebScrapingAgent

# ── 3. Install Python dependencies ──────────────────────────
echo ""
echo "[3/4] Installing Python packages..."
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install flask groq requests beautifulsoup4

echo "Packages installed successfully."

# ── 4. Configure Groq API key ────────────────────────────────
echo ""
echo "[4/4] API Key Configuration..."

if grep -q "paste_your_gsk_key_here" solar.py; then
  echo ""
  read -p "Enter your Groq API key (gsk_...): " GROQ_KEY
  if [ -n "$GROQ_KEY" ]; then
    sed -i "s|paste_your_gsk_key_here|$GROQ_KEY|g" solar.py
    echo "API key configured in solar.py."
  else
    echo "WARNING: No key entered. Edit solar.py manually before running."
  fi
else
  echo "API key already configured."
fi

echo ""
echo "============================================"
echo " Setup complete!"
echo " Run ./data/run.sh to start the application"
echo "============================================"
