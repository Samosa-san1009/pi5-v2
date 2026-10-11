#!/usr/bin/env bash
# Pull the latest code and re-run the installer (keeps settings and sessions).
#   cd ~/Documents/pi5-v2 && bash scripts/update.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
git pull --ff-only
bash scripts/install.sh
