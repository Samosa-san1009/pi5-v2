#!/usr/bin/env bash
# Mobile PLR installer for Raspberry Pi 5 (64-bit Raspberry Pi OS: Trixie or Bookworm).
#
#   cd ~/Documents/pi5-v2
#   bash scripts/install.sh
#
# Safe to re-run after every `git pull` (scripts/update.sh does exactly that).
# Steps:
#   1. removes the old IR display (it holds the camera and the TFT)
#   2. retires services from earlier PLR installs (RTC sync, pigpiod)
#   3. installs APT packages (camera, lgpio, OpenCV dev, ffmpeg, cmake ...)
#   4. enables SPI for the TFT; comments out any old I2C RTC overlay
#   5. creates .venv (uses the system camera/numpy packages) + pip packages
#   6. builds the native PuReST engine (C++)
#   7. permissions for camera, GPIO, SPI and /dev/pio0 (WS2812 driver)
#   8. installs and starts plr.service
set -euo pipefail
# Ignore packages in ~/.local (e.g. from an old pip install --break-system-packages):
# they would shadow the versions this rig installs into .venv.
export PYTHONNOUSERSITE=1

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_USER="${SUDO_USER:-$USER}"
CONFIG=/boot/firmware/config.txt
[ -f "$CONFIG" ] || CONFIG=/boot/config.txt
REBOOT=0

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mXX %s\033[0m\n' "$*"; exit 1; }

if [ "$(id -u)" -eq 0 ]; then
  die "Run as your normal user, not with sudo (the script asks for sudo where needed)."
fi

say "Checking platform"
model="$(tr -d '\0' </proc/device-tree/model 2>/dev/null || echo unknown)"
os="$(. /etc/os-release && echo "$PRETTY_NAME")"
codename="$(. /etc/os-release && echo "${VERSION_CODENAME:-unknown}")"
echo "model : $model"
echo "os    : $os ($codename, $(uname -m))"
echo "repo  : $REPO"
case "$model" in *"Raspberry Pi 5"*) ;; *) warn "Not a Raspberry Pi 5 - the WS2812 driver and pin map assume a Pi 5.";; esac
[ "$(uname -m)" = "aarch64" ] || warn "Expected 64-bit OS (aarch64)."
case "$codename" in bookworm|trixie) ;; *) warn "Untested OS release '$codename' (expected trixie or bookworm).";; esac

say "1. Removing the old IR display"
bash "$REPO/scripts/remove_old_ir_display.sh"

say "2. Retiring services from earlier installs"
for unit in plr-rtc.timer plr-rtc.service pigpiod.service; do
  if systemctl list-unit-files "$unit" 2>/dev/null | grep -q "^$unit"; then
    sudo systemctl disable --now "$unit" 2>/dev/null || true
    echo "disabled $unit"
  fi
done
for f in /etc/systemd/system/plr-rtc.timer /etc/systemd/system/plr-rtc.service; do
  [ -f "$f" ] && sudo rm -f "$f" && echo "removed $f"
done
if systemctl is-active --quiet plr.service; then
  sudo systemctl stop plr.service      # the existing unit is replaced below
fi
sudo systemctl daemon-reload

say "3. Installing APT packages"
sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-pip python3-dev build-essential \
  python3-picamera2 python3-libcamera python3-numpy python3-pil python3-flask \
  python3-lgpio python3-spidev \
  ffmpeg cmake g++ pkg-config libopencv-dev util-linux
for optional in python3-av python3-simplejpeg rpicam-apps; do
  sudo apt-get install -y "$optional" >/dev/null 2>&1 || warn "optional package $optional not installed"
done

say "4. Boot configuration ($CONFIG)"
if ! grep -qE '^\s*dtparam=spi=on' "$CONFIG"; then
  # under an explicit [all] so it never lands inside a [cm4]/[cm5]-only section
  printf '\n[all]\ndtparam=spi=on\n' | sudo tee -a "$CONFIG" >/dev/null
  echo "added: dtparam=spi=on (TFT)"
  REBOOT=1
else
  echo "SPI already enabled"
fi
# The SPI1 overlays (spi1-1cs, spi1-2cs, ...) claim GPIO18 - the left LED pin -
# as SPI1 chip select, so the WS2812 driver could not use it. Comment them out.
if grep -qE '^\s*dtoverlay=spi1-' "$CONFIG"; then
  sudo cp "$CONFIG" "$CONFIG.before-plr-spi1"
  sudo sed -i -E 's/^(\s*dtoverlay=spi1-.*)$/# \1   # removed: takes GPIO18 (left LED)/' "$CONFIG"
  echo "commented out the SPI1 overlay - it takes GPIO18, the left LED pin (backup: $CONFIG.before-plr-spi1)"
  REBOOT=1
fi
if grep -qE '^\s*dtoverlay=i2c-rtc' "$CONFIG"; then
  sudo cp "$CONFIG" "$CONFIG.before-plr"
  sudo sed -i -E 's/^(\s*dtoverlay=i2c-rtc.*)$/# \1   # external RTC module removed (PLR rig)/' "$CONFIG"
  echo "commented out the old external RTC overlay (backup: $CONFIG.before-plr)"
  REBOOT=1
fi

say "5. Python environment ($REPO/.venv)"
# --system-site-packages: picamera2, libcamera, numpy, lgpio come from APT and
# must match the OS; only the small display/LED libraries come from pip.
if [ -d "$REPO/.venv" ] && ! "$REPO/.venv/bin/python" -c "import sys" 2>/dev/null; then
  rm -rf "$REPO/.venv"
fi
python3 -m venv --system-site-packages "$REPO/.venv"
PIP="$REPO/.venv/bin/pip"
PY="$REPO/.venv/bin/python"
"$PIP" install --upgrade pip
"$PIP" install -r "$REPO/requirements.txt"
# On a Pi 5 the classic RPi.GPIO cannot work; luma.lcd must use the lgpio shim.
"$PIP" uninstall -y RPi.GPIO >/dev/null 2>&1 || true
"$PIP" install --force-reinstall --no-deps rpi-lgpio
if ! "$PY" - <<'EOF'
import importlib
ok = True
for name in ("picamera2", "libcamera", "numpy", "PIL", "flask", "lgpio", "RPi.GPIO", "luma.lcd.device",
             "adafruit_raspberry_pi5_neopixel_write"):
    try:
        importlib.import_module(name)
        print(f"   ok  {name}")
    except Exception as exc:
        ok = False
        print(f"   !!  {name}: {type(exc).__name__}: {exc}")
raise SystemExit(0 if ok else 1)
EOF
then
  warn "Some Python modules failed to import (see above). The service may not start fully."
fi

say "6. Building the PuReST engine (C++)"
bash "$REPO/scripts/build_native.sh"

say "7. Permissions"
sudo usermod -aG video,gpio,spi "$SERVICE_USER"
echo 'KERNEL=="pio[0-9]*", GROUP="gpio", MODE="0660"' | sudo tee /etc/udev/rules.d/99-plr-pio.rules >/dev/null
sudo udevadm control --reload-rules && sudo udevadm trigger || true
[ -e /dev/pio0 ] || warn "/dev/pio0 is missing: run 'sudo apt full-upgrade' and reboot, or the LEDs cannot be driven."

say "8. Installing plr.service"
sed -e "s#@USER@#$SERVICE_USER#g" -e "s#@REPO@#$REPO#g" "$REPO/scripts/plr.service" \
  | sudo tee /etc/systemd/system/plr.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable plr.service

if command -v rpicam-hello >/dev/null; then
  echo; echo "Cameras found:"; rpicam-hello --list-cameras 2>&1 | sed 's/^/   /' | head -20 || true
fi

IP="$(hostname -I | awk '{print $1}')"
if [ "$REBOOT" -eq 1 ]; then
  say "Done - REBOOT NOW: sudo reboot   (plr.service starts by itself afterwards)"
else
  sudo systemctl restart plr.service
  sleep 5
  systemctl --no-pager --lines=8 status plr.service || true
  say "Done"
fi
echo "App 'Pi server'  : $IP:5000   (also shown at the bottom of the TFT)"
echo "Health check     : curl http://$IP:5000/health"
echo "Live IR view     : http://$IP:5000/stream.mjpg"
echo "PuReST status    : curl http://$IP:5000/purest"
echo "Logs             : journalctl -u plr.service -f"
