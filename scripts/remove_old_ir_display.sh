#!/usr/bin/env bash
# Remove the old stand-alone IR display (ir_display.py + ir-display.service,
# and the test.py LED script next to it) so the camera and the TFT are free
# for plr.service.
#
#   bash scripts/remove_old_ir_display.sh            # moves old files to ~/old-ir-display-<date>/
#   bash scripts/remove_old_ir_display.sh --delete   # deletes them instead
#
# Safe to run more than once. install.sh runs it automatically.
set -uo pipefail

MODE="backup"
[ "${1:-}" = "--delete" ] && MODE="delete"
HOME_DIR="${HOME}"
BACKUP="$HOME_DIR/old-ir-display-$(date +%Y%m%d-%H%M%S)"
FOUND=0

say()  { printf '\033[1;36m-- %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }

retire() {   # retire <path> [sudo]
  local path="$1" use_sudo="${2:-}"
  [ -e "$path" ] || return 0
  FOUND=1
  if [ "$MODE" = "delete" ]; then
    $use_sudo rm -rf -- "$path" && note "deleted  $path"
  else
    mkdir -p "$BACKUP"
    $use_sudo mv -- "$path" "$BACKUP/" && note "moved    $path -> $BACKUP/"
    [ -n "$use_sudo" ] && sudo chown -R "$(id -u):$(id -g)" "$BACKUP" 2>/dev/null
  fi
}

say "1. systemd services that run the old display"
units=()
[ -f /etc/systemd/system/ir-display.service ] && units+=("ir-display.service")
while IFS= read -r f; do
  units+=("$(basename "$f")")
done < <(grep -l -E "ir_display\.py" /etc/systemd/system/*.service 2>/dev/null)
mapfile -t units < <(printf '%s\n' "${units[@]}" | sort -u | sed '/^$/d')
for unit in "${units[@]}"; do
  FOUND=1
  note "stopping and disabling $unit"
  sudo systemctl stop "$unit" 2>/dev/null
  sudo systemctl disable "$unit" 2>/dev/null
  retire "/etc/systemd/system/$unit" sudo
done
# user-level units, just in case
for f in "$HOME_DIR"/.config/systemd/user/*.service; do
  [ -f "$f" ] && grep -q "ir_display" "$f" || continue
  systemctl --user disable --now "$(basename "$f")" 2>/dev/null
  retire "$f"
done
sudo systemctl daemon-reload
sudo systemctl reset-failed 2>/dev/null

say "2. running processes"
# Python processes whose script is ir_display.py, or the LED test.py from your
# home folder / the led-test folder. argv[0] must be a python interpreter, so
# editors, shells and this script are never touched.
py_pids() {   # py_pids <script-regex>
  local pid argv0 args
  for pid in $(pgrep -f "$1" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue
    argv0="$(tr '\0' '\n' </proc/"$pid"/cmdline 2>/dev/null | head -n1)"
    case "$(basename "$argv0")" in python|python3|python3.*) ;; *) continue ;; esac
    args="$(tr '\0' ' ' </proc/"$pid"/cmdline 2>/dev/null)"
    echo "$pid|$args"
  done
}
while IFS='|' read -r pid args; do
  [ -n "$pid" ] || continue
  FOUND=1
  note "killing $pid: $args"
  sudo kill "$pid" 2>/dev/null
done < <(py_pids "ir_display\.py")
while IFS='|' read -r pid args; do
  [ -n "$pid" ] || continue
  cwd="$(readlink /proc/"$pid"/cwd 2>/dev/null)"
  case "$args" in
    *"$HOME_DIR/test.py"*|*led-test/test.py*) ;;
    *" test.py"*) [ "$cwd" = "$HOME_DIR" ] || [[ "$cwd" == *led-test* ]] || continue ;;
    *) continue ;;
  esac
  FOUND=1
  note "killing $pid: $args"
  sudo kill "$pid" 2>/dev/null
done < <(py_pids "test\.py")

say "3. other ways it could be started (cron, rc.local, desktop autostart)"
user_cron="$(crontab -l 2>/dev/null || true)"
if grep -q "ir_display" <<<"$user_cron"; then
  FOUND=1
  { grep -v "ir_display" <<<"$user_cron" || true; } | crontab - && note "removed ir_display lines from your crontab"
fi
root_cron="$(sudo crontab -l 2>/dev/null || true)"
if grep -q "ir_display" <<<"$root_cron"; then
  FOUND=1
  { grep -v "ir_display" <<<"$root_cron" || true; } | sudo crontab - && note "removed ir_display lines from root's crontab"
fi
if [ -f /etc/rc.local ] && grep -q "ir_display" /etc/rc.local; then
  FOUND=1
  sudo cp /etc/rc.local /etc/rc.local.before-plr
  sudo sed -i '/ir_display/d' /etc/rc.local && note "removed ir_display lines from /etc/rc.local (backup: /etc/rc.local.before-plr)"
fi
for f in "$HOME_DIR"/.config/autostart/*.desktop; do
  [ -f "$f" ] && grep -q "ir_display" "$f" && retire "$f"
done

say "4. old files in $HOME_DIR"
for f in "$HOME_DIR/ir_display.py" "$HOME_DIR/test.py" "$HOME_DIR/ir-display.service" \
         "$HOME_DIR/Raspberry Pi IR camera to TFT display setup - Claude.html" "$HOME_DIR/led-test"; do
  retire "$f"
done
# the old service file pointed at /home/pi; clean that copy too if it is a different user
if [ "$HOME_DIR" != "/home/pi" ] && [ -f /home/pi/ir_display.py ]; then
  retire /home/pi/ir_display.py sudo
fi

say "5. is the camera free now?"
sleep 1
busy="$(py_pids "ir_display\.py"; pgrep -af "^(rpicam|libcamera)-(hello|vid|still|raw)" 2>/dev/null)"
if [ -n "$busy" ]; then
  echo "$busy" | sed 's/^/   /'
  note "something above still holds the camera - stop it before starting plr.service"
else
  note "yes - nothing else is using the camera"
fi

if [ "$FOUND" -eq 0 ]; then
  say "Nothing from the old IR display was found - already clean."
elif [ "$MODE" = "backup" ] && [ -d "$BACKUP" ]; then
  say "Done. Old files are in $BACKUP (delete that folder once everything works)."
else
  say "Done."
fi
