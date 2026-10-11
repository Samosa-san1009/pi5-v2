# pi5-v2 – Mobile PLR on Raspberry Pi 5

Pupillary light reflex (PLR) rig. The Pi 5 flashes a WS2812 LED into each
eye, measures the true moment of every flash with two optical endstops,
records both eyes with one IR camera (shown live on a small TFT), and tracks
both pupils with **PuReST, natively in C++, on both eyes at the same time**.
The existing phone app starts sessions and shows the results. The app needs
no changes: you only type the Pi's address into its "Pi server" field.

* Wiring and every pin: [`docs/WIRING.md`](docs/WIRING.md)
* What each file does: [File layout](#file-layout) at the end

## How PuReST runs (live + offline, both eyes in parallel)

```
camera ──► every uncompressed frame ──► plr_purest --stream ──► left eye thread  (core 2) ┐
   │                                     (live, while recording)  right eye thread (core 3) ┴► result at stop
   └─► H.264 full_video.mp4 ─────────► plr_purest --input ──────► left + right threads ───► offline result
                                         (after recording)
```

* **Live:** during the recording PuReST tracks both eyes on every raw camera
  frame (better than the compressed video). When recording stops, the result
  is ready almost at once.
* **Offline:** afterwards the saved video is analysed again, every frame,
  both eyes in parallel. This is the safety net if the live pass ever had to
  drop frames.
* **What the app gets** (`purest.report`, default `auto`): the live result if
  both eyes covered at least 98 % of the frames, otherwise the app waits for
  the offline result. Both results are always saved in the session folder.
* Between sessions the live engine keeps tracking at 5 fps so the TFT shows
  the pupils while you line up the rig.

PuReST is the only pupil model. If the app sends another `pupilAnalyzer`
value (e.g. `resnet18`), it is accepted and PuReST is used.

---

## Setup (do the steps in order)

| Step | Where | What |
|---|---|---|
| 1 | PC | Put this code on GitHub |
| 2 | Pi (powered off) | Wire the hardware |
| 3 | Pi | Update the OS |
| 4 | Pi | Get the code |
| 5 | Pi | Run the installer (it also removes the old IR display) |
| 6 | Pi | Reboot and check |
| 7 | Pi / browser | Line up and calibrate |
| 8 | Phone | Run a session from the app |

### Step 1 – Put the code on GitHub (PC, once)

You received `pi5-v2.bundle`, a git bundle with one clean commit. This
replaces everything in the GitHub repo (including the old zip file).

```bash
git clone pi5-v2.bundle pi5-v2
cd pi5-v2
git remote set-url origin https://github.com/Samosa-san1009/pi5-v2.git
git push --force origin main
```

`--force` is needed once, because the new history replaces the old two
commits. Afterwards, check on GitHub that the zip file is gone and `README.md`,
`server.py`, `plr/`, `native/`, `scripts/` are there.

### Step 2 – Wire the hardware (Pi powered off)

`sudo poweroff`, unplug the power, then follow [`docs/WIRING.md`](docs/WIRING.md).
In short:

| Part | Signal → Pi pin (GPIO) | Power |
|---|---|---|
| Screen (TFT) | SCL 23 (GPIO11), SDA 19 (GPIO10), RES 22 (GPIO25), DC 18 (GPIO24), CS 24 (GPIO8) | **direct to Pi:** VCC pin 17, GND pin 6 |
| Left WS2812 | DIN 12 (GPIO18) | 5 V bus, GND bus |
| Right WS2812 | DIN 33 (GPIO13) | 5 V bus, GND bus |
| Left endstop | S 11 (GPIO17) | 3.3 V bus, GND bus |
| Right endstop | S 13 (GPIO27) | 3.3 V bus, GND bus |
| Bus feeds | 3.3 V ← pin 1, 5 V ← pin 2, GND ← pin 39 | |
| IR camera | CSI ribbon | |

* **Move the screen's VCC from pin 1 to pin 17**, because pin 1 now feeds the bus.
* **Remove the RTC module.** It is not needed. For correct dates without
  Wi-Fi, use the Pi 5's BAT connector with the official RTC battery.
* Do the multimeter checks at the end of `docs/WIRING.md` before powering on.

### Step 3 – Update the OS (Pi)

Use the 64-bit Raspberry Pi OS that Raspberry Pi Imager recommends for the
Pi 5 (Trixie; Bookworm also works).

```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

After it comes back:

```bash
cat /proc/device-tree/model; echo   # Raspberry Pi 5 Model B ...
uname -m                            # aarch64
ls -l /dev/pio0                     # must exist (WS2812 driver)
rpicam-hello --list-cameras         # the IR camera must be listed
```

If `rpicam-hello` says the camera is busy, that is the old IR display. The
installer in step 5 removes it.

### Step 4 – Get the code (Pi)

```bash
mkdir -p ~/Documents && cd ~/Documents
git clone https://github.com/Samosa-san1009/pi5-v2.git
cd pi5-v2
```

(If `~/Documents/pi5-v2` already exists from the zip: `rm -rf ~/Documents/pi5-v2`
first, then clone.)

### Step 5 – Run the installer (Pi)

Run it as your normal user, **not** with `sudo` (it asks for your password
when it needs it):

```bash
cd ~/Documents/pi5-v2
bash scripts/install.sh
```

It takes 5–15 minutes and is safe to run again. In order, it:

1. **Removes the old IR display.** It stops, disables and deletes
   `ir-display.service`, stops the running `ir_display.py` / `test.py`, and
   removes start-up entries from cron, `rc.local` and desktop autostart.
   It also moves `ir_display.py`, `test.py`, `ir-display.service` and the
   `led-test` folder out of your home folder into
   `~/old-ir-display-<date>/`. Delete that folder once everything works. To
   run only this step: `bash scripts/remove_old_ir_display.sh` (add
   `--delete` to delete instead of moving).
2. Disables leftovers from earlier PLR installs (RTC sync timer, `pigpiod`).
3. Installs the APT packages (camera stack, lgpio, OpenCV, ffmpeg, cmake …).
4. Enables SPI for the TFT and comments out any old I2C RTC overlay in
   `/boot/firmware/config.txt` (a backup is kept).
5. Creates `.venv` and installs the few pip packages (TFT and WS2812 drivers).
6. Builds the PuReST engine `native/purest/build/plr_purest`.
7. Sets permissions for the camera, GPIO, SPI and `/dev/pio0`.
8. Installs and enables `plr.service`, which starts at every boot.

It must end with `== Done`. If it stops, the message names the failing step.
Fix that and run the same command again.

### Step 6 – Reboot and check (Pi)

```bash
sudo reboot
```

After about 30 s the TFT shows the IR picture, two coloured boxes (one per
eye) and, at the bottom, the address to type into the app (e.g.
`192.168.1.42:5000`).

```bash
systemctl status plr.service --no-pager      # active (running)
curl -s localhost:5000/health | python3 -m json.tool | head -40
```

In `/health` check that `"ok": true`, the camera has `"running": true`,
`leds.ready` is true, `endstops.ready` is true, and `purest.live.running` is
true.

LED and endstop test (each LED flashes 5 times, so look away from it):

```bash
curl -s -X POST localhost:5000/selftest | python3 -m json.tool
```

Both sides should report `"detected": "5/5"`.

### Step 7 – Line up and calibrate

Open `http://<pi-address>:5000/stream.mjpg` in a browser on the same network,
or watch the TFT.

1. **Which camera.** If `rpicam-hello --list-cameras` lists more than one,
   pick the IR one:
   `curl -X POST localhost:5000/settings -H 'Content-Type: application/json' -d '{"camera":{"index":1}}'`
   then `curl -X POST localhost:5000/restart`.
2. **Screen orientation.** If the picture is rotated, try `rotate` 0–3:
   `-d '{"display":{"rotate":3}}'`, then restart.
3. **Eye boxes (ROIs).** The blue box must contain the eye the **left** LED
   shines into, and the red box the eye the right LED shines into. Each box
   should hold the whole eye with some margin. Values are fractions of the
   picture `[x, y, width, height]`:
   `-d '{"eye_roi":{"left":[0.05,0.15,0.4,0.7],"right":[0.55,0.15,0.4,0.7]}}'`
   (applies immediately). If the eyes are swapped, swap the two boxes.
4. **Pupil tracking.** With someone at the rig, the TFT draws a green circle
   on each pupil. "lost" means PuReST cannot find it: check focus, IR
   illumination and the box.
5. **Pupil size limits (recommended after the first sessions).** Look at the
   pupil diameters in the app (pixels). If a trace ever jumps to a much larger
   value (the iris outline), set limits around your real pupil sizes, e.g.
   pupils 30–70 px → `-d '{"purest":{"min_pupil_px":15,"max_pupil_px":90}}'`.
6. **Balance the LEDs** if one eye's LED looks brighter:
   `-d '{"leds":{"left":{"power_pct":85}}}'`.
7. **Check the live engine keeps up:** `curl -s localhost:5000/purest`.
   `processing_time` per eye should be well below the frame time (33 ms at
   30 fps), and `dropped_frames` should stay at 0.

Settings are saved in `device_settings.json` (not in git, so `git pull`
never overwrites them). `POST /settings/reset` restores the defaults.

### Step 8 – Run a session from the app

1. Phone on the same network as the Pi.
2. In the app's Config screen, set **Pi server** to the address shown on the
   TFT (e.g. `192.168.1.42:5000`). That is the only change from before.
3. Start the session as usual. The app shows the pupilogram and per-flash
   MCA / MCV / latency, all in pixels.

---

## Daily use

Power on the Pi, wait for the TFT picture, use the app. Nothing else needs
to be started.

* Live view: `http://<pi>:5000/stream.mjpg`
* Logs: `journalctl -u plr.service -f`
* Sessions: `~/Documents/pi5-v2/sessions/<date>_<name>/` (video, frame
  timestamps, flash timing, `pupilogram*.json`, `metrics*.json`, `session.log`)
* Restart the service: `sudo systemctl restart plr.service`

## Updating

On the PC, commit and push changes to GitHub as usual. Then on the Pi:

```bash
cd ~/Documents/pi5-v2
bash scripts/update.sh        # git pull + re-run the installer
```

Your settings and recorded sessions are kept.

---

## Reference

### Endpoints

| Method | Path | Use |
|---|---|---|
| GET | `/health` | everything at a glance |
| POST | `/session` | start a session (what the app sends) |
| GET | `/status` | stage: idle → starting → recording → analysis → done / error |
| GET | `/results` | result of the current session (what the app shows) |
| GET | `/video` | current session's video |
| POST | `/session/stop` | abort the running session |
| GET / POST | `/settings` | read / change settings (partial JSON) |
| POST | `/settings/reset` | defaults |
| GET | `/purest` | live engine: per-eye processing time, dropped frames, background job |
| GET | `/preview.jpg`, `/stream.mjpg` | live IR picture |
| POST | `/leds/test` | `{"side":"left","hex":"#FFFFFF","intensity":50,"duration":0.2}` |
| POST | `/selftest` | flash each LED, check the endstops see it |
| GET | `/sessions` | recorded sessions |
| GET | `/sessions/<id>/results?source=live\|offline` | either pass of any session |
| POST | `/sessions/<id>/reanalyze` | run offline PuReST again (`{"promote": true}` makes it the main result) |
| GET | `/time` | clock / NTP status |
| POST | `/restart` | restart the service (needed after camera, pin or display changes) |

### Settings (main ones)

| Setting | Default | Meaning |
|---|---|---|
| `purest.live` / `purest.offline` | `true` / `true` | run each pass |
| `purest.report` | `auto` | `auto`, `live` or `offline`: which result the app gets |
| `purest.live_min_coverage` | `0.98` | fraction of frames the live pass must cover per eye for `auto` |
| `purest.cpus` | `[2, 3]` | cores for the two eye threads |
| `purest.min_pupil_px` / `max_pupil_px` | none | pupil size limits in camera pixels |
| `purest.confidence_threshold` | `0.70` | lower-confidence samples are marked invalid |
| `purest.idle_preview_fps` | `5` | tracking rate between sessions (TFT circles); `0` = off |
| `eye_roi.left` / `eye_roi.right` | halves | each eye's box in the picture |
| `leds.<side>.gpio` / `power_pct` | 18 / 13, 100 | WS2812 pins and per-eye brightness |
| `endstops.<side>.gpio` | 17 / 27 | endstop signal pins |
| `camera.index`, `resolution`, `fps` | auto, 640×480, 30 | camera |
| `display.rotate` | 1 | TFT orientation |
| `session.pre_roll_s` / `post_roll_s` | 1 / 3 | recording before the first / after the last flash |

### Session files

| File | Content |
|---|---|
| `full_video.mp4` | H.264 recording |
| `frames.csv` | sensor timestamp of every frame |
| `events.json` | every flash: optical onset/offset, latency, source |
| `endstop_edges.csv` | raw endstop edges |
| `pupilogram.json` / `metrics.json` | what the app received |
| `*_live.json` / `*_offline.json` | each PuReST pass on its own |
| `manifest.json` | request, settings and hardware state at the start |
| `session.log` | log of the session |

### Troubleshooting

| Symptom | Fix |
|---|---|
| App: "Could not start the session" | Is the address the one on the TFT? Same network? `curl http://<pi>:5000/health` from a laptop. |
| `/health` camera not running | Cable; `rpicam-hello --list-cameras`; is anything else using the camera? Run `bash scripts/remove_old_ir_display.sh` again. |
| LEDs not ready / `/dev/pio0` missing | `sudo apt full-upgrade`, reboot, re-run the installer. |
| Endstop "reads ACTIVE with LEDs off" | Ambient light reaches the sensor, or the polarity is wrong: set `endstops.<side>.active` to `low`/`high`. |
| TFT stays black | VCC on pin 17? SPI enabled (re-run the installer, reboot)? Try other `display.rotate` values. |
| Trace jumps to iris size | Set `purest.max_pupil_px` (step 7.5) and tighten the eye boxes. |
| App result came from `offline` and took longer | The live pass dropped frames: check `/purest` and keep other programs off the Pi. |

### Testing without the rig (PC or Pi)

```bash
sudo apt install cmake g++ libopencv-dev ffmpeg python3-pytest python3-flask python3-numpy python3-pil
bash scripts/build_native.sh
python3 -m pytest tests -q
PLR_HARDWARE=mock python3 server.py      # simulated eyes, LEDs and endstops on port 5000
```

---

## File layout

```
server.py                    HTTP server: app endpoints, session flow (live → offline)
plr/
  settings.py                defaults (pins, PuReST, eye boxes), validation, device_settings.json
  config_adapter.py          translates the app's /session request
  camera.py                  always-on camera; feeds the live engine, records H.264; mock camera
  purest.py                  live engine (persistent native process) + offline runner
  orchestrator.py            one recording: LED schedule, optical flash timeline, frames.csv
  leds.py                    two WS2812 LEDs via the Pi 5 PIO driver
  endstops.py                kernel-timestamped optical flash detection (lgpio)
  display.py                 TFT live view with eye boxes and pupil circles
  timebase.py                clocks (monotonic for timing, NTP / Pi RTC for dates)
  analysis/results.py        pupilogram + metrics JSON (the shape the app reads)
  analysis/plr_metrics.py    baseline, latency, MCA, MCV
native/purest/
  src/plr_purest.cpp         PuReST engine: both eyes in parallel, --stream and --input modes
  third_party/...            PuRe / PuReST by Santini et al. (non-commercial licence)
scripts/
  install.sh                 one-shot installer (safe to re-run)
  update.sh                  git pull + install
  remove_old_ir_display.sh   removes ir_display.py / ir-display.service / test.py
  build_native.sh            builds plr_purest
  plr.service                systemd unit template
tools/flash_timer.py         stand-alone endstop timing tool (stop plr.service first)
tests/                       hardware-free tests (pytest)
docs/WIRING.md               full pin map and power bus
```

## Licence note

PuRe and PuReST (Thiago Santini et al., `native/purest/third_party/`) are
licensed for **non-commercial use only**. See the licence files there and
cite the papers in publications.
