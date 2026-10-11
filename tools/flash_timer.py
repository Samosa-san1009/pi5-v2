#!/usr/bin/env python3
"""
flash_timer.py - Kernel-timestamped light-flash detection on a Raspberry Pi 5.

Works with any sensor that outputs a digital edge (TEMT6000 + LM393
comparator, photodiode + comparator, optical endstop module, ...).

Timing: lgpio "alerts" use the kernel GPIO character device, which stamps
each edge when it happens, so Python scheduling delays do not affect the
recorded times. The clock domain of the timestamps is detected at startup
and reported, so they can be lined up with time.monotonic_ns() elsewhere.

WS2812 PWM: below full brightness a WS2812 flickers at its internal PWM
rate, so one flash arrives as a burst of short pulses. Pulses separated by
less than --merge-gap-us are merged into a single flash.

Install:  sudo apt install python3-lgpio
Run:      python3 flash_timer.py --gpio 17 --active-low --csv flashes.csv
          (stop plr.service first: sudo systemctl stop plr.service - it owns
          GPIO17/27 while running; or use POST /selftest instead)
"""

import argparse
import csv
import signal
import statistics
import threading
import time

import lgpio


def open_header_chip():
    """Open the gpiochip driving the 40-pin header (pinctrl-rp1 on a Pi 5).

    Recent kernels expose it as gpiochip0; older Pi 5 kernels used gpiochip4.
    """
    for chip in (0, 4):
        try:
            h = lgpio.gpiochip_open(chip)
        except lgpio.error:
            continue
        # gpio_get_chip_info -> (status, lines, name, label); label is "pinctrl-rp1".
        info = lgpio.gpio_get_chip_info(h)
        if any("rp1" in str(part) for part in info[2:]):
            return h, chip
        lgpio.gpiochip_close(h)
    raise RuntimeError("Could not find the pinctrl-rp1 gpiochip (is this a Pi 5?)")


def detect_clock(ts_ns):
    """Guess which clock an lgpio timestamp came from."""
    candidates = {
        "monotonic": time.monotonic_ns(),
        "realtime": time.time_ns(),
        "boottime": time.clock_gettime_ns(time.CLOCK_BOOTTIME),
    }
    return min(candidates, key=lambda k: abs(candidates[k] - ts_ns))


class FlashTimer:
    def __init__(self, active_low, merge_gap_ns, writer):
        self.active_level = 0 if active_low else 1
        self.merge_gap_ns = merge_gap_ns
        self.writer = writer
        self.lock = threading.Lock()
        self.clock_name = None
        self.pulse_start = None      # start of the current raw pulse
        self.flash = None            # [onset, last_offset, pulse_count]
        self.flashes = []

    def on_edge(self, chip, gpio, level, ts_ns):
        if level == 2:               # watchdog timeout, not an edge
            return
        with self.lock:
            if self.clock_name is None:
                self.clock_name = detect_clock(ts_ns)
                print(f"Timestamps are in the {self.clock_name} clock domain")
            if level == self.active_level:
                self.pulse_start = ts_ns
                if self.flash and ts_ns - self.flash[1] > self.merge_gap_ns:
                    self._close_flash()
                if self.flash is None:
                    self.flash = [ts_ns, ts_ns, 0]
            elif self.pulse_start is not None and self.flash is not None:
                self.flash[1] = ts_ns
                self.flash[2] += 1
                self.pulse_start = None

    def poll(self):
        """Close a flash once no new pulse has started within the merge gap."""
        with self.lock:
            if self.flash and self.pulse_start is None:
                now = self._now()
                if now is not None and now - self.flash[1] > self.merge_gap_ns:
                    self._close_flash()

    def _now(self):
        return {
            "monotonic": time.monotonic_ns,
            "realtime": time.time_ns,
            "boottime": lambda: time.clock_gettime_ns(time.CLOCK_BOOTTIME),
        }.get(self.clock_name, lambda: None)()

    def _close_flash(self):
        onset, offset, pulses = self.flash
        width_us = (offset - onset) / 1000
        interval_ms = ((onset - self.flashes[-1][0]) / 1e6) if self.flashes else None
        self.flashes.append((onset, width_us))
        msg = f"flash #{len(self.flashes):4d}  width {width_us:10.1f} us  pulses {pulses:4d}"
        if interval_ms is not None:
            msg += f"  since prev {interval_ms:9.3f} ms"
        print(msg)
        if self.writer:
            self.writer.writerow([len(self.flashes), onset, offset, f"{width_us:.1f}",
                                  pulses, self.clock_name])
        self.flash = None

    def summary(self):
        if not self.flashes:
            print("No flashes recorded.")
            return
        widths = [w for _, w in self.flashes]
        print(f"\n{len(widths)} flashes  width mean {statistics.mean(widths):.1f} us"
              f"  min {min(widths):.1f}  max {max(widths):.1f}"
              + (f"  stdev {statistics.stdev(widths):.1f}" if len(widths) > 1 else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpio", type=int, default=17, help="BCM GPIO number (default 17)")
    p.add_argument("--active-low", action="store_true",
                   help="sensor output goes LOW when light is detected")
    p.add_argument("--pull", choices=["up", "down", "none"], default="none",
                   help="internal bias (use 'up' for open-collector LM393 outputs)")
    p.add_argument("--merge-gap-us", type=float, default=3000,
                   help="merge pulses closer than this into one flash (default 3000)")
    p.add_argument("--debounce-us", type=int, default=0,
                   help="kernel debounce; keep 0 for sub-ms timing")
    p.add_argument("--csv", help="append flashes to this CSV file")
    args = p.parse_args()

    h, chip = open_header_chip()
    flags = {"up": lgpio.SET_PULL_UP, "down": lgpio.SET_PULL_DOWN,
             "none": lgpio.SET_PULL_NONE}[args.pull]

    csv_file = writer = None
    if args.csv:
        csv_file = open(args.csv, "a", newline="")
        writer = csv.writer(csv_file)
        if csv_file.tell() == 0:
            writer.writerow(["flash", "onset_ns", "offset_ns", "width_us", "pulses", "clock"])

    timer = FlashTimer(args.active_low, int(args.merge_gap_us * 1000), writer)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    try:
        lgpio.gpio_claim_alert(h, args.gpio, lgpio.BOTH_EDGES, flags)
        if args.debounce_us:
            lgpio.gpio_set_debounce_micros(h, args.gpio, args.debounce_us)
        cb = lgpio.callback(h, args.gpio, lgpio.BOTH_EDGES, timer.on_edge)
        print(f"Listening on gpiochip{chip} GPIO{args.gpio} "
              f"(active {'LOW' if args.active_low else 'HIGH'}, idle level "
              f"{lgpio.gpio_read(h, args.gpio)}). Ctrl-C to stop.")
        while not stop.is_set():
            timer.poll()
            stop.wait(0.05)
        cb.cancel()
        timer.poll()
        timer.summary()
    finally:
        lgpio.gpio_free(h, args.gpio)
        lgpio.gpiochip_close(h)
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    main()
