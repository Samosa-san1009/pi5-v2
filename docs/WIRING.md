# Wiring – Raspberry Pi 5

Every pin the rig uses. GPIO numbers are BCM (what the software and
`/settings` use); "Pi pin" is the physical pin on the 40-pin header. Pin 1
has the square solder pad and sits at the end of the header furthest from the
USB ports. Odd pins (1, 3, 5 …) are the inner row; even pins (2, 4, 6 …) are
the row along the board edge.

## Rules

* **Power bus:** the 3.3 V, 5 V and GND of every part **except the screen**
  go to a perfboard bus with three rails. Each rail is fed from the Pi by one
  wire.
* **Screen:** the TFT's VCC and GND go **straight to the Pi**, not the bus.
* **Signals** always go straight from the part to its GPIO pin.
* **No RTC module.** Flash timing uses the kernel's monotonic clock, which
  needs no RTC. For dates without network, plug the official Pi 5 RTC battery
  into the board's **BAT** connector (J5) instead. Disconnect the old RTC
  module and its bus wires.

## Bus feeds (Pi → bus)

| Rail | Pi pin | Feeds |
|---|---|---|
| 3.3 V rail | 1 | both endstops |
| 5 V rail | 2 | both WS2812 LEDs |
| GND rail | 39 | both endstops, both WS2812 LEDs |

## Screen – GMT020-02 2.0" ST7789 TFT (7-pin, SPI0) – direct to the Pi

| Screen pin | Function | GPIO | Pi pin |
|---|---|---|---|
| VCC | 3.3 V power (also backlight) | – | 17 |
| GND | ground | – | 6 |
| SCL | SPI0 clock (SCLK) | GPIO11 | 23 |
| SDA | SPI0 data (MOSI) | GPIO10 | 19 |
| RES | reset | GPIO25 | 22 |
| DC | data / command | GPIO24 | 18 |
| CS | SPI0 chip select (CE0) | GPIO8 | 24 |

The screen used to take its 3.3 V from pin 1. Pin 1 now feeds the 3.3 V bus,
so **move the screen's VCC to pin 17** (the Pi's second 3.3 V pin). Pin 6
stays its ground. The screen is 3.3 V only – never put it on 5 V.

## Left eye

| Part | Part pin | Connects to | GPIO | Pi pin |
|---|---|---|---|---|
| Left WS2812 | DIN | Pi | GPIO18 | 12 |
| Left WS2812 | 5V / VCC | 5 V rail | – | (bus) |
| Left WS2812 | GND | GND rail | – | (bus) |
| Left endstop | S (signal) | Pi | GPIO17 | 11 |
| Left endstop | V / VCC | **3.3 V rail** | – | (bus) |
| Left endstop | G / GND | GND rail | – | (bus) |

## Right eye

| Part | Part pin | Connects to | GPIO | Pi pin |
|---|---|---|---|---|
| Right WS2812 | DIN | Pi | GPIO13 | 33 |
| Right WS2812 | 5V / VCC | 5 V rail | – | (bus) |
| Right WS2812 | GND | GND rail | – | (bus) |
| Right endstop | S (signal) | Pi | GPIO27 | 13 |
| Right endstop | V / VCC | **3.3 V rail** | – | (bus) |
| Right endstop | G / GND | GND rail | – | (bus) |

## Camera

IR camera on a CSI connector (CAM/DISP 0 or 1) with the ribbon cable. If two
cameras are connected, set `camera.index` (see the README).

## Full header map (what is in use)

| Pi pin | Use | | Pi pin | Use |
|---|---|---|---|---|
| 1 | 3.3 V → 3.3 V bus | | 2 | 5 V → 5 V bus |
| 3 | free (GPIO2) | | 4 | free (5 V) |
| 5 | free (GPIO3) | | 6 | GND → screen GND |
| 7 | free | | 8 | free |
| 9 | free (GND) | | 10 | free |
| 11 | left endstop S (GPIO17) | | 12 | left WS2812 DIN (GPIO18) |
| 13 | right endstop S (GPIO27) | | 14 | free (GND) |
| 15 | free | | 16 | free |
| 17 | 3.3 V → screen VCC | | 18 | screen DC (GPIO24) |
| 19 | screen SDA / MOSI (GPIO10) | | 20 | free (GND) |
| 21 | reserved (SPI0 MISO, GPIO9) | | 22 | screen RES (GPIO25) |
| 23 | screen SCL / SCLK (GPIO11) | | 24 | screen CS / CE0 (GPIO8) |
| 25 | free (GND) | | 26 | free |
| 27 | do not use (ID EEPROM) | | 28 | do not use (ID EEPROM) |
| 29 | free | | 30 | free (GND) |
| 31 | free | | 32 | free |
| 33 | right WS2812 DIN (GPIO13) | | 34 | free (GND) |
| 35 | free | | 36 | free |
| 37 | free | | 38 | free |
| 39 | GND → GND bus | | 40 | free |

## Checks before the first power-up

Do these with the Pi unplugged and a multimeter.

1. **No shorts on the bus:** continuity between the 3.3 V, 5 V and GND rails
   must be open in every combination.
2. **Endstops on 3.3 V only.** Each endstop pulls its S pin up to its own
   supply. On 5 V it would put 5 V on GPIO17/27 and can damage the Pi.
3. **First power-up with the endstop S wires unplugged:** measure each S pin
   to GND with the LEDs off and with a torch shining in. It must stay between
   0 and 3.3 V. Then power off and connect S to pins 11 and 13.
4. **WS2812 data wires short** (under ~10 cm). Ideally put a 330 Ω resistor in
   series with each DIN. The Pi drives DIN at 3.3 V, which a single WS2812 on
   5 V normally accepts with short wires. If an LED flickers or shows wrong
   colours, add a 74AHCT125 level shifter.
5. **Screen VCC on pin 17, not on the bus.**
