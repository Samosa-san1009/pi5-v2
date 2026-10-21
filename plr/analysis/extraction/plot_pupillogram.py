#!/usr/bin/env python3
"""Plot complete recordings with stimulus-specific MCA and MCV selections."""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from .common import DEFAULT_OUTPUT, ROOT, number, read_csv, signal_data, smooth_paths

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"plr-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir())/"plr-cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STIM_COLORS = {"Red": "#bf3333", "Green": "#25833f", "Blue": "#315fc3", "Yellow": "#9d8300",
               "No Light": "#333333", "White": "#777777", "Purple": "#8d359a", "Indigo": "#50379d"}


def fmt(value, digits=2):
    value = number(value)
    return f"{value:.{digits}f}" if np.isfinite(value) else "—"


def plot_file(path):
    path = Path(path)
    data, meta = signal_data(path)
    eye = meta["eye"]
    mca = read_csv(path.with_name(f"{eye}_mca.csv"))
    mcv = read_csv(path.with_name(f"{eye}_mcv.csv"))
    diam, times = data["diameter_smoothed_px"], data["time_sec"]
    if not len(times):
        return None
    if not mca:
        fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
        observed = np.where(data["is_observed"] == 1, data["diameter_valid_px"], np.nan)
        ax.plot(times, observed, color="#a5adb5", lw=.8, label="Observed diameter")
        ax.plot(times, diam, color="#17486b", lw=1.4, label="Median + 1.5 Hz Butterworth")
        mask = data["is_interpolated"] == 1
        ax.scatter(times[mask], diam[mask], color="#b58123", s=15, zorder=4, label="Short-gap interpolation")
        ax.set(title=f"{meta['participant_id']} — {eye} eye — smoothed pupilogram\nNo flash timing supplied: MCA and MCV were not calculated",
               xlabel="Time from recording start (seconds)", ylabel="Pupil diameter (px)")
        ax.grid(alpha=.2)
        ax.legend(loc="best")
        destination = path.with_name(f"{eye}_pupillogram.png")
        fig.savefig(destination, dpi=130)
        plt.close(fig)
        return destination
    fig = plt.figure(figsize=(14, 9))
    grid = fig.add_gridspec(3, 1, height_ratios=(3.5, 3.5, 1.7), hspace=.35)
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
    axes[1].sharex(axes[0])
    table_ax = fig.add_subplot(grid[2])
    observed_values = np.where(data["is_observed"] == 1, data["diameter_valid_px"], np.nan)
    for ax in axes:
        ax.plot(times, observed_values, color="#b2b7bb", alpha=.65, linewidth=.6, label="Observed diameter")
        ax.plot(times, diam, color="#17486b", linewidth=1.15, label="Preprocessed diameter")
        inferred = (data["is_inferred"] == 1) | (data["is_interpolated"] == 1)
        ax.scatter(times[inferred], diam[inferred], color="#b58123", s=7, zorder=3,
                   label="Inferred / interpolated")
        ax.set_ylabel("Pupil diameter (px)")
        ax.grid(alpha=.2)
        ax.set_xlim(times[0], times[-1])
        for response in mca:
            onset = int(response["onset_frame"])
            offset = int(response["offset_frame"])
            if 0 <= onset < len(times):
                ax.axvline(times[onset], color=STIM_COLORS.get(response["stimulus"], "#315fc3"), linestyle="--", alpha=.65, linewidth=.9)
            if 0 <= offset < len(times):
                ax.axvline(times[offset], color="#888888", linestyle=":", alpha=.35, linewidth=.7)
    axes[0].set_title("MCA: local baseline to selected post-onset minimum", loc="left", fontsize=12)
    axes[1].set_title("MCV: peak contraction derivative (tangent); dashed segment = averaged velocity", loc="left", fontsize=12)
    seen_onsets, seen_selections = set(), set()
    for response, velocity in zip(mca, mcv):
        onset = int(response["onset_frame"])
        if 0 <= onset < len(times) and onset not in seen_onsets:
            label = " / ".join(r.get("event_id", r["stimulus"]) for r in mca if int(r["onset_frame"]) == onset)
            axes[0].text(times[onset]+.08, .96, label, transform=axes[0].get_xaxis_transform(),
                         color=STIM_COLORS.get(response["stimulus"], "#315fc3"), fontsize=9, fontweight="bold", va="top")
            seen_onsets.add(onset)
        minimum = number(response["minimum_frame"])
        baseline = number(response["baseline_px"])
        selection = (response["onset_frame"], response["minimum_frame"], response["baseline_px"], velocity["mcv_peak_frame"])
        if selection in seen_selections:
            continue
        seen_selections.add(selection)
        status = response["status"]
        color = "#25833f" if status == "stimulus_consistent" else "#b57912"
        if np.isfinite(minimum) and np.isfinite(baseline):
            index = int(minimum)
            start = int(response["baseline_start_frame"])
            axes[0].plot([times[start], times[onset-1]], [baseline, baseline], color=color, linewidth=2.2)
            axes[0].plot([times[onset], times[index]], [baseline, baseline], color=color, linewidth=.7, alpha=.5)
            axes[0].annotate("", xy=(times[index], diam[index]), xytext=(times[index], baseline),
                             arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.4})
            axes[0].scatter(times[index], diam[index], color=color, s=30, zorder=5)
            axes[0].annotate(f"{fmt(response['mca_px'])} px", (times[index], diam[index]),
                             xytext=(4, -13), textcoords="offset points", fontsize=9, color=color)
        peak = number(velocity["mcv_peak_frame"])
        if np.isfinite(peak):
            index = int(peak)
            rate = number(velocity["mcv_px_per_sec"])
            start, end = int(velocity["constriction_start_frame"]), int(velocity["minimum_frame"])
            radius = min(2, index-start, end-index)
            tangent_t = times[[index-radius, index+radius]]
            tangent_y = diam[index]-rate*(tangent_t-times[index])
            color = "#25833f" if velocity["status"] == "stimulus_consistent" else "#b57912"
            axes[1].plot(tangent_t, tangent_y, color=color, linewidth=3, zorder=5)
            axes[1].scatter(times[index], diam[index], s=32, color=color, zorder=6)
            axes[1].annotate(f"{rate:.2f} px/s", (times[index], diam[index]),
                             xytext=(4, 12), textcoords="offset points", fontsize=9, color=color)
            a, b = number(velocity["averaged_start_frame"]), number(velocity["averaged_end_frame"])
            if np.isfinite(a) and np.isfinite(b):
                axes[1].plot(times[[int(a), int(b)]], diam[[int(a), int(b)]], linestyle="--",
                             color="#cc6d19", linewidth=1.7, zorder=4)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[1].set_xlabel("Time from recording start (seconds)")
    columns = ["Stimulus", "Onset (s)", "MCA (px)", "MCA (%)", "MCV (px/s)", "Averaged MCV (px/s)", "MCA status", "MCV status"]
    cells = [[a.get("event_id", a["stimulus"]), fmt(a["onset_sec"]), fmt(a["mca_px"]), fmt(a["relative_constriction_percent"]),
              fmt(v["mcv_px_per_sec"]), fmt(v["mcv_averaged_px_per_sec"]),
              a["status"].replace("_", " "), v["status"].replace("_", " ")] for a, v in zip(mca, mcv)]
    table_ax.axis("off")
    table = table_ax.table(cellText=cells or [["No flash timings", *["—"]*7]], colLabels=columns, loc="center", cellLoc="center",
                           colWidths=[.10, .08, .10, .10, .10, .15, .185, .185])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d7dce0")
        if row == 0:
            cell.set_facecolor("#eaf0f4")
            cell.set_text_props(weight="bold")
    fig.suptitle(f"{meta['participant_id']} — {eye} eye — complete SoftFlow pupillogram", fontsize=14, y=.985)
    fig.subplots_adjust(left=.05, right=.985, top=.91, bottom=.07)
    fig.text(.05, .025, "Green: stimulus-consistent estimate. Amber: uncertain/no clear constriction. —: unmeasurable. "
             "Dashed: onset; dotted: offset. See CSVs for quality reasons.", fontsize=8)
    destination = path.with_name(f"{eye}_pupillogram.png")
    fig.savefig(destination, dpi=130)
    plt.close(fig)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in smooth_paths(args.output):
        print(plot_file(path), flush=True)


if __name__ == "__main__":
    main()
