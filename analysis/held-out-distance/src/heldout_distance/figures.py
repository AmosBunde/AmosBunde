"""The figure: error against distance from the training data, three studies on one vertical scale."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

OUT = Path("docs/img")
THEMES = {
    "light": dict(bg="#FFFFFF", ink="#1F2E43", muted="#5A6678", grid="#E3E7ED", fit="#BA3D1A", structure="#2F5D9B", other="#8792A2"),
    "dark": dict(bg="#0D1117", ink="#E6EDF3", muted="#9BA7B4", grid="#2A3441", fit="#EE8A66", structure="#7FA8DC", other="#8792A2"),
}


def _line(ax, c, colour, label, t, dx=1.0):
    x = [v * dx for v in c["bin_centre"]]
    lo, hi = c["relative_ci"]
    ax.fill_between(x, lo, hi, color=colour, alpha=0.16, linewidth=0)
    ax.plot(x, c["relative"], color=colour, lw=2.4, marker="o", ms=5.5, mec=t["bg"], mew=1.2, label=label, zorder=3)


def distance_figure(s: dict, theme: str) -> None:
    t = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "text.color": t["ink"], "axes.labelcolor": t["muted"],
                         "xtick.color": t["muted"], "ytick.color": t["muted"], "svg.fonttype": "path"})
    fig = plt.figure(figsize=(12.8, 4.8), facecolor=t["bg"])
    grid = fig.add_gridspec(1, 3, left=0.06, right=0.99, top=0.88, bottom=0.15, wspace=0.22)
    ax_crop = fig.add_subplot(grid[0])
    ax_view = fig.add_subplot(grid[1])
    ax_run = fig.add_subplot(grid[2], sharey=ax_view)
    models = s.get("structure_test", {}).get("models", {})

    # Crop panel: absolute macro-F1, so the level of each model is visible as well as its slope.
    for key, colour, label in (("tcn", t["fit"], "temporal CNN, raw series"), ("rf", t["other"], "forest, raw series"),
                               ("pheno", t["structure"], "forest, growth-curve parameters")):
        if key not in models:
            continue
        c = models[key]
        lo, hi = c["macro_f1_ci"]
        ax_crop.fill_between(c["bin_centre"], lo, hi, color=colour, alpha=0.16, linewidth=0)
        ax_crop.plot(c["bin_centre"], c["macro_f1"], color=colour, lw=2.4, marker="o", ms=5.5, mec=t["bg"], mew=1.2,
                     label=label, zorder=3)
    ax_crop.axhline(s["structure_test"]["majority_macro_f1"], color=t["muted"], lw=1, ls=(0, (4, 4)), zorder=1)
    ax_crop.text(1180, s["structure_test"]["majority_macro_f1"] + 0.006, "majority class", ha="right", fontsize=9,
                 color=t["muted"], style="italic")
    ax_crop.set_ylim(0.05, 0.40)
    ax_crop.set_ylabel("macro-F1 (higher is better)", fontsize=10)
    ax_crop.set_xticks([30, 100, 300, 1000])

    for c, colour, label in ((s["views"], t["fit"], "appearance: image error"), (s["views"]["depth"], t["structure"], "geometry: depth error")):
        _line(ax_view, c, colour, label, t)
    for c, colour, label in ((s["runtime"]["methods"]["gbm"], t["fit"], "gradient boosting"),
                             (s["runtime"]["methods"]["loglinear"], t["other"], "log-linear regression"),
                             (s["runtime"]["methods"]["sim"], t["structure"], "mechanistic simulator")):
        _line(ax_run, c, colour, label, t)
    for ax in (ax_view, ax_run):
        ax.axhline(1.0, color=t["muted"], lw=1, ls=(0, (4, 4)), zorder=1)
    ax_view.set_ylabel("error relative to the nearest test items\n(lower is better)", fontsize=10)
    ax_view.set_ylim(0.5, 4.4)
    ax_view.set_yticks([1, 2, 3, 4], ["1\u00d7", "2\u00d7", "3\u00d7", "4\u00d7"])
    ax_view.set_xticks([10, 15, 20, 30])
    plt.setp(ax_run.get_yticklabels(), visible=False)

    for ax, title, xlabel in ((ax_crop, "Crop-type maps", "distance to the nearest training field (m)"),
                              (ax_view, "Novel-view 3D", "angle to the nearest training camera (degrees)"),
                              (ax_run, "Runtime prediction", "test workload size \u00f7 largest training workload")):
        ax.set_facecolor(t["bg"])
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=t["ink"], pad=12)
        ax.grid(axis="y", color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(t["grid"])
        ax.tick_params(length=0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        leg = ax.legend(loc="upper left", fontsize=9.5, frameon=False, handlelength=1.6)
        for txt in leg.get_texts():
            txt.set_color(t["ink"])
    ax_crop.set_xticks([30, 100, 300, 1000]), ax_view.set_xticks([10, 15, 20, 30]), ax_run.set_xticks([0.03, 1, 10, 100])
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"distance-{theme}.svg", facecolor=t["bg"])
    fig.savefig(OUT / f"distance-{theme}.png", dpi=120, facecolor=t["bg"])
    plt.close(fig)


def make_all() -> None:
    s = json.loads(Path("results/summary.json").read_text())
    for theme in THEMES:
        distance_figure(s, theme)
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("distance-*"))))
