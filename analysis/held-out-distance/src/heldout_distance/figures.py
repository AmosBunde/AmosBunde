"""The figure: error against distance from the training data, three studies on one vertical scale."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.7), sharey=True, facecolor=t["bg"])
    panels = (
        ("Crop-type maps", "distance to the nearest training field (m)",
         [(s["crop"], t["fit"], "temporal CNN, 1 − macro-F1")]),
        ("Novel-view 3D", "angle to the nearest training camera (degrees)",
         [(s["views"], t["fit"], "appearance: image error"), (s["views"]["depth"], t["structure"], "geometry: depth error")]),
        ("Runtime prediction", "test workload size ÷ largest training workload",
         [(s["runtime"]["methods"]["gbm"], t["fit"], "gradient boosting"),
          (s["runtime"]["methods"]["loglinear"], t["other"], "log-linear regression"),
          (s["runtime"]["methods"]["sim"], t["structure"], "mechanistic simulator")]),
    )
    for ax, (title, xlabel, series) in zip(axes, panels):
        ax.set_facecolor(t["bg"])
        ax.axhline(1.0, color=t["muted"], lw=1, ls=(0, (4, 4)), zorder=1)
        for c, colour, label in series:
            _line(ax, c, colour, label, t)
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
    axes[0].set_ylabel("error, relative to the test items\nnearest the training data", fontsize=10)
    axes[0].set_ylim(0.5, 4.4)
    axes[0].set_yticks([1, 2, 3, 4], ["1×", "2×", "3×", "4×"])
    axes[0].set_xticks([30, 100, 300, 1000]), axes[1].set_xticks([10, 15, 20, 30]), axes[2].set_xticks([0.03, 1, 10, 100])
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.15, wspace=0.10)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"distance-{theme}.svg", facecolor=t["bg"])
    fig.savefig(OUT / f"distance-{theme}.png", dpi=120, facecolor=t["bg"])
    plt.close(fig)


def make_all() -> None:
    s = json.loads(Path("results/summary.json").read_text())
    for theme in THEMES:
        distance_figure(s, theme)
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("distance-*"))))
