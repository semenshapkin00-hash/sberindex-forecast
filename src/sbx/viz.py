"""Единый стиль графиков отчёта (matplotlib → SVG). Палитра проверена на различимость при
нарушениях цветовосприятия; категориальные цвета назначаются в фиксированном порядке."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#c3c2b7"           # нейтральный для «прочих» моделей
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#b9302f", "#e34948", "#f08e8d", "#f0efec", "#86b6ef", "#2a78d6", "#184f95"]  # хуже ← 0 → лучше


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.labelcolor": INK2, "lines.linewidth": 2.0, "svg.fonttype": "path",
    })


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=160)
    fig.savefig(str(path).replace(".svg", ".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)
