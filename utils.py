import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# ------ PATHS ---
DATA_DIR = Path(__file__).resolve().parent / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
PLOT_DIR = DATA_DIR / "plots"

INTERVENTIONS_DIR = DATA_DIR / "interventions"
NEURON_INTERVENTIONS_DIR = INTERVENTIONS_DIR / "neurons"
ATTENTION_INTERVENTIONS_DIR = INTERVENTIONS_DIR / "attention"


for p in [DATA_DIR, NEURON_INTERVENTIONS_DIR,
          ANALYSIS_DIR, ATTENTION_INTERVENTIONS_DIR,
          PLOT_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ------ PLOTS ------
def save_plot(filename: str | Path, 
              filetype="png", 
              folder_path=PLOT_DIR):
    folder_path.mkdir(exist_ok=True, parents=True)
    plt.savefig(folder_path / f"{filename}.{filetype}",
                dpi=600,
                bbox_inches="tight")


FONT_SIZES = {
    "PLOT_TITLE": 11,
    "LEGEND": 9.0,
    "TICK": 8.5,
    "AX_LABEL": 10,
}

PALETTE = [
    "#ff4530",
    "#00BFC4",
    "#D69724",
    "#87BC45",
    "#EDE15B",
    "#C51B8A",
    "#665191",
    "#27AEEF",
    "#B33DC6",
    "#3A0CA3",
]


def set_style(font_family="Palatino Linotype", font_scale=1.2, font_offset=0):
    global FONT_SIZES

    FONT_SIZES = {k: v + font_offset
                  for k, v in FONT_SIZES.items()}

    plt.rcParams["font.family"] = font_family
    sns.set_theme(
        style="whitegrid",
        font_scale=font_scale,
        palette=PALETTE,
        context="notebook",
        rc={
            "axes.edgecolor": "#1F1709",
            "axes.labelcolor": "#1F1709",
            "axes.labelpad": 10,
            "axes.linewidth": 0.6,
            "xtick.color": "#1F1709",
            "ytick.color": "#1F1709",
            "text.color": "#1F1709",
            # "grid.color": COLOURS["TEXT_BROWN"],
            "grid.alpha": 0.1,
            "grid.linewidth": 0.8,
            "legend.title_fontsize": FONT_SIZES["LEGEND"],
            "legend.fontsize": FONT_SIZES["LEGEND"],
            "legend.labelcolor": "#1F1709",
            "axes.labelsize": FONT_SIZES["AX_LABEL"],
            "axes.titlesize": FONT_SIZES["LEGEND"],  # plot title
            "xtick.labelsize": FONT_SIZES["TICK"],  # x tick labels
            "ytick.labelsize": FONT_SIZES["TICK"],  # y tick labels
            "figure.titlesize": FONT_SIZES["PLOT_TITLE"],
            "axes.titlesize": FONT_SIZES["PLOT_TITLE"],
            "font.family": font_family,
        },
    )
