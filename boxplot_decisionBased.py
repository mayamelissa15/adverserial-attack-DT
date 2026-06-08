# ~/swat/plot_decision_based.py
#
# Bar chart HSJA + RayS — médiane ± std sur 10 seeds
# Même style que plot_article.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("~/swat/results").expanduser()

df = pd.read_csv(RESULTS_DIR / "hsja_rays_final.csv")

MODELS = ["MLP", "LogReg", "XGBoost"]

COLORS = {
    "MLP":     "#DDEEFF",
    "LogReg":  "#E8E8E8",
    "XGBoost": "#FFE8CC",
}
EDGE = {
    "MLP":     "#5588BB",
    "LogReg":  "#888888",
    "XGBoost": "#CC7722",
}

attacks_db = ["HSJA", "RayS"]
x       = np.arange(len(attacks_db))
width   = 0.22
offsets = [-width, 0, width]

med = (df.groupby(["attack", "model"])["asr"]
         .median().reset_index().rename(columns={"asr": "med"}))
std = (df.groupby(["attack", "model"])["asr"]
         .std().reset_index().rename(columns={"asr": "std"}))
stats = med.merge(std, on=["attack", "model"])

fig, ax = plt.subplots(figsize=(5, 4.5))

for i, model in enumerate(MODELS):
    vals = [
        float(stats[(stats["attack"] == a) & (stats["model"] == model)]["med"].values[0])
        for a in attacks_db
    ]
    errs = [
        float(stats[(stats["attack"] == a) & (stats["model"] == model)]["std"].values[0])
        for a in attacks_db
    ]
    bars = ax.bar(x + offsets[i], vals, width,
                  label=model,
                  color=COLORS[model],
                  edgecolor=EDGE[model],
                  linewidth=0.8,
                  yerr=errs,
                  error_kw=dict(ecolor=EDGE[model], capsize=3, linewidth=0.9))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.022,
                f"{v:.0%}", ha="center", va="bottom",
                fontsize=7.5, color="#333333")

ax.set_xticks(x)
ax.set_xticklabels(attacks_db, fontsize=10)
ax.set_ylabel("Attack Success Rate (ASR)", fontsize=10)
ax.set_ylim(0, 0.70)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(fontsize=9, framealpha=0.5)
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.set_title("Decision-based attacks — Median ASR over 10 runs (ε = 0.1)", fontsize=11, pad=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "chart_decision_barchart.png", dpi=200, bbox_inches="tight")
plt.show()
print("✓ chart_decision_barchart.png sauvegardé")