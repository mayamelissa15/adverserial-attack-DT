# ~/swat/plot_whitebox_multirun.py
"""
Génère les boxplots whitebox pour l'article à partir de
whitebox_multirun_results.csv produit par run_whitebox_multirun.py.

Figures produites :
  chart_whitebox_boxplot.png  — boxplot ASR par attaque et modèle
  chart_whitebox_f1.png       — évolution F1 clean → adv (barres groupées)
  chart_whitebox_recall.png   — recall adversarial par attaque et modèle
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("~/swat/results").expanduser()

# ══════════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════════

df = pd.read_csv(RESULTS_DIR / "whitebox_multirun_results.csv")

MODELS  = ["MLP", "LogReg", "XGBoost"]
ATTACKS = ["FGSM", "PGD", "C&W"]

# Palette cohérente avec plot_article.py
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
MEDIAN_COLOR = {
    "MLP":     "#2255AA",
    "LogReg":  "#555555",
    "XGBoost": "#AA5500",
}


# ══════════════════════════════════════════════════════════════
# FIGURE 1 : BOXPLOT ASR
# ══════════════════════════════════════════════════════════════

def plot_whitebox_boxplot(df, filename="chart_whitebox_boxplot.png"):
    """
    Un groupe de 3 boîtes (MLP / LogReg / XGBoost) par attaque.
    Les boîtes montrent la distribution sur N_RUNS seeds.
    """
    n_attacks = len(ATTACKS)
    n_models  = len(MODELS)
    width     = 0.22
    offsets   = np.linspace(-(n_models - 1) * width / 2,
                             (n_models - 1) * width / 2,
                             n_models)
    x = np.arange(n_attacks)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, model in enumerate(MODELS):
        sub   = df[df["model"] == model]
        data  = [sub[sub["attack"] == atk]["asr"].values * 100
                 for atk in ATTACKS]

        bp = ax.boxplot(
            data,
            positions=x + offsets[i],
            widths=width * 0.85,
            patch_artist=True,
            notch=False,
            showfliers=True,
            medianprops=dict(color=MEDIAN_COLOR[model], linewidth=2),
            boxprops=dict(facecolor=COLORS[model],
                          edgecolor=EDGE[model], linewidth=0.9),
            whiskerprops=dict(color=EDGE[model], linewidth=0.9),
            capprops=dict(color=EDGE[model], linewidth=0.9),
            flierprops=dict(marker="o", markersize=3,
                            markerfacecolor=EDGE[model],
                            markeredgecolor=EDGE[model], alpha=0.6),
        )

        # Médiane en chiffre au dessus de la boîte
        for pos, vals in zip(x + offsets[i], data):
            if len(vals):
                med = np.median(vals)
                ax.text(pos, np.percentile(vals, 75) + 1.2,
                        f"{med:.0f}%",
                        ha="center", va="bottom", fontsize=7,
                        color=MEDIAN_COLOR[model])

    ax.set_xticks(x)
    ax.set_xticklabels(ATTACKS, fontsize=11)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=10)
    ax.set_ylim(-5, 115)
    ax.set_title(f"Whitebox attacks — ASR distribution over {df['seed'].nunique()} runs (ε = 0.1)",
                 fontsize=11, pad=10)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_patches = [
        mpatches.Patch(facecolor=COLORS[m], edgecolor=EDGE[m], label=m)
        for m in MODELS
    ]
    ax.legend(handles=legend_patches, fontsize=9, framealpha=0.5)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"✓ {filename} sauvegardé")


# ══════════════════════════════════════════════════════════════
# FIGURE 2 : F1 CLEAN → ADV (barres groupées)
# ══════════════════════════════════════════════════════════════

def plot_f1_drop(df, filename="chart_whitebox_f1.png"):
    """
    Pour chaque (attaque, modèle) : barre empilée F1_adv + drop F1.
    Montre visuellement la chute de F1 due à l'attaque.
    """
    x       = np.arange(len(ATTACKS))
    width   = 0.22
    offsets = np.linspace(-(len(MODELS) - 1) * width / 2,
                           (len(MODELS) - 1) * width / 2,
                           len(MODELS))

    # Médiane sur les seeds
    agg = df.groupby(["attack", "model"])[["f1_clean", "f1_adv"]].median()

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, model in enumerate(MODELS):
        f1_clean = [float(agg.loc[(atk, model), "f1_clean"]) for atk in ATTACKS]
        f1_adv   = [float(agg.loc[(atk, model), "f1_adv"])   for atk in ATTACKS]
        drop     = [c - a for c, a in zip(f1_clean, f1_adv)]

        # Partie conservée (f1_adv)
        ax.bar(x + offsets[i], f1_adv, width,
               color=COLORS[model], edgecolor=EDGE[model], linewidth=0.8,
               label=f"{model} (adv)")
        # Drop (en hachuré)
        ax.bar(x + offsets[i], drop, width, bottom=f1_adv,
               color=EDGE[model], alpha=0.35, edgecolor=EDGE[model],
               linewidth=0.6, hatch="///")

        # Valeurs
        for j, (pos, adv, cl) in enumerate(zip(x + offsets[i], f1_adv, f1_clean)):
            ax.text(pos, cl + 0.008, f"{adv:.2f}",
                    ha="center", va="bottom", fontsize=7,
                    color=MEDIAN_COLOR[model])

    ax.set_xticks(x)
    ax.set_xticklabels(ATTACKS, fontsize=11)
    ax.set_ylabel("F1 score (médiane)", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("F1 clean → adversarial — Whitebox attacks (ε = 0.1)",
                 fontsize=11, pad=10)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Légende : couleur pleine = F1_adv, hachuré = drop
    legend_patches  = [mpatches.Patch(facecolor=COLORS[m], edgecolor=EDGE[m],
                                      label=m) for m in MODELS]
    hatch_patch = mpatches.Patch(facecolor="gray", alpha=0.35,
                                 hatch="///", edgecolor="gray",
                                 label="F1 drop (attaque)")
    ax.legend(handles=legend_patches + [hatch_patch],
              fontsize=9, framealpha=0.5)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"✓ {filename} sauvegardé")


# ══════════════════════════════════════════════════════════════
# FIGURE 3 : RECALL ADVERSARIAL (barres + erreur)
# ══════════════════════════════════════════════════════════════

def plot_recall_adv(df, filename="chart_whitebox_recall.png"):
    """
    Recall adversarial médian avec barres d'erreur (IQR ou std).
    Le recall est la métrique clé pour un système de détection d'intrusion :
    un recall faible = l'attaquant passe inaperçu.
    """
    x       = np.arange(len(ATTACKS))
    width   = 0.22
    offsets = np.linspace(-(len(MODELS) - 1) * width / 2,
                           (len(MODELS) - 1) * width / 2,
                           len(MODELS))

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, model in enumerate(MODELS):
        sub  = df[df["model"] == model]
        meds = [sub[sub["attack"] == atk]["rec_adv"].median() for atk in ATTACKS]
        errs = [sub[sub["attack"] == atk]["rec_adv"].std()    for atk in ATTACKS]

        bars = ax.bar(x + offsets[i], meds, width,
                      yerr=errs, capsize=3,
                      color=COLORS[model], edgecolor=EDGE[model],
                      linewidth=0.8, label=model,
                      error_kw=dict(elinewidth=0.8, ecolor=EDGE[model]))

        for bar, v in zip(bars, meds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.018,
                    f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7,
                    color=MEDIAN_COLOR[model])

    ax.set_xticks(x)
    ax.set_xticklabels(ATTACKS, fontsize=11)
    ax.set_ylabel("Recall adversarial (médiane ± std)", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("Recall post-attaque — Whitebox attacks (ε = 0.1)",
                 fontsize=11, pad=10)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.5)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"✓ {filename} sauvegardé")


# ══════════════════════════════════════════════════════════════
# RÉSUMÉ VALEURS ARTICLE
# ══════════════════════════════════════════════════════════════

def print_article_values(df):
    print("\n=== VALEURS POUR L'ARTICLE (whitebox multi-run) ===\n")
    print(f"{'Attaque':<8} {'Modèle':<10} "
          f"{'ASR med':>8} {'ASR std':>8} "
          f"{'F1 adv':>8} {'Recall':>8}")
    print("─" * 60)

    for atk in ATTACKS:
        for model in MODELS:
            sub = df[(df["attack"] == atk) & (df["model"] == model)]
            if sub.empty:
                continue
            print(f"{atk:<8} {model:<10} "
                  f"{sub['asr'].median()*100:>7.1f}% "
                  f"{sub['asr'].std()*100:>7.1f}% "
                  f"{sub['f1_adv'].median():>8.3f} "
                  f"{sub['rec_adv'].median():>8.3f}")
        print()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Génération des figures whitebox multi-run...")

    plot_whitebox_boxplot(df)
    plot_f1_drop(df)
    plot_recall_adv(df)
    print_article_values(df)

    print(f"\n✓ Figures sauvegardées dans {RESULTS_DIR}")