# whitebox_multirun.py
"""
Whitebox multi-run — SWaT et BATADAL, eps 0.1 et 0.3.

Usage :
  python whitebox_multirun.py --dataset swat   --eps 0.1
  python whitebox_multirun.py --dataset swat   --eps 0.3
  python whitebox_multirun.py --dataset batadal --eps 0.1
  python whitebox_multirun.py --dataset batadal --eps 0.3

Sorties :
  ~/<dataset>/results/whitebox_multirun_<dataset>_eps<eps>.csv
  ~/<dataset>/results/whitebox_multirun_<dataset>_eps<eps>.json
  ~/<dataset>/results/whitebox_multirun_<dataset>_eps<eps>_tmp.csv  (checkpoint)
"""

import argparse
import numpy as np
import torch
import joblib
import pandas as pd
import json
import warnings
from pathlib import Path
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

import sys
sys.path.append(str(Path(__file__).parent))

from models import (MLP, MLPWrapper, LogRegWrapper, XGBoostWrapper,
                    build_eval_set, eval_attack)
from whitebox import (
    fgsm_mlp, fgsm_logreg, fgsm_xgb,
    pgd_mlp,  pgd_logreg,  pgd_xgb,
    cw_mlp,   cw_logreg,   cw_xgb,
    THRESHOLD_LOGIT, EPS_FD, CW_LR_XGB, CW_LR_LR,
    PGD_ALPHA_K,
)

# ══════════════════════════════════════════════════════════════
# ARGUMENTS
# ══════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",  default="swat",
                    choices=["swat", "batadal"],
                    help="Dataset cible")
parser.add_argument("--eps",      default=0.1,  type=float,
                    help="Epsilon L∞ (0.1 ou 0.3)")
parser.add_argument("--n_runs",   default=10,   type=int,
                    help="Nombre de seeds")
parser.add_argument("--fast",     action="store_true",
                    help="FAST_MODE : PGD 50×3 au lieu de 200×10")
args = parser.parse_args()

DATASET  = args.dataset
EPS      = args.eps
N_RUNS   = args.n_runs
FAST     = args.fast

# ── Chemins selon dataset ──────────────────────────────────────
SAVE_DIR    = Path(f"~/{DATASET}/artifacts").expanduser()
RESULTS_DIR = Path(f"~/{DATASET}/results").expanduser()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TAG = f"{DATASET}_eps{EPS}"   # utilisé dans tous les noms de fichiers

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS  = list(range(N_RUNS))

# ── Hyperparamètres PGD / C&W selon FAST_MODE ─────────────────
PGD_ITERS    = 50  if FAST else 200
PGD_RESTARTS = 3   if FAST else 10
CW_ITERS     = 150 if FAST else 500

# ── Taille eval set selon dataset ─────────────────────────────
# SWaT  : beaucoup d'attaques → 500
# BATADAL : seulement 219 attaques en test → 200 max
EVAL_ATK_SIZE = 200 if DATASET == "batadal" else 500
EVAL_NRM_SIZE = 500

print(f"\n{'═'*55}")
print(f"  Dataset  : {DATASET.upper()}")
print(f"  Epsilon  : {EPS}")
print(f"  N_RUNS   : {N_RUNS}")
print(f"  Device   : {DEVICE}")
print(f"  FAST     : {FAST}")
print(f"  PGD      : {PGD_ITERS} iters × {PGD_RESTARTS} restarts")
print(f"  C&W      : {CW_ITERS} iters")
print(f"  Eval atk : {EVAL_ATK_SIZE} exemples max")
print(f"  Sorties  : {RESULTS_DIR}")
print(f"{'═'*55}")


# ══════════════════════════════════════════════════════════════
# CHARGEMENT DES VICTIMES
# ══════════════════════════════════════════════════════════════

def load_victims():
    X_test = np.load(SAVE_DIR / "X_test.npy")
    y_test = np.load(SAVE_DIR / "y_test.npy")

    mlp_model = MLP(input_size=X_test.shape[1]).to(DEVICE)
    mlp_model.load_state_dict(
        torch.load(SAVE_DIR / "best_mlp.pt", map_location=DEVICE))
    mlp_model.eval()
    mlp_w = MLPWrapper(mlp_model, DEVICE)

    logreg_w = LogRegWrapper(joblib.load(SAVE_DIR / "logreg.pkl"))

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(SAVE_DIR / "xgb.json"))
    xgb_w = XGBoostWrapper(xgb_model)

    print(f"\n✓ Modèles chargés depuis {SAVE_DIR}")
    print(f"  X_test : {X_test.shape} — attaques : {y_test.sum()} / {len(y_test)}")

    return X_test, y_test, mlp_w, logreg_w, xgb_w


# ══════════════════════════════════════════════════════════════
# EVAL SET COMMUN AUX 3 MODÈLES
# ══════════════════════════════════════════════════════════════

def build_shared_eval(X_test, y_test, mlp_w, logreg_w, xgb_w, seed):
    """
    Sélectionne un eval set commun : normaux + TP détectés correctement
    par les 3 modèles simultanément.

    EVAL_ATK_SIZE est adapté au dataset (200 pour BATADAL, 500 pour SWaT)
    pour éviter replace=True sur un petit pool d'attaques.
    """
    rng        = np.random.default_rng(seed)
    idx_normal = np.where(y_test == 0)[0]
    idx_attack = np.where(y_test == 1)[0]

    preds_mlp    = mlp_w.predict(X_test[idx_attack])
    preds_logreg = logreg_w.predict(X_test[idx_attack])
    preds_xgb    = xgb_w.predict(X_test[idx_attack])
    ok_mask      = (preds_mlp == 1) & (preds_logreg == 1) & (preds_xgb == 1)
    idx_attack_ok = idx_attack[ok_mask]

    n_atk = min(EVAL_ATK_SIZE, len(idx_attack_ok))
    n_nrm = min(EVAL_NRM_SIZE, len(idx_normal))

    if n_atk == 0:
        raise ValueError(
            f"Aucun TP commun aux 3 modèles sur {DATASET} ! "
            "Vérifier les F1 des modèles entraînés.")

    sel_n  = rng.choice(idx_normal,    size=n_nrm, replace=False)
    sel_a  = rng.choice(idx_attack_ok, size=n_atk, replace=False)
    idx_ev = np.concatenate([sel_n, sel_a])
    rng.shuffle(idx_ev)

    X_eval = X_test[idx_ev]
    y_eval = y_test[idx_ev]
    mask   = (y_eval == 1)
    X_atk  = X_eval[mask].astype(np.float32)
    y_atk  = y_eval[mask]

    return X_eval, y_eval, X_atk, y_atk


# ══════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE
# ══════════════════════════════════════════════════════════════

def run():
    X_test, y_test, mlp_w, logreg_w, xgb_w = load_victims()

    victims = [
        ("MLP",     mlp_w,    False, False),
        ("LogReg",  logreg_w, True,  False),
        ("XGBoost", xgb_w,    False, True),
    ]

    all_results = []

    for seed in SEEDS:
        print(f"\n{'═'*55}")
        print(f"  SEED {seed+1}/{N_RUNS}  —  {DATASET.upper()}  eps={EPS}")
        print(f"{'═'*55}")

        set_all_seeds(seed)
        X_eval, y_eval, X_atk, y_atk = build_shared_eval(
            X_test, y_test, mlp_w, logreg_w, xgb_w, seed=seed)

        print(f"  Eval set : {len(X_eval)} exemples "
              f"({(y_eval==1).sum()} attaques, {(y_eval==0).sum()} normaux)")

        for vic_name, vic_w, is_lr, is_xgb in victims:
            print(f"\n  ── {vic_name} {'─'*(45-len(vic_name))}")

            # ── FGSM ──────────────────────────────────────────
            print("  [FGSM]")
            set_all_seeds(seed)
            if is_lr:
                X_adv = fgsm_logreg(vic_w, X_atk, y_atk, EPS)
            elif is_xgb:
                X_adv = fgsm_xgb(vic_w, X_atk, y_atk, EPS)
            else:
                X_adv = fgsm_mlp(vic_w, X_atk, y_atk, EPS)
            r = eval_attack(vic_w, X_eval, y_eval, X_adv, "FGSM", vic_name)
            r.update({"seed": seed, "family": "Whitebox",
                      "eps": EPS, "dataset": DATASET})
            all_results.append(r)
            print(f"    ASR={r['asr']*100:.1f}%  "
                  f"F1 {r['f1_clean']:.3f}→{r['f1_adv']:.3f}")

            # ── PGD ───────────────────────────────────────────
            print("  [PGD]")
            set_all_seeds(seed)
            alpha = EPS / PGD_ALPHA_K
            if is_lr:
                X_adv = pgd_logreg(vic_w, X_atk, y_atk, EPS,
                                   iters=PGD_ITERS, restarts=PGD_RESTARTS,
                                   alpha=alpha)
            elif is_xgb:
                X_adv = pgd_xgb(vic_w, X_atk, y_atk, EPS,
                                iters=PGD_ITERS, restarts=PGD_RESTARTS,
                                alpha=alpha)
            else:
                X_adv = pgd_mlp(vic_w, X_atk, y_atk, EPS,
                                iters=PGD_ITERS, restarts=PGD_RESTARTS,
                                alpha=alpha)
            r = eval_attack(vic_w, X_eval, y_eval, X_adv, "PGD", vic_name)
            r.update({"seed": seed, "family": "Whitebox",
                      "eps": EPS, "dataset": DATASET})
            all_results.append(r)
            print(f"    ASR={r['asr']*100:.1f}%  "
                  f"F1 {r['f1_clean']:.3f}→{r['f1_adv']:.3f}")

            # ── C&W ───────────────────────────────────────────
            print("  [C&W]")
            set_all_seeds(seed)
            if is_lr:
                X_adv = cw_logreg(vic_w, X_atk, y_atk, EPS, iters=CW_ITERS)
            elif is_xgb:
                X_adv = cw_xgb(vic_w, X_atk, y_atk, EPS, iters=CW_ITERS)
            else:
                X_adv = cw_mlp(vic_w, X_atk, y_atk, EPS, iters=CW_ITERS)
            r = eval_attack(vic_w, X_eval, y_eval, X_adv, "C&W", vic_name)
            r.update({"seed": seed, "family": "Whitebox",
                      "eps": EPS, "dataset": DATASET})
            all_results.append(r)
            print(f"    ASR={r['asr']*100:.1f}%  "
                  f"F1 {r['f1_clean']:.3f}→{r['f1_adv']:.3f}")

        # ── Checkpoint après chaque seed ──────────────────────
        pd.DataFrame(all_results).to_csv(
            RESULTS_DIR / f"whitebox_multirun_{TAG}_tmp.csv", index=False)
        print(f"\n  ✓ Checkpoint seed {seed} sauvegardé")

    # ══════════════════════════════════════════════════════════
    # SAUVEGARDE FINALE
    # ══════════════════════════════════════════════════════════

    df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / f"whitebox_multirun_{TAG}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV → {csv_path}")

    # ── Résumé console ────────────────────────────────────────
    summary = (df.groupby(["attack", "model"])["asr"]
                 .agg(["median", "std", "min", "max"])
                 .round(4))
    print(f"\n{'═'*55}")
    print(f"  RÉSUMÉ — {DATASET.upper()}  eps={EPS}")
    print(f"{'═'*55}")
    print(summary.to_string())

    # ── Export JSON ───────────────────────────────────────────
    out = {}
    for model_name in df["model"].unique():
        out[model_name] = {}
        sub = df[df["model"] == model_name]
        for attack_name in sub["attack"].unique():
            vals = sub[sub["attack"] == attack_name]["asr"]
            out[model_name][attack_name] = {
                "evasion_rate_median": round(float(vals.median()) * 100, 2),
                "evasion_rate_std":    round(float(vals.std())    * 100, 2),
                "evasion_rate_min":    round(float(vals.min())    * 100, 2),
                "evasion_rate_max":    round(float(vals.max())    * 100, 2),
            }

    json_path = RESULTS_DIR / f"whitebox_multirun_{TAG}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✓ JSON → {json_path}")

    return df


if __name__ == "__main__":
    run()