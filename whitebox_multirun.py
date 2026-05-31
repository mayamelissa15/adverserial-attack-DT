# ~/swat/run_whitebox_multirun.py
"""
Multi-run whitebox attacks — même pattern que run_experiments_fast.py.

Pourquoi multi-run pour le whitebox ?
─────────────────────────────────────
Les attaques whitebox sont déterministes une fois le seed fixé, mais
l'eval set varie selon le seed (build_shared_eval sélectionne 500 TP
communs parmi les vrais positifs détectés par les 3 modèles).
Faire N_RUNS seeds permet donc de :
  - mesurer la variance de l'ASR sur différents sous-ensembles de TP,
  - produire des boxplots avec whiskers pour l'article,
  - s'assurer que les résultats ne dépendent pas du choix de l'eval set.

FAST_MODE (recommandé pour le multi-run)
─────────────────────────────────────────
PGD complet (200 iters × 10 restarts) × 10 seeds = ~6h sur CPU.
FAST_MODE=True réduit à 50 iters × 3 restarts : résultats légèrement
inférieurs mais tout à fait publiables pour la variance, et ~7× plus rapide.
Mettre FAST_MODE=False pour la run "best effort" finale (1 seed suffit,
utiliser run_whitebox.py original à ce moment-là).
"""

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

# Import des fonctions d'attaque depuis run_whitebox
# (on les importe directement pour éviter la duplication)
from whitebox import (
    fgsm_mlp, fgsm_logreg, fgsm_xgb,
    pgd_mlp,  pgd_logreg,  pgd_xgb,
    cw_mlp,   cw_logreg,   cw_xgb,
    THRESHOLD_LOGIT, EPS_FD, CW_LR_XGB, CW_LR_LR, CW_ITERS,
    PGD_ALPHA_K,
)

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

SAVE_DIR    = Path("~/swat/artifacts").expanduser()
RESULTS_DIR = Path("~/swat/results").expanduser()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
EPS     = 0.1          # epsilon unique pour le multi-run (cohérent avec blackbox)
N_RUNS  = 10
SEEDS   = list(range(N_RUNS))

# ── FAST_MODE ──────────────────────────────────────────────────
# True  → PGD 50 iters × 3 restarts  (~7× plus rapide, pour la variance)
# False → PGD 200 iters × 10 restarts (paramètres complets, ~6h CPU)
FAST_MODE = True

PGD_ITERS_FAST    = 50
PGD_RESTARTS_FAST = 3
PGD_ITERS_FULL    = 200
PGD_RESTARTS_FULL = 10

# C&W : on réduit aussi les iters en FAST_MODE
CW_ITERS_FAST = 150
CW_ITERS_FULL = 500

print(f"Device   : {DEVICE}")
print(f"EPS      : {EPS}")
print(f"N_RUNS   : {N_RUNS}")
print(f"FAST_MODE: {FAST_MODE}")
if FAST_MODE:
    print(f"  PGD    : {PGD_ITERS_FAST} iters × {PGD_RESTARTS_FAST} restarts")
    print(f"  C&W    : {CW_ITERS_FAST} iters")
else:
    print(f"  PGD    : {PGD_ITERS_FULL} iters × {PGD_RESTARTS_FULL} restarts")
    print(f"  C&W    : {CW_ITERS_FULL} iters")


# ══════════════════════════════════════════════════════════════
# CHARGEMENT DES VICTIMES
# ══════════════════════════════════════════════════════════════

def load_victims():
    X_test = np.load(SAVE_DIR / "X_test.npy")
    y_test = np.load(SAVE_DIR / "y_test.npy")

    mlp_model = MLP(input_size=X_test.shape[1]).to(DEVICE)
    mlp_model.load_state_dict(torch.load(SAVE_DIR / "best_mlp.pt", map_location=DEVICE))
    mlp_model.eval()
    mlp_w = MLPWrapper(mlp_model, DEVICE)

    logreg_w = LogRegWrapper(joblib.load(SAVE_DIR / "logreg.pkl"))

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(SAVE_DIR / "xgb.json"))
    xgb_w = XGBoostWrapper(xgb_model)

    return X_test, y_test, mlp_w, logreg_w, xgb_w


# ══════════════════════════════════════════════════════════════
# EVAL SET COMMUN AUX 3 MODÈLES (copié depuis run_experiments_fast)
# ══════════════════════════════════════════════════════════════

def build_shared_eval(X_test, y_test, mlp_w, logreg_w, xgb_w, seed=42):
    """
    Sélectionne un eval set commun : 500 normaux + 500 TP détectés
    correctement par les 3 modèles simultanément.
    Même fonction que dans run_experiments_fast.py pour garantir
    la comparabilité des résultats blackbox ↔ whitebox.
    """
    rng        = np.random.default_rng(seed)
    idx_normal = np.where(y_test == 0)[0]
    idx_attack = np.where(y_test == 1)[0]

    preds_mlp    = mlp_w.predict(X_test[idx_attack])
    preds_logreg = logreg_w.predict(X_test[idx_attack])
    preds_xgb    = xgb_w.predict(X_test[idx_attack])
    ok_mask      = (preds_mlp == 1) & (preds_logreg == 1) & (preds_xgb == 1)
    idx_attack_ok = idx_attack[ok_mask]

    sel_n  = rng.choice(idx_normal,    size=500, replace=False)
    sel_a  = rng.choice(idx_attack_ok, size=min(500, len(idx_attack_ok)), replace=False)
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


def _pgd_iters():
    return PGD_ITERS_FAST    if FAST_MODE else PGD_ITERS_FULL

def _pgd_restarts():
    return PGD_RESTARTS_FAST if FAST_MODE else PGD_RESTARTS_FULL

def _cw_iters():
    return CW_ITERS_FAST     if FAST_MODE else CW_ITERS_FULL


# ══════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE
# ══════════════════════════════════════════════════════════════

def run():
    print("\nChargement des victimes...")
    X_test, y_test, mlp_w, logreg_w, xgb_w = load_victims()

    victims = [
        ("MLP",     mlp_w,    "mlp"),
        ("LogReg",  logreg_w, "lr"),
        ("XGBoost", xgb_w,    "xgb"),
    ]

    all_results = []

    for seed in SEEDS:
        print(f"\n{'═'*55}")
        print(f"  SEED {seed} / {N_RUNS - 1}")
        print(f"{'═'*55}")

        set_all_seeds(seed)
        X_eval, y_eval, X_atk, y_atk = build_shared_eval(
            X_test, y_test, mlp_w, logreg_w, xgb_w, seed=seed
        )
        print(f"  Eval set : {len(X_eval)} exemples "
              f"({(y_eval==1).sum()} attaques, {(y_eval==0).sum()} normaux)")

        for vic_name, vic_w, vic_tag in victims:
            print(f"\n  ── {vic_name} {'─'*(45-len(vic_name))}")
            is_lr  = (vic_name == "LogReg")
            is_xgb = (vic_name == "XGBoost")

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
            r["seed"]   = seed
            r["family"] = "Whitebox"
            r["eps"]    = EPS
            all_results.append(r)
            print(f"    ASR = {r['asr']*100:.1f}%  F1 {r['f1_clean']:.3f}→{r['f1_adv']:.3f}")

            # ── PGD ───────────────────────────────────────────
            print("  [PGD]")
            set_all_seeds(seed)
            iters    = _pgd_iters()
            restarts = _pgd_restarts()
            alpha    = EPS / PGD_ALPHA_K

            if is_lr:
                X_adv = pgd_logreg(vic_w, X_atk, y_atk, EPS,
                                   iters=iters, restarts=restarts, alpha=alpha)
            elif is_xgb:
                X_adv = pgd_xgb(vic_w, X_atk, y_atk, EPS,
                                iters=iters, restarts=restarts, alpha=alpha)
            else:
                X_adv = pgd_mlp(vic_w, X_atk, y_atk, EPS,
                                iters=iters, restarts=restarts, alpha=alpha)

            r = eval_attack(vic_w, X_eval, y_eval, X_adv, "PGD", vic_name)
            r["seed"]   = seed
            r["family"] = "Whitebox"
            r["eps"]    = EPS
            all_results.append(r)
            print(f"    ASR = {r['asr']*100:.1f}%  F1 {r['f1_clean']:.3f}→{r['f1_adv']:.3f}")

            # ── C&W ───────────────────────────────────────────
            print("  [C&W]")
            set_all_seeds(seed)
            cw_iters = _cw_iters()

            if is_lr:
                X_adv = cw_logreg(vic_w, X_atk, y_atk, EPS, iters=cw_iters)
            elif is_xgb:
                X_adv = cw_xgb(vic_w, X_atk, y_atk, EPS, iters=cw_iters)
            else:
                X_adv = cw_mlp(vic_w, X_atk, y_atk, EPS, iters=cw_iters)

            r = eval_attack(vic_w, X_eval, y_eval, X_adv, "C&W", vic_name)
            r["seed"]   = seed
            r["family"] = "Whitebox"
            r["eps"]    = EPS
            all_results.append(r)
            print(f"    ASR = {r['asr']*100:.1f}%  F1 {r['f1_clean']:.3f}→{r['f1_adv']:.3f}")

        # Sauvegarde intermédiaire après chaque seed
        df_tmp = pd.DataFrame(all_results)
        df_tmp.to_csv(RESULTS_DIR / "whitebox_multirun_tmp.csv", index=False)
        print(f"\n  ✓ Seed {seed} terminé — {len(all_results)} résultats cumulés")

    # ── Sauvegarde finale ──────────────────────────────────────
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "whitebox_multirun_results.csv", index=False)
    print(f"\n✓ CSV final → {RESULTS_DIR / 'whitebox_multirun_results.csv'}")

    # ── Résumé médiane ± std ───────────────────────────────────
    summary = (df.groupby(["attack", "model"])["asr"]
                 .agg(["median", "std", "min", "max"])
                 .round(4))
    print("\n=== RÉSUMÉ WHITEBOX MULTI-RUN ===")
    print(summary.to_string())

    # ── Export JSON (même format que whitebox_results.json) ────
    out = {}
    for model_name in df["model"].unique():
        out[model_name] = {}
        sub = df[df["model"] == model_name]
        for attack_name in sub["attack"].unique():
            rows = sub[sub["attack"] == attack_name]["asr"]
            out[model_name][attack_name] = {
                "evasion_rate_median": round(float(rows.median()) * 100, 2),
                "evasion_rate_std":    round(float(rows.std())    * 100, 2),
                "evasion_rate_min":    round(float(rows.min())    * 100, 2),
                "evasion_rate_max":    round(float(rows.max())    * 100, 2),
            }
    with open(RESULTS_DIR / "whitebox_multirun_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nJSON → {RESULTS_DIR / 'whitebox_multirun_results.json'}")

    return df


if __name__ == "__main__":
    df = run()