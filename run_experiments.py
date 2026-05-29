# ~/swat/run_experiments.py
#
# Lance chaque attaque 10 fois avec des seeds différentes.

# Résultat : multi_run_results.csv avec 10 lignes par (attaque, victime)


import numpy as np
import torch
import joblib
import pandas as pd
import warnings
from pathlib import Path
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent))

from models import (MLP, SmallMLP, DeepMLP,
                    MLPWrapper, LogRegWrapper, XGBoostWrapper,
                    build_eval_set, eval_attack)

from blackbox import square_attack, nes_attack, hsja, rays
from transfer  import (mi_fgsm, vmi_fgsm, ensemble_mi_fgsm,
                           train_substitute, eval_transfer)

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

SAVE_DIR   = Path("~/swat/artifacts").expanduser()
RESULTS_DIR = Path("~/swat/results").expanduser()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.45
EPS       = 0.1
N_RUNS    = 10
SEEDS     = list(range(N_RUNS))   # seeds 0..9

print(f"Device : {DEVICE}")
print(f"N_RUNS : {N_RUNS}  |  Seeds : {SEEDS}")


# ══════════════════════════════════════════════════════════════
# CHARGEMENT DES MODÈLES VICTIMES (une seule fois)
# ══════════════════════════════════════════════════════════════

def load_victims():
    X_train = np.load(SAVE_DIR / "X_train.npy")
    y_train = np.load(SAVE_DIR / "y_train.npy")
    X_test  = np.load(SAVE_DIR / "X_test.npy")
    y_test  = np.load(SAVE_DIR / "y_test.npy")

    mlp_model = MLP(input_size=X_test.shape[1]).to(DEVICE)
    mlp_model.load_state_dict(torch.load(SAVE_DIR / "best_mlp.pt", map_location=DEVICE))
    mlp_model.eval()
    mlp_w = MLPWrapper(mlp_model, DEVICE)

    logreg_w = LogRegWrapper(joblib.load(SAVE_DIR / "logreg.pkl"))

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(SAVE_DIR / "xgb.json"))
    xgb_w = XGBoostWrapper(xgb_model)

    return X_train, y_train, X_test, y_test, mlp_w, logreg_w, xgb_w


# ══════════════════════════════════════════════════════════════
# CONSTRUCTION DE L'EVAL SET (intersection des 3 victimes)
# ══════════════════════════════════════════════════════════════

def build_shared_eval(X_test, y_test, mlp_w, logreg_w, xgb_w, seed=42):
    """
    Construit un eval set partagé entre les 3 victimes :
    uniquement les exemples détectés correctement par les 3 modèles.
    Le seed contrôle le sous-échantillonnage aléatoire.
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
# HELPER : fixer tous les seeds aléatoires
# ══════════════════════════════════════════════════════════════

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════
# RUN BLACKBOX (Square, NES, HSJA, RayS)
# ══════════════════════════════════════════════════════════════

MAX_DECISION_BOUNDARY = 300

def run_blackbox_once(seed, X_eval, y_eval, X_atk, y_atk,
                      victims, eps=EPS):
    """
    Lance les 4 attaques blackbox pour UN seed donné.
    Retourne une liste de dicts {seed, attack, victim, asr, ...}
    """
    set_all_seeds(seed)
    results = []

    for vic_name, vic_w in victims:
        # Sous-échantillonnage pour HSJA/RayS (lourd en queries)
        if len(X_atk) > MAX_DECISION_BOUNDARY:
            idx_bb   = np.random.choice(len(X_atk), MAX_DECISION_BOUNDARY, replace=False)
            X_atk_bb = X_atk[idx_bb]
            y_atk_bb = y_atk[idx_bb]
        else:
            X_atk_bb, y_atk_bb, idx_bb = X_atk, y_atk, None

        # ── Square ──────────────────────────────────────────
        X_adv = square_attack(vic_w, X_atk, y_atk, eps)
        r = eval_attack(vic_w, X_eval, y_eval, X_adv,
                        "Square", vic_name)
        r["seed"] = seed
        r["family"] = "Score-based"
        results.append(r)

        # ── NES ─────────────────────────────────────────────
        X_adv = nes_attack(vic_w, X_atk, y_atk, eps)
        r = eval_attack(vic_w, X_eval, y_eval, X_adv,
                        "NES", vic_name)
        r["seed"] = seed
        r["family"] = "Score-based"
        results.append(r)

        # ── HSJA ────────────────────────────────────────────
        X_adv_bb   = hsja(vic_w, X_atk_bb, y_atk_bb, eps)
        X_adv_full = X_atk.copy()
        if idx_bb is not None:
            X_adv_full[idx_bb] = X_adv_bb
        else:
            X_adv_full = X_adv_bb
        r = eval_attack(vic_w, X_eval, y_eval, X_adv_full,
                        "HSJA", vic_name)
        r["seed"] = seed
        r["family"] = "Decision-based"
        results.append(r)

        # ── RayS ────────────────────────────────────────────
        X_adv_bb   = rays(vic_w, X_atk_bb, y_atk_bb, eps)
        X_adv_full = X_atk.copy()
        if idx_bb is not None:
            X_adv_full[idx_bb] = X_adv_bb
        else:
            X_adv_full = X_adv_bb
        r = eval_attack(vic_w, X_eval, y_eval, X_adv_full,
                        "RayS", vic_name)
        r["seed"] = seed
        r["family"] = "Decision-based"
        results.append(r)

    return results


# ══════════════════════════════════════════════════════════════
# RUN TRANSFER (MI-FGSM, VMI-FGSM, Ensemble-MI)
# Pour les attaques par transfert, la variance vient du
# ré-entraînement des substituts à chaque seed.
# ══════════════════════════════════════════════════════════════

def run_transfer_once(seed, X_train, y_train,
                      X_eval, y_eval, X_atk, y_atk,
                      victims, eps=EPS):
    """
    Re-entraîne les 3 substituts avec le seed donné,
    puis lance MI-FGSM, VMI-FGSM, Ensemble-MI.
    """
    set_all_seeds(seed)
    results = []

    # ── Entraînement substituts (source de variance principale) ──
    sub1 = train_substitute(MLP,      X_train, y_train, DEVICE,
                            f"Sub1-seed{seed}", noise_std=0.02)
    sub2 = train_substitute(SmallMLP, X_train, y_train, DEVICE,
                            f"Sub2-seed{seed}", noise_std=0.02)
    sub3 = train_substitute(DeepMLP,  X_train, y_train, DEVICE,
                            f"Sub3-seed{seed}", noise_std=0.02)

    substitutes = [
        ("Sub1-MLP",      sub1),
        ("Sub2-SmallMLP", sub2),
        ("Sub3-DeepMLP",  sub3),
    ]

    for sub_name, sub_w in substitutes:
        X_adv_mi  = mi_fgsm(sub_w,  X_atk, y_atk, eps=eps)
        X_adv_vmi = vmi_fgsm(sub_w, X_atk, y_atk, eps=eps)

        for vic_name, vic_w in victims:
            # MI-FGSM
            r = eval_transfer(X_eval, y_eval, X_adv_mi,
                              vic_w, sub_name, vic_name, "MI-FGSM")
            r["seed"]   = seed
            r["family"] = "Transfer"
            # renomme pour cohérence avec blackbox
            r["asr"]     = r.pop("asr")
            r["model"]   = r.pop("victim")
            r["attack"]  = r["attack"]
            results.append(r)

            # VMI-FGSM
            r = eval_transfer(X_eval, y_eval, X_adv_vmi,
                              vic_w, sub_name, vic_name, "VMI-FGSM")
            r["seed"]   = seed
            r["family"] = "Transfer"
            r["model"]  = r.pop("victim")
            results.append(r)

    # ── Ensemble MI-FGSM ────────────────────────────────────
    set_all_seeds(seed)  # re-fixe pour l'ensemble
    X_adv_ens = ensemble_mi_fgsm(
        [sub1, sub2, sub3], X_atk, y_atk, eps=eps,
        weights=[1/3, 1/3, 1/3]
    )
    for vic_name, vic_w in victims:
        r = eval_transfer(X_eval, y_eval, X_adv_ens,
                          vic_w, "Ensemble(S1+S2+S3)", vic_name, "Ensemble-MI")
        r["seed"]   = seed
        r["family"] = "Transfer"
        r["model"]  = r.pop("victim")
        results.append(r)

    return results


# ══════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE
# ══════════════════════════════════════════════════════════════

def run():
    print("\n" + "═"*60)
    print("  CHARGEMENT DES VICTIMES")
    print("═"*60)
    X_train, y_train, X_test, y_test, mlp_w, logreg_w, xgb_w = load_victims()

    victims = [
        ("MLP",     mlp_w),
        ("LogReg",  logreg_w),
        ("XGBoost", xgb_w),
    ]

    all_results = []

    for seed in SEEDS:
        print(f"\n{'═'*60}")
        print(f"  SEED {seed} / {N_RUNS - 1}")
        print(f"{'═'*60}")

        # Eval set légèrement différent à chaque seed
        # (sous-échantillonnage aléatoire des normaux/attaques)
        X_eval, y_eval, X_atk, y_atk = build_shared_eval(
            X_test, y_test, mlp_w, logreg_w, xgb_w, seed=seed
        )
        print(f"  Eval set : {len(X_eval)} exemples "
              f"({(y_eval==0).sum()} Normal, {(y_eval==1).sum()} Attack)")

        # ── Blackbox ────────────────────────────────────────
        print(f"\n  [Blackbox attacks — seed {seed}]")
        bb_results = run_blackbox_once(
            seed, X_eval, y_eval, X_atk, y_atk, victims
        )
        all_results.extend(bb_results)

        # ── Transfer ────────────────────────────────────────
        print(f"\n  [Transfer attacks — seed {seed}]")
        tr_results = run_transfer_once(
            seed, X_train, y_train,
            X_eval, y_eval, X_atk, y_atk, victims
        )
        all_results.extend(tr_results)

        # Sauvegarde intermédiaire après chaque seed
        # (sécurité si le run plante en cours)
        df_tmp = pd.DataFrame(all_results)
        df_tmp.to_csv(RESULTS_DIR / "multi_run_results_tmp.csv", index=False)
        print(f"  ✓ Seed {seed} terminé — {len(all_results)} résultats cumulés")

    # ── Sauvegarde finale ────────────────────────────────────
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "multi_run_results.csv", index=False)
    print(f"\n✓ Résultats finaux → {RESULTS_DIR / 'multi_run_results.csv'}")
    print(f"  Shape : {df.shape}")
    print(df.groupby(["family", "attack", "model"])["asr"].agg(["mean","std"]).round(4))

    return df


if __name__ == "__main__":
    df = run()