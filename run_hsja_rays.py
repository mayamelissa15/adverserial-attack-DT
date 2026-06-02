# ~/swat/run_hsja_rays_10seeds.py
# Lance uniquement HSJA + RayS sur 10 seeds, avec le meme protocole
# que run_experiments_fast.py

import sys
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import torch
import joblib
import pandas as pd
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent))

from models import (
    MLP,
    MLPWrapper,
    LogRegWrapper,
    XGBoostWrapper,
    eval_attack,
)

SAVE_DIR = Path("~/swat/artifacts").expanduser()
RESULTS_DIR = Path("~/swat/results").expanduser()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 0.1
N_RUNS = 10
SEEDS = list(range(N_RUNS))

# Mets None pour lancer HSJA/RayS sur tous les exemples d'attaque.
MAX_DECISION_BOUNDARY = None

OUT_CSV = RESULTS_DIR / "hsja_rays_10seeds.csv"
TMP_CSV = RESULTS_DIR / "hsja_rays_10seeds_tmp.csv"

print(f"Device : {DEVICE} | N_RUNS : {N_RUNS} | EPS : {EPS}")


def load_decision_boundary_attacks():
    """
    Importe hsja() et rays() depuis blackbox.py.
    Utile car le fichier commence par un chiffre, donc on ne peut pas faire:
        import blackbox
    """
    path = Path(__file__).parent / "blackbox.py"
    spec = importlib.util.spec_from_file_location("blackbox_02", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.hsja, module.rays


def load_victims():
    X_train = np.load(SAVE_DIR / "X_train.npy")
    y_train = np.load(SAVE_DIR / "y_train.npy")
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

    return X_train, y_train, X_test, y_test, mlp_w, logreg_w, xgb_w


def build_shared_eval(X_test, y_test, mlp_w, logreg_w, xgb_w, seed=42):
    rng = np.random.default_rng(seed)

    idx_normal = np.where(y_test == 0)[0]
    idx_attack = np.where(y_test == 1)[0]

    preds_mlp = mlp_w.predict(X_test[idx_attack])
    preds_logreg = logreg_w.predict(X_test[idx_attack])
    preds_xgb = xgb_w.predict(X_test[idx_attack])

    ok_mask = (preds_mlp == 1) & (preds_logreg == 1) & (preds_xgb == 1)
    idx_attack_ok = idx_attack[ok_mask]

    sel_n = rng.choice(idx_normal, size=500, replace=False)
    sel_a = rng.choice(idx_attack_ok, size=min(500, len(idx_attack_ok)), replace=False)

    idx_ev = np.concatenate([sel_n, sel_a])
    rng.shuffle(idx_ev)

    X_eval = X_test[idx_ev]
    y_eval = y_test[idx_ev]

    mask = y_eval == 1
    X_atk = X_eval[mask].astype(np.float32)
    y_atk = y_eval[mask]

    return X_eval, y_eval, X_atk, y_atk


def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_subsample(X_atk, y_atk, seed):
    if MAX_DECISION_BOUNDARY is None or len(X_atk) <= MAX_DECISION_BOUNDARY:
        return X_atk, y_atk, None

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_atk), MAX_DECISION_BOUNDARY, replace=False)

    return X_atk[idx], y_atk[idx], idx


def rebuild_full_attack(X_atk, X_adv_partial, idx):
    if idx is None:
        return X_adv_partial

    X_adv_full = X_atk.copy()
    X_adv_full[idx] = X_adv_partial
    return X_adv_full


def already_done(existing_df, seed, attack, model):
    if existing_df is None or existing_df.empty:
        return False

    sub = existing_df[
        (existing_df["seed"] == seed)
        & (existing_df["attack"] == attack)
        & (existing_df["model"] == model)
    ]
    return not sub.empty


def run():
    hsja, rays = load_decision_boundary_attacks()

    print("Chargement des victimes...")
    _, _, X_test, y_test, mlp_w, logreg_w, xgb_w = load_victims()

    victims = [
        ("MLP", mlp_w),
        ("LogReg", logreg_w),
        ("XGBoost", xgb_w),
    ]

    if TMP_CSV.exists():
        all_results = pd.read_csv(TMP_CSV).to_dict("records")
        existing_df = pd.DataFrame(all_results)
        print(f"Reprise depuis {TMP_CSV} avec {len(all_results)} resultats.")
    else:
        all_results = []
        existing_df = None

    for seed in SEEDS:
        print(f"\n{'=' * 60}")
        print(f"SEED {seed} / {N_RUNS - 1}")
        print(f"{'=' * 60}")

        set_all_seeds(seed)

        X_eval, y_eval, X_atk, y_atk = build_shared_eval(
            X_test, y_test, mlp_w, logreg_w, xgb_w, seed=seed
        )

        print(f"Eval set : {len(X_eval)} exemples | attaques : {len(X_atk)}")

        X_atk_bb, y_atk_bb, idx_bb = maybe_subsample(X_atk, y_atk, seed)

        if idx_bb is not None:
            print(f"HSJA/RayS : sous-echantillonnage {len(X_atk)} -> {len(X_atk_bb)}")
        else:
            print(f"HSJA/RayS : run sur tous les {len(X_atk_bb)} exemples d'attaque")

        for vic_name, vic_w in victims:
            print(f"\n[{vic_name} | seed {seed}]")

            attack_name = "HSJA"
            if not already_done(existing_df, seed, attack_name, vic_name):
                print(f"  [{attack_name}]")
                set_all_seeds(seed)

                X_adv_partial = hsja(vic_w, X_atk_bb, y_atk_bb, EPS)
                X_adv = rebuild_full_attack(X_atk, X_adv_partial, idx_bb)

                r = eval_attack(vic_w, X_eval, y_eval, X_adv, attack_name, vic_name)
                r["seed"] = seed
                r["family"] = "Decision-based"
                all_results.append(r)

                pd.DataFrame(all_results).to_csv(TMP_CSV, index=False)
                existing_df = pd.DataFrame(all_results)
            else:
                print(f"  [{attack_name}] deja fait, skip.")

            attack_name = "RayS"
            if not already_done(existing_df, seed, attack_name, vic_name):
                print(f"  [{attack_name}]")
                set_all_seeds(seed)

                X_adv_partial = rays(vic_w, X_atk_bb, y_atk_bb, EPS)
                X_adv = rebuild_full_attack(X_atk, X_adv_partial, idx_bb)

                r = eval_attack(vic_w, X_eval, y_eval, X_adv, attack_name, vic_name)
                r["seed"] = seed
                r["family"] = "Decision-based"
                all_results.append(r)

                pd.DataFrame(all_results).to_csv(TMP_CSV, index=False)
                existing_df = pd.DataFrame(all_results)
            else:
                print(f"  [{attack_name}] deja fait, skip.")

    df = pd.DataFrame(all_results)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nTermine -> {OUT_CSV}")
    print(df.groupby(["family", "attack", "model"])["asr"].agg(["mean", "std"]).round(4))

    return df


if __name__ == "__main__":
    df = run()
