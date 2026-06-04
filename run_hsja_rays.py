# ~/swat/run_hsja_rays_final.py
#
# Protocole final : HSJA + RayS, 10 seeds, eps=0.1
# - build_eval_set par modèle (pas shared)
# - MAX_DECISION_BOUNDARY = 300
# - HSJA : iters=40, n_est=100
# - RayS : iters=60, search_steps=15
# - eval_attack sur X_atk_bb uniquement (pas de reconstruction) → ASR honnête

import sys
import time
import importlib.util
import warnings
from pathlib import Path
from datetime import timedelta

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
    build_eval_set,
    eval_attack,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SAVE_DIR    = Path("~/swat/artifacts").expanduser()
RESULTS_DIR = Path("~/swat/results").expanduser()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPS        = 0.1
N_RUNS     = 10
SEEDS      = list(range(N_RUNS))

MAX_DECISION_BOUNDARY = 300

HSJA_ITERS  = 40
HSJA_N_EST  = 100
RAYS_ITERS  = 60
RAYS_SEARCH = 15

OUT_CSV = RESULTS_DIR / "hsja_rays_final.csv"
TMP_CSV = RESULTS_DIR / "hsja_rays_final_tmp.csv"

# ─────────────────────────────────────────────
# UTILS PRINT
# ─────────────────────────────────────────────

_start_global = time.time()

def elapsed():
    s = int(time.time() - _start_global)
    return str(timedelta(seconds=s))

def banner(msg, level=1):
    if level == 1:
        print(f"\n{'═'*65}")
        print(f"  {msg}  [{elapsed()}]")
        print(f"{'═'*65}")
    elif level == 2:
        print(f"\n{'─'*55}")
        print(f"  {msg}  [{elapsed()}]")
        print(f"{'─'*55}")
    else:
        print(f"  >> {msg}  [{elapsed()}]")

# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

def load_decision_boundary_attacks():
    path = Path(__file__).parent / "blackbox.py"
    spec = importlib.util.spec_from_file_location("blackbox_mod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.hsja, module.rays

def load_victims():
    banner("Chargement des artefacts", level=2)
    X_test = np.load(SAVE_DIR / "X_test.npy")
    y_test = np.load(SAVE_DIR / "y_test.npy")
    print(f"    X_test : {X_test.shape}  |  y_test : {y_test.shape}")
    print(f"    Attaques dans y_test : {(y_test==1).sum()} / {len(y_test)}")

    mlp_model = MLP(input_size=X_test.shape[1]).to(DEVICE)
    mlp_model.load_state_dict(torch.load(SAVE_DIR / "best_mlp.pt", map_location=DEVICE))
    mlp_model.eval()
    mlp_w = MLPWrapper(mlp_model, DEVICE)
    print("    MLP chargé ✓")

    logreg_w = LogRegWrapper(joblib.load(SAVE_DIR / "logreg.pkl"))
    print("    LogReg chargé ✓")

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(SAVE_DIR / "xgb.json"))
    xgb_w = XGBoostWrapper(xgb_model)
    print("    XGBoost chargé ✓")

    return X_test, y_test, mlp_w, logreg_w, xgb_w

# ─────────────────────────────────────────────
# EVAL SET (par modèle, pas shared)
# ─────────────────────────────────────────────

def get_eval_sets(X_test, y_test, mlp_w, logreg_w, xgb_w):
    banner("Construction des eval sets (par modèle)", level=2)
    victims = [
        ("MLP",     mlp_w),
        ("LogReg",  logreg_w),
        ("XGBoost", xgb_w),
    ]
    eval_sets = {}
    for name, w in victims:
        X_ev, y_ev = build_eval_set(X_test, y_test, w)
        n_atk = (y_ev == 1).sum()
        print(f"    {name:<10} eval={len(X_ev)} exemples | attaques={n_atk}")
        eval_sets[name] = (X_ev, y_ev, w)
    return eval_sets

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def subsample(X_atk, y_atk, seed):
    if len(X_atk) <= MAX_DECISION_BOUNDARY:
        return X_atk, y_atk, None
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_atk), MAX_DECISION_BOUNDARY, replace=False)
    return X_atk[idx], y_atk[idx], idx

def build_bb_eval(X_ev, y_ev, X_atk_bb, X_adv_bb):
    """
    Construit un eval set réduit aux seuls exemples bb + normaux.
    Évite la dilution de l'ASR causée par la reconstruction sur ~10k exemples.

    X_ev_bb  : normaux de X_ev  +  X_atk_bb  (exemples clean)
    y_ev_bb  : labels correspondants
    X_adv_ev : normaux de X_ev  +  X_adv_bb  (exemples adversariaux)
    """
    mask_normal = (y_ev == 0)
    X_normal    = X_ev[mask_normal]
    y_normal    = y_ev[mask_normal]

    X_ev_bb  = np.concatenate([X_normal, X_atk_bb], axis=0)
    y_ev_bb  = np.concatenate([y_normal, np.ones(len(X_atk_bb), dtype=y_ev.dtype)], axis=0)
    X_adv_ev = np.concatenate([X_normal, X_adv_bb], axis=0)

    return X_ev_bb, y_ev_bb, X_adv_ev

def already_done(df, seed, attack, model):
    if df is None or df.empty:
        return False
    return not df[(df["seed"]==seed) & (df["attack"]==attack) & (df["model"]==model)].empty

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run():
    banner(f"HSJA + RayS | eps={EPS} | {N_RUNS} seeds | device={DEVICE}", level=1)
    print(f"  Config : MAX_DB={MAX_DECISION_BOUNDARY} | "
          f"HSJA iters={HSJA_ITERS} n_est={HSJA_N_EST} | "
          f"RayS iters={RAYS_ITERS} search={RAYS_SEARCH}")
    print(f"  Évaluation : sur X_atk_bb uniquement (pas de reconstruction) → ASR non dilué")
    print(f"  Output : {OUT_CSV}")

    hsja_fn, rays_fn = load_decision_boundary_attacks()
    X_test, y_test, mlp_w, logreg_w, xgb_w = load_victims()
    eval_sets = get_eval_sets(X_test, y_test, mlp_w, logreg_w, xgb_w)

    if TMP_CSV.exists():
        all_results = pd.read_csv(TMP_CSV).to_dict("records")
        existing_df = pd.DataFrame(all_results)
        banner(f"Reprise depuis {TMP_CSV} — {len(all_results)} résultats déjà présents", level=2)
    else:
        all_results = []
        existing_df = None

    total_runs = N_RUNS * 3 * 2
    done_count = len(all_results)
    print(f"  Progression : {done_count}/{total_runs} runs déjà faits\n")

    for seed in SEEDS:
        banner(f"SEED {seed}/{N_RUNS-1}  ({done_count}/{total_runs} runs terminés)", level=1)
        set_all_seeds(seed)
        t_seed = time.time()

        for vic_name, (X_ev, y_ev, vic_w) in eval_sets.items():

            # X_atk = tous les vrais positifs du modèle (issus de build_eval_set)
            # y_ev ne contient que des 1 (build_eval_set retourne uniquement les TP)
            # mais X_ev contient aussi les négatifs → on les sépare proprement
            mask_atk = (y_ev == 1)
            X_atk    = X_ev[mask_atk].astype(np.float32)
            y_atk    = y_ev[mask_atk]

            # Sous-échantillonnage pour les attaques decision-based
            X_atk_bb, y_atk_bb, _ = subsample(X_atk, y_atk, seed)

            banner(f"[seed={seed}] {vic_name}  |  "
                   f"X_atk={len(X_atk)}  X_atk_bb={len(X_atk_bb)}  "
                   f"(eval sur bb uniquement)", level=2)

            # ── HSJA ──────────────────────────────────────────────
            attack_name = "HSJA"
            if not already_done(existing_df, seed, attack_name, vic_name):
                print(f"\n  [{attack_name}] démarrage — "
                      f"~{len(X_atk_bb)*HSJA_ITERS*HSJA_N_EST:,} appels predict prévus")
                t0 = time.time()
                set_all_seeds(seed)

                X_adv_bb = hsja_fn(vic_w, X_atk_bb, y_atk_bb, EPS,
                                   iters=HSJA_ITERS, n_est=HSJA_N_EST)

                # FIX : éval sur bb seulement, normaux + 300 exemples attaqués
                X_ev_bb, y_ev_bb, X_adv_ev = build_bb_eval(X_ev, y_ev, X_atk_bb, X_adv_bb)

                dt = time.time() - t0
                r  = eval_attack(vic_w, X_ev_bb, y_ev_bb, X_adv_ev, attack_name, vic_name)
                r["seed"]       = seed
                r["family"]     = "Decision-based"
                r["n_attacked"] = len(X_atk_bb)
                all_results.append(r)
                existing_df = pd.DataFrame(all_results)
                pd.DataFrame(all_results).to_csv(TMP_CSV, index=False)

                done_count += 1
                print(f"\n  [{attack_name}] ✓  ASR={r['asr']:.1%}  "
                      f"F1adv={r['f1_adv']:.4f}  "
                      f"durée={timedelta(seconds=int(dt))}  "
                      f"[total écoulé : {elapsed()}]")
            else:
                sub = existing_df[
                    (existing_df["seed"]==seed) &
                    (existing_df["attack"]==attack_name) &
                    (existing_df["model"]==vic_name)
                ].iloc[0]
                print(f"\n  [{attack_name}] déjà fait — "
                      f"ASR={sub['asr']:.1%}  F1adv={sub['f1_adv']:.4f}  skip.")

            # ── RayS ──────────────────────────────────────────────
            attack_name = "RayS"
            if not already_done(existing_df, seed, attack_name, vic_name):
                print(f"\n  [{attack_name}] démarrage — "
                      f"~{len(X_atk_bb)*RAYS_ITERS*RAYS_SEARCH:,} appels predict prévus")
                t0 = time.time()
                set_all_seeds(seed)

                X_adv_bb = rays_fn(vic_w, X_atk_bb, y_atk_bb, EPS,
                                   iters=RAYS_ITERS, search_steps=RAYS_SEARCH)

                # FIX : éval sur bb seulement, normaux + 300 exemples attaqués
                X_ev_bb, y_ev_bb, X_adv_ev = build_bb_eval(X_ev, y_ev, X_atk_bb, X_adv_bb)

                dt = time.time() - t0
                r  = eval_attack(vic_w, X_ev_bb, y_ev_bb, X_adv_ev, attack_name, vic_name)
                r["seed"]       = seed
                r["family"]     = "Decision-based"
                r["n_attacked"] = len(X_atk_bb)
                all_results.append(r)
                existing_df = pd.DataFrame(all_results)
                pd.DataFrame(all_results).to_csv(TMP_CSV, index=False)

                done_count += 1
                print(f"\n  [{attack_name}] ✓  ASR={r['asr']:.1%}  "
                      f"F1adv={r['f1_adv']:.4f}  "
                      f"durée={timedelta(seconds=int(dt))}  "
                      f"[total écoulé : {elapsed()}]")
            else:
                sub = existing_df[
                    (existing_df["seed"]==seed) &
                    (existing_df["attack"]==attack_name) &
                    (existing_df["model"]==vic_name)
                ].iloc[0]
                print(f"\n  [{attack_name}] déjà fait — "
                      f"ASR={sub['asr']:.1%}  F1adv={sub['f1_adv']:.4f}  skip.")

        t_seed_dt  = timedelta(seconds=int(time.time() - t_seed))
        seeds_left = N_RUNS - seed - 1
        eta        = timedelta(seconds=int((time.time() - t_seed) * seeds_left))
        print(f"\n  Seed {seed} terminée en {t_seed_dt}  |  "
              f"seeds restantes : {seeds_left}  |  ETA ≈ {eta}")

    # ── Sauvegarde finale ──────────────────────────────────────
    df = pd.DataFrame(all_results)
    df.to_csv(OUT_CSV, index=False)
    if TMP_CSV.exists():
        TMP_CSV.unlink()

    banner("RÉSULTATS FINAUX", level=1)
    summary = (
        df.groupby(["attack", "model"])["asr"]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )
    print(summary.to_string())
    print(f"\n  Fichier CSV : {OUT_CSV}")
    print(f"  Durée totale : {elapsed()}")

    return df


if __name__ == "__main__":
    df = run()