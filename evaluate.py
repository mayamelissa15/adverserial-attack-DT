# ~/swat/evaluate.py
#
# Évalue les modèles défendus contre toutes les attaques déjà calculées.
# Ne régénère RIEN — charge uniquement les X_adv.npy existants dans artifacts/.
#
# Convention des fichiers : adv_{attaque}_{Modele}_eps{epsilon}.npy
#   ex: adv_fgsm_MLP_eps0.1.npy, adv_square_LogReg_eps0.3.npy
#
# Modèles défendus attendus dans artifacts/ :
#   mlp_at_fgsm.pt, mlp_at_pgd.pt, logreg_aug_fgsm.pkl
#
# Sortie : ~/swat/results/defense_results.json
#
# Usage : python evaluate.py

import numpy as np
import torch
import joblib
import json
import re
import warnings
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

import sys

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from models import MLP, MLPWrapper, LogRegWrapper

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

SAVE_DIR    = Path("~/swat/artifacts").expanduser()
RESULTS_DIR = Path("~/swat/results").expanduser()
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD   = 0.45

EPS_PRIORITY = ["0.3", "0.5", "0.1"]

ATTACK_NAME_MAP = {
    "fgsm":         "FGSM",
    "pgd":          "PGD",
    "cw":           "CW",
    "square":       "Square",
    "nes":          "NES",
    "hsja":         "HSJA",
    "rays":         "RayS",
    "mi_fgsm":      "MI-FGSM",
    "mi-fgsm":      "MI-FGSM",
    "vmi_fgsm":     "VMI-FGSM",
    "vmi-fgsm":     "VMI-FGSM",
     "ensemble_mi":  "Ensemble-MI",   # debug debug
    "ensemble-mi":  "Ensemble-MI",
    "ensemble_vmi": "Ensemble-VMI",
    "ensemble-vmi": "Ensemble-VMI",
}

MODEL_NAME_MAP = {
    "MLP":    "MLP",
    "LogReg": "LogReg",
    "XGBoost": "XGBoost",
}

# Attaques whitebox — filtrées pour XGBoost (non différentiable)
WHITEBOX_ATTACKS = {"fgsm", "pgd", "cw"}

print(f"Device : {DEVICE}")


# ══════════════════════════════════════════════════════════════
# CHARGEMENT DES MODÈLES DÉFENDUS  (MLP + LogReg uniquement)
# ══════════════════════════════════════════════════════════════

def load_defended_models(input_size):
    """
    Charge les modèles défendus depuis artifacts/.
    Retourne un dict :
      {
        "MLP":    {"AT-FGSM": wrapper, "AT-PGD": wrapper},
        "LogReg": {"Aug-FGSM": wrapper},
      }
    """
    defended = {"MLP": {}, "LogReg": {} , "XGBoost":{}}

    # MLP AT-FGSM
    p = SAVE_DIR / "mlp_at_fgsm.pt"
    if p.exists():
        m = MLP(input_size=input_size).to(DEVICE)
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        m.eval()
        defended["MLP"]["AT-FGSM"] = MLPWrapper(m, DEVICE)
        print(f"  ✓ MLP AT-FGSM chargé")
    else:
        print(f"  ✗ mlp_at_fgsm.pt introuvable — lance defenses.py d'abord")

    # MLP AT-PGD
    p = SAVE_DIR / "mlp_at_pgd.pt"
    if p.exists():
        m = MLP(input_size=input_size).to(DEVICE)
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        m.eval()
        defended["MLP"]["AT-PGD"] = MLPWrapper(m, DEVICE)
        print(f"  ✓ MLP AT-PGD chargé")
    else:
        print(f"  ✗ mlp_at_pgd.pt introuvable — lance defenses.py d'abord")

    # LogReg Aug-FGSM
    p = SAVE_DIR / "logreg_aug_fgsm.pkl"
    if p.exists():
        defended["LogReg"]["Aug-FGSM"] = LogRegWrapper(joblib.load(p))
        print(f"  ✓ LogReg Aug-FGSM chargé")
    else:
        print(f"  ✗ logreg_aug_fgsm.pkl introuvable — lance defenses.py d'abord")

    #ici XGboost aug fgsm
    from xgboost import XGBClassifier
    from models import XGBoostWrapper
    """
    p = SAVE_DIR / "xgb_aug_fgsm.json"
    if p.exists():
        m = XGBClassifier()
        m.load_model(str(p))
        defended["XGBoost"] = {"Aug-FGSM": XGBoostWrapper(m)}
        print(f"  ✓ XGBoost Aug-FGSM chargé")
    """

    # XGBoost Aug-FGSM Itératif (option 2)
    p = SAVE_DIR / "xgb_iter_fgsm_r3.json"
    if p.exists():
        m = XGBClassifier()
        m.load_model(str(p))
        defended["XGBoost"]["Aug-FGSM-Iter"] = XGBoostWrapper(m)
        print(f"  ✓ XGBoost Aug-FGSM-Iter chargé")
    else:
        print(f"  ✗ xgb_iter_fgsm_r3.json introuvable — lance defenses.py d'abord")
        
    return defended


# ══════════════════════════════════════════════════════════════
# SCAN DES X_ADV DISPONIBLES
# ══════════════════════════════════════════════════════════════

def scan_adv_files():
    """
    Parcourt artifacts/ et groupe les fichiers adv_*.npy par (attaque, modèle).
    Ignore tous les fichiers XGBoost.
    """
    pat_sub = re.compile(
        r"adv_(.+?)_(MLP|LogReg|XGBoost)_sub_(.+?)_eps([\d.]+)\.npy",
        re.IGNORECASE
    )
    pat_old = re.compile(
        r"adv_(.+?)_(MLP|LogReg|XGBoost)_eps([\d.]+)\.npy",
        re.IGNORECASE
    )

    grouped = {}

    for f in sorted(SAVE_DIR.glob("adv_*.npy")):
        m_sub = pat_sub.match(f.name)
        m_old = pat_old.match(f.name)

        if m_sub:
            raw_attack = m_sub.group(1).lower()
            raw_model  = m_sub.group(2)
            sub_name   = m_sub.group(3)
            eps_str    = m_sub.group(4)
        elif m_old:
            raw_attack = m_old.group(1).lower()
            raw_model  = m_old.group(2)
            sub_name   = None
            eps_str    = m_old.group(3)
        else:
            continue

   
        model_clean  = MODEL_NAME_MAP.get(raw_model, raw_model)
        attack_clean = ATTACK_NAME_MAP.get(raw_attack, raw_attack.upper())

        key = (attack_clean, model_clean)
        grouped.setdefault(key, []).append({
            "path": f,
            "eps":  eps_str,
            "sub":  sub_name,
        })

    print(f"\n  {len(grouped)} paires (attaque, modèle) trouvées dans artifacts/")
    for (atk, mdl), entries in sorted(grouped.items()):
        print(f"    {atk:15s} / {mdl:8s} — {len(entries)} fichier(s)")
    return grouped


def pick_best_eps(eps_dict):
    for eps in EPS_PRIORITY:
        if eps in eps_dict:
            return eps_dict[eps], eps
    best_eps = max(eps_dict.keys(), key=float)
    return eps_dict[best_eps], best_eps


# ══════════════════════════════════════════════════════════════
# CALCUL DES MÉTRIQUES
# ══════════════════════════════════════════════════════════════

def compute_metrics(wrapper, X_test, y_test, X_adv, y_adv):
    y_pred_adv = wrapper.predict(X_adv)
    asr       = float((y_pred_adv == 0).sum()) / len(y_pred_adv)
    recall    = recall_score(y_adv, y_pred_adv, zero_division=0)
    precision = precision_score(y_adv, y_pred_adv, zero_division=0)
    f1        = f1_score(y_adv, y_pred_adv, zero_division=0)
    return {
        "evasion_rate": round(asr * 100, 2),
        "recall":       round(recall, 4),
        "precision":    round(precision, 4),
        "f1":           round(f1, 4),
    }


# ══════════════════════════════════════════════════════════════
# CHARGEMENT DES RÉSULTATS BASELINE
# ══════════════════════════════════════════════════════════════

def load_baseline_results():
    baseline = {}
    for fname in ["whitebox_results.json", "blackbox_results.json",
                  "transfer_results.json"]:
        p = RESULTS_DIR / fname
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        for model, attacks in data.items():
            for attack, metrics in attacks.items():
                baseline[(attack, model)] = metrics
    print(f"  {len(baseline)} entrées baseline chargées depuis results/")
    return baseline


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "═"*60)
    print("  CHARGEMENT")
    print("═"*60)

    X_test = np.load(SAVE_DIR / "X_test.npy")
    y_test = np.load(SAVE_DIR / "y_test.npy")
    input_size = X_test.shape[1]
    print(f"  X_test : {X_test.shape}  |  attaques dans y_test : {(y_test==1).sum()}")

    defended  = load_defended_models(input_size)
    adv_files = scan_adv_files()
    baseline  = load_baseline_results()

    print("\n" + "═"*60)
    print("  ÉVALUATION")
    print("═"*60)

    out = {}

    for (attack_name, model_name), entries in sorted(adv_files.items()):

        defenses = defended.get(model_name, {})
        if not defenses:
            continue

        def eps_rank(e):
            try:
                return -float(e["eps"])
            except ValueError:
                return 0

        entries_sorted = sorted(entries, key=eps_rank)

        print(f"\n  {attack_name:15s} / {model_name:8s}  "
              f"({len(entries_sorted)} substitut(s))")

        for defense_name, wrapper in defenses.items():

            best_metrics = None
            best_sub     = None

            for entry in entries_sorted:
                X_adv = np.load(entry["path"]).astype(np.float32)
                y_adv = np.ones(len(X_adv), dtype=int)

                metrics = compute_metrics(wrapper, X_test, y_test, X_adv, y_adv)

                if best_metrics is None or metrics["evasion_rate"] > best_metrics["evasion_rate"]:
                    best_metrics = metrics
                    best_sub     = entry["sub"] or "—"

            base    = baseline.get((attack_name, model_name), {})
            base_f1 = base.get("f1", None)
            best_metrics["delta_f1"] = (
                round(best_metrics["f1"] - base_f1, 4) if base_f1 is not None else None
            )
            best_metrics["best_sub"] = best_sub

            print(f"    {defense_name:12s} → ASR {best_metrics['evasion_rate']:6.2f}%  "
                  f"F1 {best_metrics['f1']:.4f}  "
                  f"ΔF1 {best_metrics['delta_f1']}  "
                  f"(sub={best_sub})")

            out.setdefault(model_name, {}) \
               .setdefault(defense_name, {})[attack_name] = best_metrics

    json_path = RESULTS_DIR / "defense_results.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ defense_results.json sauvegardé → {json_path}")

    print("\n" + "═"*60)
    print("  RÉSUMÉ — ASR par défense")
    print("═"*60)
    for model_name, defenses in out.items():
        for defense_name, attacks in defenses.items():
            asrs = [v["evasion_rate"] for v in attacks.values()]
            avg  = sum(asrs) / len(asrs) if asrs else 0
            bar  = "█" * int(avg / 5) + "░" * (20 - int(avg / 5))
            print(f"  {model_name}/{defense_name:<15} {bar}  {avg:5.1f}% (moy)")

    return out


if __name__ == "__main__":
    run()