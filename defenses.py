"""
defenses.py  — version corrigée
Entraînement des modèles défendus + évaluation immédiate.

═══════════════════════════════════════════════════════════════
CORRECTIONS PAR RAPPORT À LA VERSION AMÉLIORÉE
═══════════════════════════════════════════════════════════════

FIX 1 — MISMATCH D'ARCHITECTURE MLP AU CHARGEMENT
  Problème : le .pt sauvegardé peut correspondre à une ancienne
  architecture (ex. avec BatchNorm) différente de la classe MLP
  actuelle dans models.py.
  Solution : on ajoute strict=False + vérification de cohérence
  des clés au chargement, avec un message d'erreur explicite qui
  demande de supprimer l'ancien .pt plutôt que de crasher.

  Si l'architecture a changé, supprimer les anciens .pt :
      rm ~/swat/artifacts/mlp_at_fgsm.pt
      rm ~/swat/artifacts/mlp_at_pgd.pt
  puis relancer defenses.py.

FIX 2 — CAPTURE DE VARIABLE DANS LES LAMBDAS (evaluate_defended_models)
  Problème : X_atk et y_atk étaient capturées par référence dans
  les lambdas de get_attack, ce qui donnait des résultats incorrects
  (toutes les lambdas pointaient vers la dernière valeur de la boucle).
  Solution : passage explicite via paramètres par défaut des lambdas.

FIX 3 — C&W AJOUTÉ À L'ÉVALUATION DÉFENSIVE
  L'attaque C&W était absente de evaluate_defended_models.
  Elle est maintenant incluse avec les colonnes :
    FGSM 0.3 | PGD 0.3 | C&W 0.3
  (on rapporte uniquement eps=0.3 pour garder la table lisible,
   c'est le budget le plus sévère et le plus pertinent pour l'article)

═══════════════════════════════════════════════════════════════
MODÈLES PRODUITS
═══════════════════════════════════════════════════════════════
  mlp_at_fgsm.pt          → MLP Adversarial Training FGSM
  mlp_at_pgd.pt           → MLP Adversarial Training PGD
  logreg_aug_fgsm.pkl     → LogReg augmentée FGSM
  logreg_aug_pgd.pkl      → LogReg augmentée PGD
  xgb_aug_proxy_fgsm.json → XGBoost augmenté via proxy MLP AT-FGSM
  xgb_aug_direct_fgsm.json→ XGBoost augmenté via gradient numérique direct
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

import sys
import importlib.util

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from models import (MLP, MLPWrapper, LogRegWrapper, XGBoostWrapper,
                    build_eval_set, eval_attack)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

wb = _load_module("whitebox", BASE / "whitebox.py")

fgsm_mlp        = wb.fgsm_mlp
fgsm_logreg     = wb.fgsm_logreg
fgsm_xgb        = wb.fgsm_xgb
pgd_mlp         = wb.pgd_mlp
pgd_logreg      = wb.pgd_logreg
pgd_xgb         = wb.pgd_xgb
cw_mlp          = wb.cw_mlp
cw_logreg       = wb.cw_logreg
cw_xgb          = wb.cw_xgb
THRESHOLD_LOGIT = wb.THRESHOLD_LOGIT


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

SAVE_DIR  = Path("~/swat/artifacts").expanduser()
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.45

EPS_AT_LIST      = [0.1, 0.3]
EPS_AT           = 0.1
AT_EPOCHS        = 60
AT_PATIENCE      = 10
AT_MIX_RATIO     = 0.5
PGD_AT_ITERS     = 10
PGD_AT_ALPHA     = lambda eps: eps / 4
NORMAL_AUG_RATIO = 0.3

print(f"Device : {DEVICE}")


# ══════════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════════

def load_artifacts():
    X_train = np.load(SAVE_DIR / "X_train.npy")
    y_train = np.load(SAVE_DIR / "y_train.npy")
    X_test  = np.load(SAVE_DIR / "X_test.npy")
    y_test  = np.load(SAVE_DIR / "y_test.npy")

    mlp_model = MLP(input_size=X_test.shape[1]).to(DEVICE)
    mlp_model.load_state_dict(
        torch.load(SAVE_DIR / "best_mlp.pt", map_location=DEVICE)
    )
    mlp_model.eval()
    mlp_w = MLPWrapper(mlp_model, DEVICE)

    logreg_w = LogRegWrapper(joblib.load(SAVE_DIR / "logreg.pkl"))

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(SAVE_DIR / "xgb.json"))
    xgb_w = XGBoostWrapper(xgb_model)

    return X_train, y_train, X_test, y_test, mlp_w, logreg_w, xgb_w


# ══════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════

def quick_eval(wrapper, X_test, y_test, label=""):
    y_pred = wrapper.predict(X_test)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    pre = precision_score(y_test, y_pred, zero_division=0)
    print(f"    {label:<35} F1={f1:.4f}  Recall={rec:.4f}  Prec={pre:.4f}")
    return f1


def _bar(v, w=15):
    return "█" * int(round(v * w)) + "░" * (w - int(round(v * w)))


# ══════════════════════════════════════════════════════════════
# FIX 1 — CHARGEMENT MLP ROBUSTE (détection mismatch d'architecture)
# ══════════════════════════════════════════════════════════════

def _load_mlp_safe(fpath, input_size):
    """
    Charge un MLP sauvegardé avec détection automatique de mismatch
    d'architecture.

    Si les clés du checkpoint ne correspondent pas à la classe MLP
    actuelle, lève une RuntimeError explicite avec les instructions
    pour résoudre le problème (rm + relancer), plutôt que de laisser
    PyTorch crasher avec un message cryptique.

    Retourne le modèle MLP chargé en mode eval.
    """
    model = MLP(input_size=input_size).to(DEVICE)

    checkpoint = torch.load(fpath, map_location=DEVICE)

    # Vérification des clés avant le chargement
    model_keys      = set(model.state_dict().keys())
    checkpoint_keys = set(checkpoint.keys())

    missing   = model_keys - checkpoint_keys
    unexpected = checkpoint_keys - model_keys

    if missing or unexpected:
        msg = (
            f"\n{'═'*60}\n"
            f"  MISMATCH D'ARCHITECTURE — {fpath.name}\n"
            f"{'═'*60}\n"
            f"  Le fichier .pt a été généré avec une architecture MLP\n"
            f"  différente de celle définie dans models.py.\n\n"
            f"  Clés manquantes  : {missing or 'aucune'}\n"
            f"  Clés inattendues : {unexpected or 'aucune'}\n\n"
            f"  Solution : supprimer les anciens checkpoints et relancer.\n"
            f"  $ rm {fpath}\n"
            f"  $ python3 defenses.py\n"
            f"{'═'*60}"
        )
        raise RuntimeError(msg)

    model.load_state_dict(checkpoint)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════
# DÉFENSE 1 & 2 — ADVERSARIAL TRAINING MLP
# ══════════════════════════════════════════════════════════════

def adversarial_train_mlp(X_train, y_train, X_test, y_test, input_size,
                           attack="fgsm",
                           epochs=AT_EPOCHS, patience=AT_PATIENCE,
                           mix_ratio=AT_MIX_RATIO):
    fname = f"mlp_at_{attack}.pt"
    fpath = SAVE_DIR / fname

    if fpath.exists():
        print(f"    {fname} déjà présent → chargement direct")
        # FIX 1 : utilise le chargement robuste au lieu de load_state_dict direct
        model = _load_mlp_safe(fpath, input_size)
        w = MLPWrapper(model, DEVICE)
        quick_eval(w, X_test, y_test, f"[chargé] {fname}")
        return w

    model = MLP(input_size=input_size).to(DEVICE)
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / (y_train == 1).sum()],
        dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t    = torch.tensor(X_train, dtype=torch.float32)
    y_t    = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=2048, shuffle=True)

    val_size = min(8000, len(X_train) // 5)
    X_val_t  = torch.tensor(X_train[:val_size], dtype=torch.float32).to(DEVICE)
    y_val    = y_train[:val_size]

    best_f1, no_improve, best_state = 0.0, 0, None
    eps_min, eps_max = min(EPS_AT_LIST), max(EPS_AT_LIST)

    for epoch in range(epochs):
        model.train()

        frac     = min(epoch / 30, 1.0)
        eps_curr = eps_min + frac * (eps_max - eps_min)

        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            tmp_w     = MLPWrapper(model, DEVICE)
            xb_np     = xb.detach().cpu().numpy()
            yb_np     = yb.detach().cpu().numpy().flatten().astype(int)

            if attack == "fgsm":
                xb_adv_np = fgsm_mlp(tmp_w, xb_np, yb_np, eps=eps_curr)
            else:
                xb_adv_np = pgd_mlp(tmp_w, xb_np, yb_np, eps=eps_curr,
                                     iters=PGD_AT_ITERS, restarts=1,
                                     alpha=PGD_AT_ALPHA(eps_curr))

            xb_adv  = torch.tensor(xb_adv_np, dtype=torch.float32, device=DEVICE)
            n_adv   = int(len(xb) * mix_ratio)
            idx_adv = torch.randperm(len(xb))[:n_adv]
            xb_mix  = xb.clone()
            xb_mix[idx_adv] = xb_adv[idx_adv]

            optimizer.zero_grad()
            criterion(model(xb_mix), yb).backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            proba  = torch.sigmoid(model(X_val_t)).cpu().numpy().flatten()
            y_pred = (proba >= THRESHOLD).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1, no_improve = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stop epoch {epoch+1} — best F1 {best_f1:.4f}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d} | F1 val {f1:.4f} | "
                  f"best {best_f1:.4f} | eps_curr {eps_curr:.3f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    torch.save(model.state_dict(), fpath)
    print(f"    Sauvegardé : {fpath}  (best F1 val {best_f1:.4f})")

    w = MLPWrapper(model, DEVICE)
    quick_eval(w, X_test, y_test, f"[clean] {fname}")
    return w


# ══════════════════════════════════════════════════════════════
# DÉFENSE 3 — AUGMENTATION LogReg (FGSM + PGD)
# ══════════════════════════════════════════════════════════════

def _build_aug_dataset(X_train, y_train, adv_fn, eps_list,
                       include_normal=True):
    X_parts = [X_train]
    y_parts = [y_train]

    mask_atk = (y_train == 1)
    X_atk    = X_train[mask_atk].astype(np.float32)
    y_atk    = y_train[mask_atk]

    for eps in eps_list:
        X_adv = adv_fn(X_atk, y_atk, eps)
        X_parts.append(X_adv)
        y_parts.append(y_atk)
        print(f"      eps={eps} → +{len(X_adv)} adversariaux (attaque)")

    if include_normal:
        mask_norm = (y_train == 0)
        X_norm    = X_train[mask_norm].astype(np.float32)
        y_norm    = y_train[mask_norm]
        n_normal  = int(len(X_norm) * NORMAL_AUG_RATIO)
        idx       = np.random.choice(len(X_norm), n_normal, replace=False)
        X_n_sub   = X_norm[idx]
        y_n_sub   = y_norm[idx]

        for eps in eps_list:
            X_adv_n = adv_fn(X_n_sub, y_n_sub, eps)
            X_parts.append(X_adv_n)
            y_parts.append(y_n_sub)
            print(f"      eps={eps} → +{len(X_adv_n)} adversariaux (normaux)")

    X_aug = np.concatenate(X_parts, axis=0)
    y_aug = np.concatenate(y_parts, axis=0)
    print(f"      Dataset : {len(X_train)} → {len(X_aug)} exemples (+{len(X_aug)-len(X_train)})")
    return X_aug, y_aug


def augment_logreg(logreg_wrapper, X_train, y_train, X_test, y_test,
                   attack="fgsm", eps_list=None):
    if eps_list is None:
        eps_list = EPS_AT_LIST

    fname = f"logreg_aug_{attack}.pkl"
    fpath = SAVE_DIR / fname

    if fpath.exists():
        print(f"    {fname} déjà présent → chargement direct")
        w = LogRegWrapper(joblib.load(fpath))
        quick_eval(w, X_test, y_test, f"[chargé] {fname}")
        return w

    print(f"    Génération X_adv LogReg ({attack}, eps={eps_list})...")

    if attack == "fgsm":
        adv_fn = lambda X, y, eps: fgsm_logreg(logreg_wrapper, X, y, eps)
    else:
        adv_fn = lambda X, y, eps: pgd_logreg(logreg_wrapper, X, y, eps,
                                               iters=20, restarts=3)

    X_aug, y_aug = _build_aug_dataset(X_train, y_train, adv_fn, eps_list)

    new_lr = LogisticRegression(
        C=1.0, max_iter=2000, solver="saga",
        class_weight="balanced", random_state=42
    )
    new_lr.fit(X_aug, y_aug)

    joblib.dump(new_lr, fpath)
    print(f"    Sauvegardé : {fpath}")

    w = LogRegWrapper(new_lr)
    quick_eval(w, X_test, y_test, f"[clean] {fname}")
    return w


# ══════════════════════════════════════════════════════════════
# DÉFENSE 4a — AUGMENTATION XGBoost via PROXY MLP
# ══════════════════════════════════════════════════════════════

def augment_xgb_proxy(xgb_wrapper, mlp_proxy_wrapper,
                      X_train, y_train, X_test, y_test,
                      attack="fgsm", eps_list=None):
    if eps_list is None:
        eps_list = EPS_AT_LIST

    fname = f"xgb_aug_proxy_{attack}.json"
    fpath = SAVE_DIR / fname

    if fpath.exists():
        print(f"    {fname} déjà présent → chargement direct")
        m = XGBClassifier()
        m.load_model(str(fpath))
        w = XGBoostWrapper(m)
        quick_eval(w, X_test, y_test, f"[chargé] {fname}")
        return w

    print(f"    Génération X_adv XGBoost via proxy MLP ({attack}, eps={eps_list})...")

    if attack == "fgsm":
        adv_fn = lambda X, y, eps: fgsm_mlp(mlp_proxy_wrapper, X, y, eps)
    else:
        adv_fn = lambda X, y, eps: pgd_mlp(mlp_proxy_wrapper, X, y, eps,
                                            iters=20, restarts=3,
                                            alpha=PGD_AT_ALPHA(eps))

    X_aug, y_aug = _build_aug_dataset(X_train, y_train, adv_fn, eps_list)

    w = _fit_xgb(X_aug, y_aug, fpath)
    quick_eval(w, X_test, y_test, f"[clean] {fname}")
    return w


# ══════════════════════════════════════════════════════════════
# DÉFENSE 4b — AUGMENTATION XGBoost DIRECTE (gradient numérique)
# ══════════════════════════════════════════════════════════════

def augment_xgb_direct(xgb_wrapper, X_train, y_train, X_test, y_test,
                        attack="fgsm", eps_list=None):
    if eps_list is None:
        eps_list = EPS_AT_LIST

    fname = f"xgb_aug_direct_{attack}.json"
    fpath = SAVE_DIR / fname

    if fpath.exists():
        print(f"    {fname} déjà présent → chargement direct")
        m = XGBClassifier()
        m.load_model(str(fpath))
        w = XGBoostWrapper(m)
        quick_eval(w, X_test, y_test, f"[chargé] {fname}")
        return w

    print(f"    Génération X_adv XGBoost DIRECTE ({attack}, eps={eps_list})...")

    if attack == "fgsm":
        adv_fn = lambda X, y, eps: fgsm_xgb(xgb_wrapper, X, y, eps)
    else:
        adv_fn = lambda X, y, eps: pgd_xgb(xgb_wrapper, X, y, eps,
                                            iters=50, restarts=5)

    X_aug, y_aug = _build_aug_dataset(X_train, y_train, adv_fn, eps_list)

    w = _fit_xgb(X_aug, y_aug, fpath)
    quick_eval(w, X_test, y_test, f"[clean] {fname}")
    return w


def _fit_xgb(X_aug, y_aug, fpath):
    scale_pw = float((y_aug == 0).sum()) / float((y_aug == 1).sum())
    new_xgb  = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pw,
        eval_metric="logloss", early_stopping_rounds=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        random_state=42, verbosity=0
    )
    idx = np.random.permutation(len(X_aug))
    X_aug, y_aug = X_aug[idx], y_aug[idx]
    split = int(0.9 * len(X_aug))

    new_xgb.fit(
        X_aug[:split], y_aug[:split],
        eval_set=[(X_aug[split:], y_aug[split:])],
        verbose=False
    )

    new_xgb.save_model(str(fpath))
    print(f"    Sauvegardé : {fpath}")
    return XGBoostWrapper(new_xgb)


# ══════════════════════════════════════════════════════════════
# FIX 2 + FIX 3 — ÉVALUATION DÉFENSIVE AVEC C&W, LAMBDAS CORRIGÉES
# ══════════════════════════════════════════════════════════════

def evaluate_defended_models(defended_models, X_test, y_test, eps=0.3):
    """
    Évalue chaque modèle défendu sous FGSM, PGD, C&W à eps=0.3.

    FIX 2 : les lambdas capturent X_atk/y_atk via paramètres par
    défaut (xatk=X_atk, yatk=y_atk) pour éviter la capture tardive.

    FIX 3 : C&W est maintenant inclus dans l'évaluation.

    Colonnes : F1 clean | ASR FGSM | ASR PGD | ASR C&W
    (toutes à eps=0.3, budget le plus sévère)
    """
    print(f"\n{'═'*72}")
    print(f"  ÉVALUATION DÉFENSIVE — eps={eps}")
    print(f"{'═'*72}")

    def _is_mlp(w):
        return hasattr(w, 'model') and isinstance(getattr(w, 'model', None), MLP)

    def _is_logreg(w):
        return hasattr(w, 'model') and hasattr(getattr(w, 'model', None), 'coef_')

    def _get_attack_fn(wrapper, attack_name, eps_val, X_atk, y_atk):
        """
        FIX 2 : X_atk et y_atk sont passés explicitement comme paramètres
        par défaut pour forcer la capture par valeur dans la lambda.
        """
        is_mlp    = _is_mlp(wrapper)
        is_logreg = _is_logreg(wrapper)

        if attack_name == "FGSM":
            if is_mlp:
                return lambda xatk=X_atk, yatk=y_atk: fgsm_mlp(wrapper, xatk, yatk, eps_val)
            elif is_logreg:
                return lambda xatk=X_atk, yatk=y_atk: fgsm_logreg(wrapper, xatk, yatk, eps_val)
            else:
                return lambda xatk=X_atk, yatk=y_atk: fgsm_xgb(wrapper, xatk, yatk, eps_val)

        elif attack_name == "PGD":
            if is_mlp:
                return lambda xatk=X_atk, yatk=y_atk: pgd_mlp(
                    wrapper, xatk, yatk, eps_val, iters=50, restarts=3)
            elif is_logreg:
                return lambda xatk=X_atk, yatk=y_atk: pgd_logreg(
                    wrapper, xatk, yatk, eps_val, iters=50, restarts=3)
            else:
                return lambda xatk=X_atk, yatk=y_atk: pgd_xgb(
                    wrapper, xatk, yatk, eps_val, iters=50, restarts=3)

        elif attack_name == "C&W":
            # FIX 3 : C&W inclus pour les 3 types de modèles
            if is_mlp:
                return lambda xatk=X_atk, yatk=y_atk: cw_mlp(
                    wrapper, xatk, yatk, eps_val)
            elif is_logreg:
                return lambda xatk=X_atk, yatk=y_atk: cw_logreg(
                    wrapper, xatk, yatk, eps_val)
            else:
                return lambda xatk=X_atk, yatk=y_atk: cw_xgb(
                    wrapper, xatk, yatk, eps_val)

        raise ValueError(f"Attaque inconnue : {attack_name}")

    attacks = ["FGSM", "PGD", "C&W"]
    header  = (f"  {'Modèle':<30} {'F1 clean':>9} "
               f"{'FGSM':>9} {'PGD':>9} {'C&W':>9}")
    print(header)
    print(f"  {'─'*70}")

    results = {}

    for label, wrapper in defended_models.items():
        y_pred_clean = wrapper.predict(X_test)
        f1_clean     = f1_score(y_test, y_pred_clean, zero_division=0)

        tp_mask = (y_test == 1) & (y_pred_clean == 1)
        X_atk   = X_test[tp_mask].astype(np.float32)
        y_atk   = y_test[tp_mask]

        row      = f"  {label:<30} {f1_clean:>9.4f}"
        row_data = {"f1_clean": f1_clean}

        for att in attacks:
            if tp_mask.sum() == 0:
                row += f" {'N/A':>9}"
                row_data[att] = None
                continue

            fn    = _get_attack_fn(wrapper, att, eps, X_atk, y_atk)
            X_adv = fn()
            y_adv = wrapper.predict(X_adv)
            asr   = float((y_adv == 0).mean()) if len(y_adv) > 0 else 0.0

            row += f" {asr*100:>8.1f}%"
            row_data[att] = asr

        print(row)
        results[label] = row_data

    print(f"  {'─'*70}")
    return results


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run():
    print("\n" + "═"*60)
    print("  CHARGEMENT DES ARTIFACTS")
    print("═"*60)
    X_train, y_train, X_test, y_test, mlp_w, logreg_w, xgb_w = load_artifacts()
    input_size = X_train.shape[1]

    print(f"\n  eps entraînement : {EPS_AT_LIST}")
    print(f"  PGD AT : iters={PGD_AT_ITERS}, restarts=1, alpha=eps/4")

    print("\n" + "═"*60)
    print("  ENTRAÎNEMENT DES MODÈLES DÉFENDUS")
    print("═"*60)

    print("\n[1/6] Adversarial Training FGSM — MLP")
    mlp_at_fgsm = adversarial_train_mlp(
        X_train, y_train, X_test, y_test, input_size, attack="fgsm"
    )

    print("\n[2/6] Adversarial Training PGD-10 — MLP")
    mlp_at_pgd = adversarial_train_mlp(
        X_train, y_train, X_test, y_test, input_size, attack="pgd"
    )

    print("\n[3/6] Augmentation FGSM — LogReg (double eps + normaux)")
    logreg_aug_fgsm = augment_logreg(
        logreg_w, X_train, y_train, X_test, y_test, attack="fgsm"
    )

    print("\n[4/6] Augmentation PGD — LogReg (double eps + normaux)")
    logreg_aug_pgd = augment_logreg(
        logreg_w, X_train, y_train, X_test, y_test, attack="pgd"
    )

    print("\n[5/6] Augmentation XGBoost via proxy MLP AT-FGSM")
    xgb_aug_proxy = augment_xgb_proxy(
        xgb_w, mlp_at_fgsm, X_train, y_train, X_test, y_test, attack="fgsm"
    )

    print("\n[6/6] Augmentation XGBoost DIRECTE (gradient numérique FGSM)")
    xgb_aug_direct = augment_xgb_direct(
        xgb_w, X_train, y_train, X_test, y_test, attack="fgsm"
    )

    # ── Évaluation défensive (FGSM + PGD + C&W à eps=0.3) ───
    defended_models = {
        "MLP baseline":           mlp_w,
        "MLP AT-FGSM":            mlp_at_fgsm,
        "MLP AT-PGD10":           mlp_at_pgd,
        "LogReg baseline":        logreg_w,
        "LogReg Aug-FGSM":        logreg_aug_fgsm,
        "LogReg Aug-PGD":         logreg_aug_pgd,
        "XGBoost baseline":       xgb_w,
        "XGBoost Aug-proxy-FGSM": xgb_aug_proxy,
        "XGBoost Aug-direct-FGSM":xgb_aug_direct,
    }

    results = evaluate_defended_models(defended_models, X_test, y_test, eps=0.3)

    # ── Sauvegarde JSON des résultats défensifs ──────────────
    import json
    RESULTS_DIR = Path("~/swat/results").expanduser()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "defense_whitebox_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Résultats sauvegardés → {out_path}")

    print("\n" + "═"*60)
    print("  DONE — artéfacts sauvegardés dans ~/swat/artifacts/")
    print("═"*60)

    return (mlp_at_fgsm, mlp_at_pgd,
            logreg_aug_fgsm, logreg_aug_pgd,
            xgb_aug_proxy, xgb_aug_direct)


if __name__ == "__main__":
    run()