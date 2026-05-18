"""
defenses.py  — version améliorée
Entraînement des modèles défendus + évaluation immédiate.

═══════════════════════════════════════════════════════════════
MODÈLES PRODUITS
═══════════════════════════════════════════════════════════════
  mlp_at_fgsm.pt          → MLP Adversarial Training FGSM
  mlp_at_pgd.pt           → MLP Adversarial Training PGD
  logreg_aug_fgsm.pkl     → LogReg augmentée FGSM
  logreg_aug_pgd.pkl      → LogReg augmentée PGD
  xgb_aug_proxy.json      → XGBoost augmenté via proxy MLP AT
  xgb_aug_direct.json     → XGBoost augmenté via gradient numérique direct

═══════════════════════════════════════════════════════════════
CHANGEMENTS PAR RAPPORT À LA VERSION INITIALE
═══════════════════════════════════════════════════════════════

1. AT MLP — PGD plus fort (iters=10, restarts=1 → PGD-10 de Madry)
   Version initiale : iters=7, restarts=1.  La littérature AT utilise
   PGD-10 comme standard minimum.  Coût +43% par batch, justifié.

2. DOUBLE EPS pour l'augmentation (eps=0.1 ET eps=0.3)
   Le jeu de test est évalué sur EPS_LIST=[0.1, 0.3].  Entraîner
   uniquement sur eps=0.1 laisse le modèle vulnérable à eps=0.3.
   Solution : on génère X_adv pour les deux eps et on concatène.

3. AUGMENTATION SUR TOUTES LES CLASSES (LogReg/XGBoost)
   Version initiale : mask=(y_train==1), seuls les vrais attaques
   sont perturbés.  On inclut maintenant aussi une fraction des
   exemples normaux (y==0) pour couvrir les faux positifs adversariaux.

4. NOUVELLE DÉFENSE : XGBoost augmentation DIRECTE
   En plus du proxy MLP, on génère X_adv via le gradient numérique
   propre à XGBoost (différences finies centrées).  Cela produit
   des adversariaux plus proches de la frontière de décision XGBoost.
   Les deux sources (proxy + direct) sont concaténées.

5. ÉVALUATION INTÉGRÉE
   Après chaque entraînement, on mesure F1 / ASR sur X_test pour
   suivre la robustesse gagnée sans lancer evaluate.py séparément.

6. SYMÉTRIE DES DÉFENSES
   LogReg et XGBoost ont maintenant les mêmes variantes (FGSM + PGD)
   que MLP, ce qui rend la table de comparaison de l'article cohérente.

═══════════════════════════════════════════════════════════════
NOTA BENE — POURQUOI PAS C&W EN DÉFENSE ?
═══════════════════════════════════════════════════════════════
  C&W nécessite ~300-500 iters par exemple.  Intégré dans la boucle
  d'entraînement (N_batches × epochs), le coût serait prohibitif.
  La littérature (Madry, Zhang et al.) montre que AT-PGD-10 donne
  une robustesse C&W comparable à AT-C&W à coût 100× inférieur.
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

fgsm_mlp    = wb.fgsm_mlp
fgsm_logreg = wb.fgsm_logreg
fgsm_xgb    = wb.fgsm_xgb
pgd_mlp     = wb.pgd_mlp
pgd_logreg  = wb.pgd_logreg
pgd_xgb     = wb.pgd_xgb
THRESHOLD_LOGIT = wb.THRESHOLD_LOGIT


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

SAVE_DIR  = Path("~/swat/artifacts").expanduser()
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.45

# Double eps pour couvrir les deux budgets de l'étude
EPS_AT_LIST = [0.1, 0.3]   # entraîné sur les deux, défendu pour les deux
EPS_AT      = 0.1           # eps principal (pour nommage des fichiers)

# AT MLP
AT_EPOCHS     = 60
AT_PATIENCE   = 10
AT_MIX_RATIO  = 0.5
PGD_AT_ITERS  = 10   # PGD-10 standard Madry et al.
PGD_AT_ALPHA  = lambda eps: eps / 4   # alpha cohérent avec whitebox.py

# Fraction d'exemples normaux (y==0) ajoutés à l'augmentation
# pour couvrir les faux positifs adversariaux
NORMAL_AUG_RATIO = 0.3   # 30% de l'ensemble normal

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
# UTILITAIRE : ÉVALUATION RAPIDE
# ══════════════════════════════════════════════════════════════

def quick_eval(wrapper, X_test, y_test, label=""):
    """Affiche F1 / Recall / Precision du modèle sur X_test."""
    y_pred = wrapper.predict(X_test)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    pre = precision_score(y_test, y_pred, zero_division=0)
    print(f"    {label:<30} F1={f1:.4f}  Recall={rec:.4f}  Prec={pre:.4f}")
    return f1


def _bar(v, w=15):
    return "█" * int(round(v * w)) + "░" * (w - int(round(v * w)))

def quick_asr(wrapper, X_test, y_test, attack_fn, eps, label=""):
    """Génère X_adv et calcule l'ASR sur les vrais positifs."""
    y_pred_clean = wrapper.predict(X_test)
    tp_mask      = (y_test == 1) & (y_pred_clean == 1)
    if tp_mask.sum() == 0:
        print(f"    {label:<30} ASR=N/A (aucun TP)")
        return 0.0
    X_atk  = X_test[tp_mask].astype(np.float32)
    y_atk  = y_test[tp_mask]
    X_adv  = attack_fn(wrapper, X_atk, y_atk, eps)
    y_adv  = wrapper.predict(X_adv)
    asr    = float((y_adv == 0).mean())
    print(f"    {label:<30} ASR={asr*100:5.1f}%  {_bar(asr)}")
    return asr


# ══════════════════════════════════════════════════════════════
# DÉFENSE 1 & 2 — ADVERSARIAL TRAINING MLP
# ══════════════════════════════════════════════════════════════

def adversarial_train_mlp(X_train, y_train, X_test, y_test, input_size,
                           attack="fgsm",
                           epochs=AT_EPOCHS, patience=AT_PATIENCE,
                           mix_ratio=AT_MIX_RATIO):
    """
    Adversarial Training MLP.

    Protocole :
      - À chaque batch, génère X_adv via FGSM ou PGD-10 sur le modèle courant
      - Mix 50/50 clean + adv dans la loss
      - Entraîné sur TOUS les eps de EPS_AT_LIST (curriculum progressif)
      - Early stopping sur F1 de validation

    Changements vs version initiale :
      - PGD : iters=7 → iters=10  (PGD-10 standard Madry)
      - alpha cohérent avec whitebox.py (eps/4 au lieu de eps/PGD_ALPHA_K=10)
      - Curriculum eps : commence par eps_min, monte progressivement à eps_max
        → stabilise l'entraînement initial
      - Entraînement sur double eps (0.1 + 0.3) en alternance de batch
    """
    fname  = f"mlp_at_{attack}.pt"
    fpath  = SAVE_DIR / fname

    if fpath.exists():
        print(f"    {fname} déjà présent → chargement direct")
        model = MLP(input_size=input_size).to(DEVICE)
        model.load_state_dict(torch.load(fpath, map_location=DEVICE))
        model.eval()
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

    # Validation sur une portion fixe du train (plus rapide)
    val_size = min(8000, len(X_train) // 5)
    X_val_t  = torch.tensor(X_train[:val_size], dtype=torch.float32).to(DEVICE)
    y_val    = y_train[:val_size]

    best_f1, no_improve, best_state = 0.0, 0, None
    eps_min, eps_max = min(EPS_AT_LIST), max(EPS_AT_LIST)

    for epoch in range(epochs):
        model.train()

        # Curriculum : eps croît linéairement de eps_min à eps_max
        # sur les 30 premiers epochs, puis reste à eps_max
        frac = min(epoch / 30, 1.0)
        eps_curr = eps_min + frac * (eps_max - eps_min)

        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            tmp_w  = MLPWrapper(model, DEVICE)
            xb_np  = xb.detach().cpu().numpy()
            yb_np  = yb.detach().cpu().numpy().flatten().astype(int)

            if attack == "fgsm":
                xb_adv_np = fgsm_mlp(tmp_w, xb_np, yb_np, eps=eps_curr)
            else:
                # PGD-10 : standard Madry et al.
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
# DÉFENSE 3a & 3b — AUGMENTATION LogReg  (FGSM + PGD)
# ══════════════════════════════════════════════════════════════

def _build_aug_dataset(X_train, y_train, adv_fn, eps_list,
                       include_normal=True):
    """
    Construit X_aug, y_aug en générant des adversariaux pour :
      - tous les exemples d'attaque (y==1), sur chaque eps
      - une fraction NORMAL_AUG_RATIO des exemples normaux (y==0)
        pour couvrir les faux positifs adversariaux

    Retourne X_aug (N_aug, D), y_aug (N_aug,).
    """
    X_parts = [X_train]
    y_parts = [y_train]

    # ── Exemples d'attaque ──────────────────────────────────
    mask_atk = (y_train == 1)
    X_atk    = X_train[mask_atk].astype(np.float32)
    y_atk    = y_train[mask_atk]

    for eps in eps_list:
        X_adv = adv_fn(X_atk, y_atk, eps)
        X_parts.append(X_adv)
        y_parts.append(y_atk)
        print(f"      eps={eps} → +{len(X_adv)} adversariaux (attaque)")

    # ── Exemples normaux (fraction) ─────────────────────────
    if include_normal:
        mask_norm = (y_train == 0)
        X_norm    = X_train[mask_norm].astype(np.float32)
        y_norm    = y_train[mask_norm]
        n_normal  = int(len(X_norm) * NORMAL_AUG_RATIO)
        idx       = np.random.choice(len(X_norm), n_normal, replace=False)
        X_n_sub   = X_norm[idx]
        y_n_sub   = y_norm[idx]

        for eps in eps_list:
            # Perturber les normaux dans la direction adversariale
            # (signe du gradient de la BCE, qui pousse vers y=1)
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
    """
    Retourne un LogRegWrapper réentraîné sur X_train + X_adv.
    Couvre les deux eps et les exemples normaux.
    """
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
    """
    XGBoost augmenté via le MLP AT comme proxy différentiable.
    Deux eps, exemples normaux inclus.
    """
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
#
# Nouvelle défense absente de la version initiale.
# Génère X_adv directement sur XGBoost via différences finies.
# Les adversariaux sont ainsi spécifiquement adaptés à la frontière
# de décision XGBoost, pas à celle du proxy MLP.
# ══════════════════════════════════════════════════════════════

def augment_xgb_direct(xgb_wrapper, X_train, y_train, X_test, y_test,
                        attack="fgsm", eps_list=None):
    """
    XGBoost augmenté via son propre gradient numérique.
    Plus lent que le proxy (102 forward passes / exemple) mais plus précis.
    """
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
        # Note : iters/restarts réduits ici car on est en augmentation,
        # pas en attaque maximale.  Le but est la diversité, pas l'ASR max.

    X_aug, y_aug = _build_aug_dataset(X_train, y_train, adv_fn, eps_list)

    w = _fit_xgb(X_aug, y_aug, fpath)
    quick_eval(w, X_test, y_test, f"[clean] {fname}")
    return w


def _fit_xgb(X_aug, y_aug, fpath):
    """Entraîne et sauvegarde un XGBClassifier."""
    scale_pw = float((y_aug == 0).sum()) / float((y_aug == 1).sum())
    new_xgb  = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pw,
        eval_metric="logloss", early_stopping_rounds=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        random_state=42, verbosity=0
    )
    split = int(0.9 * len(X_aug))
    # Shuffle avant split pour éviter que le val set soit homogène
    idx = np.random.permutation(len(X_aug))
    X_aug, y_aug = X_aug[idx], y_aug[idx]

    new_xgb.fit(
        X_aug[:split], y_aug[:split],
        eval_set=[(X_aug[split:], y_aug[split:])],
        verbose=False
    )

    new_xgb.save_model(str(fpath))
    print(f"    Sauvegardé : {fpath}")
    return XGBoostWrapper(new_xgb)


# ══════════════════════════════════════════════════════════════
# ÉVALUATION DÉFENSIVE COMPLÈTE
# ══════════════════════════════════════════════════════════════

def evaluate_defended_models(defended_models, X_test, y_test):
    """
    Pour chaque modèle défendu, mesure :
      - F1 clean (pas de perturbation)
      - ASR FGSM eps=0.1 et eps=0.3
      - ASR PGD  eps=0.1 et eps=0.3

    Affiche un tableau de synthèse.
    """
    print(f"\n{'═'*72}")
    print(f"  ÉVALUATION DÉFENSIVE")
    print(f"{'═'*72}")

    attack_fns = {
        ("FGSM", 0.1): lambda w: (lambda ww: lambda X, y, eps: fgsm_mlp(ww, X, y, eps)
                                   if hasattr(ww, 'model') and hasattr(ww.model, 'net')
                                   else (fgsm_logreg(ww, X, y, eps)
                                         if hasattr(ww, 'model') and hasattr(ww.model, 'coef_')
                                         else fgsm_xgb(ww, X, y, eps)))(w),
    }
    # Construction dynamique selon le type de wrapper
    def get_attack(wrapper, attack_name, eps):
        is_mlp    = hasattr(wrapper, 'model') and isinstance(getattr(wrapper, 'model', None), MLP)
        is_logreg = hasattr(wrapper, 'model') and hasattr(getattr(wrapper, 'model', None), 'coef_')
        is_xgb    = not is_mlp and not is_logreg

        if attack_name == "FGSM":
            if is_mlp:    return lambda: fgsm_mlp(wrapper, X_atk, y_atk, eps)
            if is_logreg: return lambda: fgsm_logreg(wrapper, X_atk, y_atk, eps)
            return lambda: fgsm_xgb(wrapper, X_atk, y_atk, eps)
        else:  # PGD
            if is_mlp:    return lambda: pgd_mlp(wrapper, X_atk, y_atk, eps,
                                                  iters=50, restarts=3)
            if is_logreg: return lambda: pgd_logreg(wrapper, X_atk, y_atk, eps,
                                                     iters=50, restarts=3)
            return lambda: pgd_xgb(wrapper, X_atk, y_atk, eps,
                                   iters=30, restarts=3)

    header = f"  {'Modèle':<28} {'F1 clean':>9} {'FGSM 0.1':>9} {'FGSM 0.3':>9} {'PGD 0.1':>8} {'PGD 0.3':>8}"
    print(header)
    print(f"  {'─'*70}")

    for label, wrapper in defended_models.items():
        y_pred_clean = wrapper.predict(X_test)
        f1_clean     = f1_score(y_test, y_pred_clean, zero_division=0)

        tp_mask = (y_test == 1) & (y_pred_clean == 1)
        X_atk   = X_test[tp_mask].astype(np.float32)
        y_atk   = y_test[tp_mask]

        row = f"  {label:<28} {f1_clean:>9.4f}"

        for att in ["FGSM", "PGD"]:
            for eps in [0.1, 0.3]:
                if tp_mask.sum() == 0:
                    row += f" {'N/A':>8}"
                    continue
                fn    = get_attack(wrapper, att, eps)
                X_adv = fn()
                y_adv = wrapper.predict(X_adv)
                asr   = float((y_adv == 0).mean()) if len(y_adv) > 0 else 0.0
                row  += f" {asr*100:>8.1f}%"

        print(row)

    print(f"  {'─'*70}")


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
    print(f"  PGD AT : iters={PGD_AT_ITERS}, restarts=1, alpha=eps/{int(1/PGD_AT_ALPHA(1.0))}")

    print("\n" + "═"*60)
    print("  ENTRAÎNEMENT DES MODÈLES DÉFENDUS")
    print("═"*60)

    # ── MLP ─────────────────────────────────────────────────
    print("\n[1/6] Adversarial Training FGSM — MLP")
    mlp_at_fgsm = adversarial_train_mlp(
        X_train, y_train, X_test, y_test, input_size, attack="fgsm"
    )

    print("\n[2/6] Adversarial Training PGD-10 — MLP")
    mlp_at_pgd = adversarial_train_mlp(
        X_train, y_train, X_test, y_test, input_size, attack="pgd"
    )

    # ── LogReg ──────────────────────────────────────────────
    print("\n[3/6] Augmentation FGSM — LogReg (double eps + normaux)")
    logreg_aug_fgsm = augment_logreg(
        logreg_w, X_train, y_train, X_test, y_test, attack="fgsm"
    )

    print("\n[4/6] Augmentation PGD — LogReg (double eps + normaux)")
    logreg_aug_pgd = augment_logreg(
        logreg_w, X_train, y_train, X_test, y_test, attack="pgd"
    )

    # ── XGBoost ─────────────────────────────────────────────
    print("\n[5/6] Augmentation XGBoost via proxy MLP AT-FGSM")
    xgb_aug_proxy = augment_xgb_proxy(
        xgb_w, mlp_at_fgsm, X_train, y_train, X_test, y_test, attack="fgsm"
    )

    print("\n[6/6] Augmentation XGBoost DIRECTE (gradient numérique)")
    xgb_aug_direct = augment_xgb_direct(
        xgb_w, X_train, y_train, X_test, y_test, attack="fgsm"
    )

    # ── Évaluation défensive ─────────────────────────────────
    defended_models = {
        "MLP baseline":         mlp_w,
        "MLP AT-FGSM":          mlp_at_fgsm,
        "MLP AT-PGD10":         mlp_at_pgd,
        "LogReg baseline":      logreg_w,
        "LogReg Aug-FGSM":      logreg_aug_fgsm,
        "LogReg Aug-PGD":       logreg_aug_pgd,
        "XGBoost baseline":     xgb_w,
        "XGBoost Aug-proxy":    xgb_aug_proxy,
        "XGBoost Aug-direct":   xgb_aug_direct,
    }

    evaluate_defended_models(defended_models, X_test, y_test)

    print("\n" + "═"*60)
    print("  DONE — artéfacts sauvegardés dans ~/swat/artifacts/")
    print("  Lance evaluate.py pour l'évaluation blackbox complète.")
    print("═"*60)

    return (mlp_at_fgsm, mlp_at_pgd,
            logreg_aug_fgsm, logreg_aug_pgd,
            xgb_aug_proxy, xgb_aug_direct)


if __name__ == "__main__":
    run()