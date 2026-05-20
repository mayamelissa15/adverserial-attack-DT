# ~/swat/defenses.py
# Entraînement des modèles défendus :
#   - AT-FGSM MLP  → mlp_at_fgsm.pt
#   - AT-PGD  MLP  → mlp_at_pgd.pt
#   - Aug-FGSM LogReg   → logreg_aug_fgsm.pkl
#   - Aug-FGSM XGBoost  → xgb_aug_fgsm.json  (proxy = AT-FGSM MLP)
#
# Si le fichier existe déjà dans artifacts/, on le charge sans réentraîner.
# Usage : python defenses.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

import sys
import importlib.util

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from models import (MLP, MLPWrapper, LogRegWrapper, XGBoostWrapper)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

wb = _load_module("whitebox", BASE / "whitebox.py")

fgsm_mlp    = wb.fgsm_mlp
fgsm_logreg = wb.fgsm_logreg
pgd_mlp     = wb.pgd_mlp
pgd_logreg  = wb.pgd_logreg

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

SAVE_DIR = Path("~/swat/artifacts").expanduser()
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.45
EPS_AT    = 0.1   # eps utilisé pendant l'adversarial training

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
# DÉFENSE 1 & 2 — ADVERSARIAL TRAINING MLP
#
# À chaque batch :
#   1. génère X_adv via FGSM ou PGD sur le modèle courant
#   2. mélange 50/50 clean + adv
#   3. entraîne sur le mix
#
# Pourquoi pas C&W ?  C&W = 300 iters par exemple → trop lent en training.
# Pourquoi pas blackbox ?  Pas de gradient → incompatible avec le training.
# ══════════════════════════════════════════════════════════════

def adversarial_train_mlp(X_train, y_train, input_size,
                           attack="fgsm", eps=EPS_AT,
                           epochs=50, patience=7, mix_ratio=0.5):
    """
    Retourne un MLPWrapper entraîné de manière adversariale.
    Skip l'entraînement si le fichier .pt existe déjà.
    """
    fname  = f"mlp_at_{attack}.pt"
    fpath  = SAVE_DIR / fname

    # ── Skip si déjà entraîné ───────────────────────────────
    if fpath.exists():
        print(f"    {fname} déjà présent → chargement direct")
        model = MLP(input_size=input_size).to(DEVICE)
        model.load_state_dict(torch.load(fpath, map_location=DEVICE))
        model.eval()
        return MLPWrapper(model, DEVICE)

    # ── Entraînement ────────────────────────────────────────
    model = MLP(input_size=input_size).to(DEVICE)
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / (y_train == 1).sum()],
        dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_t    = torch.tensor(X_train, dtype=torch.float32)
    y_t    = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=2048, shuffle=True)

    X_val_t = torch.tensor(X_train[:5000], dtype=torch.float32).to(DEVICE)
    y_val   = y_train[:5000]

    best_f1, no_improve, best_state = 0, 0, None

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            tmp_w  = MLPWrapper(model, DEVICE)
            xb_np  = xb.detach().cpu().numpy()
            yb_np  = yb.detach().cpu().numpy().flatten().astype(int)

            if attack == "fgsm":
                xb_adv_np = fgsm_mlp(tmp_w, xb_np, yb_np, eps=eps)
            else:
                xb_adv_np = pgd_mlp(tmp_w, xb_np, yb_np, eps=eps,
                                     iters=7, restarts=1)

            xb_adv  = torch.tensor(xb_adv_np, dtype=torch.float32, device=DEVICE)
            n_adv   = int(len(xb) * mix_ratio)
            idx_adv = torch.randperm(len(xb))[:n_adv]
            xb_mix  = xb.clone()
            xb_mix[idx_adv] = xb_adv[idx_adv]

            optimizer.zero_grad()
            criterion(model(xb_mix), yb).backward()
            optimizer.step()

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
            print(f"    Epoch {epoch+1:3d} | F1 val {f1:.4f} | best {best_f1:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    torch.save(model.state_dict(), fpath)
    print(f"    Sauvegardé : {fpath}  (best F1 val {best_f1:.4f})")
    return MLPWrapper(model, DEVICE)


# ══════════════════════════════════════════════════════════════
# DÉFENSE 3 — ADVERSARIAL AUGMENTATION LogReg
#
# LogReg n'a pas de boucle itérative PyTorch.
# On génère X_adv avant le fit, on concatène, on refit.
# ══════════════════════════════════════════════════════════════

def augment_logreg(logreg_wrapper, X_train, y_train,
                   attack="fgsm", eps=EPS_AT):
    """
    Retourne un LogRegWrapper réentraîné sur X_train + X_adv.
    Skip si le fichier .pkl existe déjà.
    """
    fname = f"logreg_aug_{attack}.pkl"
    fpath = SAVE_DIR / fname

    if fpath.exists():
        print(f"    {fname} déjà présent → chargement direct")
        return LogRegWrapper(joblib.load(fpath))

    mask  = (y_train == 1)
    X_atk = X_train[mask].astype(np.float32)
    y_atk = y_train[mask]

    print(f"    Génération X_adv LogReg ({attack}, eps={eps}) "
          f"sur {len(X_atk)} exemples attack...")

    if attack == "fgsm":
        X_adv = fgsm_logreg(logreg_wrapper, X_atk, y_atk, eps=eps)
    else:
        X_adv = pgd_logreg(logreg_wrapper, X_atk, y_atk, eps=eps,
                            iters=20, restarts=3)

    X_aug = np.concatenate([X_train, X_adv], axis=0)
    y_aug = np.concatenate([y_train, y_atk], axis=0)
    print(f"    Dataset : {len(X_train)} → {len(X_aug)} exemples")

    new_lr = LogisticRegression(
        C=1.0, max_iter=1000, solver="saga",
        class_weight="balanced", random_state=42
    )
    new_lr.fit(X_aug, y_aug)

    joblib.dump(new_lr, fpath)
    print(f"    Sauvegardé : {fpath}")
    return LogRegWrapper(new_lr)


# ══════════════════════════════════════════════════════════════
# DÉFENSE 4 — ADVERSARIAL AUGMENTATION XGBoost via proxy MLP
#
# XGBoost non différentiable → pas de gradient direct.
# On génère X_adv via le MLP AT (proxy), puis on refit XGBoost.
# ══════════════════════════════════════════════════════════════
"""
def augment_xgboost(xgb_wrapper, mlp_proxy_wrapper,
                    X_train, y_train,
                    attack="fgsm", eps=EPS_AT):
    
    fname = f"xgb_aug_{attack}.json"
    fpath = SAVE_DIR / fname

    if fpath.exists():
        print(f"    {fname} déjà présent → chargement direct")
        m = XGBClassifier()
        m.load_model(str(fpath))
        return XGBoostWrapper(m)

    mask  = (y_train == 1)
    X_atk = X_train[mask].astype(np.float32)
    y_atk = y_train[mask]

    print(f"    Génération X_adv XGBoost via proxy MLP ({attack}, eps={eps}) "
          f"sur {len(X_atk)} exemples attack...")

    if attack == "fgsm":
        X_adv = fgsm_mlp(mlp_proxy_wrapper, X_atk, y_atk, eps=eps)
    else:
        X_adv = pgd_mlp(mlp_proxy_wrapper, X_atk, y_atk, eps=eps,
                         iters=20, restarts=3)

    X_aug = np.concatenate([X_train, X_adv], axis=0)
    y_aug = np.concatenate([y_train, y_atk], axis=0)
    print(f"    Dataset : {len(X_train)} → {len(X_aug)} exemples")

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
    new_xgb.fit(
        X_aug[:split], y_aug[:split],
        eval_set=[(X_aug[split:], y_aug[split:])],
        verbose=False
    )

    new_xgb.save_model(str(fpath))
    print(f"    Sauvegardé : {fpath}")
    return XGBoostWrapper(new_xgb)"""

# ══════════════════════════════════════════════════════════════
# DÉFENSE 5 — ADVERSARIAL AUGMENTATION ITÉRATIVE XGBoost
#
# Contrairement à augment_xgboost() qui utilise un proxy MLP,
# ici on génère X_adv directement sur XGBoost courant à chaque
# round via gradient numérique (différences finies centrées).
#
# Round k :
#   1. XGB_k génère X_adv via fgsm_xgb (grad numérique)
#   2. On concat X_train + tous les X_adv vus jusqu'ici
#   3. On refit → XGB_{k+1}
#
# Avantage vs proxy : les X_adv correspondent aux vraies
# failles de XGBoost, pas à celles du MLP.
# ══════════════════════════════════════════════════════════════

def augment_xgboost_iterative(xgb_wrapper, X_train, y_train,
                               attack="fgsm", eps=EPS_AT,
                               n_rounds=3):
    """
    Adversarial augmentation itérative sur XGBoost.
    Génère X_adv sur le modèle courant à chaque round (pas de proxy).
    Skip si le fichier final existe déjà.
    """
    fname = f"xgb_iter_{attack}_r{n_rounds}.json"
    fpath = SAVE_DIR / fname

    if fpath.exists():
        print(f"    {fname} déjà présent → chargement direct")
        m = XGBClassifier()
        m.load_model(str(fpath))
        return XGBoostWrapper(m)

    # Import des attaques XGBoost depuis whitebox.py
    fgsm_xgb = wb.fgsm_xgb
    pgd_xgb  = wb.pgd_xgb

    current_wrapper = xgb_wrapper
    X_aug = X_train.copy()
    y_aug = y_train.copy()

    mask  = (y_train == 1)
    X_atk = X_train[mask].astype(np.float32)
    y_atk = y_train[mask]

    for r in range(1, n_rounds + 1):
        print(f"\n    ── Round {r}/{n_rounds} ──────────────────────────")

        # 1. Génère X_adv sur le modèle COURANT (pas de proxy)
        print(f"    Génération X_adv sur XGB courant ({attack}, eps={eps})...")
        if attack == "fgsm":
            X_adv = fgsm_xgb(current_wrapper, X_atk, y_atk, eps=eps)
        else:
            X_adv = pgd_xgb(current_wrapper, X_atk, y_atk, eps=eps,
                             iters=20, restarts=3)

        # 2. Accumule — on garde tous les X_adv des rounds précédents
        X_aug = np.concatenate([X_aug, X_adv], axis=0)
        y_aug = np.concatenate([y_aug, y_atk], axis=0)
        print(f"    Dataset cumulé : {len(X_aug)} exemples")

        # 3. Refit XGBoost sur le dataset augmenté
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
        new_xgb.fit(
            X_aug[:split], y_aug[:split],
            eval_set=[(X_aug[split:], y_aug[split:])],
            verbose=False
        )

        current_wrapper = XGBoostWrapper(new_xgb)
        print(f"    XGB_round{r} fitté ✓")

    # Sauvegarde du modèle final
    current_wrapper.model.save_model(str(fpath))
    print(f"\n    Sauvegardé : {fpath}")
    return current_wrapper

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run():
    print("\n" + "═"*60)
    print("  CHARGEMENT DES ARTIFACTS")
    print("═"*60)
    X_train, y_train, X_test, y_test, mlp_w, logreg_w, xgb_w = load_artifacts()
    input_size = X_train.shape[1]

    print("\n" + "═"*60)
    print("  ENTRAÎNEMENT DES MODÈLES DÉFENDUS")
    print("═"*60)

    print("\n[1/4] Adversarial Training FGSM — MLP")
    mlp_at_fgsm = adversarial_train_mlp(
        X_train, y_train, input_size, attack="fgsm", eps=EPS_AT
    )

    print("\n[2/4] Adversarial Training PGD — MLP")
    mlp_at_pgd = adversarial_train_mlp(
        X_train, y_train, input_size, attack="pgd", eps=EPS_AT
    )

    print("\n[3/4] Adversarial Augmentation FGSM — LogReg")
    logreg_aug = augment_logreg(logreg_w, X_train, y_train, attack="fgsm")

    """print("\n[4/4] Adversarial Augmentation FGSM ( basique )— XGBoost (proxy = AT-FGSM MLP)")
    xgb_aug = augment_xgboost(xgb_w, mlp_at_fgsm, X_train, y_train, attack="fgsm")"""

    #on debug avec ca 
    print("\n[5/5] Adversarial Augmentation Itérative FGSM — XGBoost (self, 3 rounds)")
    xgb_iter = augment_xgboost_iterative( xgb_w, X_train, y_train, attack="fgsm", eps=EPS_AT, n_rounds=3)

    print("\n" + "═"*60)
    print("  DONE — lance maintenant evaluate.py")
    print("═"*60)

    return mlp_at_fgsm, mlp_at_pgd, logreg_aug, xgb_iter


if __name__ == "__main__":
    run()