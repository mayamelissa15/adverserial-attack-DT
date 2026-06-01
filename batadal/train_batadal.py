# ~/swat/batadal/train_batadal.py
"""
Entraînement MLP / LogReg / XGBoost sur BATADAL.

Structure BATADAL :
  dataset03 : 8761 exemples normaux uniquement (ATT_FLAG = 0)
  dataset04 : 3958 normaux (ATT_FLAG = -999) + 219 attaques (ATT_FLAG = 1)

Stratégie :
  - Train : dataset03 (normal) + exemples normaux de dataset04
  - Test  : dataset04 complet  (-999 → 0, 1 → 1)
  
  On garde dataset04 entier en test pour avoir les attaques,
  et on enrichit le training avec ses normaux pour équilibrer.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split  # ← ajouter ça

warnings.filterwarnings("ignore")

import sys
sys.path.append(str(Path(__file__).parent.parent))  # accès à models.py dans ~/swat/
from models import MLP

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

DATA_DIR  = Path("~/swat/batadal/data").expanduser()
SAVE_DIR  = Path("~/batadal/artifacts").expanduser()
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.45
EPOCHS    = 50
BATCH     = 256
LR        = 1e-3

# ══════════════════════════════════════════════════════════════
# CHARGEMENT & PRÉPARATION
# ══════════════════════════════════════════════════════════════
# ici on a merge comme dans swat 
def load_batadal():
    d03 = pd.read_csv(DATA_DIR / "BATADAL_dataset03.csv", skipinitialspace=True)
    d04 = pd.read_csv(DATA_DIR / "BATADAL_dataset04.csv", skipinitialspace=True)

    drop_cols    = ["DATETIME", "ATT_FLAG"]
    feature_cols = [c for c in d03.columns if c not in drop_cols]

    d03["label"] = 0
    d04["label"] = (d04["ATT_FLAG"] == 1).astype(int)

    # Jumeler les deux datasets comme SWaT
    full = pd.concat([d03, d04], ignore_index=True)
    X    = full[feature_cols].values.astype(np.float32)
    y    = full["label"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    joblib.dump(scaler, SAVE_DIR / "scaler.pkl")

    print(f"Train : {X_train.shape} — attaques : {y_train.sum()} / {len(y_train)}")
    print(f"Test  : {X_test.shape}  — attaques : {y_test.sum()} / {len(y_test)}")
    print(f"Features : {len(feature_cols)} → {feature_cols}")

    return X_train, y_train, X_test, y_test

# ══════════════════════════════════════════════════════════════
# ENTRAÎNEMENT MLP
# ══════════════════════════════════════════════════════════════

def train_mlp(X_train, y_train, X_test, y_test, input_size):
    print(f"\n{'─'*50}")
    print("  MLP")
    print(f"{'─'*50}")

    model = MLP(input_size=input_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Poids pour déséquilibre (219 attaques vs ~12k normaux)
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)

    best_f1   = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr))
        X_tr, y_tr = X_tr[perm], y_tr[perm]

        for i in range(0, len(X_tr), BATCH):
            xb = X_tr[i:i+BATCH].to(DEVICE)
            yb = y_tr[i:i+BATCH].to(DEVICE).view(-1, 1)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE))
            proba  = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        preds = (proba >= THRESHOLD).astype(int)
        f1    = f1_score(y_test, preds, zero_division=0)

        if f1 > best_f1:
            best_f1    = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} — F1 test : {f1:.4f}  (best: {best_f1:.4f})")

    model.load_state_dict(best_state)
    torch.save(best_state, SAVE_DIR / "best_mlp.pt")
    print(f"  ✓ MLP sauvegardé — best F1 = {best_f1:.4f}")

    # Rapport final
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE))
        proba  = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
    preds = (proba >= THRESHOLD).astype(int)
    print(classification_report(y_test, preds, zero_division=0))

    return model


# ══════════════════════════════════════════════════════════════
# ENTRAÎNEMENT LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════

def train_logreg(X_train, y_train, X_test, y_test):
    print(f"\n{'─'*50}")
    print("  Logistic Regression")
    print(f"{'─'*50}")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",   # compense le déséquilibre
        C=1.0,
        solver="lbfgs"
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= THRESHOLD).astype(int)
    f1    = f1_score(y_test, preds, zero_division=0)

    joblib.dump(model, SAVE_DIR / "logreg.pkl")
    print(f"  ✓ LogReg sauvegardé — F1 = {f1:.4f}")
    print(classification_report(y_test, preds, zero_division=0))

    return model


# ══════════════════════════════════════════════════════════════
# ENTRAÎNEMENT XGBOOST
# ══════════════════════════════════════════════════════════════

def train_xgboost(X_train, y_train, X_test, y_test):
    print(f"\n{'─'*50}")
    print("  XGBoost")
    print(f"{'─'*50}")

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale = n_neg / n_pos   # compense le déséquilibre

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= THRESHOLD).astype(int)
    f1    = f1_score(y_test, preds, zero_division=0)

    model.save_model(str(SAVE_DIR / "xgb.json"))
    print(f"  ✓ XGBoost sauvegardé — F1 = {f1:.4f}")
    print(classification_report(y_test, preds, zero_division=0))

    return model


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print(f"\n{'═'*50}")
    print(f"  BATADAL — Entraînement")
    print(f"  Device : {DEVICE}")
    print(f"{'═'*50}")

    X_train, y_train, X_test, y_test = load_batadal()

    # Sauvegarde X_test / y_test pour whitebox/blackbox
    np.save(SAVE_DIR / "X_test.npy", X_test)
    np.save(SAVE_DIR / "y_test.npy", y_test)
    print(f"\n✓ X_test.npy / y_test.npy sauvegardés → {SAVE_DIR}")

    input_size = X_train.shape[1]  # 43 features BATADAL

    train_mlp(X_train, y_train, X_test, y_test, input_size)
    train_logreg(X_train, y_train, X_test, y_test)
    train_xgboost(X_train, y_train, X_test, y_test)

    print(f"\n{'═'*50}")
    print(f"  ✓ Tous les artifacts dans {SAVE_DIR}")
    print(f"{'═'*50}")


if __name__ == "__main__":
    main()