# ~/swat/03_transfer.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import pandas as pd
import warnings
from pathlib import Path
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent))
from models import (MLP, SmallMLP, DeepMLP,
                    MLPWrapper, LogRegWrapper, XGBoostWrapper,
                    build_eval_set, eval_attack)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SAVE_DIR  = Path("~/swat/artifacts").expanduser()
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.45
EPS       = 0.1
ITERS     = 40

print(f"Device : {DEVICE}")


# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

def load_artifacts():
    X_train   = np.load(SAVE_DIR / "X_train.npy")
    y_train   = np.load(SAVE_DIR / "y_train.npy")
    X_test    = np.load(SAVE_DIR / "X_test.npy")
    y_test    = np.load(SAVE_DIR / "y_test.npy")

    mlp_model = MLP(input_size=X_test.shape[1]).to(DEVICE)
    mlp_model.load_state_dict(torch.load(SAVE_DIR / "best_mlp.pt", map_location=DEVICE))
    mlp_model.eval()
    mlp_w = MLPWrapper(mlp_model, DEVICE)

    logreg   = joblib.load(SAVE_DIR / "logreg.pkl")
    logreg_w = LogRegWrapper(logreg)

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(SAVE_DIR / "xgb.json"))
    xgb_w = XGBoostWrapper(xgb_model)

    return X_train, y_train, X_test, y_test, mlp_w, logreg_w, xgb_w


# ─────────────────────────────────────────────
# ENTRAÎNEMENT SUBSTITUT
# ─────────────────────────────────────────────

def train_substitute(arch_class, X_train, y_train, device, name="Sub", noise_std=0.02):
    model      = arch_class(input_size=X_train.shape[1]).to(device)
    X_t        = torch.tensor(X_train, dtype=torch.float32)
    y_t        = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / (y_train == 1).sum()],
        dtype=torch.float32
    ).to(device)

    loader    = DataLoader(TensorDataset(X_t, y_t), batch_size=2048, shuffle=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_val_t = torch.tensor(X_train[:5000], dtype=torch.float32).to(device)
    y_val   = y_train[:5000]

    best_f1, patience, no_improve, best_state = 0, 5, 0, None

    for epoch in range(30):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if noise_std > 0:
                xb = xb + torch.randn_like(xb) * noise_std
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
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
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  {name} — best F1 : {best_f1:.4f}")
    return MLPWrapper(model, device)


# ─────────────────────────────────────────────
# MI-FGSM
# ─────────────────────────────────────────────

def mi_fgsm(sub_wrapper, X_atk, y_atk, eps, iters=ITERS, mu=1.0):
    alpha  = 2 * eps / iters
    device = sub_wrapper.device
    x_orig = torch.tensor(X_atk, dtype=torch.float32, device=device)
    y_t    = torch.tensor(y_atk, dtype=torch.float32, device=device).view(-1, 1)
    x_adv  = x_orig.clone()
    g      = torch.zeros_like(x_orig)

    sub_wrapper.model.eval()

    for _ in range(iters):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = sub_wrapper.model(x_adv)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, y_t, reduction='sum'
        )
        loss.backward()

        grad      = x_adv.grad.data
        grad_norm = grad / (grad.abs().sum(dim=1, keepdim=True) + 1e-12)
        g         = mu * g + grad_norm
        x_adv     = x_adv.detach() + alpha * g.sign()
        x_adv     = torch.clamp(x_adv, x_orig - eps, x_orig + eps)

    return x_adv.cpu().numpy()

def ensemble_mi_fgsm(sub_wrappers, X_atk, y_atk, eps,
                     iters=ITERS, mu=1.0, weights=None):
    if weights is None:
        weights = [1.0 / len(sub_wrappers)] * len(sub_wrappers)
    alpha  = 2 * eps / iters
    device = sub_wrappers[0].device
    x_orig = torch.tensor(X_atk, dtype=torch.float32, device=device)
    y_t    = torch.tensor(y_atk, dtype=torch.float32, device=device).view(-1, 1)
    x_adv  = x_orig.clone()
    g      = torch.zeros_like(x_orig)

    for sub in sub_wrappers:
        sub.model.eval()

    for _ in range(iters):
        x_adv_d       = x_adv.detach()
        grad_ensemble = torch.zeros_like(x_orig)

        for w, sub in zip(weights, sub_wrappers):
            x_inp  = x_adv_d.requires_grad_(True)
            logits = sub.model(x_inp)
            loss   = nn.functional.binary_cross_entropy_with_logits(
                         logits, y_t, reduction='sum')
            loss.backward()
            grad_cur = x_inp.grad.data.clone()
            grad_norm = grad_cur / (grad_cur.abs().sum(dim=1, keepdim=True) + 1e-12)
            grad_ensemble += w * grad_norm

        g     = mu * g + grad_ensemble
        x_adv = x_adv_d + alpha * g.sign()
        x_adv = torch.clamp(x_adv, x_orig - eps, x_orig + eps).detach()

    return x_adv.cpu().numpy()

# ─────────────────────────────────────────────
# VMI-FGSM
# ─────────────────────────────────────────────

def vmi_fgsm(sub_wrapper, X_atk, y_atk, eps,
             iters=ITERS, mu=1.0, beta=0.3, n_neighbors=10):
    alpha  = 2 * eps / iters
    device = sub_wrapper.device
    x_orig = torch.tensor(X_atk, dtype=torch.float32, device=device)
    y_t    = torch.tensor(y_atk, dtype=torch.float32, device=device).view(-1, 1)
    x_adv  = x_orig.clone()
    g      = torch.zeros_like(x_orig)

    sub_wrapper.model.eval()

    for _ in range(iters):
        x_adv_d = x_adv.detach()

        # Gradient au point courant
        x_inp  = x_adv_d.requires_grad_(True)
        logits = sub_wrapper.model(x_inp)
        loss   = nn.functional.binary_cross_entropy_with_logits(
                     logits, y_t, reduction='sum')
        loss.backward()
        grad_cur = x_inp.grad.data.clone()

        # Moyenne des gradients dans le voisinage
        grad_neigh = torch.zeros_like(grad_cur)
        for _ in range(n_neighbors):
            noise    = torch.empty_like(x_adv_d).uniform_(-beta * eps, beta * eps)
            x_n      = (x_adv_d + noise).detach().requires_grad_(True)
            logits_n = sub_wrapper.model(x_n)
            loss_n   = nn.functional.binary_cross_entropy_with_logits(
                           logits_n, y_t, reduction='sum')
            loss_n.backward()
            grad_neigh += x_n.grad.data
        grad_neigh /= n_neighbors

        # Correction de variance
        grad_var  = grad_cur - grad_neigh
        grad_used = grad_cur + beta * grad_var

        # CORRECTION : normaliser grad_cur et grad_var séparément
        # pour éviter les annulations qui explosent la norme L1
        #we are debugging here 

        grad_used = grad_cur + beta * (grad_cur - grad_neigh)
        grad_used_norm = grad_used / (grad_used.abs().sum(dim=1, keepdim=True) + 1e-12)

        g     = mu * g + grad_used_norm

        x_adv = x_adv_d + alpha * g.sign()
        x_adv = torch.clamp(x_adv, x_orig - eps, x_orig + eps).detach()

    return x_adv.cpu().numpy()


def ensemble_vmi_fgsm(sub_wrappers, X_atk, y_atk, eps,
                      iters=ITERS, mu=1.0, beta=0.3, n_neighbors=10,
                      weights=None):
    # n_neighbors=20 au lieu de 10 — cohérent avec vmi_fgsm seul
    if weights is None:
        weights = [1.0 / len(sub_wrappers)] * len(sub_wrappers)
    alpha  = 2 * eps / iters
    device = sub_wrappers[0].device
    x_orig = torch.tensor(X_atk, dtype=torch.float32, device=device)
    y_t    = torch.tensor(y_atk, dtype=torch.float32, device=device).view(-1, 1)
    x_adv  = x_orig.clone()
    g      = torch.zeros_like(x_orig)

    for sub in sub_wrappers:
        sub.model.eval()

    for _ in range(iters):
        x_adv_d       = x_adv.detach()
        grad_ensemble = torch.zeros_like(x_orig)

        for w, sub in zip(weights, sub_wrappers):
            x_inp  = x_adv_d.requires_grad_(True)
            logits = sub.model(x_inp)
            loss   = nn.functional.binary_cross_entropy_with_logits(
                         logits, y_t, reduction='sum')
            loss.backward()
            grad_cur = x_inp.grad.data.clone()

            grad_neigh = torch.zeros_like(grad_cur)
            for _ in range(n_neighbors):
                noise    = torch.empty_like(x_adv_d).uniform_(-beta * eps, beta * eps)
                x_n      = (x_adv_d + noise).detach().requires_grad_(True)
                logits_n = sub.model(x_n)
                loss_n   = nn.functional.binary_cross_entropy_with_logits(
                               logits_n, y_t, reduction='sum')
                loss_n.backward()
                grad_neigh += x_n.grad.data
            grad_neigh /= n_neighbors

            grad_var  = grad_cur - grad_neigh

            # CORRECTION : même normalisation séparée que vmi_fgsm
            grad_used = grad_cur + beta * (grad_cur - grad_neigh)
            grad_used_norm = grad_used / (grad_used.abs().sum(dim=1, keepdim=True) + 1e-12)

            grad_ensemble += w * grad_used_norm

        g     = mu * g + grad_ensemble
        x_adv = x_adv_d + alpha * g.sign()
        x_adv = torch.clamp(x_adv, x_orig - eps, x_orig + eps).detach()

    return x_adv.cpu().numpy()


    
def save_all_adv(adv_store, eps):
    """
    vu qu'on a eu un probleme ( le asr apres avoir ete defendu est plus elevé que avant ce qui est illogique ) on Sauvegarde TOUS les X_adv générés, un fichier par (attaque, substitut, victime).
    adv_store : dict {(sub_name, attack_name): X_adv}

    Pour les attaques transfer, le X_adv n'est pas spécifique à une victime 
    il est généré contre le substitut, pas contre la victime. On l'associe aux
    3 victimes en copiant le même tableau (le nom de victime vient de la cible
    visée, pas du X_adv lui-même).
    On stocke donc (sub, attack) → X_adv, et on tag avec chaque victime.
    """
    victim_names = ["MLP", "LogReg", "XGBoost"]

    for (sub_name, attack_name), X_adv in adv_store.items():
        # Normalise les noms pour les noms de fichiers
        fname_attack = attack_name.lower().replace("-", "_")
        # Sub1-MLP → Sub1-MLP (on garde tel quel, mais on remplace les
        # caractères invalides pour un nom de fichier)
        fname_sub = sub_name.replace("(", "").replace(")", "").replace("+", "-")

        for vic_name in victim_names:
            fname = (
                f"adv_{fname_attack}_{vic_name}"
                f"_sub_{fname_sub}_eps{eps}.npy"
            )
            path = SAVE_DIR / fname
            np.save(path, X_adv)
            print(f"  ✓ {fname}")

# ─────────────────────────────────────────────
# EVAL TRANSFERT
# ─────────────────────────────────────────────

def eval_transfer(X_eval, y_eval, X_adv_atk,
                  victim_wrapper, sub_name, victim_name, attack_name):
    from sklearn.metrics import f1_score, recall_score, precision_score

    mask             = (y_eval == 1)
    X_adv_full       = X_eval.copy()
    X_adv_full[mask] = X_adv_atk

    p_clean  = victim_wrapper.predict_proba(X_eval)
    p_adv    = victim_wrapper.predict_proba(X_adv_full)
    pr_clean = (p_clean >= THRESHOLD).astype(int)
    pr_adv   = (p_adv   >= THRESHOLD).astype(int)

    f1_clean  = f1_score(y_eval, pr_clean,  zero_division=0)
    f1_adv    = f1_score(y_eval, pr_adv,    zero_division=0)
    rec_clean = recall_score(y_eval, pr_clean, zero_division=0)
    rec_adv   = recall_score(y_eval, pr_adv,   zero_division=0)

    p_att  = victim_wrapper.predict_proba(X_adv_atk)
    pr_att = (p_att >= THRESHOLD).astype(int)
    asr    = float(np.mean(pr_att == 0))
    linf   = float(np.abs(X_adv_atk - X_eval[mask]).max())

    print(f"  [{attack_name}] {sub_name} → {victim_name:10s} | "
          f"ASR {asr:.1%} | "
          f"F1 {f1_clean:.3f}→{f1_adv:.3f} | "
          f"Recall {rec_clean:.3f}→{rec_adv:.3f} | "
          f"L∞ {linf:.4f}")

    return {
        "attack":      attack_name,
        "substitute":  sub_name,
        "victim":      victim_name,
        "asr":         round(asr,       4),
        "f1_clean":    round(f1_clean,  4),
        "f1_adv":      round(f1_adv,    4),
        "rec_clean":   round(rec_clean, 4),
        "rec_adv":     round(rec_adv,   4),
        "delta_f1":    round(f1_adv - f1_clean, 4),
        "linf":        round(linf,      4),
    }


# ─────────────────────────────────────────────
# AFFICHAGE
# ─────────────────────────────────────────────

def print_results(df):
    victims  = list(df["victim"].unique())
    attacks  = sorted(df["attack"].unique())
    col_w    = 26
    label_w  = 16

    header = f"{'Attaque / Substitut':<{label_w}}" + "".join(
        f"  {v:^{col_w}}" for v in victims
    )
    sep = "═" * len(header)

    print(f"\n{sep}")
    print("RÉSUMÉ TRANSFERT")
    print(sep)
    print(header)

    for atk in attacks:
        sub_df = df[df["attack"] == atk]
        subs   = sorted(sub_df["substitute"].unique())
        print(f"  ── {atk}")
        for sub in subs:
            row_str = f"    {sub:<{label_w - 4}}"
            for v in victims:
                cell = sub_df[(sub_df["substitute"] == sub) & (sub_df["victim"] == v)]
                if cell.empty:
                    row_str += "  " + " " * col_w
                else:
                    r = cell.iloc[0]
                    row_str += (
                        f"  {r['asr']:>6.1%} {r['f1_adv']:>6.4f} "
                        f"{r['rec_adv']:>5.3f} {r['delta_f1']:>+6.3f}  "
                    )
            print(row_str)
        print()
    print(sep)


# ─────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ─────────────────────────────────────────────

def run():
    X_train, y_train, X_test, y_test, mlp_w, logreg_w, xgb_w = load_artifacts()

    victims = [
        ("MLP",     mlp_w),
        ("LogReg",  logreg_w),
        ("XGBoost", xgb_w),
    ]

    print("\n=== Entraînement substituts ===")
    sub1 = train_substitute(MLP,      X_train, y_train, DEVICE, "Sub1-MLP")
    sub2 = train_substitute(SmallMLP, X_train, y_train, DEVICE, "Sub2-SmallMLP")
    sub3 = train_substitute(DeepMLP,  X_train, y_train, DEVICE, "Sub3-DeepMLP")
    substitutes = [
        ("Sub1-MLP",      sub1),
        ("Sub2-SmallMLP", sub2),
        ("Sub3-DeepMLP",  sub3),
    ]

    print("\n=== Build eval set (intersection des 3 victimes) ===")
    rng        = np.random.default_rng(42)
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
    print(f"  Eval : {X_eval.shape[0]} exemples "
          f"({(y_eval==0).sum()} Normal, {(y_eval==1).sum()} Attack)")

    results  = []
    # Stocke tous les X_adv générés : {(sub_name, attack_name): X_adv}
    adv_store = {}

    for sub_name, sub_w in substitutes:
        print(f"\n{'─'*60}")
        print(f"Substitut : {sub_name}")
        print(f"{'─'*60}")

        X_adv_mi  = mi_fgsm(sub_w,  X_atk, y_atk, eps=EPS)
        X_adv_vmi = vmi_fgsm(sub_w, X_atk, y_atk, eps=EPS)

        # Stocke pour sauvegarde ultérieure
        adv_store[(sub_name, "MI-FGSM")]  = X_adv_mi
        adv_store[(sub_name, "VMI-FGSM")] = X_adv_vmi

        for vic_name, vic_w in victims:
            results.append(eval_transfer(X_eval, y_eval, X_adv_mi,
                                         vic_w, sub_name, vic_name, "MI-FGSM"))
            results.append(eval_transfer(X_eval, y_eval, X_adv_vmi,
                                         vic_w, sub_name, vic_name, "VMI-FGSM"))

    print(f"\n{'─'*60}")
    print("Ensemble VMI-FGSM (Sub1+Sub2+Sub3)")
    print(f"{'─'*60}")
    X_adv_ens = ensemble_mi_fgsm(
        [sub1, sub2, sub3], X_atk, y_atk, eps=EPS,
        weights=[1/3, 1/3, 1/3]
    )
    adv_store[("Ensemble(S1+S2+S3)", "Ensemble-MI")] = X_adv_ens

    for vic_name, vic_w in victims:
        results.append(eval_transfer(X_eval, y_eval, X_adv_ens,
                                     vic_w, "Ensemble(S1+S2+S3)", vic_name, "Ensemble-MI"))

    df = pd.DataFrame(results)
    df.to_csv(SAVE_DIR / "transfer_results.csv", index=False)

    print_results(df)

    print("\n=== Sauvegarde des X_adv (tous les substituts) ===")
    save_all_adv(adv_store, EPS)
    

    return df


if __name__ == "__main__":
    df = run()

    import json

    RESULTS_DIR = Path("~/swat/results").expanduser()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out = {}
    for victim in df["victim"].unique():
        out[victim] = {}
        sub = df[df["victim"] == victim]
        for attack in sub["attack"].unique():
            best_row = sub[sub["attack"] == attack].sort_values("asr", ascending=False).iloc[0]
            out[victim][attack] = {
                "evasion_rate": round(best_row["asr"] * 100, 2),
                "recall":       round(best_row["rec_adv"], 4),
                "f1":           round(best_row["f1_adv"], 4),
                "delta_f1":     round(best_row["delta_f1"], 4),
                "best_sub":     best_row["substitute"],
            }

    with open(RESULTS_DIR / "transfer_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nJSON sauvegardé → {RESULTS_DIR / 'transfer_results.json'}")
