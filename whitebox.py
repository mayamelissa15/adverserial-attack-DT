import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import pandas as pd
import warnings
from pathlib import Path
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

import sys
sys.path.append(str(Path(__file__).parent))
from models import MLP, MLPWrapper, LogRegWrapper, XGBoostWrapper, build_eval_set, eval_attack


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
print('test')
SAVE_DIR  = Path("~/swat/artifacts").expanduser()
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.45
EPS_LIST  = [0.1, 0.3]

THRESHOLD_LOGIT = float(np.log(THRESHOLD / (1 - THRESHOLD)))

PGD_ITERS    = 200
PGD_RESTARTS = 10
PGD_ALPHA_K  = 4        # alpha = eps / PGD_ALPHA_K

EPS_FD       = 1e-3
EPS_FD_LIST  = [5e-4, 1e-3, 2e-3, 5e-3]
SMOOTH_GRAD  = False
SMOOTH_K     = 10
SMOOTH_SIGMA = 0.01

CW_LR_XGB   = 0.005
CW_LR_LR    = 0.01
CW_ITERS    = 500


# ══════════════════════════════════════════════════════════════
# GRADIENT XGBoost
# ══════════════════════════════════════════════════════════════

def _grad_xgb_vote(wrapper, X, eps_fd_list=EPS_FD_LIST):
    grads     = np.stack([wrapper.grad_numerical(X, e) for e in eps_fd_list], axis=0)
    grad_mean = grads.mean(axis=0)
    sign_vote = np.sign(np.sign(grads).sum(axis=0))
    return grad_mean, sign_vote


def _grad_xgb_smooth(wrapper, X, sigma=SMOOTH_SIGMA, K=SMOOTH_K, eps_fd=EPS_FD):
    grads = []
    for _ in range(K):
        noise = np.random.normal(0, sigma, X.shape).astype(np.float32)
        grads.append(wrapper.grad_numerical(X + noise, eps_fd))
    return np.mean(grads, axis=0)


def _grad_xgb(wrapper, X, mode="vote", eps_fd=EPS_FD):
    if mode == "vote":
        grad, _ = _grad_xgb_vote(wrapper, X)
        return grad
    elif mode == "smooth":
        return _grad_xgb_smooth(wrapper, X)
    else:
        return wrapper.grad_numerical(X, eps_fd)


# ══════════════════════════════════════════════════════════════
# FGSM
# ══════════════════════════════════════════════════════════════

def fgsm_mlp(wrapper, X_atk, y_atk, eps):
    x_orig = torch.tensor(X_atk, dtype=torch.float32, device=wrapper.device)
    y      = torch.tensor(y_atk, dtype=torch.float32, device=wrapper.device).view(-1, 1)
    x_adv  = x_orig.clone().detach().requires_grad_(True)
    wrapper.model.eval()
    nn.BCEWithLogitsLoss()(wrapper.model(x_adv), y).backward()
    delta       = torch.clamp(eps * x_adv.grad.sign(), -eps, eps)
    x_adv_final = torch.clamp(x_orig + delta, x_orig - eps, x_orig + eps)
    return x_adv_final.detach().cpu().numpy()


def fgsm_logreg(wrapper, X_atk, y_atk, eps):
    g     = wrapper.grad_bce(X_atk, y_atk)
    delta = eps * np.sign(g)
    return np.clip(X_atk + delta, X_atk - eps, X_atk + eps)


def fgsm_xgb(wrapper, X_atk, y_atk, eps):
    if SMOOTH_GRAD:
        grad = _grad_xgb_smooth(wrapper, X_atk)
        sign = np.sign(grad)
    else:
        _, sign = _grad_xgb_vote(wrapper, X_atk)
    delta = eps * sign
    return np.clip(X_atk + delta, X_atk - eps, X_atk + eps)


# ══════════════════════════════════════════════════════════════
# PGD
# ══════════════════════════════════════════════════════════════

def pgd_mlp(wrapper, X_atk, y_atk, eps,
            iters=PGD_ITERS, restarts=PGD_RESTARTS, alpha=None):
    """
    PGD multi-restart MLP.

    FIX RNG : chaque restart crée son bruit via torch.zeros_like().uniform_()
    sur le générateur courant — le state avance naturellement entre restarts,
    garantissant des initialisations différentes sans re-seed externe.

    FIX best_logits : initialisé à +inf pour forcer une vraie compétition
    entre restarts plutôt que contre les logits clean.
    """
    alpha  = alpha or (eps / PGD_ALPHA_K)
    device = wrapper.device

    x_orig = torch.tensor(X_atk, dtype=torch.float32, device=device)
    y_t    = torch.tensor(y_atk, dtype=torch.float32, device=device).view(-1, 1)

    wrapper.model.eval()

    best_adv    = x_orig.clone()
    best_logits = torch.full((len(X_atk),), float("+inf"), device=device)

    for r in range(restarts):
        # ✅ FIX RNG : nouveau tensor à chaque restart → state RNG progresse
        noise = torch.zeros_like(x_orig).uniform_(-eps, eps)
        x_adv = torch.clamp(x_orig + noise, x_orig - eps, x_orig + eps).detach()

        for _ in range(iters):
            x_adv = x_adv.detach().requires_grad_(True)
            logits = wrapper.model(x_adv)
            loss   = nn.BCEWithLogitsLoss()(logits, y_t)
            loss.backward()

            with torch.no_grad():
                x_adv = x_adv + alpha * x_adv.grad.sign()
                x_adv = torch.clamp(x_adv, x_orig - eps, x_orig + eps)

        with torch.no_grad():
            final_logits = wrapper.model(x_adv).squeeze(-1)

        improved = final_logits < best_logits
        best_adv[improved]    = x_adv[improved]
        best_logits[improved] = final_logits[improved]

        n_evaded = (best_logits < THRESHOLD_LOGIT).sum().item()
        print(f"      restart {r+1:02d}/{restarts} — "
              f"évadés cumulés : {n_evaded}/{len(X_atk)} "
              f"({100*n_evaded/len(X_atk):.1f}%)")

    return best_adv.cpu().numpy()


def pgd_logreg(wrapper, X_atk, y_atk, eps,
               iters=PGD_ITERS, restarts=PGD_RESTARTS, alpha=None):
    """
    PGD multi-restart LogReg.

    FIX RNG : np.random.uniform() avance naturellement le state numpy
    entre restarts — pas de re-seed, chaque restart part d'un point différent.

    FIX best_logits : initialisé à +inf.
    """
    alpha = alpha or (eps / PGD_ALPHA_K)

    best_adv    = X_atk.copy().astype(np.float32)
    best_logits = np.full(len(X_atk), float("+inf"), dtype=np.float64)

    for r in range(restarts):
        # ✅ FIX RNG : np.random avance entre restarts sans re-seed
        noise = np.random.uniform(-eps, eps, X_atk.shape).astype(np.float32)
        x_adv = (X_atk + noise).astype(np.float32)
        x_adv = np.clip(x_adv, X_atk - eps, X_atk + eps)

        for _ in range(iters):
            g     = wrapper.grad_bce(x_adv, y_atk)
            x_adv = x_adv + alpha * np.sign(g)
            x_adv = np.clip(x_adv, X_atk - eps, X_atk + eps)

        logits   = wrapper.logits_np(x_adv)
        improved = logits < best_logits
        best_adv[improved]    = x_adv[improved].astype(np.float32)
        best_logits[improved] = logits[improved]

        n_evaded = (best_logits < THRESHOLD_LOGIT).sum()
        print(f"      restart {r+1:02d}/{restarts} — "
              f"évadés : {n_evaded}/{len(X_atk)} "
              f"({100*n_evaded/len(X_atk):.1f}%)")

    return best_adv


def pgd_xgb(wrapper, X_atk, y_atk, eps,
            iters=PGD_ITERS, restarts=PGD_RESTARTS, alpha=None):
    """
    PGD XGBoost.

    FIX RNG : même logique que pgd_logreg — np.random avance naturellement.
    FIX best_logits : initialisé à +inf pour cohérence.
    """
    alpha       = alpha or (eps / PGD_ALPHA_K)
    best_adv    = X_atk.copy().astype(np.float32)
    best_logits = np.full(len(X_atk), float("+inf"), dtype=np.float64)

    for r in range(restarts):
        # ✅ FIX RNG : np.random avance entre restarts sans re-seed
        noise = np.random.uniform(-eps, eps, X_atk.shape).astype(np.float32)
        x_adv = (X_atk + noise).astype(np.float32)
        x_adv = np.clip(x_adv, X_atk - eps, X_atk + eps)

        for _ in range(iters):
            g     = wrapper.grad_numerical(x_adv, EPS_FD)
            x_adv = x_adv + alpha * np.sign(g)
            x_adv = np.clip(x_adv, X_atk - eps, X_atk + eps)

        logits   = wrapper.logits_np(x_adv)
        improved = logits < best_logits
        best_adv[improved]    = x_adv[improved]
        best_logits[improved] = logits[improved]

        n_evaded = (best_logits < THRESHOLD_LOGIT).sum()
        print(f"      restart {r+1:02d}/{restarts} — "
              f"évadés : {n_evaded}/{len(X_atk)} "
              f"({100*n_evaded/len(X_atk):.1f}%)")

    return best_adv


# ══════════════════════════════════════════════════════════════
# C&W
# ══════════════════════════════════════════════════════════════

def cw_logreg(wrapper, X_atk, y_atk, eps,
              lr=CW_LR_LR, iters=CW_ITERS, kappa=0.0):
    """C&W LogReg — Adam sur les logits, pas de warm-start nécessaire."""
    X_adv = X_atk.copy().astype(np.float64)
    m = np.zeros_like(X_adv)
    v = np.zeros_like(X_adv)
    b1, b2, ep_adam = 0.9, 0.999, 1e-8

    for t in range(1, iters + 1):
        logits = wrapper.logits_np(X_adv)
        active = (logits - THRESHOLD_LOGIT + kappa > 0).astype(np.float64)
        grad_log = wrapper.grad_logit(X_adv)
        grad     = active[:, None] * grad_log

        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)

        X_adv -= lr * m_hat / (np.sqrt(v_hat) + ep_adam)
        X_adv  = np.clip(X_adv, X_atk - eps, X_atk + eps)

        if t % 100 == 0:
            n_evaded = (wrapper.logits_np(X_adv) < THRESHOLD_LOGIT).sum()
            print(f"      iter {t:4d} — évadés : {n_evaded}/{len(X_atk)} "
                  f"({100*n_evaded/len(X_atk):.1f}%)")

    return X_adv.astype(np.float32)


def cw_mlp(wrapper, X_atk, y_atk, eps,
           lr=0.005, iters=300, kappa=0.0, lam=0.0, batch_size=128):
    """
    C&W MLP — warm-start depuis pgd_mlp (maintenant corrigé).

    Le fix RNG de pgd_mlp se propage ici : le warm-start explore
    vraiment différents points de départ.
    """
    device = wrapper.device
    wrapper.model.eval()

    # Warm-start depuis PGD corrigé
    X_init = pgd_mlp(wrapper, X_atk, y_atk, eps, iters=20, restarts=3,
                     alpha=eps / PGD_ALPHA_K)
    print(f"      [C&W warm-start PGD terminé]")

    X_tensor_orig = torch.tensor(X_atk,  dtype=torch.float32)
    X_tensor_init = torch.tensor(X_init, dtype=torch.float32)
    X_adv_np      = X_init.copy()

    for start in range(0, len(X_atk), batch_size):
        end      = min(start + batch_size, len(X_atk))
        x_orig_b = X_tensor_orig[start:end].to(device)
        x_init_b = X_tensor_init[start:end].to(device)

        delta_init = (x_init_b - x_orig_b).clamp(-eps + 1e-6, eps - 1e-6)
        w_init     = torch.atanh(delta_init / (eps + 1e-8))
        w          = w_init.detach().clone().requires_grad_(True)

        optimizer = optim.Adam([w], lr=lr)

        for _ in range(iters):
            optimizer.zero_grad()
            delta   = eps * torch.tanh(w)
            x_adv_b = x_orig_b + delta
            logits  = wrapper.model(x_adv_b).squeeze(-1)
            loss    = torch.clamp(logits - THRESHOLD_LOGIT + kappa, min=0).mean()
            if lam > 0:
                loss = loss + lam * (delta ** 2).sum(dim=1).mean()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            delta_final         = torch.clamp(eps * torch.tanh(w), -eps, eps)
            X_adv_np[start:end] = (x_orig_b + delta_final).cpu().numpy()

    return X_adv_np


def cw_xgb(wrapper, X_atk, y_atk, eps,
           lr=CW_LR_XGB, iters=CW_ITERS, kappa=0.0):
    """C&W XGBoost — warm-start depuis pgd_xgb (maintenant corrigé)."""
    print("      [C&W XGBoost : warm-start PGD...]")
    X_adv = pgd_xgb(wrapper, X_atk, y_atk, eps,
                    iters=20, restarts=3).astype(np.float64)
    print(f"      [C&W warm-start PGD terminé]")

    m = np.zeros_like(X_adv)
    v = np.zeros_like(X_adv)
    b1, b2, ep_adam = 0.9, 0.999, 1e-8

    for t in range(1, iters + 1):
        logits   = wrapper.logits_np(X_adv)
        active   = (logits - THRESHOLD_LOGIT + kappa > 0).astype(np.float64)
        grad_log = wrapper.grad_numerical(X_adv, EPS_FD)
        grad     = active[:, None] * grad_log

        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)

        X_adv -= lr * m_hat / (np.sqrt(v_hat) + ep_adam)
        X_adv  = np.clip(X_adv, X_atk - eps, X_atk + eps)

        if t % 100 == 0:
            n_evaded = (wrapper.logits_np(X_adv) < THRESHOLD_LOGIT).sum()
            print(f"      iter {t:4d} — évadés : {n_evaded}/{len(X_atk)} "
                  f"({100*n_evaded/len(X_atk):.1f}%)")

    return X_adv.astype(np.float32)


# ══════════════════════════════════════════════════════════════
# AFFICHAGE
# ══════════════════════════════════════════════════════════════

def _bar(value, width=20):
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)

def _asr_label(asr):
    if asr >= 0.6:  return "✓ FORT  "
    if asr >= 0.35: return "~ MOYEN "
    return                  "✗ FAIBLE"

def print_result(r):
    asr = r["asr"]
    print(
        f"  {r['attack']:<14} {r['model']:<8} │ "
        f"{_asr_label(asr)} {_bar(asr)} {asr*100:5.1f}% │ "
        f"F1 {r['f1_clean']:.3f}→{r['f1_adv']:.3f} │ "
        f"Recall adv {r['rec_adv']:.3f} │ "
        f"L∞ {r['linf']:.4f}"
    )

def print_summary(df):
    col_w = 60
    print(f"\n{'═'*col_w}")
    print(f"  RÉSUMÉ WHITEBOX")
    print(f"{'═'*col_w}")
    for eps in EPS_LIST:
        print(f"\n  ┌─ eps = {eps} {'─'*(col_w-12)}┐")
        for model in ["MLP", "LogReg", "XGBoost"]:
            sub = df[(df.model == model) & df.attack.str.contains(str(eps))]
            if sub.empty:
                continue
            print(f"  │  {model}")
            for _, row in sub.sort_values("attack").iterrows():
                asr = row["asr"]
                print(f"  │    {row['attack']:<14} ASR {asr*100:5.1f}%  "
                      f"{_bar(asr, 15)}  "
                      f"F1 {row['f1_clean']:.3f}→{row['f1_adv']:.3f}  "
                      f"Recall {row['rec_adv']:.3f}")
        print(f"  └{'─'*(col_w-2)}┘")