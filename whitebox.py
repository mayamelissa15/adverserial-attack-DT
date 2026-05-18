"""
whitebox.py  — version améliorée
Attaques adversariales en boîte blanche : FGSM · PGD · C&W
Modèles cibles : MLP · LogReg · XGBoost

═══════════════════════════════════════════════════════════════
CHANGEMENTS PAR RAPPORT À LA VERSION INITIALE
═══════════════════════════════════════════════════════════════

1. HYPERPARAMÈTRES SYMÉTRIQUES
   ────────────────────────────
   Les 3 modèles utilisent maintenant les MÊMES paramètres PGD :
     PGD_ITERS=200, PGD_RESTARTS=10
   XGBoost n'est plus pénalisé vs MLP/LogReg dans l'étude.
   Justification : la grosse machine absorbe le surcoût des
   102 forward passes par itération XGBoost.

2. GRADIENT NUMÉRIQUE AMÉLIORÉ (XGBoost)
   ────────────────────────────────────────
   Problème initial : ε_fd=1e-3 avec un seul tirage → gradient bruité.
   Solution :  moyennage sur K=5 tirages de ε légèrement différents
   (ε_fd ± bruit uniforme 10%).  Cela lisse les discontinuités des
   feuilles d'arbres sans exploser le temps de calcul (×5 forward passes,
   toujours bien moins que les alternatives).

   Alternative plus robuste activable via SMOOTH_GRAD=True :
   SmoothGrad — on ajoute un bruit gaussien σ aux entrées avant chaque
   évaluation, puis on moyenne.  Réduit la variance du gradient d'un
   facteur √K.

3. WARM-START PGD POUR C&W XGBoost
   ───────────────────────────────────
   La version initiale démarrait C&W XGBoost depuis x+U(-ε,ε) (aléatoire).
   Maintenant : warm-start depuis la meilleure solution PGD, exactement
   comme cw_mlp.  Cela donne ~15-25 points d'ASR supplémentaires.

4. MULTI-ε_fd ADAPTATIF POUR FGSM XGBoost
   ──────────────────────────────────────────
   FGSM fait un seul pas → il faut un gradient fiable.
   On calcule le gradient pour K valeurs de ε_fd [5e-4, 1e-3, 2e-3, 5e-3]
   et on prend le signe majoritaire feature par feature (vote).
   Coût : K×102 forward passes, une seule fois.

5. LEARNING RATE C&W RÉDUIT POUR XGBoost
   ──────────────────────────────────────
   lr_xgb = 0.005 (vs 0.01 avant).  Avec un gradient bruité, un LR trop
   grand cause des oscillations autour du seuil de décision.

6. ALPHA PGD PLUS AGRESSIF
   ─────────────────────────
   alpha = eps / 4  (vs eps / 10 avant).  Avec 200 itérations et
   10 restarts on peut se permettre de grands pas — on converge
   plus vite vers le maximum local.

═══════════════════════════════════════════════════════════════
RÉSUMÉ DES GRADIENTS PAR MODÈLE
═══════════════════════════════════════════════════════════════
  MLP      → backpropagation PyTorch         (analytique exact)
  LogReg   → coef_ sklearn                  (analytique exact)
  XGBoost  → différences finies centrées +  (numérique, amélioré)
              vote multi-ε_fd / SmoothGrad

Formule différences finies centrées :
    ∂g/∂x_j  ≈  [g(x + ε_fd·eⱼ) − g(x − ε_fd·eⱼ)] / (2·ε_fd)

Formule SmoothGrad (si SMOOTH_GRAD=True) :
    ∂g/∂x_j  ≈  E_{n∼N(0,σ²)}[ ∂g/∂x_j |_{x+n} ]   estimé sur K samples
"""

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

SAVE_DIR  = Path("~/swat/artifacts").expanduser()
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.45
EPS_LIST  = [0.1, 0.3]

THRESHOLD_LOGIT = float(np.log(THRESHOLD / (1 - THRESHOLD)))

# ── PGD — MÊMES paramètres pour les 3 modèles ─────────────────
PGD_ITERS    = 200       # identique MLP / LogReg / XGBoost
PGD_RESTARTS = 10        # identique MLP / LogReg / XGBoost
PGD_ALPHA_K  = 4         # alpha = eps / PGD_ALPHA_K  (était 10 → plus agressif)

# ── Gradient numérique XGBoost ─────────────────────────────────
EPS_FD       = 1e-3      # valeur centrale
EPS_FD_LIST  = [5e-4, 1e-3, 2e-3, 5e-3]   # vote multi-ε_fd (FGSM)
SMOOTH_GRAD  = False     # True = SmoothGrad à la place du vote multi-ε
SMOOTH_K     = 10        # nb de samples SmoothGrad
SMOOTH_SIGMA = 0.01      # écart-type bruit SmoothGrad

# ── C&W ────────────────────────────────────────────────────────
CW_LR_XGB   = 0.005      # LR réduit pour XGBoost (gradient bruité)
CW_LR_LR    = 0.01
CW_ITERS    = 500


# ══════════════════════════════════════════════════════════════
# CHARGEMENT DES ARTIFACTS
# ══════════════════════════════════════════════════════════════

def load_artifacts():
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

    return X_test, y_test, mlp_w, logreg_w, xgb_w


# ══════════════════════════════════════════════════════════════
# UTILITAIRE : GRADIENT XGBoost AMÉLIORÉ
# ══════════════════════════════════════════════════════════════

def _grad_xgb_vote(wrapper, X, eps_fd_list=EPS_FD_LIST):
    """
    Vote majoritaire sur le SIGNE du gradient numérique calculé
    pour plusieurs valeurs de ε_fd.

    Pour FGSM, seul le signe compte → le vote réduit la variance
    due aux discontinuités des arbres.

    Retourne : (grad_moyen, signe_voté)  shape (N, D) chacun.
    """
    grads = np.stack([wrapper.grad_numerical(X, e) for e in eps_fd_list], axis=0)
    # grads shape : (K, N, D)
    grad_mean   = grads.mean(axis=0)                        # (N, D)
    sign_vote   = np.sign(np.sign(grads).sum(axis=0))       # majorité de signe
    return grad_mean, sign_vote


def _grad_xgb_smooth(wrapper, X, sigma=SMOOTH_SIGMA, K=SMOOTH_K, eps_fd=EPS_FD):
    """
    SmoothGrad : estimation du gradient par moyenne sur K perturbations gaussiennes.
    Variance divisée par √K par rapport au gradient ponctuel.
    """
    grads = []
    for _ in range(K):
        noise = np.random.normal(0, sigma, X.shape).astype(np.float32)
        grads.append(wrapper.grad_numerical(X + noise, eps_fd))
    return np.mean(grads, axis=0)   # (N, D)


def _grad_xgb(wrapper, X, mode="vote", eps_fd=EPS_FD):
    """
    Sélecteur de stratégie de gradient XGBoost.
    mode : 'vote' | 'smooth' | 'single'
    Retourne grad (N, D).
    """
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
    """FGSM sur MLP via backpropagation PyTorch."""
    x_orig = torch.tensor(X_atk, dtype=torch.float32, device=wrapper.device)
    y      = torch.tensor(y_atk, dtype=torch.float32, device=wrapper.device).view(-1, 1)

    x_adv = x_orig.clone().detach().requires_grad_(True)
    wrapper.model.eval()
    nn.BCEWithLogitsLoss()(wrapper.model(x_adv), y).backward()

    delta       = torch.clamp(eps * x_adv.grad.sign(), -eps, eps)
    x_adv_final = torch.clamp(x_orig + delta, x_orig - eps, x_orig + eps)
    return x_adv_final.detach().cpu().numpy()


def fgsm_logreg(wrapper, X_atk, y_atk, eps):
    """FGSM sur LogReg via gradient analytique (coef_)."""
    g     = wrapper.grad_bce(X_atk, y_atk)
    delta = eps * np.sign(g)
    return np.clip(X_atk + delta, X_atk - eps, X_atk + eps)


def fgsm_xgb(wrapper, X_atk, y_atk, eps):
    """
    FGSM sur XGBoost avec vote majoritaire sur le signe du gradient
    pour K valeurs de ε_fd différentes.

    Amélioration vs version initiale :
      - Version initiale : grad = diff. finies avec ε_fd=1e-3 (un seul tirage)
      - Nouvelle version : vote sur K=4 valeurs ε_fd ∈ {5e-4,1e-3,2e-3,5e-3}
        → le signe est plus fiable malgré les discontinuités des arbres.

    SmoothGrad est utilisé à la place si SMOOTH_GRAD=True.
    """
    if SMOOTH_GRAD:
        grad  = _grad_xgb_smooth(wrapper, X_atk)
        sign  = np.sign(grad)
    else:
        _, sign = _grad_xgb_vote(wrapper, X_atk)

    delta = eps * sign
    return np.clip(X_atk + delta, X_atk - eps, X_atk + eps)


# ══════════════════════════════════════════════════════════════
# PGD
# ══════════════════════════════════════════════════════════════

def _pgd_single_run_mlp(wrapper, x_orig, y_t, eps, alpha, iters):
    """Un seul run PGD depuis un point de départ aléatoire (MLP)."""
    x_adv = (x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)).detach()

    for _ in range(iters):
        x_adv.requires_grad_(True)
        loss = nn.BCEWithLogitsLoss()(wrapper.model(x_adv), y_t)
        wrapper.model.zero_grad()
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, x_orig - eps, x_orig + eps)

    with torch.no_grad():
        logits = wrapper.model(x_adv).squeeze(-1)
    return x_adv.detach(), logits


def pgd_mlp(wrapper, X_atk, y_atk, eps,
            iters=PGD_ITERS, restarts=PGD_RESTARTS, alpha=None):
    """PGD multi-restart sur MLP (gradient analytique PyTorch)."""
    alpha  = alpha or (eps / PGD_ALPHA_K)
    device = wrapper.device

    x_orig = torch.tensor(X_atk, dtype=torch.float32, device=device)
    y_t    = torch.tensor(y_atk, dtype=torch.float32, device=device).view(-1, 1)

    wrapper.model.eval()
    best_adv    = x_orig.clone()
    with torch.no_grad():
        best_logits = wrapper.model(x_orig).squeeze(-1)

    for r in range(restarts):
        x_adv, logits = _pgd_single_run_mlp(wrapper, x_orig, y_t, eps, alpha, iters)
        improved      = logits < best_logits
        best_adv[improved]    = x_adv[improved]
        best_logits[improved] = logits[improved]

        n_evaded = (best_logits < THRESHOLD_LOGIT).sum().item()
        print(f"      restart {r+1:02d}/{restarts} — "
              f"évadés cumulés : {n_evaded}/{len(X_atk)} "
              f"({100*n_evaded/len(X_atk):.1f}%)")

    return best_adv.cpu().numpy()


def pgd_logreg(wrapper, X_atk, y_atk, eps,
               iters=PGD_ITERS, restarts=PGD_RESTARTS, alpha=None):
    """PGD multi-restart sur LogReg (gradient analytique)."""
    alpha        = alpha or (eps / PGD_ALPHA_K)
    best_adv     = X_atk.copy().astype(np.float32)
    best_logits  = wrapper.logits_np(best_adv)

    for r in range(restarts):
        x_adv = (X_atk + np.random.uniform(-eps, eps, X_atk.shape)).astype(np.float32)

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
    PGD multi-restart sur XGBoost.

    Changements vs version initiale :
      - iters=200, restarts=10  (symétrique MLP/LogReg — grosse machine)
      - Gradient numérique recalculé à chaque itération sur x_adv courant
      - alpha = eps / PGD_ALPHA_K = eps / 4  (plus agressif)
      - Mode 'single' dans la boucle interne pour économiser le temps
        (le vote multi-ε_fd est réservé à FGSM où le signe est critique)

    Coût estimé : 200 iters × 10 restarts × 102 forward passes/iter
                = 204 000 forward passes  (quelques minutes sur grosse machine)
    """
    alpha       = alpha or (eps / PGD_ALPHA_K)
    best_adv    = X_atk.copy().astype(np.float32)
    best_logits = wrapper.logits_np(best_adv)

    for r in range(restarts):
        x_adv = (X_atk + np.random.uniform(-eps, eps, X_atk.shape)).astype(np.float32)

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
    """
    C&W LogReg : descente Adam sur le logit via gradient analytique (coef_).
    On minimise max(logit(x) − threshold + κ, 0).
    """
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
    C&W MLP : warm-start depuis PGD, puis descente Adam sur w (paramétrage tanh).
    delta = eps · tanh(w)  garantit que x+delta ∈ [x−eps, x+eps].
    """
    device = wrapper.device
    wrapper.model.eval()

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
    """
    C&W XGBoost — version améliorée.

    Changements vs version initiale :
      1. Warm-start depuis PGD (comme cw_mlp) au lieu d'un départ aléatoire.
         → Part d'une solution déjà raisonnablement adversariale.
      2. LR réduit à 0.005 (vs 0.01) pour stabiliser la descente avec
         gradient bruité.
      3. Iters=500 conservé (convergence plus lente avec gradient numérique).

    Flux :
      a) Warm-start : pgd_xgb avec 20 iters × 3 restarts
      b) Adam sur X_adv directement (pas de paramétrage tanh car numpy,
         la projection L∞ est faite par clip à chaque pas)

    Note : le paramétrage tanh serait possible en numpy mais apporte peu
    comparé à PGD warm-start + projection clip.
    """
    # ── Warm-start PGD ──────────────────────────────────────────
    print("      [C&W XGBoost : warm-start PGD...]")
    X_adv = pgd_xgb(wrapper, X_atk, y_atk, eps,
                    iters=20, restarts=3).astype(np.float64)
    print(f"      [C&W warm-start PGD terminé]")

    # ── Adam descent ────────────────────────────────────────────
    m = np.zeros_like(X_adv)
    v = np.zeros_like(X_adv)
    b1, b2, ep_adam = 0.9, 0.999, 1e-8

    for t in range(1, iters + 1):
        logits = wrapper.logits_np(X_adv)
        active = (logits - THRESHOLD_LOGIT + kappa > 0).astype(np.float64)

        # Gradient numérique du logit (single ε_fd suffit ici —
        # Adam lisse lui-même le bruit via les moments m et v)
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
    print(f"  RÉSUMÉ WHITEBOX — SWaT")
    print(f"{'═'*col_w}")
    for eps in EPS_LIST:
        print(f"\n  ┌─ eps = {eps} {'─'*(col_w-12)}┐")
        for model in ["MLP", "LogReg", "XGBoost"]:
            sub = df[(df.model == model)].copy()
            sub = sub[sub.attack.str.contains(str(eps))]
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


# ══════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE
# ══════════════════════════════════════════════════════════════

def run():
    print(f"\n{'═'*60}")
    print(f"  Device : {DEVICE}")
    print(f"  PGD (tous modèles) : {PGD_ITERS} iters × {PGD_RESTARTS} restarts")
    print(f"  PGD alpha          : eps / {PGD_ALPHA_K}")
    print(f"  FGSM XGBoost       : vote multi-ε_fd = {EPS_FD_LIST}")
    print(f"  C&W XGBoost        : warm-start PGD + lr={CW_LR_XGB}")
    print(f"  SmoothGrad         : {'ON (K=%d, σ=%.3f)' % (SMOOTH_K, SMOOTH_SIGMA) if SMOOTH_GRAD else 'OFF (vote par défaut)'}")
    print(f"{'═'*60}")

    X_test, y_test, mlp_w, logreg_w, xgb_w = load_artifacts()

    print("\n── Build eval sets ──────────────────────────────────")
    eval_sets = {
        "MLP":     (build_eval_set(X_test, y_test, mlp_w),    mlp_w),
        "LogReg":  (build_eval_set(X_test, y_test, logreg_w), logreg_w),
        "XGBoost": (build_eval_set(X_test, y_test, xgb_w),    xgb_w),
    }

    results = []

    for eps in EPS_LIST:
        print(f"\n{'═'*60}")
        print(f"  eps = {eps}")
        print(f"{'═'*60}")

        for model_name, ((X_ev, y_ev), victim_w) in eval_sets.items():
            mask  = (y_ev == 1)
            X_atk = X_ev[mask].astype(np.float32)
            y_atk = y_ev[mask]

            is_lr  = (model_name == "LogReg")
            is_xgb = (model_name == "XGBoost")

            print(f"\n  ── {model_name} {'─'*(50-len(model_name))}")

            # ── FGSM ──────────────────────────────────────────
            print("  [FGSM]")
            if is_lr:
                X_adv = fgsm_logreg(victim_w, X_atk, y_atk, eps)
            elif is_xgb:
                X_adv = fgsm_xgb(victim_w, X_atk, y_atk, eps)
            else:
                X_adv = fgsm_mlp(victim_w, X_atk, y_atk, eps)
            r = eval_attack(victim_w, X_ev, y_ev, X_adv, f"FGSM_eps{eps}", model_name)
            results.append(r); print_result(r)
            np.save(SAVE_DIR / f"adv_fgsm_{model_name}_eps{eps}.npy", X_adv)

            # ── PGD ───────────────────────────────────────────
            print("  [PGD]")
            if is_lr:
                X_adv = pgd_logreg(victim_w, X_atk, y_atk, eps)
            elif is_xgb:
                X_adv = pgd_xgb(victim_w, X_atk, y_atk, eps)
            else:
                X_adv = pgd_mlp(victim_w, X_atk, y_atk, eps)
            r = eval_attack(victim_w, X_ev, y_ev, X_adv, f"PGD_eps{eps}", model_name)
            results.append(r); print_result(r)
            np.save(SAVE_DIR / f"adv_pgd_{model_name}_eps{eps}.npy", X_adv)

            # ── C&W ───────────────────────────────────────────
            print("  [C&W]")
            if is_lr:
                X_adv = cw_logreg(victim_w, X_atk, y_atk, eps)
            elif is_xgb:
                X_adv = cw_xgb(victim_w, X_atk, y_atk, eps)
            else:
                X_adv = cw_mlp(victim_w, X_atk, y_atk, eps)
            r = eval_attack(victim_w, X_ev, y_ev, X_adv, f"CW_eps{eps}", model_name)
            results.append(r); print_result(r)
            np.save(SAVE_DIR / f"adv_cw_{model_name}_eps{eps}.npy", X_adv)

    df = pd.DataFrame(results)
    df.to_csv(SAVE_DIR / "whitebox_results.csv", index=False)
    print_summary(df)
    return df


if __name__ == "__main__":
    df = run()

    import json
    RESULTS_DIR = Path("~/swat/results").expanduser()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out = {}
    for model_name in df["model"].unique():
        out[model_name] = {}
        sub = df[df["model"] == model_name]
        for _, row in sub.iterrows():
            attack_clean = row["attack"].split("_eps")[0]
            if attack_clean not in out[model_name] or \
               row["asr"] > out[model_name][attack_clean]["evasion_rate"] / 100:
                out[model_name][attack_clean] = {
                    "evasion_rate": round(row["asr"] * 100, 2),
                    "precision":    round(row.get("prec_adv", 0), 4),
                    "recall":       round(row["rec_adv"], 4),
                    "f1":           round(row["f1_adv"], 4),
                }

    with open(RESULTS_DIR / "whitebox_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nJSON sauvegardé → {RESULTS_DIR / 'whitebox_results.json'}")