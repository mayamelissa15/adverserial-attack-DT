

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score


# ══════════════════════════════════════════════════════════════
# ARCHITECTURE MLP
# ══════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """
    Perceptron multicouche binaire.
    Architecture : input → 128 → 64 → 32 → 1 (logit)
    """
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),          nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)

# ══════════════════════════════════════════════════════════════
# ARCHITECTURES SUBSTITUTS
# ══════════════════════════════════════════════════════════════

class SmallMLP(nn.Module):
    """
    MLP plus petit que MLP — substitut léger.
    Architecture : input → 64 → 32 → 1 (logit)
    """
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class DeepMLP(nn.Module):
    """
    MLP plus profond que MLP — substitut expressif.
    Architecture : input → 256 → 128 → 64 → 32 → 1 (logit)
    """
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),        nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),          nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)
# ══════════════════════════════════════════════════════════════
# WRAPPER MLP
# ══════════════════════════════════════════════════════════════

class MLPWrapper:
    """Wrapper PyTorch exposant l'interface commune aux attaques."""

    def __init__(self, model, device):
        self.model  = model
        self.device = device

    def predict(self, X, threshold=0.45):
        x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x_t).squeeze(-1)
            proba  = torch.sigmoid(logits).cpu().numpy()
        return (proba >= threshold).astype(int)

    def predict_proba(self, X):
        x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x_t).squeeze(-1)
            proba  = torch.sigmoid(logits).cpu().numpy()
        return proba

    def logits_np(self, X):
        x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model(x_t).squeeze(-1).cpu().numpy()


# ══════════════════════════════════════════════════════════════
# WRAPPER LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════

class LogRegWrapper:
    """
    Wrapper sklearn LogisticRegression.
    Le gradient analytique du logit par rapport à x est simplement
    le vecteur de coefficients  w  du modèle linéaire.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)

    def predict(self, X, threshold=0.45):
        return (self.model.predict_proba(X)[:, 1] >= threshold).astype(int)

    def logits_np(self, X):
        return X @ self.model.coef_[0] + self.model.intercept_[0]

    def grad_bce(self, X, y):
        """
        Gradient de la BCE par rapport à x :
            ∇_x L_BCE = (p - y) · w
        Utilisé par FGSM/PGD.
        """
        p    = 1 / (1 + np.exp(-self.logits_np(X)))  # (N,)
        return (p - y)[:, None] * self.model.coef_    # (N, D)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
        
    def grad_logit(self, X):
        """
        Gradient du logit par rapport à x :
            ∂logit/∂x = w   (constant pour un modèle linéaire)
        Utilisé par C&W.
        """
        return np.tile(self.model.coef_, (len(X), 1))  # (N, D)


# ══════════════════════════════════════════════════════════════
# WRAPPER XGBOOST  —  gradient NUMÉRIQUE
# ══════════════════════════════════════════════════════════════

class XGBoostWrapper:
    """
    Wrapper XGBoost avec gradient numérique par différences finies centrées.

    Pourquoi différences finies ?
    ─────────────────────────────
    XGBoost est un ensemble d'arbres binaires.  Les arbres sont des fonctions
    constantes par morceaux : leur dérivée exacte est 0 presque partout et
    indéfinie aux nœuds de décision.  Il est donc impossible d'utiliser la
    backpropagation classique.

    Solution : on approxime le gradient du logit  g = log(p/(1-p))  en
    évaluant le modèle deux fois par feature, en déplaçant la valeur de ±ε_fd :

        ∂g/∂x_j  ≈  [g(x + ε_fd·eⱼ) − g(x − ε_fd·eⱼ)] / (2·ε_fd)

    C'est la formule des différences finies centrées d'ordre 2, qui annule
    l'erreur de premier ordre et donne une approximation en O(ε_fd²).

    Coût : 2 × D forward passes par batch (102 pour SWaT à 51 features).
    Valeur recommandée : ε_fd = 1e-3  (compromis précision / bruit numérique).
    """

    def __init__(self, model):
        self.model = model

    # ── Prédictions ───────────────────────────────────────────

    def predict(self, X, threshold=0.45):
        return (self.predict_proba(X) >= threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def logits_np(self, X):
        """Logit = log(p / (1−p)), valeur réelle utilisée comme score continu."""
        p = np.clip(self.predict_proba(X), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    # ── Gradient numérique ────────────────────────────────────

    def grad_numerical(self, X, eps_fd=1e-3):
        """
        Gradient numérique du logit par différences finies centrées.

        Paramètres
        ----------
        X      : array (N, D)  — batch d'entrées
        eps_fd : float         — pas de différentiation (défaut 1e-3)

        Retourne
        --------
        grad : array (N, D)  — gradient approché ∂logit/∂x
        """
        X    = X.astype(np.float64)
        grad = np.zeros_like(X)

        for j in range(X.shape[1]):
            X_plus        = X.copy()
            X_minus       = X.copy()
            X_plus[:, j]  += eps_fd
            X_minus[:, j] -= eps_fd
            grad[:, j]    = (
                self.logits_np(X_plus) - self.logits_np(X_minus)
            ) / (2 * eps_fd)

        return grad                                    # (N, D)

    # ── Alias pour compatibilité avec les fonctions d'attaque ─

    def grad_bce(self, X, y, eps_fd=1e-3):
        """
        Approximation du gradient BCE via le gradient du logit.
        grad_BCE ≈ (p − y) · grad_logit
        """
        p        = np.clip(self.predict_proba(X), 1e-7, 1 - 1e-7)
        grad_log = self.grad_numerical(X, eps_fd)
        return (p - y)[:, None] * grad_log

    def grad_logit(self, X, eps_fd=1e-3):
        """Alias direct vers grad_numerical (pour cohérence avec LogRegWrapper)."""
        return self.grad_numerical(X, eps_fd)


# ══════════════════════════════════════════════════════════════
# UTILITAIRES D'ÉVALUATION
# ══════════════════════════════════════════════════════════════

    """
    Retourne (X_eval, y_eval) : uniquement les vrais positifs du modèle clean,
    c'est-à-dire les exemples d'attaque que le modèle détecte correctement.
    Ce sont les seuls exemples sur lesquels une attaque adversariale a du sens.
    """
def build_eval_set(X_test, y_test, wrapper, threshold=0.45):
    y_pred = wrapper.predict(X_test, threshold)
    mask = (y_test == 1) & (y_pred == 1)
    return X_test[mask].astype(np.float32), y_test[mask]


def eval_attack(wrapper, X_full, y_full, X_adv, attack_name, model_name,
                threshold=0.45):
    """
    Évalue l'efficacité d'une attaque adversariale.

    Métriques retournées
    --------------------
    asr      : Attack Success Rate — fraction des attaques qui trompent le modèle
    f1_clean : F1 sur le jeu complet sans perturbation
    f1_adv   : F1 après substitution des exemples adversariaux
    rec_adv  : Recall après attaque
    prec_adv : Precision après attaque
    linf     : Norme L∞ moyenne de la perturbation
    """
    # Indices des vrais positifs dans X_full
    y_pred_clean = wrapper.predict(X_full)
    tp_mask      = (y_full == 1) & (y_pred_clean == 1)

    # Construit le jeu évalué : remplace les TP par leurs versions adversariales
    X_eval       = X_full.copy()
    X_eval[tp_mask] = X_adv

    y_pred_adv   = wrapper.predict(X_eval)

    # ASR = fraction des TP initiaux maintenant mal classifiés
    asr = float((y_pred_adv[tp_mask] == 0).mean()) if tp_mask.sum() > 0 else 0.0

    f1_clean = f1_score(y_full, y_pred_clean, zero_division=0)
    f1_adv   = f1_score(y_full, y_pred_adv,   zero_division=0)
    rec_adv  = recall_score(y_full, y_pred_adv,    zero_division=0)
    prec_adv = precision_score(y_full, y_pred_adv, zero_division=0)
    linf     = float(np.max(np.abs(X_adv - X_full[tp_mask]))) if tp_mask.sum() > 0 else 0.0

    return {
        "attack":    attack_name,
        "model":     model_name,
        "asr":       asr,
        "f1_clean":  f1_clean,
        "f1_adv":    f1_adv,
        "rec_adv":   rec_adv,
        "prec_adv":  prec_adv,
        "linf":      linf,
    }