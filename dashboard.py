"""
dashboard.py — SWaT Adversarial Attacks Results Dashboard
Usage: streamlit run dashboard.py
"""

import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("~/swat/results").expanduser()
MODELS = ["MLP", "LogReg", "XGBoost"]
WHITEBOX_ATTACKS  = ["FGSM", "PGD", "CW"]
BLACKBOX_ATTACKS  = ["Square", "NES", "HSJA", "RayS"]
TRANSFER_ATTACKS  = ["MI-FGSM", "VMI-FGSM", "Ensemble-MI"]
ALL_ATTACKS = WHITEBOX_ATTACKS + BLACKBOX_ATTACKS + TRANSFER_ATTACKS

COLOR_MAP = {
    "MLP":     "#2563EB",   # blue-600
    "LogReg":  "#0EA5E9",   # sky-500
    "XGBoost": "#6366F1",   # indigo-500
}

DEFENSE_COLOR_MAP = {
    "AT-FGSM":  "#2563EB",
    "AT-PGD":   "#93C5FD",
    "Aug-FGSM": "#0EA5E9",
}

# ── Plotly template clair (évite le fond noir sur onglets blancs) ──────────────
PLOT_TEMPLATE = "plotly_white"


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def load_all_results():
    data = {}
    for fname in ["train_results.json", "whitebox_results.json",
                  "blackbox_results.json", "transfer_results.json"]:
        key  = fname.replace("_results.json", "")
        path = RESULTS_DIR / fname
        data[key] = json.load(open(path)) if path.exists() else {}
 
    # ← CHANGEMENT : defense_results.json (pas defense_whitebox_results.json)
    defense_path = RESULTS_DIR / "defense_results.json"
    data["defense"] = json.load(open(defense_path)) if defense_path.exists() else {}
    return data


def evasion_rate(results, model, attack):
    try:
        return results[model][attack]["evasion_rate"]
    except (KeyError, TypeError):
        return None


def get_evasion_df(results):
    rows = []
    for category, attacks in [("whitebox", WHITEBOX_ATTACKS),
                               ("blackbox", BLACKBOX_ATTACKS),
                               ("transfer", TRANSFER_ATTACKS)]:
        res = results.get(category, {})
        for model in MODELS:
            for attack in attacks:
                val = evasion_rate(res, model, attack)
                rows.append({"model": model, "attack": attack,
                              "category": category, "evasion_rate": val})
    return pd.DataFrame(rows)


def get_clean_accuracy(results):
    train = results.get("train", {})
    return {m: train.get(m, {}).get("clean_accuracy") for m in MODELS}


def get_defense_df(defense_data: dict, baseline_data: dict) -> pd.DataFrame:
    """
    Construit un DataFrame avec une ligne par (model, defense, attack).
 
    Colonnes :
      model, defense, attack,
      asr_baseline  (depuis Baseline ou whitebox_results),
      asr_defended  (depuis la défense active),
      delta_asr, f1_defended, delta_f1, recall
    """
    rows = []
 
    for model_name, defenses in defense_data.items():
 
        # ── ASR baseline : depuis la clé "Baseline" ou whitebox_results ──
        baseline_attacks = defenses.get("Baseline", {})
 
        for defense_name, attacks in defenses.items():
            if defense_name == "Baseline":
                # On ne crée pas de ligne pour Baseline lui-même,
                # il sert uniquement de référence
                continue
 
            for attack_name, metrics in attacks.items():
                # ASR baseline : chercher dans Baseline d'abord,
                # puis dans whitebox_results
                asr_base = None
                if attack_name in baseline_attacks:
                    asr_base = baseline_attacks[attack_name].get("evasion_rate")
                else:
                    for cat_key in ["whitebox", "blackbox", "transfer"]:
                        try:
                            asr_base = baseline_data[cat_key][model_name][attack_name]["evasion_rate"]
                            break
                        except (KeyError, TypeError):
                            continue
 
                asr_def  = metrics.get("evasion_rate")
                f1_def   = metrics.get("f1")
                delta_f1 = metrics.get("delta_f1")
                recall   = metrics.get("recall")
 
                delta_asr = (
                    round(asr_def - asr_base, 2)
                    if asr_def is not None and asr_base is not None
                    else None
                )
 
                rows.append({
                    "model":        model_name,
                    "defense":      defense_name,
                    "attack":       attack_name,
                    "asr_baseline": asr_base,
                    "asr_defended": asr_def,
                    "delta_asr":    delta_asr,
                    "f1_defended":  f1_def,
                    "delta_f1":     delta_f1,
                    "recall":       recall,
                })
 
    return pd.DataFrame(rows)
 
 
# ══════════════════════════════════════════════════════════════
# TEST RAPIDE (hors Streamlit)
# ══════════════════════════════════════════════════════════════

# ── Page layout ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SWaT · Digital Twin Adversarial Dashboard",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Fond général blanc ── */
    [data-testid="stAppViewContainer"] { background: #f8fafc; }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    /* ── Cartes métriques ── */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-size: 11px;
        color: #64748b;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.2;
    }

    /* ── Titres de section ── */
    .section-title {
        font-size: 12px;
        font-weight: 700;
        color: #94a3b8;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 1.5rem 0 0.75rem;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 6px;
    }

    /* ── Tabs ───────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 12px;
    gap: 10px;                  /* espace entre tabs */
    padding: 6px;
    border: 1px solid #e2e8f0;
}

/* tab normal */
.stTabs [data-baseweb="tab"] {
    color: #475569 !important;
    font-weight: 500;
    border-radius: 10px;
    padding: 10px 18px;
    transition: all 0.18s ease;
}

/* hover léger */
.stTabs [data-baseweb="tab"]:hover {
    background: #f1f5f9;
    color: #0f172a !important;
}

/* tab active */
.stTabs [aria-selected="true"] {
    background: #eff6ff !important;
    color: #2563EB !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
}

    /* ── Helpers delta ── */
    .delta-neg { color: #16a34a; font-weight: 600; }
    .delta-pos { color: #dc2626; font-weight: 600; }

    /* ── Titre principal ── */
    h1 { color: #0f172a !important; }
</style>
""", unsafe_allow_html=True)


# ── Load ───────────────────────────────────────────────────────────────────────
results    = load_all_results()
data_loaded = any(v for k, v in results.items() if k != "defense")
defense_data = results.get("defense", {})

with st.sidebar:

    # ── Logo centré ──
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("assets/logo-cran.png", use_container_width=True)

    # ── Titre projet ──
    st.markdown("""
    <div style="text-align:center; margin: 0.5rem 0 0.25rem;">
        <div style="font-size:15px; font-weight:600; color:#0f172a;">Securing Digital Twins</div>
        <div style="font-size:11px; color:#64748b; letter-spacing:0.04em;">Université de Lorraine · CRAN</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Étudiante ──
    st.markdown("""
    <div style="margin-bottom:0.75rem;">
        <div style="font-size:10px; font-weight:700; color:#94a3b8; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:8px;">Étudiante</div>
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:32px; height:32px; border-radius:50%; background:#eff6ff; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:600; color:#2563EB; flex-shrink:0;">MM</div>
            <div style="font-size:13px; color:#0f172a; font-weight:500;">Mellaz Maya</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Encadrantes ──
    st.markdown("""
    <div style="margin-bottom:1rem;">
        <div style="font-size:10px; font-weight:700; color:#94a3b8; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:8px;">Encadrantes</div>
        <div style="display:flex; flex-direction:column; gap:8px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:32px; height:32px; border-radius:50%; background:#f0f9ff; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:600; color:#0EA5E9; flex-shrink:0;">LL</div>
                <div style="font-size:12px; color:#475569;">Louail Lemia</div>
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:32px; height:32px; border-radius:50%; background:#eef2ff; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:600; color:#6366F1; flex-shrink:0;">MD</div>
                <div style="font-size:12px; color:#475569;">Meroua Daoudi</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if not data_loaded:
        st.warning(f"Aucun résultat trouvé dans `{RESULTS_DIR}`")

    st.markdown("**Models**")
    selected_models = st.multiselect("Select models", MODELS,
                                     default=MODELS, label_visibility="collapsed")

    st.markdown("**Attack categories**")
    show_whitebox = st.checkbox("White-box (FGSM, PGD, C&W)", value=True)
    show_blackbox = st.checkbox("Black-box (Square, NES, HSJA, RayS)", value=True)
    show_transfer = st.checkbox("Transfer (MI-FGSM, VMI-FGSM, Ensemble)", value=True)

    st.divider()
    st.markdown("**Result files**")
    for fname in ["train_results.json", "whitebox_results.json",
                  "blackbox_results.json", "transfer_results.json",
                  "defense_results.json"]:
        path = RESULTS_DIR / fname
        icon = "🟢" if path.exists() else "🔵"
        st.markdown(f"{icon} `{fname}`")

    if st.button("🔄 Reload results"):
        st.cache_data.clear()
        st.rerun()

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("# Securing Digital Twin : Adversarial Robustness Dashboard")
st.markdown(
    "<span style='color:#000000;font-size:28 px;'>"
    "Attacking & Defending AI model in Digital Twins : White-box & Black-box Adversarial Evaluation "
   
    "</span>",
    unsafe_allow_html=True
)

if not data_loaded:
    st.info("Lance tes scripts pipeline et place les JSON dans `~/swat/results/`.")
    st.stop()

df         = get_evasion_df(results)
clean_acc  = get_clean_accuracy(results)

active_attacks = []
if show_whitebox: active_attacks += WHITEBOX_ATTACKS
if show_blackbox: active_attacks += BLACKBOX_ATTACKS
if show_transfer: active_attacks += TRANSFER_ATTACKS

df_filtered = df[df["model"].isin(selected_models) &
                 df["attack"].isin(active_attacks)].copy()

# ── Shared plot layout defaults (fond transparent, axes lisibles) ───────────────
PLOT_LAYOUT = dict(
    template=PLOT_TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,250,252,1)",  # f8fafc — légèrement grisé
    font=dict(color="#0f172a"),
)

# ── KPI row ────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Model baseline — clean accuracy</p>',
            unsafe_allow_html=True)
kpi_cols = st.columns(len(selected_models))
for i, model in enumerate(selected_models):
    acc     = clean_acc.get(model)
    val_str = f"{acc:.1f}%" if acc is not None else "—"
    kpi_cols[i].markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{model}</div>
        <div class="metric-value" style="color:{COLOR_MAP[model]}">{val_str}</div>
        <div style="font-size:12px;color:#64748b;margin-top:4px;">clean accuracy</div>
    </div>
    """, unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Evasion overview",
    " Heatmap",
    " Per-attack detail",
    " Defenses",
    " Raw data",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — Grouped bar chart
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Evasion rate by attack & model</p>',
                unsafe_allow_html=True)

    fig = go.Figure()
    for model in selected_models:
        sub = df_filtered[df_filtered["model"] == model]
        fig.add_trace(go.Bar(
            name=model, x=sub["attack"], y=sub["evasion_rate"],
            marker_color=COLOR_MAP[model],
            text=[f"{v:.1f}%" if v is not None else "N/A" for v in sub["evasion_rate"]],
            textposition="outside",
        ))

    sep_positions = [len(WHITEBOX_ATTACKS) - 0.5,
                     len(WHITEBOX_ATTACKS) + len(BLACKBOX_ATTACKS) - 0.5]
    for x in sep_positions:
        fig.add_vline(x=x, line_dash="dot", line_color="#cbd5e1", line_width=1)
    fig.add_annotation(x=1, y=1.08, xref="x", yref="paper",
                       text="White-box", showarrow=False, font=dict(size=11, color="#64748b"))
    fig.add_annotation(x=len(WHITEBOX_ATTACKS)+1.5, y=1.08, xref="x", yref="paper",
                       text="Black-box", showarrow=False, font=dict(size=11, color="#64748b"))
    fig.add_annotation(x=len(WHITEBOX_ATTACKS)+len(BLACKBOX_ATTACKS)+1, y=1.08,
                       xref="x", yref="paper",
                       text="Transfer", showarrow=False, font=dict(size=11, color="#64748b"))

    fig.update_layout(
        **PLOT_LAYOUT,
        barmode="group",
        yaxis=dict(title="Evasion rate (%)", range=[0, 115]),
        xaxis_title="Attack",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420, margin=dict(t=80, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    if not df_filtered["evasion_rate"].dropna().empty:
        st.markdown('<p class="section-title">Extremes</p>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        top = df_filtered.nlargest(3, "evasion_rate")[["attack", "model", "evasion_rate"]]
        bot = df_filtered.nsmallest(3, "evasion_rate")[["attack", "model", "evasion_rate"]]
        with col_a:
            st.markdown("**Highest evasion**")
            st.dataframe(top.assign(evasion_rate=top["evasion_rate"].map("{:.1f}%".format))
                         .rename(columns={"evasion_rate": "rate"}),
                         hide_index=True, use_container_width=True)
        with col_b:
            st.markdown("**Lowest evasion (hardest to fool)**")
            st.dataframe(bot.assign(evasion_rate=bot["evasion_rate"].map("{:.1f}%".format))
                         .rename(columns={"evasion_rate": "rate"}),
                         hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — Heatmap
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Evasion rate heatmap — attacks × models</p>',
                unsafe_allow_html=True)

    pivot = df_filtered.pivot_table(index="attack", columns="model",
                                    values="evasion_rate")
    pivot = pivot.reindex(
        index=[a for a in ALL_ATTACKS if a in pivot.index],
        columns=[m for m in MODELS if m in pivot.columns]
    )

    fig2 = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0, "#eff6ff"], [0.4, "#3b82f6"], [1, "#1e3a8a"]],
        zmin=0, zmax=100,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "N/A" for v in row]
              for row in pivot.values],
        texttemplate="%{text}", textfont=dict(size=13, color="#0f172a"),
        colorbar=dict(title="Evasion %", ticksuffix="%"),
    ))
    fig2.update_layout(
        **PLOT_LAYOUT,
        height=420,
        margin=dict(l=80, r=20, t=30, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — Per-attack detail
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Per-attack detailed metrics</p>',
                unsafe_allow_html=True)

    attack_choice = st.selectbox("Select attack", active_attacks)

    detail_rows = []
    for cat_key, attacks in [("whitebox", WHITEBOX_ATTACKS),
                              ("blackbox", BLACKBOX_ATTACKS),
                              ("transfer", TRANSFER_ATTACKS)]:
        if attack_choice in attacks:
            cat_res = results.get(cat_key, {})
            for model in selected_models:
                metrics = cat_res.get(model, {}).get(attack_choice, {})
                if metrics:
                    detail_rows.append({"model": model, **metrics})
            break

    if detail_rows:
        det_df = pd.DataFrame(detail_rows).set_index("model")
        numeric_cols = det_df.select_dtypes(include="number").columns
        st.dataframe(det_df.style.format({col: "{:.4f}" for col in numeric_cols},
                                          na_rep="N/A"),
                     use_container_width=True)

        radar_keys = [k for k in ["evasion_rate", "precision", "recall", "f1"]
                      if k in det_df.columns]
        if len(radar_keys) >= 3:
            fig3 = go.Figure()
            for model in det_df.index:
                vals     = [det_df.loc[model, k] for k in radar_keys]
                vals_pct = [(v * 100 if v <= 1 else v) for v in vals]
                fig3.add_trace(go.Scatterpolar(
                    r=vals_pct + [vals_pct[0]],
                    theta=radar_keys + [radar_keys[0]],
                    name=model,
                    line=dict(color=COLOR_MAP.get(model, "#2563EB"), width=2),
                    fill="toself", opacity=0.4,
                ))
            fig3.update_layout(
                **PLOT_LAYOUT,
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                height=380,
            )
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info(f"Pas de métriques détaillées pour `{attack_choice}`.")

    if attack_choice in BLACKBOX_ATTACKS:
        st.markdown('<p class="section-title">Query budget</p>', unsafe_allow_html=True)
        bb_res = results.get("blackbox", {})
        q_data = {m: bb_res.get(m, {}).get(attack_choice, {}).get("n_queries")
                  for m in selected_models}
        q_df   = pd.DataFrame({"model": list(q_data.keys()),
                                "avg_queries": list(q_data.values())})
        fig_q  = px.bar(q_df, x="model", y="avg_queries", color="model",
                        color_discrete_map=COLOR_MAP, template=PLOT_TEMPLATE,
                        labels={"avg_queries": "Avg queries to evasion"})
        fig_q.update_layout(
            **PLOT_LAYOUT,
            showlegend=False, height=280,
        )
        st.plotly_chart(fig_q, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — DEFENSES
# ══════════════════════════════════════════════════════════════
with tab4:

    if not defense_data:
        st.info("Pas encore de résultats de défense. Lance `evaluate.py` d'abord.")
        st.stop()

    def_df = get_defense_df(defense_data, results)

    st.markdown('<p class="section-title">Filtres</p>', unsafe_allow_html=True)
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sel_models_def = st.multiselect(
            "Modèle", MODELS,
            default=[m for m in MODELS if m in def_df["model"].unique()],
            key="def_model"
        )
    with col_f2:
        avail_defenses = sorted(def_df["defense"].unique().tolist())
        sel_defenses = st.multiselect(
            "Défense", avail_defenses, default=avail_defenses, key="def_defense"
        )
    with col_f3:
        avail_attacks = sorted(def_df["attack"].unique().tolist())
        sel_attacks_def = st.multiselect(
            "Attaque", avail_attacks, default=avail_attacks, key="def_attack"
        )

    df_def_filtered = def_df[
        def_df["model"].isin(sel_models_def) &
        def_df["defense"].isin(sel_defenses) &
        def_df["attack"].isin(sel_attacks_def)
    ].copy()

    # ── Section 1 : Comparaison avant / après par attaque ───
    st.markdown('<p class="section-title">Avant / après défense — ASR par attaque</p>',
                unsafe_allow_html=True)

    for model_name in sel_models_def:
        sub = df_def_filtered[df_def_filtered["model"] == model_name].copy()
        if sub.empty:
            continue

        st.markdown(f"**{model_name}**")

        fig_ba = go.Figure()

        base_vals = sub.groupby("attack")["asr_baseline"].first().reset_index()
        fig_ba.add_trace(go.Bar(
            name="Baseline",
            x=base_vals["attack"],
            y=base_vals["asr_baseline"],
            marker_color="#cbd5e1",
            opacity=0.9,
            text=[f"{v:.1f}%" if v is not None else "N/A"
                  for v in base_vals["asr_baseline"]],
            textposition="outside",
        ))

        for defense_name in sub["defense"].unique():
            d = sub[sub["defense"] == defense_name]
            color = DEFENSE_COLOR_MAP.get(defense_name, "#2563EB")
            fig_ba.add_trace(go.Bar(
                name=defense_name,
                x=d["attack"],
                y=d["asr_defended"],
                marker_color=color,
                text=[f"{v:.1f}%" if v is not None else "N/A"
                      for v in d["asr_defended"]],
                textposition="outside",
            ))

        fig_ba.update_layout(
            **PLOT_LAYOUT,
            barmode="group",
            yaxis=dict(title="ASR (%)", range=[0, 115]),
            xaxis_title="Attaque",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            height=380,
            margin=dict(t=60, b=20),
        )
        st.plotly_chart(fig_ba, use_container_width=True)

    # ── Section 2 : Delta ASR ───────────────────────────────
    st.markdown('<p class="section-title">Réduction de l\'ASR (baseline − défendu)</p>',
                unsafe_allow_html=True)
    st.caption("Valeur négative = la défense a réduit le taux d'évasion (bien). "
               "Valeur positive = la défense a empiré les choses.")

    df_delta = df_def_filtered.copy()
    df_delta["reduction"] = df_delta.apply(
        lambda r: round(r["asr_baseline"] - r["asr_defended"], 2)
        if (r["asr_baseline"] is not None and r["asr_defended"] is not None)
        else None,
        axis=1
    )

    fig_delta = go.Figure()
    for defense_name in df_delta["defense"].unique():
        d     = df_delta[df_delta["defense"] == defense_name]
        color = DEFENSE_COLOR_MAP.get(defense_name, "#2563EB")
        fig_delta.add_trace(go.Bar(
            name=defense_name,
            x=[f"{r['model']}/{r['attack']}" for _, r in d.iterrows()],
            y=d["reduction"],
            marker_color=color,
            text=[f"{v:+.1f}pp" if v is not None else "N/A"
                  for v in d["reduction"]],
            textposition="outside",
        ))

    fig_delta.add_hline(y=0, line_dash="dot", line_color="#94a3b8", line_width=1)
    fig_delta.update_layout(
        **PLOT_LAYOUT,
        barmode="group",
        yaxis=dict(title="Réduction ASR (pp)"),
        xaxis_title="Modèle / Attaque",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        height=420,
        margin=dict(t=60, b=60),
    )
    st.plotly_chart(fig_delta, use_container_width=True)

    # ── Section 3 : Tableau récap ────────────────────────────
    st.markdown('<p class="section-title">Tableau récapitulatif</p>',
                unsafe_allow_html=True)

    display_cols = ["model", "defense", "attack",
                    "asr_baseline", "asr_defended", "delta_asr",
                    "f1_defended", "delta_f1", "recall"]

    disp = df_def_filtered[display_cols].copy()
    disp = disp.rename(columns={
        "model":        "Modèle",
        "defense":      "Défense",
        "attack":       "Attaque",
        "asr_baseline": "ASR baseline (%)",
        "asr_defended": "ASR défendu (%)",
        "delta_asr":    "ΔASR",
        "f1_defended":  "F1 défendu",
        "delta_f1":     "ΔF1",
        "recall":       "Recall",
    })

    def color_delta(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        if val < 0:
            return "color: #16a34a; font-weight: 600"
        elif val > 0:
            return "color: #dc2626; font-weight: 600"
        return ""

    styled = disp.style \
        .format({
            "ASR baseline (%)": lambda x: f"{x:.1f}%" if x is not None else "N/A",
            "ASR défendu (%)":  lambda x: f"{x:.1f}%" if x is not None else "N/A",
            "ΔASR":             lambda x: f"{x:+.1f}pp" if x is not None else "N/A",
            "F1 défendu":       lambda x: f"{x:.4f}" if x is not None else "N/A",
            "ΔF1":              lambda x: f"{x:+.4f}" if x is not None else "N/A",
            "Recall":           lambda x: f"{x:.4f}" if x is not None else "N/A",
        }, na_rep="N/A") \
        .map(color_delta, subset=["ΔASR", "ΔF1"])

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Section 4 : KPIs résumé par défense ─────────────────
    st.markdown('<p class="section-title">Résumé par défense</p>',
                unsafe_allow_html=True)

    summary_cols = st.columns(len(sel_defenses)) if sel_defenses else []
    for i, defense_name in enumerate(sel_defenses):
        d   = df_def_filtered[df_def_filtered["defense"] == defense_name]
        if d.empty:
            continue
        avg_asr_base = d["asr_baseline"].mean()
        avg_asr_def  = d["asr_defended"].mean()
        avg_red      = (avg_asr_base - avg_asr_def) if (avg_asr_base and avg_asr_def) else None
        color        = DEFENSE_COLOR_MAP.get(defense_name, "#2563EB")

        summary_cols[i].markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{defense_name}</div>
            <div class="metric-value" style="color:{color}">
                {avg_asr_def:.1f}%
            </div>
            <div style="font-size:12px;color:#64748b;margin-top:4px;">
                ASR moyen défendu<br>
                baseline : {avg_asr_base:.1f}% →
                <span style="color:{'#16a34a' if avg_red and avg_red > 0 else '#dc2626'}">
                    {f'{avg_red:+.1f}pp' if avg_red is not None else 'N/A'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — Raw data
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">All evasion rates</p>', unsafe_allow_html=True)
    display_df = df[df["model"].isin(selected_models) &
                    df["attack"].isin(active_attacks)].copy()
    display_df["evasion_rate"] = display_df["evasion_rate"].map(
        lambda x: f"{x:.2f}%" if x is not None else "N/A"
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown('<p class="section-title">Raw JSON</p>', unsafe_allow_html=True)
    for key, val in results.items():
        with st.expander(f"{key}_results.json"):
            st.json(val)