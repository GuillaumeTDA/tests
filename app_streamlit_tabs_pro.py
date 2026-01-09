# -*- coding: utf-8 -*-
"""
Banking Book Simulator - Streamlit App (v2)
- Onglets (tabs)
- 4 graphiques de distributions (CHR, notional, ECL S1, ECL S2)
- Variantes "pro" : log-scale ECL, boxplot ECL par CHR, heatmap migration (si CHR_new existe),
  KPIs additionnels, export CSV/Parquet
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Import de tes fonctions depuis ton module
from bbs_v3 import (
    CONFIG,
    generate_loan_df,
    add_forward_pd_columns,
    compute_ecl_s1_s2_forward_pd,
)

st.set_page_config(page_title="Banking Book Simulator - IFRS9", layout="wide")
st.title("Banking Book Simulator — Paramétrage & génération portefeuille")

# =========================
# Sidebar - Paramètres
# =========================
st.sidebar.header("Paramètres de simulation")

segment = st.sidebar.selectbox("Segment", ["Corporate", "SME", "Large Corporate"], index=0)
arrete = st.sidebar.text_input("Date d'arrêté", value=str(CONFIG.get("Arrete", "Q42000")))

n = st.sidebar.number_input(
    "Nombre de lignes (n)",
    min_value=100,
    max_value=2_000_000,
    value=int(CONFIG.get("n", 1000)),
    step=100
)

seed = st.sidebar.number_input(
    "Seed",
    min_value=0,
    max_value=10_000_000,
    value=int(CONFIG.get("seed", 42)),
    step=1
)

st.sidebar.subheader("Taux")
ecb_rate = st.sidebar.number_input(
    "Taux BCE (ecb_rate, % - paramètre)",
    min_value=-5.0,
    max_value=15.0,
    value=float(CONFIG.get("ecb_rate", 3.75)),
    step=0.05
)
alea_std = st.sidebar.number_input(
    "Volatilité aléa taux (std, points de %)",
    min_value=0.0,
    max_value=5.0,
    value=float(CONFIG.get("interest_alea_std", 0.25)),
    step=0.05
)

st.sidebar.subheader("Dates")
orig_start = st.sidebar.date_input(
    "Origination start",
    value=pd.to_datetime(CONFIG.get("origination_start", "2015-01-01"))
)
orig_end = st.sidebar.date_input(
    "Origination end",
    value=pd.to_datetime(CONFIG.get("origination_end", "2022-12-31"))
)

st.sidebar.subheader("Maturité")
min_tenor = st.sidebar.number_input(
    "Tenor min (années)",
    min_value=0,
    max_value=50,
    value=int(CONFIG.get("min_tenor_years", 1)),
    step=1
)
max_tenor = st.sidebar.number_input(
    "Tenor max (années)",
    min_value=1,
    max_value=50,
    value=int(CONFIG.get("max_tenor_years", 10)),
    step=1
)

st.sidebar.subheader("Notional")
notional_mean = st.sidebar.number_input(
    "Notional mean",
    min_value=1_000.0,
    max_value=1e9,
    value=float(CONFIG.get("notional_mean", 1_000_000)),
    step=10_000.0
)
notional_std = st.sidebar.number_input(
    "Notional std",
    min_value=0.0,
    max_value=1e9,
    value=float(CONFIG.get("notional_std", 300_000)),
    step=10_000.0
)
notional_min = st.sidebar.number_input(
    "Notional min",
    min_value=0.0,
    max_value=1e9,
    value=float(CONFIG.get("notional_min", 10_000)),
    step=1_000.0
)

st.sidebar.subheader("IFRS9 / ECL")
horizon_years = st.sidebar.slider(
    "Horizon PD forward (années)",
    min_value=1,
    max_value=30,
    value=10,
    step=1
)
term_slope = st.sidebar.number_input(
    "Pente PD forward (term_slope)",
    min_value=-0.50,
    max_value=1.00,
    value=float(0.05),
    step=0.01
)
pd_cap = st.sidebar.number_input(
    "Cap PD forward (par année)",
    min_value=0.01,
    max_value=1.00,
    value=float(0.50),
    step=0.01
)

stage2_thr = st.sidebar.slider(
    "Seuil Stage 2 (CHR > seuil)",
    min_value=1,
    max_value=10,
    value=6,
    step=1
)

discount_rate_on = st.sidebar.checkbox("Actualiser (discounting)", value=False)
disc = None
if discount_rate_on:
    disc = st.sidebar.number_input(
        "Discount rate annuel",
        min_value=-0.10,
        max_value=0.30,
        value=0.03,
        step=0.005
    )

st.sidebar.subheader("Affichage / perf")
max_rows_display = st.sidebar.slider("Nombre de lignes à afficher (table)", 10, 500, 50, 10)
plot_sample = st.sidebar.slider("Échantillon max pour graphiques (downsample)", 10_000, 300_000, 100_000, 10_000)
use_log_ecl = st.sidebar.checkbox("ECL en échelle log (graphiques)", value=True)

# Bouton génération
run = st.sidebar.button("Générer le portefeuille")

# =========================
# Génération
# =========================
@st.cache_data(show_spinner=False)
def build_df(cfg: dict, horizon: int, slope: float, cap: float, stage_thr: int, disc_rate):
    df = generate_loan_df(cfg, use_ecb_api=False)
    df = add_forward_pd_columns(df, base_pd_col="PD", horizon_years=horizon, term_slope=slope, cap=cap)

    pd_cols = [f"PD_FWD_{i}Y" for i in range(1, horizon + 1)]
    df = compute_ecl_s1_s2_forward_pd(
        df,
        pd_fwd_cols=pd_cols,
        stage2_chr_threshold=stage_thr,
        discount_rate=disc_rate
    )
    return df

def _downsample(df: pd.DataFrame, n_max: int, seed: int) -> pd.DataFrame:
    if len(df) <= n_max:
        return df
    return df.sample(n=n_max, random_state=seed)

def _format_int(x: float) -> str:
    try:
        return f"{x:,.0f}"
    except Exception:
        return str(x)

def _safe_log_series(s: pd.Series) -> pd.Series:
    return np.log10(1.0 + s.clip(lower=0.0))

def _migration_heatmap(df: pd.DataFrame, src_col="CHR", dst_col="CHR_new") -> pd.DataFrame:
    tab = pd.crosstab(df[src_col], df[dst_col], normalize="index")
    idx = sorted(tab.index.tolist())
    cols = sorted(tab.columns.tolist())
    return tab.reindex(index=idx, columns=cols).fillna(0.0)

if run:
    # Construire un cfg dérivé de CONFIG
    cfg = dict(CONFIG)
    cfg["segment"] = segment
    cfg["Arrete"] = arrete
    cfg["n"] = int(n)
    cfg["seed"] = int(seed)
    cfg["ecb_rate"] = float(ecb_rate)
    cfg["interest_alea_std"] = float(alea_std)
    cfg["origination_start"] = str(orig_start)
    cfg["origination_end"] = str(orig_end)
    cfg["min_tenor_years"] = int(min_tenor)
    cfg["max_tenor_years"] = int(max_tenor)
    cfg["notional_mean"] = float(notional_mean)
    cfg["notional_std"] = float(notional_std)
    cfg["notional_min"] = float(notional_min)

    with st.spinner("Génération en cours..."):
        loan_df = build_df(cfg, horizon_years, float(term_slope), float(pd_cap), int(stage2_thr), disc)

    share_s2 = loan_df["stage"].eq("S2").mean() * 100 if "stage" in loan_df.columns else np.nan
    share_s3 = loan_df["stage"].eq("S3").mean() * 100 if "stage" in loan_df.columns else np.nan

    st.success(f"Portefeuille généré : {len(loan_df):,} lignes")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ECL total", _format_int(loan_df["ECL"].sum()))
    c2.metric("ECL S1 total", _format_int(loan_df["ECL_S1"].sum()))
    c3.metric("ECL S2 total", _format_int(loan_df["ECL_S2"].sum()))
    c4.metric("Part Stage 2", f"{share_s2:.1f}%" if not np.isnan(share_s2) else "NA")
    c5.metric("Part Stage 3", f"{share_s3:.1f}%" if not np.isnan(share_s3) else "NA")

    tab1, tab2, tab3 = st.tabs(["Aperçu portefeuille", "Distributions (4 graphiques)", "Analyses (pro)"])

    with tab1:
        st.subheader("Aperçu (premières lignes)")
        st.dataframe(loan_df.head(max_rows_display), use_container_width=True)

        st.subheader("Contrôles rapides")
        cc1, cc2 = st.columns(2)

        with cc1:
            st.write("Répartition CHR")
            st.dataframe(
                loan_df["CHR"].value_counts(normalize=True).sort_index().to_frame("share"),
                use_container_width=True
            )

        with cc2:
            st.write("Répartition LGD_code")
            st.dataframe(
                loan_df["LGD_code"].value_counts(normalize=True).to_frame("share"),
                use_container_width=True
            )

        st.subheader("Exports")
        csv = loan_df.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger CSV", csv, file_name="loan_df.csv", mime="text/csv")

        try:
            import io
            buf = io.BytesIO()
            loan_df.to_parquet(buf, index=False)
            st.download_button("Télécharger Parquet", buf.getvalue(), file_name="loan_df.parquet", mime="application/octet-stream")
        except Exception:
            st.info("Export Parquet indisponible (dépendances manquantes).")

    with tab2:
        st.subheader("Distributions clés (4 graphiques)")
        plot_df = _downsample(loan_df, plot_sample, seed=int(seed)).copy()

        if use_log_ecl:
            plot_df["_ECL_S1_plot"] = _safe_log_series(plot_df["ECL_S1"])
            plot_df["_ECL_S2_plot"] = _safe_log_series(plot_df["ECL_S2"])
            ecl_s1_col, ecl_s2_col = "_ECL_S1_plot", "_ECL_S2_plot"
            ecl_s1_label, ecl_s2_label = "log10(1 + ECL_S1)", "log10(1 + ECL_S2)"
        else:
            ecl_s1_col, ecl_s2_col = "ECL_S1", "ECL_S2"
            ecl_s1_label, ecl_s2_label = "ECL_S1", "ECL_S2"

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            tmp = plot_df["CHR"].value_counts().sort_index().reset_index()
            tmp.columns = ["CHR", "Nombre"]
            fig = px.bar(tmp, x="CHR", y="Nombre", title="Distribution des CHR")
            st.plotly_chart(fig, use_container_width=True)

        with g2:
            fig = px.histogram(plot_df, x="notional", nbins=60, title="Distribution des notionnels")
            st.plotly_chart(fig, use_container_width=True)

        with g3:
            d = plot_df.loc[plot_df["stage"].eq("S1")] if "stage" in plot_df.columns else plot_df
            fig = px.histogram(d, x=ecl_s1_col, nbins=60, title="Distribution ECL Stage 1", labels={ecl_s1_col: ecl_s1_label})
            st.plotly_chart(fig, use_container_width=True)

        with g4:
            d = plot_df.loc[plot_df["stage"].eq("S2")] if "stage" in plot_df.columns else plot_df
            fig = px.histogram(d, x=ecl_s2_col, nbins=60, title="Distribution ECL Stage 2", labels={ecl_s2_col: ecl_s2_label})
            st.plotly_chart(fig, use_container_width=True)

        st.caption("Note: échantillonnage éventuel pour performance (paramètre sidebar).")

    with tab3:
        st.subheader("Analyses avancées (pro)")
        plot_df = _downsample(loan_df, plot_sample, seed=int(seed)).copy()

        st.markdown("**Boxplot ECL par CHR**")
        if use_log_ecl:
            plot_df["ECL_log"] = _safe_log_series(plot_df["ECL"])
            y_col, y_lab = "ECL_log", "log10(1 + ECL)"
        else:
            y_col, y_lab = "ECL", "ECL"

        fig = px.box(plot_df, x="CHR", y=y_col, points="outliers", title="Distribution ECL par CHR", labels={y_col: y_lab})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**ECL moyen par CHR et par stage**")
        grp = loan_df.groupby(["CHR", "stage"], dropna=False)["ECL"].mean().reset_index()
        fig = px.line(grp, x="CHR", y="ECL", color="stage", markers=True, title="ECL moyen par CHR (split par stage)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Heatmap migration CHR → CHR_new (si colonne disponible)**")
        if "CHR_new" in loan_df.columns:
            heat = _migration_heatmap(loan_df, "CHR", "CHR_new")
            fig = px.imshow(
                heat.values,
                x=heat.columns.astype(str),
                y=heat.index.astype(str),
                aspect="auto",
                title="Matrice de migration réalisée (normalisée par CHR source)",
                labels=dict(x="CHR_new", y="CHR", color="probabilité")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune colonne CHR_new détectée. Exécute d’abord la simulation de migration pour activer cette vue.")

        st.markdown("**Diagnostics (quantiles)**")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.write("Notional")
            st.dataframe(loan_df["notional"].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).to_frame("quantile"), use_container_width=True)
        with d2:
            st.write("ECL")
            st.dataframe(loan_df["ECL"].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).to_frame("quantile"), use_container_width=True)
        with d3:
            st.write("PD forward (min/max)")
            pd_cols = [c for c in loan_df.columns if c.startswith("PD_FWD_")]
            if pd_cols:
                st.dataframe(loan_df[pd_cols].agg(["min", "max"]).T.head(30), use_container_width=True)
            else:
                st.info("Pas de colonnes PD_FWD_* détectées.")

else:
    st.info("Renseigne les paramètres dans la sidebar puis clique sur « Générer le portefeuille ».")
