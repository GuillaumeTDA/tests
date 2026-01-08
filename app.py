# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 20:06:13 2026

@author: guill
"""

import streamlit as st
import pandas as pd

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
n = st.sidebar.number_input("Nombre de lignes (n)", min_value=100, max_value=2_000_000, value=int(CONFIG.get("n", 1000)), step=100)

seed = st.sidebar.number_input("Seed", min_value=0, max_value=10_000_000, value=int(CONFIG.get("seed", 42)), step=1)

st.sidebar.subheader("Taux")
ecb_rate = st.sidebar.number_input("Taux BCE (ecb_rate, %)", min_value=-5.0, max_value=15.0, value=float(CONFIG.get("ecb_rate", 3.75)), step=0.05)
alea_std = st.sidebar.number_input("Volatilité aléa taux (std, points de %)", min_value=0.0, max_value=5.0, value=float(CONFIG.get("interest_alea_std", 0.25)), step=0.05)

st.sidebar.subheader("Dates")
orig_start = st.sidebar.date_input("Origination start", value=pd.to_datetime(CONFIG.get("origination_start", "2015-01-01")))
orig_end = st.sidebar.date_input("Origination end", value=pd.to_datetime(CONFIG.get("origination_end", "2022-12-31")))

st.sidebar.subheader("Maturité")
min_tenor = st.sidebar.number_input("Tenor min (années)", min_value=0, max_value=50, value=int(CONFIG.get("min_tenor_years", 1)), step=1)
max_tenor = st.sidebar.number_input("Tenor max (années)", min_value=1, max_value=50, value=int(CONFIG.get("max_tenor_years", 10)), step=1)

st.sidebar.subheader("Notional")
notional_mean = st.sidebar.number_input("Notional mean", min_value=1_000.0, max_value=1e9, value=float(CONFIG.get("notional_mean", 1_000_000)), step=10_000.0)
notional_std = st.sidebar.number_input("Notional std", min_value=0.0, max_value=1e9, value=float(CONFIG.get("notional_std", 300_000)), step=10_000.0)
notional_min = st.sidebar.number_input("Notional min", min_value=0.0, max_value=1e9, value=float(CONFIG.get("notional_min", 10_000)), step=1_000.0)

st.sidebar.subheader("IFRS9 / ECL")
horizon_years = st.sidebar.slider("Horizon PD forward (années)", min_value=1, max_value=30, value=10, step=1)
term_slope = st.sidebar.number_input("Pente PD forward (term_slope)", min_value=-0.50, max_value=1.00, value=float(0.05), step=0.01)
pd_cap = st.sidebar.number_input("Cap PD forward (par année)", min_value=0.01, max_value=1.00, value=float(0.50), step=0.01)

stage2_thr = st.sidebar.slider("Seuil Stage 2 (CHR > seuil)", min_value=1, max_value=10, value=6, step=1)
discount_rate = st.sidebar.checkbox("Actualiser (discounting)", value=False)
disc = None
if discount_rate:
    disc = st.sidebar.number_input("Discount rate annuel", min_value=-0.10, max_value=0.30, value=0.03, step=0.005)

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

    st.success(f"Portefeuille généré : {len(loan_df):,} lignes")

    c1, c2, c3, c4 , c5 = st.columns(5)
    c1.metric("ECL total", f"{loan_df['ECL'].sum():,.0f}")
    c2.metric("ECL S1 total", f"{loan_df['ECL_S1'].sum():,.0f}")
    c3.metric("ECL S2 total", f"{loan_df['ECL_S2'].sum():,.0f}")
    c4.metric("Part Stage 2", f"{(loan_df['stage'].eq('S2').mean()*100):.1f}%")
    c5.metric("Part Stage 2", f"{(loan_df['stage'].eq('S2').mean()*100):.1f}%")

    st.subheader("Aperçu (premières lignes)")
    st.dataframe(loan_df.head(50), use_container_width=True)

    st.subheader("Contrôles rapides")
    st.write("Répartition CHR")
    st.dataframe(loan_df["CHR"].value_counts(normalize=True).sort_index().to_frame("share"), use_container_width=True)

    st.write("Répartition LGD_code")
    st.dataframe(loan_df["LGD_code"].value_counts(normalize=True).to_frame("share"), use_container_width=True)

    # Download CSV
    csv = loan_df.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger CSV", csv, file_name="loan_df.csv", mime="text/csv")

else:
    st.info("Renseigne les paramètres dans la sidebar puis clique sur « Générer le portefeuille ».")
