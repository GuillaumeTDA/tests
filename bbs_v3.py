# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 16:18:17 2026

@author: guill
"""

import numpy as np
import pandas as pd
import random
import string
import uuid


# =============================================================================
# PARAMÈTRES (centralisés)
# =============================================================================
CONFIG = {
    # Taille
    "segment": "Corporate",
    "Arrete" : "Q42000",
    "n": 1_000,
    "ecb_rate": 3.75,   # en % annuel, paramétrable
    # IDs
    "counterparty_id_length": 10,
    "counterparty_id_alphabet": string.ascii_uppercase + string.digits,
    "seed": 42,

    # Dates origination
    "origination_start": "2015-01-01",
    "origination_end": "2022-12-31",

    # Maturité
    "min_tenor_years": 1,
    "max_tenor_years": 25,

    # Notional ~ N(mean, std), tronqué
    "notional_mean": 1_000_000,
    "notional_std": 300_000,
    "notional_min": 10_000,

    # Currency
    "currencies": ["EUR", "USD", "GBP"],
    "currency_probs": [0.60, 0.25, 0.15],

    # Interest rate type
    "rate_types": ["fixed", "floating"],
    "rate_type_probs": [0.65, 0.35],

    # Day count
    "day_count_convention": "ACT/365",

    # Reset freq (floating only)
    "reset_freq_values": ["1M", "3M", "6M", "12M"],
    "reset_freq_probs": [0.10, 0.55, 0.30, 0.05],

    # Interest rate = max(0, ecb_dfr + alea)
    "interest_alea_mean": 0.0,
    "interest_alea_std": 0.25,

    # Amortization
    "amortization_types": ["linear", "annuity", "bullet", "interest_only"],
    "amortization_probs": [0.35, 0.45, 0.10, 0.10],

    # Payment frequency
    "payment_freqs": ["monthly", "quarterly", "annual"],
    "payment_freq_probs": [0.60, 0.30, 0.10],

    # Grace period (mois)
    "grace_values": [0, 3, 6, 12],
    "grace_probs": [0.60, 0.20, 0.15, 0.05],

    # Prepayment rate
    "prepay_mean": 0.03,
    "prepay_std": 0.015,
    "prepay_min": 0.0,
    "prepay_max": 0.20,

    # IFRS 9 - CHR / PD
    "chr_max": 10,
    "chr_probs": [0.05, 0.08, 0.12, 0.15, 0.15, 0.15, 0.10, 0.08, 0.07, 0.05],
    "chr_to_pd": {1: 0.001, 2: 0.002, 3: 0.004, 4: 0.007, 5: 0.012,
                  6: 0.020, 7: 0.035, 8: 0.060, 9: 0.100, 10: 1},

    # LGD
    "lgd_codes": ["LGD_1", "LGD_2", "LGD_3", "LGD_4", "LGD_5"],
    "lgd_code_probs": [0.20, 0.25, 0.25, 0.20, 0.10],
    "lgd_code_to_value": {"LGD_1": 0.15, "LGD_2": 0.25, "LGD_3": 0.40, "LGD_4": 0.55, "LGD_5": 0.70},
}


# =============================================================================
# COLONNES
# =============================================================================
LOAN_COLUMNS = [
    "loan_id", "counterparty_id",
    "origination_date", "maturity_date", "notional", "currency",
    "interest_rate_type", "interest_rate", "day_count_convention", "rate_reset_frequency",
    "amortization_type", "payment_frequency", "first_payment_date",
    "grace_period", "prepayment_rate",
    # IFRS9
    "CHR", "PD", "LGD_code", "LGD",
    # (optionnel mais utile)
    "residual_maturity_days",
]


# =============================================================================
# UTILITAIRES
# =============================================================================
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_empty_loan_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(index=np.arange(n), columns=LOAN_COLUMNS)


def generate_counterparty_ids(n: int, length: int, alphabet: str) -> np.ndarray:
    # Tirage avec remise => doublons possibles
    # random.choices est ok, mais sur 1M lignes c’est plus rapide de vectoriser via numpy
    # Ici, on fait un tirage de caractères puis on reshape.
    chars = np.random.choice(list(alphabet), size=(n, length))
    return np.array(["".join(row) for row in chars], dtype=object)


def generate_loan_ids_uuid(n: int) -> np.ndarray:
    return np.array([uuid.uuid4().hex.upper() for _ in range(n)], dtype=object)


def generate_origination_dates(n: int, start: str, end: str) -> pd.Series:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    days = (end_dt - start_dt).days
    random_days = np.random.randint(0, days + 1, size=n)
    return start_dt + pd.to_timedelta(random_days, unit="D")


def generate_maturity_dates(origination: pd.Series, min_years: int, max_years: int) -> pd.Series:
    min_days = int(min_years * 365)
    max_days = int(max_years * 365)
    tenor_days = np.random.randint(min_days, max_days + 1, size=len(origination))
    maturity = origination + pd.to_timedelta(tenor_days, unit="D")
    # garantie strictement supérieur
    maturity = maturity.where(maturity > origination, origination + pd.to_timedelta(1, unit="D"))
    return maturity


def add_residual_maturity_days(df: pd.DataFrame) -> None:
    df["residual_maturity_days"] = (df["maturity_date"] - df["origination_date"]).dt.days


def generate_notional(n: int, mean: float, std: float, min_amount: float) -> np.ndarray:
    x = np.random.normal(loc=mean, scale=std, size=n)
    return np.maximum(x, min_amount)


def generate_choice(n: int, values: list, probs: list) -> np.ndarray:
    return np.random.choice(values, size=n, p=probs)


def generate_first_payment_date(origination: pd.Series, payment_frequency: pd.Series) -> pd.Series:
    # Vectorisé (évite apply)
    months_map = {"monthly": 1, "quarterly": 3, "annual": 12}
    months = payment_frequency.map(months_map).astype(int)
    return origination + pd.to_timedelta(months * 30, unit="D")  # approx rapide
    # Si tu veux exact “calendar month”, on peut faire une version plus précise mais plus coûteuse.


def assign_reset_frequency(df: pd.DataFrame, values: list, probs: list) -> None:
    mask_float = df["interest_rate_type"].eq("floating")
    df.loc[mask_float, "rate_reset_frequency"] = np.random.choice(values, size=int(mask_float.sum()), p=probs)
    df.loc[~mask_float, "rate_reset_frequency"] = pd.NA


def generate_interest_rate(df: pd.DataFrame, base_col: str, alea_mean: float, alea_std: float) -> None:
    alea = np.random.normal(loc=alea_mean, scale=alea_std, size=len(df))
    df["interest_rate"] = np.maximum(0.0, df[base_col].astype(float) + alea)


def assign_ifrs9_chr_pd(df: pd.DataFrame, chr_max: int, chr_probs: list, chr_to_pd: dict) -> None:
    chr_values = list(range(1, chr_max + 1))
    df["CHR"] = np.random.choice(chr_values, size=len(df), p=chr_probs)
    df["PD"] = df["CHR"].map(chr_to_pd).astype(float)


def assign_lgd(df: pd.DataFrame, lgd_codes: list, lgd_probs: list, lgd_map: dict) -> None:
    df["LGD_code"] = np.random.choice(lgd_codes, size=len(df), p=lgd_probs)
    df["LGD"] = df["LGD_code"].map(lgd_map).astype(float)


# =============================================================================
# (Optionnel) ECB DFR — nécessite internet
# =============================================================================
def add_ecb_dfr_from_api(df: pd.DataFrame,
                         ecb_flow: str = "FM",
                         ecb_series_key: str = "D.U2.EUR.4F.KR.DFR.LEV",
                         start: str = "1999-01-01",
                         end: str = "2030-12-31") -> pd.DataFrame:
    url = (
        f"https://data-api.ecb.europa.eu/service/data/{ecb_flow}/{ecb_series_key}"
        f"?startPeriod={start}&endPeriod={end}&format=csvdata"
    )
    rates = pd.read_csv(url)
    rates = rates.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "ecb_dfr"})
    rates["date"] = pd.to_datetime(rates["date"])
    rates["ecb_dfr"] = pd.to_numeric(rates["ecb_dfr"], errors="coerce")
    rates = rates.sort_values("date")[["date", "ecb_dfr"]].dropna()

    out = df.sort_values("origination_date").copy()
    out = pd.merge_asof(out, rates, left_on="origination_date", right_on="date", direction="backward").drop(columns=["date"])
    return out


def assign_ecb_rate(df: pd.DataFrame, ecb_rate: float) -> None:
    """
    Assigne un taux BCE constant à tous les loans.
    """
    df["ecb_dfr"] = float(ecb_rate)

# =============================================================================
# GÉNÉRATEUR PRINCIPAL
# =============================================================================
def generate_loan_df(cfg: dict, use_ecb_api: bool = False) -> pd.DataFrame:
    set_seeds(cfg["seed"])
    n = cfg["n"]

    df = make_empty_loan_df(n)

    # IDs
    df["counterparty_id"] = generate_counterparty_ids(n, cfg["counterparty_id_length"], cfg["counterparty_id_alphabet"])
    df["loan_id"] = generate_loan_ids_uuid(n)

    # Dates
    df["origination_date"] = generate_origination_dates(n, cfg["origination_start"], cfg["origination_end"])
    df["maturity_date"] = generate_maturity_dates(df["origination_date"], cfg["min_tenor_years"], cfg["max_tenor_years"])
    add_residual_maturity_days(df)

    # Notional / currency
    df["notional"] = generate_notional(n, cfg["notional_mean"], cfg["notional_std"], cfg["notional_min"])
    df["currency"] = generate_choice(n, cfg["currencies"], cfg["currency_probs"])

    # Taux
    df["interest_rate_type"] = generate_choice(n, cfg["rate_types"], cfg["rate_type_probs"])
    df["day_count_convention"] = cfg["day_count_convention"]
    assign_reset_frequency(df, cfg["reset_freq_values"], cfg["reset_freq_probs"])

    # # ECB DFR
    # if use_ecb_api:
    #     df = add_ecb_dfr_from_api(df)
    # else:
    #     # fallback si pas d’API : base neutre à 0 (tu peux remplacer par un proxy)
    #     df["ecb_dfr"] = 0.0
    assign_ecb_rate(df, cfg["ecb_rate"])
    generate_interest_rate(df, base_col="ecb_dfr", alea_mean=cfg["interest_alea_mean"], alea_std=cfg["interest_alea_std"])

    # Amortization / fréquence / 1ère échéance
    df["amortization_type"] = generate_choice(n, cfg["amortization_types"], cfg["amortization_probs"])
    df["payment_frequency"] = generate_choice(n, cfg["payment_freqs"], cfg["payment_freq_probs"])
    df["first_payment_date"] = generate_first_payment_date(df["origination_date"], df["payment_frequency"])

    # Options
    df["grace_period"] = generate_choice(n, cfg["grace_values"], cfg["grace_probs"])
    prepay = np.random.normal(cfg["prepay_mean"], cfg["prepay_std"], size=n)
    df["prepayment_rate"] = np.clip(prepay, cfg["prepay_min"], cfg["prepay_max"])

    # IFRS9
    assign_ifrs9_chr_pd(df, cfg["chr_max"], cfg["chr_probs"], cfg["chr_to_pd"])
    assign_lgd(df, cfg["lgd_codes"], cfg["lgd_code_probs"], cfg["lgd_code_to_value"])

    df["Asset_Class"] = cfg["segment"]
    df["Date_arrete"] = cfg["Arrete"]
    return df


# =============================================================================
# EXÉCUTION
# =============================================================================
loan_df = generate_loan_df(CONFIG, use_ecb_api=False)
loan_df.head()




def add_forward_pd_columns(
    loan_df,
    base_pd_col="PD",
    horizon_years=10,
    term_slope=0.05,      # +5% par an (multiplie la PD forward)
    cap=0.50,             # borne max par période
    prefix="PD_FWD_"
):
    df = loan_df.copy()
    base_pd = df[base_pd_col].astype(float).to_numpy()  # PD annuelle de base

    for t in range(1, horizon_years + 1):
        # PD forward année t = PD_base * (1 + slope)^(t-1)
        pd_t = base_pd * ((1.0 + term_slope) ** (t - 1))
        pd_t = np.clip(pd_t, 0.0, cap)
        df[f"{prefix}{t}Y"] = pd_t

    return df

loan_df = add_forward_pd_columns(
    loan_df,
    base_pd_col="PD",
    horizon_years=10,
    term_slope=0.05,
    cap=0.50
)

def compute_ecl_s1_s2_forward_pd(
    loan_df: pd.DataFrame,
    pd_fwd_cols: list[str],
    stage2_chr_threshold: int = 6,
    ead_col: str = "notional",
    lgd_col: str = "LGD",
    discount_rate: float | None = None,   # ex: 0.03 pour 3% ; None => pas d’actualisation
    out_ecl_col: str = "ECL"
) -> pd.DataFrame:
    """
    Calcule ECL S1 (12m) et ECL S2 (lifetime) à partir de PD forward (courbe).
    
    Requis dans loan_df :
      - CHR, ead_col, lgd_col
      - colonnes PD forward listées dans pd_fwd_cols (ex: ["PD_FWD_1Y","PD_FWD_2Y",...])
    PD forward = probas conditionnelles par période (année).
    """

    df = loan_df.copy()

    # Stage (simple règle paramétrable)
    df["stage"] = np.where(df["CHR"] > stage2_chr_threshold, "S2", "S1")

    # Matrices
    pd_fwd = df[pd_fwd_cols].astype(float).to_numpy()  # shape (n, T)
    n, T = pd_fwd.shape

    # Survie au début de chaque période: S_{t-1} = Π_{k< t} (1 - q_k)
    # q_t = PD forward période t
    survival = np.ones((n, T))
    if T > 1:
        survival[:, 1:] = np.cumprod(1.0 - pd_fwd[:, :-1], axis=1)

    # Facteurs d’actualisation (optionnels)
    if discount_rate is None:
        dfactors = np.ones((1, T))
    else:
        years = np.arange(1, T + 1, dtype=float)
        dfactors = 1.0 / np.power(1.0 + float(discount_rate), years)
        dfactors = dfactors.reshape(1, T)

    # ECL par période = EAD * LGD * S_{t-1} * q_t * DF_t
    ead = df[ead_col].astype(float).to_numpy().reshape(-1, 1)
    lgd = df[lgd_col].astype(float).to_numpy().reshape(-1, 1)

    ecl_terms = ead * lgd * survival * pd_fwd * dfactors

    # Stage 1 = 12m => première période uniquement
    df["ECL_S1"] = ecl_terms[:, 0]

    # Stage 2 = lifetime => somme sur toutes les périodes disponibles
    df["ECL_S2"] = ecl_terms.sum(axis=1)

    # ECL final selon stage
    df[out_ecl_col] = np.where(df["stage"] == "S1", df["ECL_S1"], df["ECL_S2"])

    return df

pd_cols = [f"PD_FWD_{i}Y" for i in range(1, 11)]  # 10 ans forward

loan_df = compute_ecl_s1_s2_forward_pd(
    loan_df,
    pd_fwd_cols=pd_cols,
    stage2_chr_threshold=6,
    discount_rate=None  # ou 0.03
)
