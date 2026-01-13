# app.py - OpenAI Usage Analyzer (Tier selector + cached input + robust datetime + model normalization)
#
# Run:
#   pip install streamlit pandas numpy
#   streamlit run app.py

import json
import re
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------
# Page
# -----------------------
st.set_page_config(page_title="OpenAI Usage Analyzer", page_icon="💸", layout="wide")
st.title("💸 OpenAI Usage Analyzer")
st.caption(
    "Sube tus CSVs de **Usage** de OpenAI. Obtén costos por mes y en dónde se usaron los tokens. "
    "Suma **cuotas fijas** como ChatGPT Plus a tus reportes."
)

# -----------------------
# Persistence files
# -----------------------
PRICING_FILE = Path("pricing_config.json")
FIXED_FEES_FILE = Path("fixed_fees.json")  # [{"description","amount_usd","start_month","end_month"}]

MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
DATE_SUFFIX_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")  # e.g. o3-2025-04-16
CHATGPT_PREFIX_RE = re.compile(r"^chatgpt-")

# -----------------------
# Default pricing by Tier (TEXT TOKENS) - per 1M tokens
# Structure: model -> (input_per_1M, cached_input_per_1M, output_per_1M)
# NOTE: Keep this limited to TEXT token pricing. If your CSV is for audio/image, costs may not align.
# -----------------------
DEFAULT_PRICING_BY_TIER = {
    "Batch": {
        "gpt-5.2": (0.875, 0.0875, 7.00),
        "gpt-5.1": (0.625, 0.0625, 5.00),
        "gpt-5": (0.625, 0.0625, 5.00),
        "gpt-5-mini": (0.125, 0.0125, 1.00),
        "gpt-5-nano": (0.025, 0.0025, 0.20),
        "gpt-5.2-pro": (10.50, np.nan, 84.00),
        "gpt-5-pro": (7.50, np.nan, 60.00),
        "gpt-4.1": (1.00, np.nan, 4.00),
        "gpt-4.1-mini": (0.20, np.nan, 0.80),
        "gpt-4.1-nano": (0.05, np.nan, 0.20),
        "gpt-4o": (1.25, np.nan, 5.00),
        "gpt-4o-2024-05-13": (2.50, np.nan, 7.50),
        "gpt-4o-mini": (0.075, np.nan, 0.30),
        "o1": (7.50, np.nan, 30.00),
        "o1-pro": (75.00, np.nan, 300.00),
        "o3-pro": (10.00, np.nan, 40.00),
        "o3": (1.00, np.nan, 4.00),
        "o3-deep-research": (5.00, np.nan, 20.00),
        "o4-mini": (0.55, np.nan, 2.20),
        "o4-mini-deep-research": (1.00, np.nan, 4.00),
        "o3-mini": (0.55, np.nan, 2.20),
        "o1-mini": (0.55, np.nan, 2.20),
        "computer-use-preview": (1.50, np.nan, 6.00),
        # Legacy (Batch table includes these; keep if you see them in exports)
        "gpt-4-turbo-2024-04-09": (5.00, np.nan, 15.00),
        "gpt-3.5-turbo": (0.25, np.nan, 0.75),  # gpt-3.5-turbo-0125 shown; close enough if export says generic
    },
    "Flex": {
        "gpt-5.2": (0.875, 0.0875, 7.00),
        "gpt-5.1": (0.625, 0.0625, 5.00),
        "gpt-5": (0.625, 0.0625, 5.00),
        "gpt-5-mini": (0.125, 0.0125, 1.00),
        "gpt-5-nano": (0.025, 0.0025, 0.20),
        "o3": (1.00, 0.25, 4.00),
        "o4-mini": (0.55, 0.138, 2.20),
    },
    "Standard": {
        "gpt-5.2": (1.75, 0.175, 14.00),
        "gpt-5.1": (1.25, 0.125, 10.00),
        "gpt-5": (1.25, 0.125, 10.00),
        "gpt-5-mini": (0.25, 0.025, 2.00),
        "gpt-5-nano": (0.05, 0.005, 0.40),

        "gpt-5.2-chat-latest": (1.75, 0.175, 14.00),
        "gpt-5.1-chat-latest": (1.25, 0.125, 10.00),
        "gpt-5-chat-latest": (1.25, 0.125, 10.00),

        "gpt-5.1-codex-max": (1.25, 0.125, 10.00),
        "gpt-5.1-codex": (1.25, 0.125, 10.00),
        "gpt-5-codex": (1.25, 0.125, 10.00),

        "gpt-5.2-pro": (21.00, np.nan, 168.00),
        "gpt-5-pro": (15.00, np.nan, 120.00),

        "gpt-4.1": (2.00, 0.50, 8.00),
        "gpt-4.1-mini": (0.40, 0.10, 1.60),
        "gpt-4.1-nano": (0.10, 0.025, 0.40),

        "gpt-4o": (2.50, 1.25, 10.00),
        "gpt-4o-2024-05-13": (5.00, np.nan, 15.00),
        "gpt-4o-mini": (0.15, 0.075, 0.60),

        "gpt-realtime": (4.00, 0.40, 16.00),
        "gpt-realtime-mini": (0.60, 0.06, 2.40),
        "gpt-4o-realtime-preview": (5.00, 2.50, 20.00),
        "gpt-4o-mini-realtime-preview": (0.60, 0.30, 2.40),

        "gpt-audio": (2.50, np.nan, 10.00),
        "gpt-audio-mini": (0.60, np.nan, 2.40),
        "gpt-4o-audio-preview": (2.50, np.nan, 10.00),
        "gpt-4o-mini-audio-preview": (0.15, np.nan, 0.60),

        "o1": (15.00, 7.50, 60.00),
        "o1-pro": (150.00, np.nan, 600.00),
        "o3-pro": (20.00, np.nan, 80.00),
        "o3": (2.00, 0.50, 8.00),
        "o3-deep-research": (10.00, 2.50, 40.00),
        "o4-mini": (1.10, 0.275, 4.40),
        "o4-mini-deep-research": (2.00, 0.50, 8.00),
        "o3-mini": (1.10, 0.55, 4.40),
        "o1-mini": (1.10, 0.55, 4.40),

        "gpt-5.1-codex-mini": (0.25, 0.025, 2.00),
        "codex-mini-latest": (1.50, 0.375, 6.00),

        "gpt-5-search-api": (1.25, 0.125, 10.00),
        "gpt-4o-mini-search-preview": (0.15, np.nan, 0.60),
        "gpt-4o-search-preview": (2.50, np.nan, 10.00),

        "computer-use-preview": (3.00, np.nan, 12.00),

        # Legacy/commonly exported
        "chatgpt-4o-latest": (5.00, np.nan, 15.00),
        "gpt-4-turbo-2024-04-09": (10.00, np.nan, 30.00),
        "gpt-3.5-turbo": (0.50, np.nan, 1.50),
    },
    "Priority": {
        "gpt-5.2": (3.50, 0.35, 28.00),
        "gpt-5.1": (2.50, 0.25, 20.00),
        "gpt-5": (2.50, 0.25, 20.00),
        "gpt-5-mini": (0.45, 0.045, 3.60),

        "gpt-5.1-codex-max": (2.50, 0.25, 20.00),
        "gpt-5.1-codex": (2.50, 0.25, 20.00),
        "gpt-5-codex": (2.50, 0.25, 20.00),

        "gpt-4.1": (3.50, 0.875, 14.00),
        "gpt-4.1-mini": (0.70, 0.175, 2.80),
        "gpt-4.1-nano": (0.20, 0.05, 0.80),

        "gpt-4o": (4.25, 2.125, 17.00),
        "gpt-4o-2024-05-13": (8.75, np.nan, 26.25),
        "gpt-4o-mini": (0.25, 0.125, 1.00),

        "o3": (3.50, 0.875, 14.00),
        "o4-mini": (2.00, 0.50, 8.00),
    },
}

DEFAULT_TIER = "Standard"

# -----------------------
# Pricing helpers (backward compatible)
# pricing_config.json supports either:
#   model -> [input, output]  (old)
#   model -> [input, cached_input, output] (new)
# -----------------------
def load_pricing_config(fallback_dict):
    if PRICING_FILE.exists():
        try:
            data = json.loads(PRICING_FILE.read_text(encoding="utf-8"))
            fixed = {}
            for k, v in data.items():
                if isinstance(v, (list, tuple)) and len(v) == 3:
                    fixed[k] = (float(v[0]), float(v[1]) if v[1] is not None else np.nan, float(v[2]))
                elif isinstance(v, (list, tuple)) and len(v) == 2:
                    fixed[k] = (float(v[0]), float(v[0]), float(v[1]))
            if fixed:
                return fixed
        except Exception:
            st.warning("No se pudo leer 'pricing_config.json'. Se usarán precios predeterminados del Tier.")
    return dict(fallback_dict)


def save_pricing_config(d):
    out = {k: [float(v[0]), (None if pd.isna(v[1]) else float(v[1])), float(v[2])] for k, v in d.items()}
    try:
        PRICING_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception as e:
        st.error(f"No se pudo guardar 'pricing_config.json': {e}")
        return False


def pricing_dict_to_df(pr_dict):
    rows = []
    for model, (inp, cached, out) in pr_dict.items():
        rows.append(
            {
                "model": model,
                "input_per_1M": float(inp),
                "cached_input_per_1M": (np.nan if pd.isna(cached) else float(cached)),
                "output_per_1M": float(out),
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def df_to_pricing_dict(df):
    pr = {}
    for _, r in df.iterrows():
        m = str(r.get("model", "")).strip()
        if not m:
            continue
        try:
            i = float(r.get("input_per_1M", np.nan))
            c = r.get("cached_input_per_1M", np.nan)
            c = np.nan if pd.isna(c) else float(c)
            o = float(r.get("output_per_1M", np.nan))
        except Exception:
            continue
        if np.isnan(i) or np.isnan(o):
            continue
        pr[m] = (i, c, o)
    return pr


# -----------------------
# Fixed fees helpers
# -----------------------
def load_fixed_fees():
    if FIXED_FEES_FILE.exists():
        try:
            data = json.loads(FIXED_FEES_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                fixed = []
                for item in data:
                    desc = str(item.get("description", "")).strip()
                    amt = float(item.get("amount_usd", 0.0))
                    sm = str(item.get("start_month", "")).strip()
                    em = str(item.get("end_month", "")).strip()
                    fixed.append({"description": desc, "amount_usd": amt, "start_month": sm, "end_month": em})
                return fixed
        except Exception:
            st.warning("No se pudo leer 'fixed_fees.json'. Se ignorarán cuotas fijas.")
    return []


def save_fixed_fees(fees_list):
    try:
        FIXED_FEES_FILE.write_text(json.dumps(fees_list, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception as e:
        st.error(f"No se pudo guardar 'fixed_fees.json': {e}")
        return False


def fixed_fees_to_df(fees):
    return pd.DataFrame(fees, columns=["description", "amount_usd", "start_month", "end_month"])


def df_to_fixed_fees(df):
    fees = []
    for _, r in df.iterrows():
        desc = str(r.get("description", "")).strip()
        amt = r.get("amount_usd", np.nan)
        sm = str(r.get("start_month", "")).strip()
        em = str(r.get("end_month", "")).strip()
        if not desc or pd.isna(amt):
            continue
        try:
            amt = float(amt)
        except Exception:
            continue
        fees.append({"description": desc, "amount_usd": amt, "start_month": sm, "end_month": em})
    return fees


def month_to_period(m: str):
    if not m:
        return None
    m = str(m).strip()
    if not MONTH_RE.match(m):
        return None
    try:
        return pd.Period(m, freq="M")
    except Exception:
        return None


def month_in_range_period(month_period, start_month, end_month):
    if month_period is None:
        return False
    sp = month_to_period(start_month) if start_month else None
    ep = month_to_period(end_month) if end_month else None
    if sp is not None and month_period < sp:
        return False
    if ep is not None and month_period > ep:
        return False
    return True


def validate_fee_months(fees):
    msgs = []
    for i, fee in enumerate(fees, start=1):
        sm = (fee.get("start_month") or "").strip()
        em = (fee.get("end_month") or "").strip()
        if sm and not MONTH_RE.match(sm):
            msgs.append(f"Fila {i}: start_month inválido ('{sm}'). Usa formato YYYY-MM.")
        if em and not MONTH_RE.match(em):
            msgs.append(f"Fila {i}: end_month inválido ('{em}'). Usa formato YYYY-MM o vacío.")
        if sm and em and MONTH_RE.match(sm) and MONTH_RE.match(em):
            sp = month_to_period(sm)
            ep = month_to_period(em)
            if sp and ep and ep < sp:
                msgs.append(f"Fila {i}: end_month ('{em}') es menor que start_month ('{sm}').")
    return msgs


# -----------------------
# CSV normalization
# -----------------------
def guess_col(cols, candidates):
    lcols = [c.lower() for c in cols]
    for name in candidates:
        if name in lcols:
            return cols[lcols.index(name)]
    for c in cols:
        cl = c.lower()
        if any(name in cl for name in candidates):
            return c
    return None


def normalize_columns(df):
    cols = list(df.columns)
    mapping = {}

    # Prefer ISO columns first (prevents epoch mis-parse)
    mapping["date"] = guess_col(cols, ["start_time_iso", "usage_start_time_iso", "created_at", "timestamp_iso", "date_iso"])
    if mapping["date"] is None:
        mapping["date"] = guess_col(cols, ["date", "day", "timestamp", "created", "usage_date", "start_time", "usage_start_time", "time"])

    mapping["model"] = guess_col(cols, ["model", "gpt_model", "engine"])
    mapping["endpoint"] = guess_col(cols, ["endpoint", "operation", "api_endpoint", "request_type"])
    mapping["project"] = guess_col(cols, ["project", "project_name", "project id", "project_id"])
    mapping["user"] = guess_col(cols, ["user", "user_email", "email", "actor", "owner"])
    mapping["org"] = guess_col(cols, ["org", "organization", "organization_id", "organization name"])
    mapping["api_key_id"] = guess_col(cols, ["api_key_id", "api key id", "key_id", "key id", "api_key", "api key"])

    mapping["input"] = guess_col(cols, ["prompt tokens", "input tokens", "input_tokens", "prompt_tokens", "tokens_in"])
    mapping["cached_input"] = guess_col(cols, ["cached input", "cached_input", "cached_input_tokens", "input_cached_tokens"])
    mapping["output"] = guess_col(cols, ["completion tokens", "output tokens", "output_tokens", "tokens_out"])
    mapping["total"] = guess_col(cols, ["total tokens", "total_tokens", "tokens_total"])
    mapping["cost"] = guess_col(cols, ["cost", "usd", "amount_usd", "cost_usd", "price_usd"])

    nd = pd.DataFrame(index=df.index)

    # Parse date robustly
    if mapping["date"] is not None:
        s = df[mapping["date"]]
        if pd.api.types.is_numeric_dtype(s):
            nd["date"] = pd.to_datetime(s, errors="coerce", unit="s", utc=True).dt.tz_convert(None)
        else:
            nd["date"] = pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None)
    else:
        nd["date"] = pd.NaT

    def safe_text(colname):
        if mapping[colname] is not None:
            ser = df[mapping[colname]]
        else:
            ser = pd.Series(pd.NA, index=df.index, dtype="string")
        ser = (
            ser.astype("string")
            .replace({"None": pd.NA, "nan": pd.NA, "NaN": pd.NA, "null": pd.NA, "NULL": pd.NA, "": pd.NA})
            .str.strip()
        )
        return ser

    for c in ["model", "endpoint", "project", "user", "org", "api_key_id"]:
        nd[c] = safe_text(c)

    for k in ["input", "cached_input", "output", "total"]:
        if mapping.get(k) is not None:
            nd[k] = pd.to_numeric(df[mapping[k]], errors="coerce")
        else:
            nd[k] = np.nan

    if mapping["cost"] is not None:
        nd["cost"] = pd.to_numeric(df[mapping["cost"]], errors="coerce")
    else:
        nd["cost"] = np.nan

    if nd["total"].isna().all():
        nd["total"] = nd[["input", "output"]].sum(axis=1, skipna=True)

    nd["month"] = nd["date"].dt.to_period("M").astype(str)
    nd.loc[nd["date"].isna(), "month"] = "Unknown"

    return nd, mapping


# -----------------------
# Model normalization & costing (vectorized)
# -----------------------
def normalize_model_name(model_series: pd.Series) -> pd.Series:
    s = model_series.astype("string").fillna("")
    # Strip suffix date and keep pricing keys consistent
    s = s.str.replace(DATE_SUFFIX_RE, "", regex=True)

    # Keep chatgpt-* as-is for pricing (since pricing table includes chatgpt-4o-latest)
    # But also support cases where you might have stripped it elsewhere.
    # Here we only do an alias if the model comes as "4o-latest".
    stripped = s.str.replace(CHATGPT_PREFIX_RE, "", regex=True)
    s = s.where(stripped != "4o-latest", "chatgpt-4o-latest")

    return s


def build_rate_maps(pricing_dict):
    in_map, cached_map, out_map = {}, {}, {}
    for m, (i, c, o) in pricing_dict.items():
        try:
            in_map[str(m)] = float(i)
            out_map[str(m)] = float(o)
            cached_map[str(m)] = (np.nan if pd.isna(c) else float(c))
        except Exception:
            continue
    return in_map, cached_map, out_map


def add_estimated_cost(df, pricing_dict):
    if "cost" not in df.columns:
        df["cost"] = np.nan

    in_map, cached_map, out_map = build_rate_maps(pricing_dict)

    model_norm = normalize_model_name(df["model"])
    in_rate = model_norm.map(in_map)
    out_rate = model_norm.map(out_map)
    cached_rate = model_norm.map(cached_map)

    # If cached_rate missing but input exists, fall back to input_rate (conservative)
    cached_rate = cached_rate.where(~cached_rate.isna(), in_rate)

    tokens_in = pd.to_numeric(df["input"], errors="coerce").fillna(0.0)
    tokens_cached = pd.to_numeric(df.get("cached_input", np.nan), errors="coerce").fillna(0.0)
    tokens_out = pd.to_numeric(df["output"], errors="coerce").fillna(0.0)

    cost_est = (
        (tokens_in / 1_000_000.0) * in_rate
        + (tokens_cached / 1_000_000.0) * cached_rate
        + (tokens_out / 1_000_000.0) * out_rate
    )
    df["cost_estimated"] = cost_est

    df["cost_variable"] = df["cost"]
    df.loc[df["cost_variable"].isna(), "cost_variable"] = df.loc[df["cost_variable"].isna(), "cost_estimated"]
    df["cost_final"] = df["cost_variable"]

    return df


# -----------------------
# Summaries
# -----------------------
def summarize(df):
    monthly = df.groupby("month", dropna=False)["cost_variable"].sum().reset_index().sort_values("month")

    by_model = (
        df.groupby("model", dropna=False)
        .agg(
            cost=("cost_variable", "sum"),
            input_tokens=("input", "sum"),
            cached_input_tokens=("cached_input", "sum"),
            output_tokens=("output", "sum"),
            total_tokens=("total", "sum"),
            rows=("model", "count"),
        )
        .reset_index()
        .sort_values("cost", ascending=False)
    )

    by_endpoint = (
        df.groupby("endpoint", dropna=False)
        .agg(cost=("cost_variable", "sum"), total_tokens=("total", "sum"), rows=("endpoint", "count"))
        .reset_index()
        .sort_values("cost", ascending=False)
    )

    by_project = (
        df.groupby("project", dropna=False)
        .agg(cost=("cost_variable", "sum"), total_tokens=("total", "sum"), rows=("project", "count"))
        .reset_index()
        .sort_values("cost", ascending=False)
    )

    by_user = (
        df.groupby("user", dropna=False)
        .agg(cost=("cost_variable", "sum"), total_tokens=("total", "sum"), rows=("user", "count"))
        .reset_index()
        .sort_values("cost", ascending=False)
    )

    by_key = (
        df.groupby("api_key_id", dropna=False)
        .agg(cost=("cost_variable", "sum"), total_tokens=("total", "sum"), rows=("api_key_id", "count"))
        .reset_index()
        .sort_values("cost", ascending=False)
    )

    daily = (
        df.loc[~df["date"].isna()]
        .groupby(df.loc[~df["date"].isna(), "date"].dt.date)["cost_variable"]
        .sum()
        .reset_index()
        .rename(columns={"date": "day"})
        .sort_values("day")
    )

    return monthly, by_model, by_endpoint, by_project, by_user, by_key, daily


def format_money(x):
    try:
        if pd.isna(x):
            return ""
        return f"${x:,.2f}"
    except Exception:
        return str(x)


def parse_comma_list(s):
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown(
        "1) **Sube tus CSVs** de Usage.\n"
        "2) Selecciona **Tier**.\n"
        "3) Ajusta filtros.\n"
        "4) Edita y guarda tarifas (incluye cached input).\n"
        "5) Agrega cuotas fijas mensuales."
    )

    files = st.file_uploader("CSV(s) de Usage", type=["csv"], accept_multiple_files=True)

    st.markdown("---")
    st.subheader("Tier (Service Tier)")

    if "selected_tier" not in st.session_state:
        st.session_state.selected_tier = DEFAULT_TIER

    selected_tier = st.selectbox(
        "Selecciona el Tier para precios predeterminados",
        options=list(DEFAULT_PRICING_BY_TIER.keys()),
        index=list(DEFAULT_PRICING_BY_TIER.keys()).index(st.session_state.selected_tier),
        key="tier_selector",
    )
    st.session_state.selected_tier = selected_tier
    tier_defaults = DEFAULT_PRICING_BY_TIER[selected_tier]

    st.caption(
        "Importante: el Tier sólo afecta los **predeterminados**. "
        "El cálculo usa lo que tengas cargado en tu tabla de precios (pricing_config)."
    )

    st.markdown("---")
    st.subheader("Tarifas por modelo (persistentes)")

    if "pricing_dict" not in st.session_state:
        st.session_state.pricing_dict = load_pricing_config(tier_defaults)

    cols_tier_btn = st.columns(2)
    with cols_tier_btn[0]:
        if st.button("⬇️ Aplicar predeterminados del Tier"):
            st.session_state.pricing_dict = dict(tier_defaults)
            st.success(f"Se cargaron predeterminados del Tier: {selected_tier}. (No guardado aún)")
    with cols_tier_btn[1]:
        if st.button("↩️ Recargar desde pricing_config.json"):
            st.session_state.pricing_dict = load_pricing_config(tier_defaults)
            st.success("Tarifas recargadas desde archivo (o predeterminadas del Tier si no existe).")

    pr_df = pricing_dict_to_df(st.session_state.pricing_dict)
    edited_pr = st.data_editor(
        pr_df,
        num_rows="dynamic",
        use_container_width=True,
        key="pricing_editor",
        column_config={
            "model": st.column_config.TextColumn("Modelo"),
            "input_per_1M": st.column_config.NumberColumn("Input / 1M", min_value=0.0, step=0.01),
            "cached_input_per_1M": st.column_config.NumberColumn("Cached input / 1M", min_value=0.0, step=0.01),
            "output_per_1M": st.column_config.NumberColumn("Output / 1M", min_value=0.0, step=0.01),
        },
    )

    cols_btn = st.columns(3)
    with cols_btn[0]:
        if st.button("💾 Guardar tarifas"):
            new_dict = df_to_pricing_dict(edited_pr)
            if not new_dict:
                st.error("No hay filas válidas para guardar.")
            else:
                if save_pricing_config(new_dict):
                    st.session_state.pricing_dict = new_dict
                    st.success("Tarifas guardadas en 'pricing_config.json'.")
    with cols_btn[1]:
        if st.button("🧹 Restaurar predeterminadas (Tier)"):
            st.session_state.pricing_dict = dict(tier_defaults)
            st.success(f"Se restauraron predeterminadas del Tier: {selected_tier}. (No guardado aún)")
    with cols_btn[2]:
        export_json_pr = json.dumps(
            {k: [float(v[0]), (None if pd.isna(v[1]) else float(v[1])), float(v[2])] for k, v in st.session_state.pricing_dict.items()},
            indent=2,
            ensure_ascii=False,
        )
        st.download_button("⬇️ Exportar precios (JSON)", export_json_pr.encode("utf-8"), file_name="pricing_config.json", mime="application/json")

    uploaded_json_pr = st.file_uploader("⬆️ Importar precios (JSON)", type=["json"], key="pricing_importer")
    if uploaded_json_pr is not None:
        try:
            imp = json.loads(uploaded_json_pr.read().decode("utf-8"))
            fixed = {}
            for k, v in imp.items():
                if isinstance(v, (list, tuple)) and len(v) == 3:
                    fixed[k] = (float(v[0]), float(v[1]) if v[1] is not None else np.nan, float(v[2]))
                elif isinstance(v, (list, tuple)) and len(v) == 2:
                    fixed[k] = (float(v[0]), float(v[0]), float(v[1]))
            st.session_state.pricing_dict = fixed
            st.success("JSON importado. Presiona **Guardar tarifas** si deseas persistirlo.")
        except Exception as e:
            st.error(f"No se pudo importar el JSON: {e}")

    st.markdown("---")
    st.subheader("Cuotas fijas mensuales (persistentes)")
    if "fixed_fees" not in st.session_state:
        st.session_state.fixed_fees = load_fixed_fees()

    ff_df = fixed_fees_to_df(st.session_state.fixed_fees)
    edited_ff = st.data_editor(
        ff_df,
        num_rows="dynamic",
        use_container_width=True,
        key="fixed_fees_editor",
        column_config={
            "description": st.column_config.TextColumn("Descripción"),
            "amount_usd": st.column_config.NumberColumn("Monto USD (mensual)", min_value=0.0, step=0.50),
            "start_month": st.column_config.TextColumn("Inicio (YYYY-MM)"),
            "end_month": st.column_config.TextColumn("Fin (YYYY-MM, opcional)"),
        },
    )

    cols_ff = st.columns(3)
    with cols_ff[0]:
        if st.button("💾 Guardar cuotas fijas"):
            new_fees = df_to_fixed_fees(edited_ff)
            errors = validate_fee_months(new_fees)
            if errors:
                for msg in errors:
                    st.error(msg)
            else:
                if save_fixed_fees(new_fees):
                    st.session_state.fixed_fees = new_fees
                    st.success("Cuotas fijas guardadas en 'fixed_fees.json'.")
    with cols_ff[1]:
        if st.button("↩️ Cargar cuotas guardadas"):
            st.session_state.fixed_fees = load_fixed_fees()
            st.success("Cuotas fijas recargadas desde 'fixed_fees.json'.")
    with cols_ff[2]:
        if st.button("🧹 Limpiar cuotas (no guarda)"):
            st.session_state.fixed_fees = []
            st.info("Lista limpiada (no se guardó).")

    st.markdown("---")
    st.subheader("Filtros")
    use_date_filter = st.checkbox("Activar filtro por fechas", value=False)
    if use_date_filter:
        date_min = st.date_input("Fecha inicial", value=date.today().replace(day=1), key="date_min")
        date_max = st.date_input("Fecha final", value=date.today(), key="date_max")
    else:
        date_min, date_max = None, None

    sel_models = st.text_input("Filtrar modelos (coma-separado, opcional)")
    sel_endpoints = st.text_input("Filtrar endpoints (coma-separado, opcional)")
    sel_projects = st.text_input("Filtrar proyectos (coma-separado, opcional)")
    sel_users = st.text_input("Filtrar usuarios (coma-separado, opcional)")


# -----------------------
# Load & normalize
# -----------------------
frames, mappings = [], []
if files:
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            f.seek(0)
            df = pd.read_csv(f, encoding="utf-8", engine="python", on_bad_lines="skip")
        nd, mp = normalize_columns(df)
        frames.append(nd)
        mappings.append((f.name, mp))

if not frames:
    st.info("Sube uno o más CSVs para comenzar.")
    st.stop()

raw = pd.concat(frames, ignore_index=True)
original_rows = len(raw)

# Filters
if date_min:
    raw = raw[~raw["date"].isna()]
    raw = raw[raw["date"].dt.date >= date_min]
if date_max:
    raw = raw[~raw["date"].isna()]
    raw = raw[raw["date"].dt.date <= date_max]

filters = {
    "model": parse_comma_list(sel_models),
    "endpoint": parse_comma_list(sel_endpoints),
    "project": parse_comma_list(sel_projects),
    "user": parse_comma_list(sel_users),
}
for col, vals in filters.items():
    if vals:
        raw = raw[raw[col].isin(vals)]

filtered_rows = len(raw)

# Costing
raw = add_estimated_cost(raw, st.session_state.pricing_dict)

# Summaries
monthly, by_model, by_endpoint, by_project, by_user, by_key, daily = summarize(raw)

# Fixed fees allocation (by month)
months_present = [m for m in monthly["month"].tolist() if m != "Unknown"]
months_present_periods = [(m, month_to_period(m)) for m in months_present]

fees = st.session_state.fixed_fees or []
fees_by_month = {m: 0.0 for m in months_present}
for m_str, m_per in months_present_periods:
    for fee in fees:
        amt = float(fee.get("amount_usd", 0.0) or 0.0)
        sm = (fee.get("start_month", "") or "").strip()
        em = (fee.get("end_month", "") or "").strip()
        if month_in_range_period(m_per, sm, em):
            fees_by_month[m_str] += amt

fees_df = pd.DataFrame({"month": months_present, "fixed_fees": [fees_by_month[m] for m in months_present]})
monthly_merged = monthly.merge(fees_df, on="month", how="left")
monthly_merged["fixed_fees"] = monthly_merged["fixed_fees"].fillna(0.0)
monthly_merged = monthly_merged.rename(columns={"cost_variable": "variable_cost"})
monthly_merged["total_cost"] = monthly_merged["variable_cost"] + monthly_merged["fixed_fees"]

# Quality signals
rows_missing_date = int(raw["date"].isna().sum())
rows_no_cost = int(raw["cost_variable"].isna().sum())

st.success("Datos cargados correctamente.")
st.write("**Tier seleccionado:**", f"`{st.session_state.selected_tier}`")
st.write("**Archivos mapeados** (para transparencia de columnas detectadas):")
with st.expander("Ver mapeos de columnas"):
    for fname, mp in mappings:
        st.code(json.dumps({"file": fname, **mp}, indent=2, ensure_ascii=False))

with st.expander("Calidad de datos y filtros aplicados", expanded=True):
    st.write(
        f"- Filas originales: **{original_rows:,}**\n"
        f"- Filas después de filtros: **{filtered_rows:,}**\n"
        f"- Filas sin fecha: **{rows_missing_date:,}**\n"
        f"- Filas sin costo (ni provisto ni estimado): **{rows_no_cost:,}**"
    )
    if rows_no_cost > 0:
        st.warning(
            "Hay filas sin costo estimado. Usualmente es porque el modelo no existe en tu tabla de precios "
            "o el nombre del modelo no coincide."
        )

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Gasto total (var + fijo)", format_money(monthly_merged["total_cost"].sum()))
with col2:
    st.metric("Gasto variable", format_money(monthly_merged["variable_cost"].sum()))
with col3:
    st.metric("Cuotas fijas", format_money(monthly_merged["fixed_fees"].sum()))
with col4:
    days_with_usage = int(raw.loc[~raw["date"].isna(), "date"].dt.date.nunique())
    st.metric("Días con uso", days_with_usage)

st.markdown("## 📅 Gasto por mes (variable + fijo)")
st.dataframe(
    monthly_merged.assign(
        variable_cost=monthly_merged["variable_cost"].map(format_money),
        fixed_fees=monthly_merged["fixed_fees"].map(format_money),
        total_cost=monthly_merged["total_cost"].map(format_money),
    ),
    use_container_width=True,
)
st.bar_chart(monthly_merged.set_index("month")[["variable_cost", "fixed_fees", "total_cost"]])

st.markdown("## 🤖 Por modelo (variable)")
st.dataframe(by_model.assign(cost=by_model["cost"].map(format_money)), use_container_width=True)

st.markdown("## 🔌 Por endpoint (variable)")
st.dataframe(by_endpoint.assign(cost=by_endpoint["cost"].map(format_money)), use_container_width=True)

st.markdown("## 📁 Por proyecto (variable)")
st.dataframe(by_project.assign(cost=by_project["cost"].map(format_money)), use_container_width=True)

st.markdown("## 👤 Por usuario (variable)")
st.dataframe(by_user.assign(cost=by_user["cost"].map(format_money)), use_container_width=True)

st.markdown("## 🔑 Por API key (variable)")
st.dataframe(by_key.assign(cost=by_key["cost"].map(format_money)), use_container_width=True)

st.markdown("## 📆 Gasto diario (variable)")
if len(daily) == 0:
    st.info("No hay fechas válidas para el resumen diario.")
else:
    st.dataframe(daily.assign(cost_variable=daily["cost_variable"].map(format_money)).rename(columns={"cost_variable": "cost"}), use_container_width=True)
    st.line_chart(daily.set_index("day")["cost_variable"])

st.markdown("## ⬇️ Exportar resúmenes")
colA, colB, colC, colD, colE, colF, colG, colH, colI = st.columns(9)
with colA:
    st.download_button("Mensual (CSV)", to_csv_bytes(monthly_merged), file_name="summary_monthly_total.csv", mime="text/csv")
with colB:
    st.download_button("Por modelo (CSV)", to_csv_bytes(by_model), file_name="summary_by_model.csv", mime="text/csv")
with colC:
    st.download_button("Por endpoint (CSV)", to_csv_bytes(by_endpoint), file_name="summary_by_endpoint.csv", mime="text/csv")
with colD:
    st.download_button("Por proyecto (CSV)", to_csv_bytes(by_project), file_name="summary_by_project.csv", mime="text/csv")
with colE:
    st.download_button("Por usuario (CSV)", to_csv_bytes(by_user), file_name="summary_by_user.csv", mime="text/csv")
with colF:
    st.download_button("Por API key (CSV)", to_csv_bytes(by_key), file_name="summary_by_api_key.csv", mime="text/csv")
with colG:
    st.download_button("Diario (CSV)", to_csv_bytes(daily), file_name="summary_daily.csv", mime="text/csv")
with colH:
    st.download_button("Cuotas (CSV)", to_csv_bytes(pd.DataFrame(st.session_state.fixed_fees)), file_name="fixed_fees.csv", mime="text/csv")
with colI:
    st.download_button("Dataset normalizado (CSV)", to_csv_bytes(raw), file_name="usage_normalized_with_costs.csv", mime="text/csv")

st.markdown("---")
st.caption(
    "Tarifas incluyen **input**, **cached input**, y **output** (text tokens). "
    "El selector **Tier** controla únicamente los predeterminados; el cálculo usa tu tabla actual (pricing_config)."
)
