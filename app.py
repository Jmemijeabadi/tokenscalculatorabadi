
# streamlit_openai_usage_app.py (with pricing persistence)
# Run: streamlit run streamlit_openai_usage_app.py
import io, json, re, sys, time, zipfile
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="OpenAI Usage Analyzer", page_icon="💸", layout="wide")
st.title("💸 OpenAI Usage Analyzer")
st.caption("Sube tus CSVs de **Usage** de OpenAI (API/Projects). Obtén costos por mes y en dónde se usaron los tokens.")

# -----------------------
# Default pricing (USD per 1M tokens)
# -----------------------
DEFAULT_PRICING = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-2024-05-13": (5.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4-turbo": (3.00, 6.00),
    "gpt-4-turbo-2024-04-09": (5.00, 10.00),
    "gpt-4.1": (5.00, 15.00),
    "gpt-4.1-mini": (1.00, 2.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o3": (25.00, 100.00),
}

PRICING_FILE = Path("pricing_config.json")

def load_pricing_config():
    """Load pricing dict from pricing_config.json if present; else return DEFAULT_PRICING copy."""
    if PRICING_FILE.exists():
        try:
            data = json.loads(PRICING_FILE.read_text(encoding="utf-8"))
            # ensure tuples
            fixed = {k: tuple(v) for k, v in data.items()}
            return fixed
        except Exception:
            st.warning("No se pudo leer 'pricing_config.json'. Se usarán precios predeterminados.")
    return dict(DEFAULT_PRICING)

def save_pricing_config(d):
    try:
        PRICING_FILE.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception as e:
        st.error(f"No se pudo guardar 'pricing_config.json': {e}")
        return False

def pricing_dict_to_df(pr_dict):
    rows = []
    for model, (inp, out) in pr_dict.items():
        rows.append({"model": model, "input_per_1M": float(inp), "output_per_1M": float(out)})
    df = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    return df

def df_to_pricing_dict(df):
    pr = {}
    for _, r in df.iterrows():
        m = str(r.get("model", "")).strip()
        if not m:
            continue
        try:
            i = float(r.get("input_per_1M", np.nan))
            o = float(r.get("output_per_1M", np.nan))
        except Exception:
            continue
        if np.isnan(i) or np.isnan(o):
            continue
        pr[m] = (i, o)
    return pr

def guess_datetime_col(cols):
    cand = [c for c in cols if c.lower() in ("date", "day", "timestamp", "created_at", "time")]
    return cand[0] if cand else None

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
    mapping = {}
    cols = list(df.columns)
    mapping['date'] = guess_datetime_col(cols) or guess_col(cols, ["created", "usage_date"])
    mapping['model'] = guess_col(cols, ["model", "gpt_model", "engine"])
    mapping['endpoint'] = guess_col(cols, ["endpoint", "operation", "api_endpoint", "request_type"])
    mapping['project'] = guess_col(cols, ["project", "project_name", "project id", "project_id"])
    mapping['user'] = guess_col(cols, ["user", "user_email", "email", "actor", "owner"])
    mapping['org'] = guess_col(cols, ["org", "organization", "organization_id", "organization name"])
    mapping['input'] = guess_col(cols, ["prompt tokens", "input tokens", "input_tokens", "prompt_tokens", "tokens_in"])
    mapping['output'] = guess_col(cols, ["completion tokens", "output tokens", "output_tokens", "tokens_out"])
    mapping['total'] = guess_col(cols, ["total tokens", "total_tokens", "tokens_total"])
    mapping['cost'] = guess_col(cols, ["cost", "usd", "amount_usd", "cost_usd", "price_usd"])

    nd = pd.DataFrame(index=df.index)

    if mapping['date'] is not None:
        nd["date"] = pd.to_datetime(df[mapping['date']], errors="coerce")
    else:
        nd["date"] = pd.NaT

    def safe_text_col(colname):
        if mapping[colname] is not None:
            ser = df[mapping[colname]]
        else:
            ser = pd.Series(pd.NA, index=df.index, dtype="string")
        ser = ser.astype("string").replace({"None": pd.NA, "nan": pd.NA, "NaN": pd.NA}).str.strip()
        return ser

    for c in ["model", "endpoint", "project", "user", "org"]:
        nd[c] = safe_text_col(c)

    for k in ["input", "output", "total"]:
        if mapping[k] is not None:
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

    return nd, mapping

def estimate_cost(row, pricing_map):
    if not pd.isna(row.get("cost", np.nan)):
        return row["cost"]
    model = str(row.get("model", "") or "").strip()
    price = None
    if model in pricing_map:
        price = pricing_map[model]
    else:
        base = re.split(r"[:@]", model)[0]
        if base in pricing_map:
            price = pricing_map[base]
    if price is None:
        return np.nan
    in_rate, out_rate = price
    tokens_in = row.get("input", np.nan)
    tokens_out = row.get("output", np.nan)
    if pd.isna(tokens_in) and pd.isna(tokens_out):
        return np.nan
    v_in = (tokens_in or 0) / 1_000_000.0
    v_out = (tokens_out or 0) / 1_000_000.0
    return v_in * in_rate + v_out * out_rate

def add_estimated_cost(df, pricing_dict):
    if "cost" not in df.columns:
        df["cost"] = np.nan
    df["cost_estimated"] = df.apply(lambda r: estimate_cost(r, pricing_dict), axis=1)
    df["cost_final"] = df["cost"]
    df.loc[df["cost_final"].isna(), "cost_final"] = df.loc[df["cost_final"].isna(), "cost_estimated"]
    return df

def summarize(df):
    monthly = df.groupby("month", dropna=False)["cost_final"].sum().reset_index().sort_values("month")
    by_model = df.groupby("model", dropna=False).agg(
        cost=("cost_final", "sum"),
        input_tokens=("input", "sum"),
        output_tokens=("output", "sum"),
        total_tokens=("total", "sum"),
        rows=("model", "count"),
    ).reset_index().sort_values("cost", ascending=False)
    by_endpoint = df.groupby("endpoint", dropna=False).agg(
        cost=("cost_final", "sum"),
        total_tokens=("total", "sum"),
        rows=("endpoint", "count"),
    ).reset_index().sort_values("cost", ascending=False)
    by_project = df.groupby("project", dropna=False).agg(
        cost=("cost_final", "sum"),
        total_tokens=("total", "sum"),
        rows=("project", "count"),
    ).reset_index().sort_values("cost", ascending=False)
    by_user = df.groupby("user", dropna=False).agg(
        cost=("cost_final", "sum"),
        total_tokens=("total", "sum"),
        rows=("user", "count"),
    ).reset_index().sort_values("cost", ascending=False)
    daily = df.groupby(df["date"].dt.date)["cost_final"].sum().reset_index().rename(columns={"date":"day"}).sort_values("day")
    return monthly, by_model, by_endpoint, by_project, by_user, daily

def format_money(x):
    try:
        if pd.isna(x):
            return ""
        return f"${x:,.2f}"
    except Exception:
        return str(x)

# -----------------------
# Sidebar: Pricing persistence + filters
# -----------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown("1) **Sube tus CSVs** de Usage de OpenAI.\n2) Ajusta filtros.\n3) **Define y guarda tarifas por modelo**.")

    files = st.file_uploader("CSV(s) de Usage", type=["csv"], accept_multiple_files=True)

    st.markdown("---")
    st.subheader("Tarifas por modelo (persistentes)")

    # Load pricing into session_state once
    if "pricing_dict" not in st.session_state:
        st.session_state.pricing_dict = load_pricing_config()

    # Editor
    pr_df = pricing_dict_to_df(st.session_state.pricing_dict)
    edited = st.data_editor(
        pr_df,
        num_rows="dynamic",
        use_container_width=True,
        key="pricing_editor",
        column_config={
            "model": st.column_config.TextColumn("Modelo"),
            "input_per_1M": st.column_config.NumberColumn("Costo entrada / 1M", min_value=0.0, step=0.01),
            "output_per_1M": st.column_config.NumberColumn("Costo salida / 1M", min_value=0.0, step=0.01),
        }
    )

    cols_btn = st.columns(3)
    with cols_btn[0]:
        if st.button("💾 Guardar tarifas"):
            new_dict = df_to_pricing_dict(edited)
            if not new_dict:
                st.error("No hay filas válidas para guardar.")
            else:
                ok = save_pricing_config(new_dict)
                if ok:
                    st.session_state.pricing_dict = new_dict
                    st.success("Tarifas guardadas en 'pricing_config.json'.")

    with cols_btn[1]:
        if st.button("↩️ Cargar desde archivo"):
            st.session_state.pricing_dict = load_pricing_config()
            st.success("Tarifas recargadas desde 'pricing_config.json'.")

    with cols_btn[2]:
        if st.button("🧹 Restaurar predeterminadas"):
            st.session_state.pricing_dict = dict(DEFAULT_PRICING)
            st.success("Se restauraron las tarifas predeterminadas (no guardadas aún).")

    # Import/Export JSON
    st.caption("Importar/Exportar JSON")
    export_json = json.dumps(st.session_state.pricing_dict, indent=2, ensure_ascii=False)
    st.download_button("⬇️ Exportar JSON", export_json.encode("utf-8"), file_name="pricing_config.json", mime="application/json")
    uploaded_json = st.file_uploader("⬆️ Importar JSON", type=["json"], key="pricing_importer")
    if uploaded_json is not None:
        try:
            imp = json.loads(uploaded_json.read().decode("utf-8"))
            fixed = {k: tuple(v) for k, v in imp.items()}
            st.session_state.pricing_dict = fixed
            st.success("JSON importado. Recuerda presionar **Guardar tarifas** para persistirlo en disco.")
        except Exception as e:
            st.error(f"No se pudo importar el JSON: {e}")

    st.markdown("---")
    st.subheader("Filtros")
    date_min = st.date_input("Fecha inicial", value=None)
    date_max = st.date_input("Fecha final", value=None)
    sel_models = st.text_input("Filtrar modelos (coma-separado, opcional)")
    sel_endpoints = st.text_input("Filtrar endpoints (coma-separado, opcional)")
    sel_projects = st.text_input("Filtrar proyectos (coma-separado, opcional)")
    sel_users = st.text_input("Filtrar usuarios (coma-separado, opcional)")

# -----------------------
# Load & normalize data
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
        frames.append(nd); mappings.append((f.name, mp))

if not frames:
    st.info("Sube uno o más CSVs para comenzar.")
    st.stop()

raw = pd.concat(frames, ignore_index=True)

if date_min:
    raw = raw[raw["date"].dt.date >= date_min]
if date_max:
    raw = raw[raw["date"].dt.date <= date_max]

def parse_comma_list(s):
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]

filters = {
    "model": parse_comma_list(sel_models),
    "endpoint": parse_comma_list(sel_endpoints),
    "project": parse_comma_list(sel_projects),
    "user": parse_comma_list(sel_users),
}
for col, vals in filters.items():
    if vals:
        raw = raw[raw[col].isin(vals)]

# Costs using persistent pricing
raw = add_estimated_cost(raw, st.session_state.pricing_dict)

# -----------------------
# Summaries & UI
# -----------------------
monthly, by_model, by_endpoint, by_project, by_user, daily = summarize(raw)

st.success("Datos cargados correctamente.")
st.write("**Archivos mapeados** (para transparencia de columnas detectadas):")
with st.expander("Ver mapeos de columnas"):
    for fname, mp in mappings:
        st.code(json.dumps({"file": fname, **mp}, indent=2, ensure_ascii=False))

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Gasto total", format_money(raw["cost_final"].sum()))
with col2:
    st.metric("Tokens totales", f"{int(np.nan_to_num(raw['total'].sum())):,}")
with col3:
    st.metric("Modelos únicos", raw["model"].nunique())
with col4:
    st.metric("Días con uso", raw["date"].dt.date.nunique())

st.markdown("## 📅 Gasto por mes")
st.dataframe(monthly.assign(**{"cost_final": monthly["cost_final"].map(format_money)}).rename(columns={"cost_final":"cost"}), use_container_width=True)
st.bar_chart(monthly.set_index("month")["cost_final"])

st.markdown("## 🤖 Por modelo")
st.dataframe(by_model.assign(cost=by_model["cost"].map(format_money)), use_container_width=True)

st.markdown("## 🔌 Por endpoint")
st.dataframe(by_endpoint.assign(cost=by_endpoint["cost"].map(format_money)), use_container_width=True)

st.markdown("## 📁 Por proyecto")
st.dataframe(by_project.assign(cost=by_project["cost"].map(format_money)), use_container_width=True)

st.markdown("## 👤 Por usuario")
st.dataframe(by_user.assign(cost=by_user["cost"].map(format_money)), use_container_width=True)

st.markdown("## 📆 Gasto diario")
st.dataframe(daily.assign(cost_final=daily["cost_final"].map(format_money)).rename(columns={"cost_final":"cost"}), use_container_width=True)
st.line_chart(daily.set_index("day")["cost_final"])

# Downloads
st.markdown("## ⬇️ Exportar resúmenes")
def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

colA, colB, colC, colD, colE, colF = st.columns(6)
with colA:
    st.download_button("Mensual (CSV)", to_csv_bytes(monthly), file_name="summary_monthly.csv", mime="text/csv")
with colB:
    st.download_button("Por modelo (CSV)", to_csv_bytes(by_model), file_name="summary_by_model.csv", mime="text/csv")
with colC:
    st.download_button("Por endpoint (CSV)", to_csv_bytes(by_endpoint), file_name="summary_by_endpoint.csv", mime="text/csv")
with colD:
    st.download_button("Por proyecto (CSV)", to_csv_bytes(by_project), file_name="summary_by_project.csv", mime="text/csv")
with colE:
    st.download_button("Por usuario (CSV)", to_csv_bytes(by_user), file_name="summary_by_user.csv", mime="text/csv")
with colF:
    st.download_button("Diario (CSV)", to_csv_bytes(daily), file_name="summary_daily.csv", mime="text/csv")

st.markdown("---")
st.caption("Las tarifas se guardan en 'pricing_config.json'. Puedes editarlas en la tabla, importar/exportar JSON o restaurar valores predeterminados.")
