st.set_page_config(page_title="OpenAI Usage Analyzer", page_icon="💸", layout="wide")

st.title("💸 OpenAI Usage Analyzer")
st.caption("Sube tus CSVs de **Usage** de OpenAI (API/Projects). Obtén costos por mes y en dónde se usaron los tokens.")

# -----------------------
# Helper: default pricing (USD per 1M tokens)
# Update as needed. If your CSV already has a 'cost' column, that will be used.
# Otherwise, we estimate using this mapping.
# -----------------------
DEFAULT_PRICING = {
    # model_key: (input_per_million, output_per_million)
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-2024-05-13": (5.00, 15.00),  # older pricing example
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4-turbo": (3.00, 6.00),
    "gpt-4-turbo-2024-04-09": (5.00, 10.00),  # older pricing example
    "gpt-4.1": (5.00, 15.00),
    "gpt-4.1-mini": (1.00, 2.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o3": (25.00, 100.00),
}

def guess_datetime_col(cols):
    cand = [c for c in cols if c.lower() in ("date", "day", "timestamp", "created_at", "time")]
    return cand[0] if cand else None

def guess_col(cols, candidates):
    # Return first match in candidates by lowercase exact equality or contains.
    lcols = [c.lower() for c in cols]
    for name in candidates:
        if name in lcols:
            return cols[lcols.index(name)]
    # contains search
    for c in cols:
        cl = c.lower()
        if any(name in cl for name in candidates):
            return c
    return None

def normalize_columns(df):
    # Standardize expected columns if present
    mapping = {}
    cols = list(df.columns)
    # date/time
    mapping['date'] = guess_datetime_col(cols) or guess_col(cols, ["created", "usage_date"])
    # model
    mapping['model'] = guess_col(cols, ["model", "gpt_model", "engine"])
    # operation/endpoint
    mapping['endpoint'] = guess_col(cols, ["endpoint", "operation", "api_endpoint", "request_type"])
    # project
    mapping['project'] = guess_col(cols, ["project", "project_name", "project id", "project_id"])
    # user
    mapping['user'] = guess_col(cols, ["user", "user_email", "email", "actor", "owner"])
    # org
    mapping['org'] = guess_col(cols, ["org", "organization", "organization_id", "organization name"])
    # tokens
    mapping['input'] = guess_col(cols, ["prompt tokens", "input tokens", "input_tokens", "prompt_tokens", "tokens_in"])
    mapping['output'] = guess_col(cols, ["completion tokens", "output tokens", "output_tokens", "tokens_out"])
    mapping['total'] = guess_col(cols, ["total tokens", "total_tokens", "tokens_total"])
    # cost
    mapping['cost'] = guess_col(cols, ["cost", "usd", "amount_usd", "cost_usd", "price_usd"])

    # Build normalized dataframe
    nd = pd.DataFrame()
    if mapping['date'] is not None:
        nd["date"] = pd.to_datetime(df[mapping['date']], errors="coerce")
    else:
        nd["date"] = pd.NaT
    nd["model"] = df[mapping['model']] if mapping['model'] is not None else None
    nd["endpoint"] = df[mapping['endpoint']] if mapping['endpoint'] is not None else None
    nd["project"] = df[mapping['project']] if mapping['project'] is not None else None
    nd["user"] = df[mapping['user']] if mapping['user'] is not None else None
    nd["org"] = df[mapping['org']] if mapping['org'] is not None else None

    for k in ["input", "output", "total"]:
        if mapping[k] is not None:
            nd[k] = pd.to_numeric(df[mapping[k]], errors="coerce")
        else:
            nd[k] = np.nan

    if mapping["cost"] is not None:
        nd["cost"] = pd.to_numeric(df[mapping["cost"]], errors="coerce")
    else:
        nd["cost"] = np.nan

    # Derive total if missing
    if nd["total"].isna().all():
        nd["total"] = nd[["input", "output"]].sum(axis=1, skipna=True)

    # Derive month
    nd["month"] = nd["date"].dt.to_period("M").astype(str)

    # Clean strings
    for c in ["model", "endpoint", "project", "user", "org"]:
        if nd[c].dtype == object:
            nd[c] = nd[c].astype(str).replace({"None": np.nan, "nan": np.nan}).str.strip()

    return nd, mapping

def estimate_cost(row, pricing_map):
    if not pd.isna(row.get("cost", np.nan)):
        return row["cost"]
    model = str(row.get("model", "") or "").strip()
    # fuzzy match: try exact, then base prefix before ":" or "@"
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
    # Convert tokens to millions
    v_in = (tokens_in or 0) / 1_000_000.0
    v_out = (tokens_out or 0) / 1_000_000.0
    return v_in * in_rate + v_out * out_rate

def add_estimated_cost(df, pricing_json):
    # Merge user pricing
    try:
        custom = json.loads(pricing_json) if pricing_json else {}
    except Exception:
        custom = {}
    pricing = dict(DEFAULT_PRICING)
    pricing.update(custom)

    if "cost" not in df.columns:
        df["cost"] = np.nan
    df["cost_estimated"] = df.apply(lambda r: estimate_cost(r, pricing), axis=1)
    # prefer actual cost when present
    df["cost_final"] = df["cost"]
    df.loc[df["cost_final"].isna(), "cost_final"] = df.loc[df["cost_final"].isna(), "cost_estimated"]
    return df

def summarize(df):
    # Monthly totals
    monthly = df.groupby("month", dropna=False)["cost_final"].sum().reset_index().sort_values("month")

    # By model
    by_model = df.groupby("model", dropna=False).agg(
        cost=("cost_final", "sum"),
        input_tokens=("input", "sum"),
        output_tokens=("output", "sum"),
        total_tokens=("total", "sum"),
        rows=("model", "count"),
    ).reset_index().sort_values("cost", ascending=False)

    # By endpoint
    by_endpoint = df.groupby("endpoint", dropna=False).agg(
        cost=("cost_final", "sum"),
        total_tokens=("total", "sum"),
        rows=("endpoint", "count"),
    ).reset_index().sort_values("cost", ascending=False)

    # By project
    by_project = df.groupby("project", dropna=False).agg(
        cost=("cost_final", "sum"),
        total_tokens=("total", "sum"),
        rows=("project", "count"),
    ).reset_index().sort_values("cost", ascending=False)

    # By user
    by_user = df.groupby("user", dropna=False).agg(
        cost=("cost_final", "sum"),
        total_tokens=("total", "sum"),
        rows=("user", "count"),
    ).reset_index().sort_values("cost", ascending=False)

    # Daily top (for drilling)
    daily = df.groupby(df["date"].dt.date)["cost_final"].sum().reset_index().rename(columns={"date":"day"}).sort_values("day")

    return monthly, by_model, by_endpoint, by_project, by_user, daily

def format_money(x):
    try:
        if pd.isna(x):
            return ""
        return f"${x:,.2f}"
    except Exception:
        return str(x)

with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown("1) **Sube tus CSVs** de Usage de OpenAI.\n2) Ajusta filtros.\n3) (Opcional) Proporciona tarifas personalizadas (JSON).")
    files = st.file_uploader("CSV(s) de Usage", type=["csv"], accept_multiple_files=True)
    st.markdown("---")
    st.subheader("Tarifas personalizadas (JSON opcional)")
    pricing_json = st.text_area(
        "Ejemplo: {\"gpt-4o\": [2.5, 10], \"gpt-3.5-turbo\": [0.5, 1.5]}",
        height=120
    )
    st.caption("Si tu CSV ya trae columna 'cost', se usará esa. Si no, se estimará con estas tarifas.")
    st.markdown("---")
    st.subheader("Filtros")
    date_min = st.date_input("Fecha inicial", value=None)
    date_max = st.date_input("Fecha final", value=None)
    sel_models = st.text_input("Filtrar modelos (coma-separado, opcional)")
    sel_endpoints = st.text_input("Filtrar endpoints (coma-separado, opcional)")
    sel_projects = st.text_input("Filtrar proyectos (coma-separado, opcional)")
    sel_users = st.text_input("Filtrar usuarios (coma-separado, opcional)")

# Load & normalize
frames = []
mappings = []
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

# Apply filters
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

# Add costs
raw = add_estimated_cost(raw, pricing_json)

# Summaries
monthly, by_model, by_endpoint, by_project, by_user, daily = summarize(raw)

st.success("Datos cargados correctamente.")
st.write("**Archivos mapeados** (para transparencia de columnas detectadas):")
with st.expander("Ver mapeos de columnas"):
    for fname, mp in mappings:
        st.code(json.dumps({"file": fname, **mp}, indent=2, ensure_ascii=False))

# KPI
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

# Downloadable exports
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
st.caption("Tip: si tu CSV no trae columna de 'cost', el app estima el costo usando tarifas por modelo (editables en la barra lateral).")
