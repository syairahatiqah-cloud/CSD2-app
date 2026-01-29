# app.py
# ============================================================
# Streamlit App: Early Warning Signals (EWS) using Critical Slowing Down (CSD)
#
# IMPROVEMENTS ADDED based on your friend's dashboard concept:
# ✅ Step 1 Trend Extraction uses GAUSSIAN filter (Sigma in HOURS) if SciPy available
#    - fallback: centered rolling mean/median (if SciPy not available)
# ✅ Step 2 Residual = Raw - Trend
# ✅ Step 3 Rolling Metrics: Variance + Lag-1 Autocorrelation (AR1) + PSD Low-Freq
# ✅ Step 4 Kendall’s Tau trend score for Variance/AR1/PSD (tau in [-1, +1])
#
# Tabs:
# TAB 1: Analysis Plot (Step 1–5) with HTML + PNG download for each step
# TAB 2: Data & Statistics (missingness, stats, histogram, ADF stationarity test*)
# TAB 3: Lead Time Analysis (threshold crossing → flood peak lead time + confusion metrics)
# TAB 4: Memory Analysis (ACF plot, return map Xt vs Xt-1, simple red-noise PSD compare)
#
# *ADF requires statsmodels (optional). App will still run without it.
#
# Notes:
# - CSD indicators are computed on RESIDUAL series (recommended by CSD theory).
# - Hover shows FULL datetime.
# - Robust CSV + robust datetime parsing.
# ============================================================

import io
import re
import zipfile
import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -----------------------------
# Optional imports
# -----------------------------
_HAS_SCIPY = False
_HAS_STATS = False
gaussian_filter1d = None
scipy_welch = None
kendalltau = None

try:
    from scipy.ndimage import gaussian_filter1d as _gaussian_filter1d
    from scipy.signal import welch as _welch
    from scipy.stats import kendalltau as _kendalltau
    gaussian_filter1d = _gaussian_filter1d
    scipy_welch = _welch
    kendalltau = _kendalltau
    _HAS_SCIPY = True
except Exception:
    pass

try:
    from statsmodels.tsa.stattools import adfuller as _adfuller
    _HAS_STATS = True
except Exception:
    _adfuller = None


# -----------------------------
# Robust CSV loader
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "utf-16", "cp1252", "latin1"]
    seps = [",", ";", "\t"]
    last_err = None

    for enc in encodings:
        for sep in seps:
            try:
                return pd.read_csv(
                    io.BytesIO(file_bytes),
                    dtype=str,
                    keep_default_na=False,
                    encoding=enc,
                    sep=sep,
                    engine="python",
                )
            except Exception as e:
                last_err = e

    try:
        return pd.read_csv(
            io.BytesIO(file_bytes),
            dtype=str,
            keep_default_na=False,
            encoding="latin1",
            sep=None,
            engine="python",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV. Last error: {last_err}") from e


# -----------------------------
# Datetime parsing (R-like order)
# -----------------------------
def parse_dt_series(s: pd.Series, tz: str = "Asia/Kuala_Lumpur") -> pd.Series:
    s = s.astype(str).str.strip()

    fmts = [
        "%d/%m/%Y %H:%M", "%d/%m/%Y %H",
        "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"
    ]

    dt = pd.Series(pd.NaT, index=s.index)
    remaining = dt.isna()

    for f in fmts:
        if not remaining.any():
            break
        parsed = pd.to_datetime(s[remaining], format=f, errors="coerce")
        dt.loc[remaining] = parsed
        remaining = dt.isna()

    if remaining.any():
        dt.loc[remaining] = pd.to_datetime(s[remaining], errors="coerce", dayfirst=True)

    if tz:
        try:
            if getattr(dt.dt, "tz", None) is None:
                dt = dt.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
            else:
                dt = dt.dt.tz_convert(tz)
        except Exception:
            pass

    return dt


def safe_name(col: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(col)).strip("_") or "value"


# ============================================================
# Step 1: Trend extraction (Gaussian preferred)
# ============================================================
def extract_trend(series: pd.Series, sigma_hours: int, method_fallback: str = "mean") -> pd.Series:
    """
    Preferred: Gaussian filter with sigma (in samples = hours) when SciPy available.
    Fallback: centered rolling mean/median.
    """
    s = series.astype(float).to_numpy()
    out = np.full_like(s, np.nan, dtype=float)

    good = np.isfinite(s)
    if good.sum() < 5:
        return pd.Series(out, index=series.index)

    # Fill missing for filtering (linear interpolate + edge fill)
    s2 = pd.Series(s).interpolate(limit_direction="both").to_numpy()

    if _HAS_SCIPY and gaussian_filter1d is not None:
        sig = max(1.0, float(sigma_hours))  # sigma in samples (hourly)
        # "reflect" reduces edge artifacts
        t = gaussian_filter1d(s2, sigma=sig, mode="reflect")
        out[:] = t
        out[~good] = np.nan  # keep missing as missing
        return pd.Series(out, index=series.index)

    # Fallback rolling
    win = max(3, int(sigma_hours))
    ss = pd.Series(s, index=series.index)
    if method_fallback == "median":
        tr = ss.rolling(win, center=True, min_periods=max(2, win // 3)).median()
    else:
        tr = ss.rolling(win, center=True, min_periods=max(2, win // 3)).mean()
    return tr


# ============================================================
# CSD indicators (computed on residual)
# ============================================================
def calc_ar1(w: np.ndarray) -> float:
    w = w[np.isfinite(w)]
    if w.size < 2:
        return np.nan
    if np.nanstd(w) == 0:
        return np.nan
    return float(np.corrcoef(w[:-1], w[1:])[0, 1])


def psd_lowfreq_fft(w: np.ndarray, low_k: int = 8) -> float:
    w = w[np.isfinite(w)]
    if w.size < 4:
        return np.nan
    w = w - np.nanmean(w)
    spec = np.abs(np.fft.rfft(w)) ** 2
    if spec.size <= 1:
        return np.nan
    k = min(low_k, spec.size - 1)  # exclude DC
    if k < 1:
        return np.nan
    return float(np.nanmean(spec[1:k + 1]))


def psd_lowfreq_welch(w: np.ndarray, fs: float = 1.0) -> float:
    w = w[np.isfinite(w)]
    if w.size < 8:
        return np.nan
    w = w - np.nanmean(w)
    freqs, pxx = scipy_welch(w, fs=fs, nperseg=min(256, w.size))
    if len(freqs) < 3:
        return np.nan
    start = 1
    take = max(2, int(0.10 * (len(freqs) - start)))
    end = min(len(freqs), start + take)
    return float(np.nanmean(pxx[start:end]))


def csd_roll(x: np.ndarray, dt: pd.Series, window_hours: int, valid_frac: float = 0.8,
             psd_method: str = "FFT", low_k_cap: int = 8) -> pd.DataFrame:
    win = int(window_hours)
    low_k = min(low_k_cap, max(2, win // 10))  # dynamic

    n = len(x)
    out_len = n - win + 1
    if out_len < 1:
        return pd.DataFrame()

    var_roll = np.full(out_len, np.nan, dtype=float)
    ar1_roll = np.full(out_len, np.nan, dtype=float)
    psd_low  = np.full(out_len, np.nan, dtype=float)

    min_good = max(4, int(np.floor(valid_frac * win)))

    for i in range(out_len):
        w = x[i:i + win]
        good = np.isfinite(w).sum()
        if good < min_good:
            continue

        w2 = w[np.isfinite(w)]
        var_roll[i] = float(np.var(w2, ddof=1)) if w2.size > 1 else np.nan
        ar1_roll[i] = calc_ar1(w2)

        if psd_method == "Welch" and _HAS_SCIPY and scipy_welch is not None:
            psd_low[i] = psd_lowfreq_welch(w2, fs=1.0)
        else:
            psd_low[i] = psd_lowfreq_fft(w2, low_k=low_k)

    return pd.DataFrame({
        "datetime_start": dt.iloc[:out_len].values,
        "datetime_end":   dt.iloc[win - 1: win - 1 + out_len].values,
        "window_hours":   win,
        "low_k_used":     low_k,
        "Variance":       var_roll,
        "AR1":            ar1_roll,
        "PSD_LowFreq":    psd_low
    })


# ============================================================
# Kendall's Tau (Step 4)
# ============================================================
def kendall_tau_trend(y: pd.Series) -> float:
    """
    Kendall tau of indicator vs time index.
    tau ~ +1 means strong increasing trend (warning).
    """
    y = pd.Series(y).astype(float)
    y = y[np.isfinite(y)]
    if len(y) < 8:
        return np.nan

    x = np.arange(len(y), dtype=float)

    if _HAS_SCIPY and kendalltau is not None:
        tau, _p = kendalltau(x, y.to_numpy())
        return float(tau)

    # fallback (simple approx): compute tau using sign concordance (O(n^2), OK for small)
    arr = y.to_numpy()
    n = len(arr)
    if n > 2000:  # avoid too slow
        arr = arr[::2]
        n = len(arr)
    conc = 0
    disc = 0
    for i in range(n - 1):
        di = arr[i+1:] - arr[i]
        conc += int((di > 0).sum())
        disc += int((di < 0).sum())
    denom = conc + disc
    return (conc - disc) / denom if denom > 0 else np.nan


# ============================================================
# Helper: thresholds + warnings
# ============================================================
def to_float_or_nan(s: str) -> float:
    s = (s or "").strip()
    if s == "":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


# ============================================================
# Lead time analysis helpers
# ============================================================
def simple_peak_indices(x: np.ndarray, min_separation: int = 24) -> np.ndarray:
    """
    Simple local maxima with min separation (hours).
    Used ONLY for lead time analysis (not plotted).
    """
    x = np.asarray(x, float)
    good = np.isfinite(x)
    idxs = []
    last = -10**9
    for i in range(1, len(x) - 1):
        if not (good[i-1] and good[i] and good[i+1]):
            continue
        if x[i] > x[i-1] and x[i] > x[i+1]:
            if i - last >= min_separation:
                idxs.append(i)
                last = i
    return np.array(idxs, dtype=int)


# ============================================================
# Plot helpers + exports
# ============================================================
HOVER_FMT = "%Y-%m-%d %H:%M:%S"

def plotly_line(x, y, name, hover_label=None, color=None, width=2):
    lbl = hover_label or name
    line = dict(width=width)
    if color is not None:
        line["color"] = color
    return go.Scatter(
        x=x, y=y, mode="lines", name=name,
        line=line,
        hovertemplate=f"datetime: %{{x|{HOVER_FMT}}}<br>{lbl}: %{{y:.6g}}<extra></extra>"
    )

def add_threshold_line(fig: go.Figure, y: float, label: str):
    if y is None or (isinstance(y, float) and np.isnan(y)):
        return
    fig.add_hline(
        y=y, line_width=1.5, line_dash="dash",
        annotation_text=label, annotation_position="top left"
    )

def fig_to_html_bytes(fig: go.Figure) -> bytes:
    return fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")

def mpl_timeseries(dt, ys, labels, title, ylabel):
    """
    Matplotlib plot with multiple lines (for PNG export).
    ys: list of arrays/Series
    """
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for y, lab in zip(ys, labels):
        ax.plot(dt, y, label=lab)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("DateTime")
    ax.set_ylabel(ylabel)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, linestyle="--", alpha=0.35)
    if len(labels) > 1:
        ax.legend()
    fig.tight_layout()
    return fig

def fig_to_png_bytes(fig: plt.Figure, dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# App
# ============================================================
st.set_page_config(page_title="CSD Early Warning (EWS)", layout="wide")
st.title("Early Warning Signals (EWS) — Critical Slowing Down (CSD)")

with st.sidebar:
    st.header("1) Configuration")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    st.divider()
    st.header("2) Columns & Parsing")
    tz = st.selectbox("Time zone", ["Asia/Kuala_Lumpur", "UTC"], index=0)

    st.divider()
    st.header("3) Trend Extraction (Step 1)")
    st.caption("CSD works on fluctuations → we remove slow trend to get residuals.")
    trend_method_fallback = st.selectbox("Fallback trend method (if no SciPy)", ["mean", "median"], index=0)
    sigma_hours = st.slider("Gaussian Sigma (HOURS)", 3, 24 * 60, 24 * 7, 1)
    st.caption("Example: 168 hours = 7 days (as in your friend's dashboard).")
    st.info(f"SciPy Gaussian available: {_HAS_SCIPY}")

    st.divider()
    st.header("4) Rolling indicators (Step 3)")
    window_choices = [6, 12, 24, 36, 48, 72, 96, 120, 144, 168]
    windows = st.multiselect("Rolling windows (hours)", options=window_choices, default=[72])
    valid_frac = st.slider("Min valid fraction per window", 0.50, 1.00, 0.80, 0.05)

    psd_method = st.selectbox(
        "PSD method",
        options=(["Welch (SciPy)"] if _HAS_SCIPY else []) + ["FFT (fast)"],
        index=0
    )
    psd_method_key = "Welch" if psd_method.startswith("Welch") else "FFT"

    st.divider()
    st.header("5) Thresholds (for warnings / lead time)")
    st.caption("Red dashed threshold lines (like your friend’s screenshot). Leave blank to disable.")
    thr_var = st.text_input("Variance threshold", value="")
    thr_ar1 = st.text_input("Autocorrelation threshold (AR1)", value="")
    thr_psd = st.text_input("PSD_LowFreq threshold", value="")
    warn_rule = st.selectbox("Warning rule", ["Any indicator exceeds", "All indicators exceed"], index=0)

    st.divider()
    run = st.button("Run Analysis", type="primary")

def add_log(msg: str):
    st.session_state.setdefault("log", [])
    st.session_state["log"].append(msg)

if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

raw_df = load_csv_bytes(uploaded.getvalue())
if raw_df.empty:
    st.error("Your CSV is empty.")
    st.stop()

cols = list(raw_df.columns)
c1, c2 = st.columns(2)
with c1:
    dt_col = st.selectbox("Datetime column", options=cols, index=0)
with c2:
    default_val_idx = 1 if len(cols) > 1 else 0
    val_col = st.selectbox("Value column (WL/SF)", options=cols, index=default_val_idx)

if not windows:
    st.warning("Please select at least one rolling window.")
    st.stop()

# ============================================================
# RUN
# ============================================================
if run:
    st.session_state["log"] = []
    add_log("---- Run started ----")
    add_log(f"File: {uploaded.name}")
    add_log(f"Datetime col: {dt_col} | Value col: {val_col}")
    add_log(f"Timezone: {tz}")
    add_log(f"Sigma (hours): {sigma_hours} | SciPy Gaussian: {_HAS_SCIPY}")
    add_log(f"Rolling windows: {windows} | valid_frac={valid_frac} | PSD={psd_method_key}")

    df = raw_df.copy()
    df[dt_col] = parse_dt_series(df[dt_col], tz=tz)
    df[val_col] = pd.to_numeric(df[val_col].replace("", np.nan), errors="coerce")
    df = df[df[dt_col].notna()].sort_values(dt_col).reset_index(drop=True)

    if len(df) < max(windows) + 5:
        st.warning("Your series is short compared to your rolling window. Try smaller windows.")

    wl = df[val_col].astype(float)
    trend = extract_trend(wl, sigma_hours=int(sigma_hours), method_fallback=trend_method_fallback)
    resid = wl - trend

    df["WL"] = wl
    df["Trend"] = trend
    df["Residual"] = resid

    add_log(f"Datetime range: {df[dt_col].iloc[0]} to {df[dt_col].iloc[-1]}")
    add_log(f"Rows: {len(df)}")

    # Rolling CSD on residual
    dt = df[dt_col]
    x = df["Residual"].to_numpy(dtype=float)

    res_list = []
    for w in sorted(map(int, windows)):
        add_log(f"Computing rolling indicators for window={w}h")
        out = csd_roll(
            x=x, dt=dt, window_hours=w, valid_frac=float(valid_frac),
            psd_method=psd_method_key, low_k_cap=8
        )
        if not out.empty:
            res_list.append(out)

    if not res_list:
        st.error("No rolling results produced. Reduce window length or reduce missingness.")
        st.stop()

    res_all = pd.concat(res_list, ignore_index=True)
    res_all["datetime_start"] = pd.to_datetime(res_all["datetime_start"], errors="coerce")
    res_all["datetime_end"] = pd.to_datetime(res_all["datetime_end"], errors="coerce")

    # Thresholds + warnings
    t_var = to_float_or_nan(thr_var)
    t_ar1 = to_float_or_nan(thr_ar1)
    t_psd = to_float_or_nan(thr_psd)

    def exceeds(row) -> dict:
        v_ok = (not np.isnan(t_var)) and np.isfinite(row["Variance"]) and (row["Variance"] >= t_var)
        a_ok = (not np.isnan(t_ar1)) and np.isfinite(row["AR1"]) and (row["AR1"] >= t_ar1)
        p_ok = (not np.isnan(t_psd)) and np.isfinite(row["PSD_LowFreq"]) and (row["PSD_LowFreq"] >= t_psd)
        return {"var": v_ok, "ar1": a_ok, "psd": p_ok}

    flags = res_all.apply(exceeds, axis=1, result_type="expand")
    res_all["exceed_var"] = flags["var"].fillna(False)
    res_all["exceed_ar1"] = flags["ar1"].fillna(False)
    res_all["exceed_psd"] = flags["psd"].fillna(False)

    enabled = [not np.isnan(t_var), not np.isnan(t_ar1), not np.isnan(t_psd)]
    enabled_count = int(sum(enabled))

    if enabled_count == 0:
        res_all["warning"] = False
    else:
        if warn_rule == "All indicators exceed":
            conds = []
            if not np.isnan(t_var): conds.append(res_all["exceed_var"])
            if not np.isnan(t_ar1): conds.append(res_all["exceed_ar1"])
            if not np.isnan(t_psd): conds.append(res_all["exceed_psd"])
            res_all["warning"] = np.logical_and.reduce(conds)
        else:
            cols_any = []
            if not np.isnan(t_var): cols_any.append("exceed_var")
            if not np.isnan(t_ar1): cols_any.append("exceed_ar1")
            if not np.isnan(t_psd): cols_any.append("exceed_psd")
            res_all["warning"] = res_all[cols_any].any(axis=1)

    # Kendall tau (per window)
    tau_rows = []
    for w in sorted(res_all["window_hours"].unique().astype(int).tolist()):
        dfw = res_all[res_all["window_hours"] == w].copy()
        tau_rows.append({
            "window_hours": w,
            "tau_variance": kendall_tau_trend(dfw["Variance"]),
            "tau_ar1": kendall_tau_trend(dfw["AR1"]),
            "tau_psd": kendall_tau_trend(dfw["PSD_LowFreq"]),
        })
    tau_df = pd.DataFrame(tau_rows)

    add_log("---- Run finished ----")

    st.session_state["df_main"] = df
    st.session_state["res_all"] = res_all
    st.session_state["tau_df"] = tau_df
    st.session_state["thr"] = {"var": t_var, "ar1": t_ar1, "psd": t_psd}
    st.session_state["dt_col"] = dt_col
    st.session_state["val_col"] = val_col
    st.session_state["sigma_hours"] = int(sigma_hours)

# Use prior results
df_main = st.session_state.get("df_main")
res_all = st.session_state.get("res_all")
tau_df = st.session_state.get("tau_df")

if df_main is None or res_all is None:
    st.warning("Click **Run Analysis** to compute indicators.")
    st.stop()

dt_col = st.session_state["dt_col"]
val_col = st.session_state["val_col"]
thr = st.session_state["thr"]
sigma_hours = st.session_state["sigma_hours"]

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["Analysis Plot", "Data & Statistics", "Lead Time Analysis", "Memory Analysis"]
)

# ============================================================
# TAB 1: Analysis Plot (Step 1–5 + Kendall tau)
# ============================================================
with tab1:
    st.subheader("TAB 1 — Analysis Plot (Core Workflow)")

    available_windows = sorted(res_all["window_hours"].unique().astype(int).tolist())
    default_w = available_windows[0]
    plot_w = st.selectbox("Select rolling window (hours)", available_windows, index=0)

    dfw = res_all[res_all["window_hours"] == int(plot_w)].copy().sort_values("datetime_start")

    # --- Step 1: Trend Extraction ---
    st.markdown(f"### Step 1: Trend Extraction (Gaussian Sigma = {sigma_hours} hours)")
    fig1 = go.Figure()
    fig1.add_trace(plotly_line(df_main[dt_col], df_main["WL"], "Raw Data", hover_label="Raw", color="#9e9e9e", width=2))
    fig1.add_trace(plotly_line(df_main[dt_col], df_main["Trend"], "Trend", hover_label="Trend", color="#1f77b4", width=3))
    fig1.update_layout(height=420, hovermode="x unified", xaxis_title="DateTime", yaxis_title=str(val_col))
    fig1.update_xaxes(hoverformat=HOVER_FMT)
    st.plotly_chart(fig1, use_container_width=True)

    c1a, c1b = st.columns(2)
    c1a.download_button("Download HTML (Step 1)", fig_to_html_bytes(fig1), f"Step1_Trend_{safe_name(val_col)}.html", "text/html")
    png1 = fig_to_png_bytes(mpl_timeseries(df_main[dt_col], [df_main["WL"], df_main["Trend"]], ["Raw", "Trend"],
                                           f"Step 1 Trend (Sigma={sigma_hours}h)", str(val_col)))
    c1b.download_button("Download PNG (Step 1)", png1, f"Step1_Trend_{safe_name(val_col)}.png", "image/png")

    # --- Step 2: Residual ---
    st.markdown("### Step 2: Residual (WL − Trend)")
    fig2 = go.Figure()
    fig2.add_trace(plotly_line(df_main[dt_col], df_main["Residual"], "Residual", hover_label="Residual"))
    fig2.update_layout(height=360, hovermode="x unified", xaxis_title="DateTime", yaxis_title="Residual")
    fig2.update_xaxes(hoverformat=HOVER_FMT)
    st.plotly_chart(fig2, use_container_width=True)

    c2a, c2b = st.columns(2)
    c2a.download_button("Download HTML (Step 2)", fig_to_html_bytes(fig2), f"Step2_Residual_{safe_name(val_col)}.html", "text/html")
    png2 = fig_to_png_bytes(mpl_timeseries(df_main[dt_col], [df_main["Residual"]], ["Residual"],
                                           "Step 2 Residual", "Residual"))
    c2b.download_button("Download PNG (Step 2)", png2, f"Step2_Residual_{safe_name(val_col)}.png", "image/png")

    # --- Step 3: Variance + AR1 (Rolling Metrics) ---
    st.markdown(f"### Step 3: Rolling Metrics (window = {plot_w}h)")
    st.caption("These are the main Early Warning Indicators: Variance ↑ and AR1 ↑.")

    # Variance
    st.markdown("#### 3A) Rolling Variance")
    fig3 = go.Figure()
    fig3.add_trace(plotly_line(dfw["datetime_start"], dfw["Variance"], "Variance", hover_label="Variance"))
    add_threshold_line(fig3, thr["var"], "Variance threshold")
    fig3.update_layout(height=320, hovermode="x unified", xaxis_title="DateTime", yaxis_title="Variance")
    fig3.update_xaxes(hoverformat=HOVER_FMT)
    st.plotly_chart(fig3, use_container_width=True)
    c3a, c3b = st.columns(2)
    c3a.download_button("Download HTML (Variance)", fig_to_html_bytes(fig3), f"Step3A_Variance_{plot_w}h.html", "text/html")
    png3 = fig_to_png_bytes(mpl_timeseries(dfw["datetime_start"], [dfw["Variance"]], ["Variance"],
                                           f"Rolling Variance ({plot_w}h)", "Variance"))
    c3b.download_button("Download PNG (Variance)", png3, f"Step3A_Variance_{plot_w}h.png", "image/png")

    # AR1
    st.markdown("#### 3B) Rolling Lag-1 Autocorrelation (AR1)")
    fig5 = go.Figure()
    fig5.add_trace(plotly_line(dfw["datetime_start"], dfw["AR1"], "AR1", hover_label="AR1"))
    fig5.add_hline(y=0, line_width=1, line_dash="dash")
    add_threshold_line(fig5, thr["ar1"], "AR1 threshold")
    fig5.update_layout(height=320, hovermode="x unified", xaxis_title="DateTime", yaxis_title="AR1")
    fig5.update_xaxes(hoverformat=HOVER_FMT)
    st.plotly_chart(fig5, use_container_width=True)
    c5a, c5b = st.columns(2)
    c5a.download_button("Download HTML (AR1)", fig_to_html_bytes(fig5), f"Step3B_AR1_{plot_w}h.html", "text/html")
    png5 = fig_to_png_bytes(mpl_timeseries(dfw["datetime_start"], [dfw["AR1"]], ["AR1"],
                                           f"Rolling AR1 ({plot_w}h)", "AR1"))
    c5b.download_button("Download PNG (AR1)", png5, f"Step3B_AR1_{plot_w}h.png", "image/png")

    # --- Step 4: PSD (Low-Frequency) ---
    st.markdown(f"### Step 4: PSD Low-Frequency (window = {plot_w}h)")
    fig4 = go.Figure()
    fig4.add_trace(plotly_line(dfw["datetime_start"], dfw["PSD_LowFreq"], "PSD_LowFreq", hover_label="PSD_LowFreq"))
    add_threshold_line(fig4, thr["psd"], "PSD threshold")
    fig4.update_layout(height=320, hovermode="x unified", xaxis_title="DateTime", yaxis_title="PSD_LowFreq")
    fig4.update_xaxes(hoverformat=HOVER_FMT)
    st.plotly_chart(fig4, use_container_width=True)
    c4a, c4b = st.columns(2)
    c4a.download_button("Download HTML (PSD)", fig_to_html_bytes(fig4), f"Step4_PSD_{plot_w}h.html", "text/html")
    png4 = fig_to_png_bytes(mpl_timeseries(dfw["datetime_start"], [dfw["PSD_LowFreq"]], ["PSD_LowFreq"],
                                           f"PSD Low-Frequency ({plot_w}h)", "PSD_LowFreq"))
    c4b.download_button("Download PNG (PSD)", png4, f"Step4_PSD_{plot_w}h.png", "image/png")

    # --- Step 5: Kendall’s Tau Score ---
    st.markdown("### Step 5: Kendall’s Tau Score (Trend Strength)")
    st.caption("Tau close to +1 indicates a strong increasing trend in indicators (strong warning signal).")

    tau_row = tau_df[tau_df["window_hours"] == int(plot_w)].copy()
    if tau_row.empty:
        st.info("Tau not available for this window.")
    else:
        tau_row = tau_row.iloc[0]
        cA, cB, cC = st.columns(3)
        cA.metric("Tau (Variance)", f"{tau_row['tau_variance']:.3f}" if np.isfinite(tau_row['tau_variance']) else "NA")
        cB.metric("Tau (AR1)", f"{tau_row['tau_ar1']:.3f}" if np.isfinite(tau_row['tau_ar1']) else "NA")
        cC.metric("Tau (PSD)", f"{tau_row['tau_psd']:.3f}" if np.isfinite(tau_row['tau_psd']) else "NA")

        st.download_button(
            "Download Kendall Tau (all windows) CSV",
            data=tau_df.to_csv(index=False).encode("utf-8"),
            file_name="kendall_tau_by_window.csv",
            mime="text/csv"
        )

# ============================================================
# TAB 2: Data & Statistics (EDA + ADF)
# ============================================================
with tab2:
    st.subheader("TAB 2 — Data & Statistics (Quality Check)")

    total = len(df_main)
    miss_wl = int(df_main["WL"].isna().sum())
    miss_pct = 100.0 * miss_wl / total if total else np.nan

    cA, cB, cC = st.columns(3)
    cA.metric("Total rows", total)
    cB.metric("Missing WL", miss_wl)
    cC.metric("Missing WL (%)", f"{miss_pct:.2f}%")

    st.markdown("### Descriptive statistics")
    stats = df_main[["WL", "Trend", "Residual"]].describe().T
    st.dataframe(stats, use_container_width=True)
    st.download_button("Download stats CSV", stats.to_csv().encode("utf-8"), "series_stats.csv", "text/csv")

    st.markdown("### Distribution (Histogram) — Water Level")
    hist_bins = st.slider("Histogram bins", 10, 200, 50, 5)
    wl_clean = df_main["WL"].dropna().astype(float)
    fig_h, ax = plt.subplots(figsize=(10, 4))
    ax.hist(wl_clean.to_numpy(), bins=int(hist_bins))
    ax.set_title("Histogram of Water Level", fontweight="bold")
    ax.set_xlabel(str(val_col))
    ax.set_ylabel("Frequency")
    fig_h.tight_layout()
    st.pyplot(fig_h, use_container_width=True)
    st.download_button("Download histogram PNG", fig_to_png_bytes(fig_h), "hist_wl.png", "image/png")

    st.markdown("### Stationarity test (ADF) — Residual")
    if not _HAS_STATS or _adfuller is None:
        st.info("ADF test requires `statsmodels`. If you want this feature, install: statsmodels")
    else:
        resid = df_main["Residual"].dropna().astype(float).to_numpy()
        if resid.size < 50:
            st.warning("Too few residual points for a stable ADF test.")
        else:
            try:
                adf_stat, pval, usedlag, nobs, crit, _icbest = _adfuller(resid, autolag="AIC")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("ADF Statistic", f"{adf_stat:.4f}")
                c2.metric("p-value", f"{pval:.4g}")
                c3.metric("Used lags", int(usedlag))
                c4.metric("N obs", int(nobs))
                st.write("Critical values:", crit)
                st.caption("Interpretation: p-value < 0.05 suggests stationarity (often residual should be closer to stationary).")
            except Exception as e:
                st.error(f"ADF failed: {e}")

# ============================================================
# TAB 3: Lead Time Analysis (threshold crossing → peak)
# ============================================================
with tab3:
    st.subheader("TAB 3 — Lead Time Analysis")
    st.caption(
        "This estimates how early the warning happens before the next flood peak. "
        "Peaks are detected from WL (not plotted), warnings from indicator threshold crossings."
    )

    available_windows = sorted(res_all["window_hours"].unique().astype(int).tolist())
    default_w = available_windows[0]
    lt_w = st.selectbox("Use rolling window for warnings", available_windows, index=0)

    dfw = res_all[res_all["window_hours"] == int(lt_w)].copy().sort_values("datetime_start")

    # Peak detection settings
    peak_sep = st.slider("Min separation between peaks (hours)", 6, 168, 24, 6)
    lookback = st.slider("Max lookback (hours) before peak to search warning", 6, 24 * 14, 72, 6)

    wl = df_main["WL"].to_numpy(float)
    peaks_idx = simple_peak_indices(wl, min_separation=int(peak_sep))
    if peaks_idx.size == 0:
        st.warning("No peaks detected. Try reducing separation or check your WL series.")
        st.stop()

    peaks = df_main.loc[peaks_idx, [dt_col, "WL"]].rename(columns={dt_col: "peak_datetime", "WL": "peak_WL"}).reset_index(drop=True)

    # Warning timestamps: first time each indicator crosses threshold
    def first_crossing_times(series: pd.Series, thr_val: float) -> pd.Series:
        if np.isnan(thr_val):
            return pd.Series([], dtype="datetime64[ns]")
        s = pd.Series(series).astype(float)
        return dfw.loc[(np.isfinite(s)) & (s >= thr_val), "datetime_start"]

    cross_var = first_crossing_times(dfw["Variance"], thr["var"])
    cross_ar1 = first_crossing_times(dfw["AR1"], thr["ar1"])
    cross_psd = first_crossing_times(dfw["PSD_LowFreq"], thr["psd"])

    # Combine warning times based on rule
    def warning_times():
        enabled = []
        if not np.isnan(thr["var"]): enabled.append(("var", dfw["exceed_var"]))
        if not np.isnan(thr["ar1"]): enabled.append(("ar1", dfw["exceed_ar1"]))
        if not np.isnan(thr["psd"]): enabled.append(("psd", dfw["exceed_psd"]))
        if not enabled:
            return []

        if warn_rule == "All indicators exceed":
            cond = np.logical_and.reduce([b.to_numpy() for _, b in enabled])
        else:
            cond = np.logical_or.reduce([b.to_numpy() for _, b in enabled])

        return dfw.loc[cond, "datetime_start"].dropna().sort_values().to_list()

    warn_times = warning_times()

    rows = []
    for _, r in peaks.iterrows():
        peak_dt = pd.to_datetime(r["peak_datetime"])
        start_dt = peak_dt - pd.Timedelta(hours=int(lookback))

        candidates = [t for t in warn_times if (t >= start_dt) and (t <= peak_dt)]
        if not candidates:
            rows.append({
                "peak_datetime": peak_dt,
                "peak_WL": r["peak_WL"],
                "warning_datetime": pd.NaT,
                "lead_time_hours": np.nan
            })
        else:
            warn_dt = candidates[0]
            lead_hr = (peak_dt - warn_dt) / pd.Timedelta(hours=1)
            rows.append({
                "peak_datetime": peak_dt,
                "peak_WL": r["peak_WL"],
                "warning_datetime": warn_dt,
                "lead_time_hours": float(lead_hr)
            })

    lead_df = pd.DataFrame(rows)
    st.dataframe(lead_df, use_container_width=True)

    # Performance metrics (simple event-based)
    # Event is "peak exists" (always yes). We check if warning detected within lookback.
    TP = int(lead_df["lead_time_hours"].notna().sum())  # peaks warned
    FN = int(lead_df["lead_time_hours"].isna().sum())   # peaks not warned

    # False positives: warnings not followed by a peak within horizon
    # Define horizon = lookback hours after warning must contain a peak
    peak_times = pd.to_datetime(peaks["peak_datetime"]).sort_values().to_list()
    FP = 0
    for wdt in warn_times:
        wdt = pd.to_datetime(wdt)
        end_dt = wdt + pd.Timedelta(hours=int(lookback))
        has_peak = any((pt >= wdt) and (pt <= end_dt) for pt in peak_times)
        if not has_peak:
            FP += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Positives (TP)", TP)
    c2.metric("False Negatives (FN)", FN)
    c3.metric("False Positives (FP)", FP)
    c4.metric("Recall (TP/(TP+FN))", f"{recall:.2f}" if np.isfinite(recall) else "NA")

    st.caption("FP here = warnings not followed by any peak within the selected horizon (simple definition).")

    st.download_button(
        "Download lead time table CSV",
        data=lead_df.to_csv(index=False).encode("utf-8"),
        file_name=f"lead_time_{lt_w}h.csv",
        mime="text/csv"
    )

# ============================================================
# TAB 4: Memory Analysis (ACF + return map + red noise hint)
# ============================================================
with tab4:
    st.subheader("TAB 4 — Memory Analysis")
    st.caption("Memory in CSD is mainly autocorrelation/persistence. We show ACF + return map Xt vs Xt-1.")

    resid = df_main["Residual"].dropna().astype(float).to_numpy()
    if resid.size < 50:
        st.warning("Too few residual points for memory analysis.")
        st.stop()

    max_lag = st.slider("ACF max lag", 10, 200, 48, 2)

    # ACF calculation (no statsmodels required)
    def acf_np(x, nlags):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        x = x - np.mean(x)
        n = len(x)
        if n < nlags + 2:
            nlags = max(1, n - 2)
        denom = np.dot(x, x)
        out = [1.0]
        for k in range(1, nlags + 1):
            out.append(float(np.dot(x[:-k], x[k:]) / denom) if denom != 0 else np.nan)
        return np.array(out)

    acf_vals = acf_np(resid, int(max_lag))
    lags = np.arange(len(acf_vals))

    st.markdown("### ACF Plot (Correlogram)")
    fig_acf = go.Figure()
    fig_acf.add_trace(go.Bar(
        x=lags, y=acf_vals,
        hovertemplate="lag: %{x}<br>ACF: %{y:.4f}<extra></extra>",
        name="ACF"
    ))
    fig_acf.update_layout(height=360, xaxis_title="Lag (hours)", yaxis_title="ACF", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_acf, use_container_width=True)
    st.download_button("Download HTML (ACF)", fig_to_html_bytes(fig_acf), "acf.html", "text/html")

    # Return map Xt vs Xt-1
    st.markdown("### Return Map (Xₜ vs Xₜ₋₁)")
    x_t = resid[1:]
    x_tm1 = resid[:-1]
    fig_ret = go.Figure()
    fig_ret.add_trace(go.Scatter(
        x=x_tm1, y=x_t, mode="markers",
        hovertemplate="X(t-1): %{x:.4f}<br>X(t): %{y:.4f}<extra></extra>",
        name="Return map"
    ))
    fig_ret.update_layout(height=420, xaxis_title="X(t-1)", yaxis_title="X(t)", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_ret, use_container_width=True)
    st.download_button("Download HTML (Return map)", fig_to_html_bytes(fig_ret), "return_map.html", "text/html")

    # Red noise hint: compare PSD of residual vs white noise
    st.markdown("### Red Noise Hint (PSD comparison)")
    n = min(4096, resid.size)
    resid_seg = resid[-n:]
    white = np.random.normal(0, np.std(resid_seg), size=n)

    def psd_curve(arr):
        arr = arr - np.mean(arr)
        if _HAS_SCIPY and scipy_welch is not None:
            f, p = scipy_welch(arr, fs=1.0, nperseg=min(256, len(arr)))
            return f, p
        # FFT fallback
        spec = np.abs(np.fft.rfft(arr)) ** 2
        f = np.fft.rfftfreq(len(arr), d=1.0)
        return f, spec

    f1, p1 = psd_curve(resid_seg)
    f2, p2 = psd_curve(white)

    fig_psd = go.Figure()
    fig_psd.add_trace(go.Scatter(x=f1, y=p1, mode="lines", name="Residual PSD"))
    fig_psd.add_trace(go.Scatter(x=f2, y=p2, mode="lines", name="White noise PSD"))
    fig_psd.update_layout(
        height=380,
        xaxis_title="Frequency (1/hour)",
        yaxis_title="Power",
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig_psd, use_container_width=True)
    st.download_button("Download HTML (PSD compare)", fig_to_html_bytes(fig_psd), "red_noise_psd_compare.html", "text/html")

# Logs
with st.expander("Messages / Logs", expanded=False):
    logs = st.session_state.get("log", ["Upload a CSV and click Run Analysis."])
    st.code("\n".join(logs), language="text")

st.caption("CSD/EWS workflow: Trend extraction → Residual → Rolling Variance/AR1/PSD → Kendall Tau → Lead time + Memory analysis.")
