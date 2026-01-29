# app.py
# ============================================================
# Streamlit Dashboard: Early Warning System (Critical Slowing Down)
# - Upload CSV
# - Select datetime + value column
# - Step 1: Trend extraction (rolling mean/median) + residual series
# - Step 2: CSD rolling indicators on RESIDUAL (Variance / PSD low-freq / AR1)
# - Dashboard-style layout (like your friend's):
#     * Water level + trend + (optional) peak markers
#     * Residual series
#     * Variance + threshold + warnings
#     * PSD low-freq + threshold + warnings
#     * AR1 + threshold + warnings
#
# Exports:
# - Interactive HTML (dashboard)
# - PNG (dashboard snapshot) via Matplotlib for individual charts
# - CSV + ZIP outputs (per window)
#
# Key UX:
# - Hover tooltip shows FULL datetime (date + time)
# - Robust datetime parse (explicit formats first; dayfirst fallback)
# - Robust CSV load (encodings + separators)
# ============================================================

import io
import re
import zipfile
import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Optional: better PSD via Welch if SciPy exists
try:
    from scipy.signal import welch as scipy_welch
    _HAS_SCIPY = True
except Exception:
    scipy_welch = None
    _HAS_SCIPY = False


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

    # final fallback: separator auto-detect, forgiving encoding
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

    # timezone handling (safe)
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


# -----------------------------
# Step 1: Trend extraction
# -----------------------------
def extract_trend(series: pd.Series, win_hours: int, method: str = "mean") -> pd.Series:
    """
    Centered rolling trend; keeps index aligned.
    method: 'mean' or 'median'
    """
    s = series.astype(float)
    win = max(3, int(win_hours))
    if method == "median":
        return s.rolling(win, center=True, min_periods=max(2, win // 3)).median()
    return s.rolling(win, center=True, min_periods=max(2, win // 3)).mean()


# -----------------------------
# CSD helpers
# -----------------------------
def calc_ar1(w: np.ndarray) -> float:
    w = w[np.isfinite(w)]
    if w.size < 2:
        return np.nan
    if np.nanstd(w) == 0:
        return np.nan
    return float(np.corrcoef(w[:-1], w[1:])[0, 1])


def psd_lowfreq_fft(w: np.ndarray, low_k: int = 8) -> float:
    """
    Low-frequency PSD proxy via rFFT power (excludes DC).
    """
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
    """
    Welch PSD low-frequency mean (first ~10% of non-zero frequencies).
    Requires SciPy.
    """
    w = w[np.isfinite(w)]
    if w.size < 8:
        return np.nan
    w = w - np.nanmean(w)
    freqs, pxx = scipy_welch(w, fs=fs, nperseg=min(256, w.size))
    if len(freqs) < 3:
        return np.nan
    # exclude DC (freq=0), take first 10% bins (at least 2 bins)
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

        if psd_method == "Welch" and _HAS_SCIPY:
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


# -----------------------------
# Peak marker (simple)
# -----------------------------
def simple_peak_indices(x: np.ndarray, min_separation: int = 24) -> np.ndarray:
    """
    Very simple local maxima with minimum separation.
    Not a scientific peak detector—just for dashboard markers.
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


# -----------------------------
# Plot helpers (FULL datetime hover)
# -----------------------------
HOVER_FMT = "%Y-%m-%d %H:%M:%S"

def plot_line(df: pd.DataFrame, xcol: str, ycol: str, name: str,
              hover_y_label: str = None) -> go.Scatter:
    lbl = hover_y_label or ycol
    return go.Scatter(
        x=df[xcol],
        y=df[ycol],
        mode="lines",
        name=name,
        hovertemplate=f"datetime: %{{x|{HOVER_FMT}}}<br>{lbl}: %{{y:.6g}}<extra></extra>"
    )

def add_threshold(fig: go.Figure, y: float, row: int, col: int, label: str):
    if y is None or (isinstance(y, float) and np.isnan(y)):
        return
    fig.add_hline(y=y, line_width=1.5, line_dash="dash", row=row, col=col,
                  annotation_text=label, annotation_position="top left")


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="CSD Early Warning Dashboard", layout="wide")
st.title("Early Warning System — Critical Slowing Down (CSD) Dashboard")

with st.sidebar:
    st.header("1) Configuration")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    st.divider()
    st.header("2) Columns & Parsing")
    tz = st.selectbox("Time zone", ["Asia/Kuala_Lumpur", "UTC"], index=0)

    st.divider()
    st.header("3) Trend Extraction (Step 1)")
    trend_method = st.selectbox("Trend method", ["mean", "median"], index=0)
    trend_sigma_days = st.slider("Trend smoothing (Sigma) in DAYS", 1, 30, 7, 1)
    st.caption("Sigma days ≈ rolling window length (days) for trend line.")

    st.divider()
    st.header("4) CSD Rolling Indicators")
    window_choices = [6, 12, 24, 36, 48, 72, 96, 120, 144, 168]
    windows = st.multiselect("Rolling windows (hours)", options=window_choices, default=[24, 72, 168])
    valid_frac = st.slider("Min valid fraction per window", 0.50, 1.00, 0.80, 0.05)

    psd_method = st.selectbox(
        "PSD method",
        options=(["Welch (SciPy)"] if _HAS_SCIPY else []) + ["FFT (fast)"],
        index=0
    )
    psd_method_key = "Welch" if psd_method.startswith("Welch") else "FFT"

    st.divider()
    st.header("5) Thresholds & Warning")
    st.caption("Leave blank to disable a threshold.")
    thr_var = st.text_input("Variance threshold", value="")
    thr_psd = st.text_input("PSD_LowFreq threshold", value="")
    thr_ar1 = st.text_input("AR1 threshold", value="")

    warn_rule = st.selectbox("Warning rule", ["Any indicator exceeds", "All indicators exceed"], index=0)

    st.divider()
    st.header("6) Optional Markers")
    show_peaks = st.checkbox("Mark simple WL peaks", value=True)
    peak_sep_hours = st.slider("Min peak separation (hours)", 6, 168, 24, 6)

    run = st.button("Run Dashboard", type="primary")


def _to_float_or_nan(s: str) -> float:
    s = (s or "").strip()
    if s == "":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


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

if run:
    st.session_state["log"] = []
    add_log("---- Run started ----")
    add_log(f"File: {uploaded.name}")
    add_log(f"Datetime col: {dt_col} | Value col: {val_col}")
    add_log(f"Timezone: {tz}")
    add_log(f"Trend: {trend_method}, sigma_days={trend_sigma_days}")
    add_log(f"Windows: {', '.join(map(str, windows))} hours | valid_frac={valid_frac}")
    add_log(f"PSD method: {psd_method_key}")
    add_log("---- Parsing data ----")

    df = raw_df.copy()

    # Parse datetime + numeric
    df[dt_col] = parse_dt_series(df[dt_col], tz=tz)
    df[val_col] = pd.to_numeric(df[val_col].replace("", np.nan), errors="coerce")

    df = df[df[dt_col].notna()].sort_values(dt_col).reset_index(drop=True)

    # Create trend + residual
    sigma_hours = int(trend_sigma_days * 24)
    wl = df[val_col].astype(float)
    trend = extract_trend(wl, win_hours=sigma_hours, method=trend_method)
    resid = wl - trend

    df["WL"] = wl
    df["Trend"] = trend
    df["Residual"] = resid

    # Log ranges
    add_log(f"Parsed datetime range: {df[dt_col].iloc[0]}  to  {df[dt_col].iloc[-1]}")
    add_log(f"Rows after parse: {len(df)}")

    # Compute CSD on residual (recommended)
    dt = df[dt_col]
    x = df["Residual"].to_numpy(dtype=float)

    res_list = []
    for w in sorted(map(int, windows)):
        add_log(f"Processing window: {w} hours")
        out = csd_roll(
            x=x, dt=dt, window_hours=w, valid_frac=float(valid_frac),
            psd_method=psd_method_key, low_k_cap=8
        )
        if not out.empty:
            res_list.append(out)

    if not res_list:
        add_log("No results produced (too few rows / too many missing).")
        st.error("No results produced. Try smaller windows or reduce 'Min valid fraction per window'.")
        st.stop()

    res_all = pd.concat(res_list, ignore_index=True)
    res_all["datetime_start"] = pd.to_datetime(res_all["datetime_start"], errors="coerce")
    res_all["datetime_end"] = pd.to_datetime(res_all["datetime_end"], errors="coerce")

    # Thresholds
    t_var = _to_float_or_nan(thr_var)
    t_psd = _to_float_or_nan(thr_psd)
    t_ar1 = _to_float_or_nan(thr_ar1)

    # Determine warnings per-row (per window result row)
    def exceeds(row) -> dict:
        v_ok = (not np.isnan(t_var)) and np.isfinite(row["Variance"]) and (row["Variance"] >= t_var)
        p_ok = (not np.isnan(t_psd)) and np.isfinite(row["PSD_LowFreq"]) and (row["PSD_LowFreq"] >= t_psd)
        a_ok = (not np.isnan(t_ar1)) and np.isfinite(row["AR1"]) and (row["AR1"] >= t_ar1)
        return {"var": v_ok, "psd": p_ok, "ar1": a_ok}

    flags = res_all.apply(exceeds, axis=1, result_type="expand")
    res_all["exceed_var"] = flags["var"].fillna(False)
    res_all["exceed_psd"] = flags["psd"].fillna(False)
    res_all["exceed_ar1"] = flags["ar1"].fillna(False)

    enabled_count = int(not np.isnan(t_var)) + int(not np.isnan(t_psd)) + int(not np.isnan(t_ar1))
    if enabled_count == 0:
        res_all["warning"] = False
    else:
        if warn_rule == "All indicators exceed":
            # only for enabled thresholds
            conds = []
            if not np.isnan(t_var): conds.append(res_all["exceed_var"])
            if not np.isnan(t_psd): conds.append(res_all["exceed_psd"])
            if not np.isnan(t_ar1): conds.append(res_all["exceed_ar1"])
            res_all["warning"] = np.logical_and.reduce(conds)
        else:
            res_all["warning"] = res_all[["exceed_var", "exceed_psd", "exceed_ar1"]].any(axis=1)

    add_log(f"Done. CSD rows: {len(res_all)}")
    add_log("---- Run finished ----")

    st.session_state["df_main"] = df
    st.session_state["dt_col"] = dt_col
    st.session_state["val_col"] = val_col
    st.session_state["res_all"] = res_all
    st.session_state["thr"] = {"var": t_var, "psd": t_psd, "ar1": t_ar1}
    st.session_state["psd_method_key"] = psd_method_key
    st.session_state["sigma_hours"] = sigma_hours

# Use previous results if exist
df_main = st.session_state.get("df_main", None)
res_all = st.session_state.get("res_all", None)
if df_main is None or res_all is None:
    st.warning("Click **Run Dashboard** to compute indicators.")
    st.stop()

dt_col = st.session_state["dt_col"]
val_col = st.session_state["val_col"]
thr = st.session_state["thr"]
sigma_hours = st.session_state["sigma_hours"]

# -----------------------------
# Tabs (Dashboard-first)
# -----------------------------
tab_dash, tab_data, tab_msgs = st.tabs(["Dashboard", "Data & Downloads", "Messages"])

with tab_dash:
    st.subheader("Dashboard View")

    available_windows = sorted(res_all["window_hours"].unique().astype(int).tolist())
    default_w = 72 if 72 in available_windows else available_windows[0]

    plot_w = st.selectbox("Select rolling window (hours)", options=available_windows, index=available_windows.index(default_w))

    # Filter results for this window
    dfw = res_all[res_all["window_hours"] == int(plot_w)].copy()
    if dfw.empty:
        st.error("No data for this window.")
        st.stop()

    # Build a dashboard-like multi-panel plot
    fig = make_subplots(
        rows=3, cols=2,
        shared_xaxes=False,
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
        subplot_titles=(
            f"Water Level + Trend (Sigma={sigma_hours//24}d)", "Residual (WL - Trend)",
            f"Variance (window={plot_w}h)", f"PSD Low-Freq (window={plot_w}h)",
            f"AR1 (window={plot_w}h)", "Warnings (indicator exceedance)"
        )
    )

    # Panel 1: WL + trend
    fig.add_trace(go.Scatter(
        x=df_main[dt_col], y=df_main["WL"], mode="lines",
        name="Water Level",
        hovertemplate=f"datetime: %{{x|{HOVER_FMT}}}<br>WL: %{{y:.6g}}<extra></extra>"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_main[dt_col], y=df_main["Trend"], mode="lines",
        name="Trend",
        hovertemplate=f"datetime: %{{x|{HOVER_FMT}}}<br>Trend: %{{y:.6g}}<extra></extra>"
    ), row=1, col=1)

    # Optional peaks
    if show_peaks:
        idx = simple_peak_indices(df_main["WL"].to_numpy(float), min_separation=int(peak_sep_hours))
        if idx.size > 0:
            fig.add_trace(go.Scatter(
                x=df_main.loc[idx, dt_col],
                y=df_main.loc[idx, "WL"],
                mode="markers",
                name="WL peaks",
                hovertemplate=f"datetime: %{{x|{HOVER_FMT}}}<br>WL peak: %{{y:.6g}}<extra></extra>"
            ), row=1, col=1)

    # Panel 2: Residual
    fig.add_trace(go.Scatter(
        x=df_main[dt_col], y=df_main["Residual"], mode="lines",
        name="Residual",
        hovertemplate=f"datetime: %{{x|{HOVER_FMT}}}<br>Residual: %{{y:.6g}}<extra></extra>"
    ), row=1, col=2)

    # Panel 3: Variance
    fig.add_trace(plot_line(dfw, "datetime_start", "Variance", "Variance", "Variance"), row=2, col=1)
    add_threshold(fig, thr["var"], row=2, col=1, label="Variance threshold")

    # Panel 4: PSD
    fig.add_trace(plot_line(dfw, "datetime_start", "PSD_LowFreq", "PSD_LowFreq", "PSD_LowFreq"), row=2, col=2)
    add_threshold(fig, thr["psd"], row=2, col=2, label="PSD threshold")

    # Panel 5: AR1
    fig.add_trace(plot_line(dfw, "datetime_start", "AR1", "AR1", "AR1"), row=3, col=1)
    fig.add_hline(y=0, line_width=1, line_dash="dash", row=3, col=1)
    add_threshold(fig, thr["ar1"], row=3, col=1, label="AR1 threshold")

    # Panel 6: Warnings as markers (where warning=True)
    warn_pts = dfw[dfw["warning"] == True].copy()
    if warn_pts.empty:
        fig.add_trace(go.Scatter(
            x=dfw["datetime_start"], y=np.zeros(len(dfw)),
            mode="lines", name="No warnings",
            hoverinfo="skip", showlegend=False
        ), row=3, col=2)
    else:
        # plot warning markers along y=1 line
        fig.add_trace(go.Scatter(
            x=warn_pts["datetime_start"],
            y=np.ones(len(warn_pts)),
            mode="markers",
            name="Warning",
            hovertemplate=(
                f"datetime: %{{x|{HOVER_FMT}}}<br>"
                f"warning: TRUE<br>"
                f"exceed_var: %{{customdata[0]}}<br>"
                f"exceed_psd: %{{customdata[1]}}<br>"
                f"exceed_ar1: %{{customdata[2]}}<extra></extra>"
            ),
            customdata=np.stack([warn_pts["exceed_var"], warn_pts["exceed_psd"], warn_pts["exceed_ar1"]], axis=1)
        ), row=3, col=2)

    fig.update_layout(
        height=980,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=70, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0)
    )
    # Ensure hoverformat
    fig.update_xaxes(hoverformat=HOVER_FMT)

    st.plotly_chart(fig, use_container_width=True)

    # Download dashboard HTML
    html_bytes = fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
    st.download_button(
        "Download interactive dashboard HTML",
        data=html_bytes,
        file_name=f"CSD_Dashboard_{safe_name(val_col)}_{plot_w}h.html",
        mime="text/html"
    )

    # Quick summary
    cA, cB, cC, cD = st.columns(4)
    with cA:
        st.metric("Window (h)", int(plot_w))
    with cB:
        st.metric("Warnings (count)", int(dfw["warning"].sum()))
    with cC:
        st.metric("Variance exceed", int(dfw["exceed_var"].sum()))
    with cD:
        st.metric("AR1 exceed", int(dfw["exceed_ar1"].sum()))

with tab_data:
    st.subheader("Data & Downloads")

    st.markdown("### CSD Results (all windows)")
    st.dataframe(res_all, use_container_width=True, height=520)

    combined_csv = res_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download combined CSD CSV",
        data=combined_csv,
        file_name=f"CSD_{safe_name(val_col)}_combined.csv",
        mime="text/csv"
    )

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for w in sorted(res_all["window_hours"].unique()):
            dfwin = res_all[res_all["window_hours"] == w]
            zf.writestr(f"CSD_{int(w)}h.csv", dfwin.to_csv(index=False))
    zip_buf.seek(0)

    st.download_button(
        "Download per-window ZIP (CSVs)",
        data=zip_buf.getvalue(),
        file_name=f"CSD_{safe_name(val_col)}_per_window.zip",
        mime="application/zip"
    )

    st.markdown("### Parsed series (WL / Trend / Residual)")
    st.dataframe(df_main[[dt_col, "WL", "Trend", "Residual"]], use_container_width=True, height=350)

    parsed_csv = df_main[[dt_col, "WL", "Trend", "Residual"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download parsed series CSV (WL+Trend+Residual)",
        data=parsed_csv,
        file_name=f"Parsed_{safe_name(val_col)}_WL_Trend_Residual.csv",
        mime="text/csv"
    )

with tab_msgs:
    st.subheader("Messages")
    logs = st.session_state.get("log", ["Upload a CSV to begin."])
    st.code("\n".join(logs), language="text")

st.caption("CSD dashboard: Trend extraction → residual → rolling Variance / PSD low-frequency / AR1 + thresholds + warnings.")
