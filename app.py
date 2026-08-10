"""睡眠品質報告 — Streamlit App"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from charts import (
    draw_cycles, draw_hypnogram, draw_norm_comparison, draw_spectra, draw_stage_pie,
)
from sleep_core import (
    NORMS, STAGES, STAGE_COLORS, STAGE_Y, STAGE_ZH,
    compute_metrics, detect_cycles, find_hypnogram, fmt_hm,
    generate_demo_night, generate_multi_night, parse_csv, parse_edf,
    stage_spectra,
)

st.set_page_config(page_title="睡眠報告", page_icon="🌙", layout="wide")

UPLOAD_DIR = Path(__file__).resolve().parent / ".uploads"

st.markdown("""
<style>
  .block-container {padding-top:1.2rem; padding-bottom:1rem;}
  div[data-testid="stHorizontalBlock"] {gap:.35rem;}
  .metric-box {
    border-radius:12px; padding:18px 14px 12px; text-align:center;
    border:1px solid rgba(128,128,128,.15);
  }
  .metric-val {font-size:2rem; font-weight:700; line-height:1.15;}
  .metric-label {font-size:.78rem; color:#8A8F98; margin-top:2px;}
  .metric-badge {
    display:inline-block; font-size:.7rem; padding:1px 8px;
    border-radius:9px; margin-top:4px; font-weight:600;
  }
  .badge-normal  {background:#d4edda; color:#155724;}
  .badge-good    {background:#d1ecf1; color:#0c5460;}
  .badge-warning {background:#fff3cd; color:#856404;}
  .badge-alert   {background:#f8d7da; color:#721c24;}
  @media print {
    [data-testid="stSidebar"],[data-testid="stHeader"],
    button,.stDeployButton {display:none !important;}
    .block-container {padding:0!important;max-width:100%!important;}
  }
</style>
""", unsafe_allow_html=True)

RATING_LABEL = {"normal": "正常", "good": "優良", "warning": "偏低", "alert": "注意"}


# ================================================================ Sidebar

def save_upload(uploaded) -> Path:
    UPLOAD_DIR.mkdir(exist_ok=True)
    digest = hashlib.md5(uploaded.getbuffer()).hexdigest()[:10]
    target = UPLOAD_DIR / f"{digest}-{uploaded.name}"
    if not target.exists():
        target.write_bytes(uploaded.getbuffer())
    return target


with st.sidebar:
    st.markdown("### 🌙 睡眠報告")
    tab_choice = st.radio("模式", ["單夜報告", "多夜趨勢"], horizontal=True)
    st.divider()

    if tab_choice == "單夜報告":
        source = st.radio("資料來源", ["Demo 範例", "EDF 範例檔 (SC4002)", "上傳 EDF", "上傳 CSV"])
    else:
        source = st.radio("資料來源", ["Demo 範例 (7 夜)", "上傳多個 CSV"])


# ================================================================ 單夜報告

def render_metric_card(label, value, unit="", rating=None):
    badge = ""
    if rating:
        badge = f'<div class="metric-badge badge-{rating}">{RATING_LABEL.get(rating, rating)}</div>'
    st.markdown(
        f'<div class="metric-box">'
        f'<div class="metric-val">{value}<span style="font-size:.9rem;font-weight:400;"> {unit}</span></div>'
        f'<div class="metric-label">{label}</div>{badge}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="計算各階段功率頻譜…")
def load_spectra(path_str: str, mtime_ns: int, stages_key: tuple, epoch_len: float):
    return stage_spectra(path_str, list(stages_key), epoch_len)


def render_single_report(stages, epoch_len, start_time, name, spectra=None):
    metrics = compute_metrics(stages, epoch_len)
    cycles = detect_cycles(stages, epoch_len)
    end_time = start_time + timedelta(seconds=len(stages) * epoch_len)

    st.markdown(
        f"<h2 style='margin-bottom:0;'>🌙 睡眠品質報告</h2>"
        f"<p style='color:#8A8F98;margin-top:2px;'>"
        f"{name} · {start_time:%Y-%m-%d %H:%M} – {end_time:%H:%M} · "
        f"紀錄時長 {fmt_hm(metrics['tib'])}</p>",
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    with cols[0]:
        render_metric_card("總睡眠時間 TST", fmt_hm(metrics["tst"]), "", metrics["ratings"].get("tst"))
    with cols[1]:
        render_metric_card("睡眠效率 SE", f"{metrics['se']:.1f}", "%", metrics["ratings"].get("se"))
    with cols[2]:
        render_metric_card("入睡潛伏期 SOL", f"{metrics['sol']:.0f}", "min", metrics["ratings"].get("sol"))
    with cols[3]:
        render_metric_card("睡後醒來 WASO", f"{metrics['waso']:.0f}", "min", metrics["ratings"].get("waso"))
    with cols[4]:
        render_metric_card("覺醒次數", f"{metrics['n_awakenings']}", "次")

    st.markdown("#### 睡眠結構圖 Hypnogram")
    st.plotly_chart(draw_hypnogram(stages, epoch_len, start_time), use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 睡眠分期比例")
        st.plotly_chart(draw_stage_pie(metrics), use_container_width=True, config={"displaylogo": False})
    with c2:
        st.markdown("#### 與正常值比較")
        st.plotly_chart(draw_norm_comparison(metrics), use_container_width=True, config={"displaylogo": False})

    if cycles:
        st.markdown(f"#### 睡眠週期分析（共 {len(cycles)} 個週期）")
        fig_c = draw_cycles(cycles, epoch_len, start_time)
        if fig_c:
            st.plotly_chart(fig_c, use_container_width=True, config={"displaylogo": False})

    if spectra is not None and spectra[0] is not None and spectra[1]:
        fig_sp, ch = draw_spectra(spectra)
        st.markdown(f"#### 各階段 EEG 功率頻譜（{ch}）")
        st.plotly_chart(fig_sp, use_container_width=True, config={"displaylogo": False})

    st.markdown("#### 詳細數據")
    detail = {
        "指標": [
            "紀錄時長 TIB", "總睡眠時間 TST", "睡眠效率 SE",
            "入睡潛伏期 SOL", "睡後醒來 WASO", "覺醒次數",
            "REM 潛伏期", "睡眠週期數",
            "── N1 時間", "── N2 時間", "── N3 深睡時間", "── REM 時間", "── 清醒時間",
            "── N1 佔比 (TST)", "── N2 佔比 (TST)", "── N3 佔比 (TST)", "── REM 佔比 (TST)",
        ],
        "數值": [
            fmt_hm(metrics["tib"]), fmt_hm(metrics["tst"]), f"{metrics['se']:.1f} %",
            f"{metrics['sol']:.1f} min", f"{metrics['waso']:.1f} min", str(metrics["n_awakenings"]),
            fmt_hm(metrics["rem_latency"]), str(len(cycles)),
            fmt_hm(metrics["stage_min"]["N1"]), fmt_hm(metrics["stage_min"]["N2"]),
            fmt_hm(metrics["stage_min"]["N3"]), fmt_hm(metrics["stage_min"]["REM"]),
            fmt_hm(metrics["stage_min"]["W"]),
            f"{metrics['stage_pct']['N1']:.1f} %", f"{metrics['stage_pct']['N2']:.1f} %",
            f"{metrics['stage_pct']['N3']:.1f} %", f"{metrics['stage_pct']['REM']:.1f} %",
        ],
        "正常範圍": [
            "—", "7h–9h", "> 85 %",
            "< 30 min", "< 30 min", "—",
            "—", "4–6",
            "—", "—", "—", "—", "—",
            "2–5 %", "45–55 %", "13–23 %", "20–25 %",
        ],
    }
    st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True, height=660)

    st.markdown(
        '<div style="text-align:center;margin-top:1rem;">'
        '<button onclick="window.print()" style="padding:8px 24px;border-radius:8px;'
        'border:1px solid #888;background:transparent;cursor:pointer;font-size:.9rem;">'
        '🖨 列印報告 / 匯出 PDF</button></div>',
        unsafe_allow_html=True,
    )
    return metrics


# ================================================================ 多夜趨勢

def render_multi_night(nights_data):
    records = []
    for stages, epoch_len, start_time, name in nights_data:
        m = compute_metrics(stages, epoch_len)
        m["name"] = name
        m["date"] = start_time
        records.append(m)

    df = pd.DataFrame(records)

    st.markdown("<h2>🌙 多夜睡眠趨勢</h2>", unsafe_allow_html=True)
    st.caption(f"共 {len(records)} 晚紀錄")

    fig_tst = go.Figure()
    fig_tst.add_trace(go.Scatter(
        x=df["date"], y=df["tst"] / 60, mode="lines+markers",
        line=dict(color="#3A82C4", width=2), marker=dict(size=8),
        hovertemplate="%{x|%m/%d}<br>TST: %{y:.1f}h<extra></extra>",
    ))
    lo, hi = NORMS["tst"][0] / 60, NORMS["tst"][1] / 60
    fig_tst.add_hrect(y0=lo, y1=hi, fillcolor="rgba(90,168,90,.1)", line_width=0)
    fig_tst.update_layout(
        title="總睡眠時間 (小時)", height=280,
        margin=dict(l=50, r=20, t=40, b=30),
        yaxis_title="小時", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig_tst.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,.12)")
    fig_tst.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,.12)")

    fig_se = go.Figure()
    fig_se.add_trace(go.Scatter(
        x=df["date"], y=df["se"], mode="lines+markers",
        line=dict(color="#5BA85A", width=2), marker=dict(size=8),
        hovertemplate="%{x|%m/%d}<br>SE: %{y:.1f}%<extra></extra>",
    ))
    fig_se.add_hrect(y0=NORMS["se"][0], y1=100, fillcolor="rgba(90,168,90,.1)", line_width=0)
    fig_se.update_layout(
        title="睡眠效率 (%)", height=280,
        margin=dict(l=50, r=20, t=40, b=30),
        yaxis_title="%", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig_se.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,.12)")
    fig_se.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,.12)")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_tst, use_container_width=True, config={"displaylogo": False})
    with c2:
        st.plotly_chart(fig_se, use_container_width=True, config={"displaylogo": False})

    fig_stages = go.Figure()
    for s in ["N1", "N2", "N3", "REM"]:
        pcts = [r["stage_pct"][s] for r in records]
        fig_stages.add_trace(go.Bar(
            x=[r["date"] for r in records], y=pcts,
            name=STAGE_ZH[s], marker_color=STAGE_COLORS[s],
            hovertemplate=f"{STAGE_ZH[s]}: %{{y:.1f}}%<extra></extra>",
        ))
    fig_stages.update_layout(
        barmode="stack", title="各期佔比趨勢", height=320,
        margin=dict(l=50, r=20, t=40, b=30),
        yaxis_title="% of TST", legend=dict(orientation="h", y=1.12),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig_stages.update_xaxes(showgrid=False)
    fig_stages.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,.12)")
    st.plotly_chart(fig_stages, use_container_width=True, config={"displaylogo": False})

    c3, c4 = st.columns(2)
    fig_sol = go.Figure()
    fig_sol.add_trace(go.Scatter(
        x=df["date"], y=df["sol"], mode="lines+markers",
        line=dict(color="#E07B39", width=2), marker=dict(size=8),
    ))
    fig_sol.add_hrect(y0=0, y1=NORMS["sol"][1], fillcolor="rgba(90,168,90,.1)", line_width=0)
    fig_sol.update_layout(
        title="入睡潛伏期 (min)", height=260,
        margin=dict(l=50, r=20, t=40, b=30),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig_sol.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,.12)")
    fig_sol.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,.12)")

    fig_waso = go.Figure()
    fig_waso.add_trace(go.Scatter(
        x=df["date"], y=df["waso"], mode="lines+markers",
        line=dict(color="#C6427F", width=2), marker=dict(size=8),
    ))
    fig_waso.add_hrect(y0=0, y1=NORMS["waso"][1], fillcolor="rgba(90,168,90,.1)", line_width=0)
    fig_waso.update_layout(
        title="睡後醒來 WASO (min)", height=260,
        margin=dict(l=50, r=20, t=40, b=30),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig_waso.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,.12)")
    fig_waso.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,.12)")

    with c3:
        st.plotly_chart(fig_sol, use_container_width=True, config={"displaylogo": False})
    with c4:
        st.plotly_chart(fig_waso, use_container_width=True, config={"displaylogo": False})

    st.markdown("#### 每夜摘要")
    summary = []
    for r in records:
        summary.append({
            "日期": r["date"].strftime("%m/%d %H:%M"),
            "名稱": r["name"],
            "TST": fmt_hm(r["tst"]),
            "SE": f"{r['se']:.1f}%",
            "SOL": f"{r['sol']:.0f}m",
            "WASO": f"{r['waso']:.0f}m",
            "N3%": f"{r['stage_pct']['N3']:.1f}",
            "REM%": f"{r['stage_pct']['REM']:.1f}",
        })
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)


# ================================================================ Main

if tab_choice == "單夜報告":
    stages = None
    edf_signal: Path | None = None  # 有原始訊號才能算功率頻譜
    if source == "Demo 範例":
        stages, epoch_len, start_time, name = generate_demo_night()
    elif source == "EDF 範例檔 (SC4002)":
        edf_dir = Path(__file__).resolve().parent.parent / "EDF檢視器"
        sig = edf_dir / "SC4002E0-PSG.edf"
        hyp = edf_dir / "SC4002EC-Hypnogram.edf"
        if sig.is_file() and hyp.is_file():
            try:
                stages, epoch_len, start_time, name = parse_edf(sig, hyp)
                edf_signal = sig
            except Exception as e:
                st.error(f"EDF 讀取失敗：{e}")
        else:
            st.error(f"找不到 EDF 範例檔：{edf_dir}")
    elif source == "上傳 EDF":
        with st.sidebar:
            sig_file = st.file_uploader("訊號檔 (.edf)", type=["edf"], key="edf_sig")
            hyp_file = st.file_uploader("分期檔 Hypnogram (.edf)", type=["edf"], key="edf_hyp")
        if sig_file and hyp_file:
            sig_path = save_upload(sig_file)
            hyp_path = save_upload(hyp_file)
            try:
                stages, epoch_len, start_time, name = parse_edf(sig_path, hyp_path)
                edf_signal = sig_path
            except Exception as e:
                st.error(f"EDF 讀取失敗：{e}")
        elif sig_file and not hyp_file:
            sig_path = save_upload(sig_file)
            hyp_path = find_hypnogram(sig_path)
            if hyp_path:
                st.sidebar.caption(f"自動找到分期檔：{hyp_path.name}")
                try:
                    stages, epoch_len, start_time, name = parse_edf(sig_path, hyp_path)
                    edf_signal = sig_path
                except Exception as e:
                    st.error(f"EDF 讀取失敗：{e}")
            else:
                st.info("請上傳對應的 Hypnogram 分期檔。")
        else:
            st.info("請在左側上傳 EDF 訊號檔與分期檔。")
    elif source == "上傳 CSV":
        with st.sidebar:
            csv_file = st.file_uploader("CSV 檔案", type=["csv", "txt"], key="csv_up")
            st.caption("格式：每行一個 epoch 的分期\n(W / N1 / N2 / N3 / REM)")
        if csv_file:
            try:
                content = csv_file.getvalue().decode("utf-8")
                stages, epoch_len, start_time, name = parse_csv(content)
            except Exception as e:
                st.error(f"CSV 讀取失敗：{e}")
        else:
            st.info("請在左側上傳 CSV 檔案。")

    if stages:
        spectra = None
        if edf_signal is not None:
            try:
                spectra = load_spectra(
                    str(edf_signal), edf_signal.stat().st_mtime_ns, tuple(stages), epoch_len
                )
            except Exception as e:
                st.warning(f"功率頻譜計算失敗：{e}")
        render_single_report(stages, epoch_len, start_time, name, spectra=spectra)

else:
    if source == "Demo 範例 (7 夜)":
        nights = generate_multi_night(7)
        render_multi_night(nights)
    else:
        with st.sidebar:
            csv_files = st.file_uploader(
                "上傳多個 CSV", type=["csv", "txt"],
                accept_multiple_files=True, key="multi_csv",
            )
        if csv_files:
            nights = []
            for f in csv_files:
                try:
                    content = f.getvalue().decode("utf-8")
                    nights.append(parse_csv(content))
                except Exception as e:
                    st.warning(f"{f.name} 解析失敗：{e}")
            if nights:
                nights.sort(key=lambda x: x[2])
                render_multi_night(nights)
        else:
            st.info("請在左側上傳多個 CSV 檔案以進行趨勢分析。")
