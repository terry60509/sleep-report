"""睡眠報告的 Plotly 圖表 — 純繪圖,不依賴 Streamlit,app 與 HTML 匯出共用。"""

from __future__ import annotations

from datetime import timedelta

import plotly.graph_objects as go

from sleep_core import NORMS, STAGES, STAGE_COLORS, STAGE_Y, STAGE_ZH, fmt_hm

BAND_MARKS = [  # (名稱, 下限Hz, 上限Hz)
    ("δ", 0.5, 4), ("θ", 4, 8), ("α", 8, 12), ("σ", 12, 16), ("β", 16, 30),
]


def stage_runs(stages):
    runs, cur, start = [], stages[0], 0
    for i in range(1, len(stages)):
        if stages[i] != cur:
            runs.append((start, i, cur))
            cur, start = stages[i], i
    runs.append((start, len(stages), cur))
    return runs


def draw_hypnogram(stages, epoch_len, start_time):
    runs = stage_runs(stages)
    fig = go.Figure()
    for rs, re, stg in runs:
        t0 = start_time + timedelta(seconds=rs * epoch_len)
        t1 = start_time + timedelta(seconds=re * epoch_len)
        y = STAGE_Y.get(stg, 0)
        color = STAGE_COLORS.get(stg, "#8A8F98")
        fig.add_shape(
            type="rect", x0=t0, x1=t1, y0=y - 0.38, y1=y + 0.38,
            fillcolor=color, opacity=0.7, line_width=0, layer="below",
        )
    times = [start_time + timedelta(seconds=i * epoch_len) for i in range(len(stages))]
    ys = [STAGE_Y.get(s, 0) for s in stages]
    fig.add_trace(go.Scatter(
        x=times, y=ys, mode="lines", line=dict(color="rgba(80,80,80,.45)", width=1, shape="hv"),
        hovertemplate="%{x|%H:%M}<br>%{text}<extra></extra>",
        text=[STAGE_ZH.get(s, s) for s in stages],
    ))
    fig.update_layout(
        height=220, margin=dict(l=60, r=20, t=10, b=30),
        yaxis=dict(
            tickvals=[1, 2, 3, 4, 5], ticktext=["N3", "N2", "N1", "REM", "W"],
            range=[0.3, 5.7], fixedrange=True,
        ),
        xaxis=dict(tickformat="%H:%M", dtick=3600000),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False, dragmode="pan",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,.1)")
    return fig


def draw_stage_pie(metrics):
    sleep_stages = ["N1", "N2", "N3", "REM"]
    vals = [metrics["stage_pct"][s] for s in sleep_stages]
    labels = [STAGE_ZH[s] for s in sleep_stages]
    colors = [STAGE_COLORS[s] for s in sleep_stages]
    fig = go.Figure(go.Pie(
        labels=labels, values=vals, hole=0.52,
        marker=dict(colors=colors),
        textinfo="label+percent", textposition="outside",
        textfont=dict(size=12),
        hovertemplate="%{label}<br>%{value:.1f}%<br>%{customdata}<extra></extra>",
        customdata=[fmt_hm(metrics["stage_min"][s]) for s in sleep_stages],
    ))
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=f'<b>{fmt_hm(metrics["tst"])}</b><br><span style="font-size:11px">總睡眠</span>',
            x=0.5, y=0.5, font_size=18, showarrow=False,
        )],
    )
    return fig


def draw_norm_comparison(metrics):
    items = [
        ("N1 %", metrics["stage_pct"]["N1"], NORMS["n1_pct"], STAGE_COLORS["N1"]),
        ("N2 %", metrics["stage_pct"]["N2"], NORMS["n2_pct"], STAGE_COLORS["N2"]),
        ("N3 %", metrics["stage_pct"]["N3"], NORMS["n3_pct"], STAGE_COLORS["N3"]),
        ("REM %", metrics["stage_pct"]["REM"], NORMS["rem_pct"], STAGE_COLORS["REM"]),
        ("SE %", metrics["se"], NORMS["se"], "#5BA85A"),
    ]
    fig = go.Figure()
    names = [it[0] for it in items]
    for i, (name, val, (lo, hi), color) in enumerate(items):
        fig.add_shape(
            type="rect", x0=lo, x1=hi, y0=i - 0.3, y1=i + 0.3,
            fillcolor="rgba(128,128,128,.12)", line_width=0,
        )
        fig.add_trace(go.Scatter(
            x=[val], y=[i], mode="markers+text",
            marker=dict(size=14, color=color, symbol="diamond"),
            text=[f"{val:.1f}"], textposition="top center", textfont=dict(size=11),
            hovertemplate=f"{name}: {val:.1f}%<br>正常範圍: {lo}-{hi}%<extra></extra>",
            showlegend=False,
        ))
    fig.update_layout(
        height=300, margin=dict(l=60, r=20, t=10, b=30),
        yaxis=dict(tickvals=list(range(len(names))), ticktext=names, fixedrange=True),
        xaxis=dict(title="百分比 (%)", range=[0, max(60, max(it[1] for it in items) + 5)]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,.12)")
    fig.update_yaxes(showgrid=False)
    return fig


def draw_cycles(cycles, epoch_len, start_time):
    if not cycles:
        return None
    fig = go.Figure()
    for c in cycles:
        fig.add_trace(go.Bar(
            x=[c["nrem_min"]], y=[f"Cycle {c['number']}"], orientation="h",
            marker_color="#2557A0", name="NREM", showlegend=(c["number"] == 1),
            text=[f"NREM {fmt_hm(c['nrem_min'])}"], textposition="inside",
            textfont=dict(color="white", size=11),
            hovertemplate=f"NREM: {fmt_hm(c['nrem_min'])}<extra></extra>",
        ))
        if c["rem_min"] > 0:
            fig.add_trace(go.Bar(
                x=[c["rem_min"]], y=[f"Cycle {c['number']}"], orientation="h",
                marker_color="#C6427F", name="REM", showlegend=(c["number"] == 1),
                text=[f"REM {fmt_hm(c['rem_min'])}"], textposition="inside",
                textfont=dict(color="white", size=11),
                hovertemplate=f"REM: {fmt_hm(c['rem_min'])}<extra></extra>",
            ))
    fig.update_layout(
        barmode="stack", height=50 + len(cycles) * 50,
        margin=dict(l=70, r=20, t=10, b=30),
        xaxis=dict(title="分鐘"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,.12)")
    fig.update_yaxes(showgrid=False)
    return fig


def draw_spectra(spectra):
    freqs, psds, ch = spectra
    fig = go.Figure()
    for name, lo, hi in BAND_MARKS:
        fig.add_vrect(x0=lo, x1=hi, fillcolor="rgba(128,128,128,.05)", line_width=0,
                      annotation_text=name, annotation_position="top",
                      annotation_font=dict(size=11, color="#8A8F98"))
    for s in STAGES:
        if s not in psds:
            continue
        fig.add_trace(go.Scatter(
            x=freqs, y=psds[s], mode="lines", name=STAGE_ZH.get(s, s),
            line=dict(color=STAGE_COLORS[s], width=2),
            hovertemplate=f"{STAGE_ZH.get(s, s)}<br>%{{x:.2f}} Hz<br>%{{y:.2f}} µV²/Hz<extra></extra>",
        ))
    fig.update_layout(
        height=340, margin=dict(l=60, r=20, t=24, b=40),
        xaxis=dict(title="頻率 (Hz)", range=[0, freqs[-1]]),
        yaxis=dict(title="功率密度 (µV²/Hz)", type="log"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,.12)")
    return fig, ch
