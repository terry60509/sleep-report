"""從 EDF 產出單檔 HTML 睡眠報告(不需開 Streamlit)。

用法:
    python3 export_report.py [PSG.edf] [Hypnogram.edf] [輸出.html]

省略引數時,預設讀 ../EDF檢視器/ 下的 SC4002 範例檔,輸出 sample_report.html。
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.io as pio
from plotly.offline import get_plotlyjs

from charts import (
    draw_cycles, draw_hypnogram, draw_norm_comparison, draw_spectra, draw_stage_pie,
)
from sleep_core import (
    compute_metrics, detect_cycles, fmt_hm, parse_edf, stage_spectra,
)

RATING_LABEL = {"normal": "正常", "good": "優良", "warning": "偏低", "alert": "注意"}
RATING_COLOR = {
    "normal": ("#d4edda", "#155724"), "good": ("#d1ecf1", "#0c5460"),
    "warning": ("#fff3cd", "#856404"), "alert": ("#f8d7da", "#721c24"),
}

CSS = """
body {font-family: -apple-system, "PingFang TC", "Microsoft JhengHei", sans-serif;
      max-width: 1080px; margin: 0 auto; padding: 24px 20px 60px; color: #1a1a2e;}
h1 {margin-bottom: 2px;} h3 {margin: 28px 0 6px;}
.sub {color: #8A8F98; margin-top: 0;}
.cards {display: flex; gap: 10px; margin: 18px 0;}
.card {flex: 1; border: 1px solid rgba(128,128,128,.2); border-radius: 12px;
       padding: 16px 12px 12px; text-align: center;}
.card .val {font-size: 1.7rem; font-weight: 700;}
.card .val small {font-size: .85rem; font-weight: 400;}
.card .label {font-size: .78rem; color: #8A8F98; margin-top: 2px;}
.badge {display: inline-block; font-size: .7rem; padding: 1px 8px; border-radius: 9px;
        margin-top: 4px; font-weight: 600;}
.row {display: flex; gap: 16px;} .row > div {flex: 1; min-width: 0;}
table.detail {border-collapse: collapse; width: 100%; font-size: .88rem;}
table.detail th, table.detail td {border-bottom: 1px solid rgba(128,128,128,.18);
        padding: 6px 10px; text-align: left;}
table.detail th {color: #8A8F98; font-weight: 600;}
@media print {body {max-width: 100%;}}
"""


def card(label: str, value: str, unit: str, rating: str | None) -> str:
    badge = ""
    if rating:
        bg, fg = RATING_COLOR.get(rating, ("#eee", "#333"))
        badge = (f'<div class="badge" style="background:{bg};color:{fg};">'
                 f"{RATING_LABEL.get(rating, rating)}</div>")
    unit_html = f"<small> {unit}</small>" if unit else ""
    return (f'<div class="card"><div class="val">{value}{unit_html}</div>'
            f'<div class="label">{label}</div>{badge}</div>')


def fig_html(fig) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"displaylogo": False})


def main() -> None:
    base = Path(__file__).resolve().parent
    default_dir = base.parent / "EDF檢視器"
    sig = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir / "SC4002E0-PSG.edf"
    hyp = Path(sys.argv[2]) if len(sys.argv) > 2 else default_dir / "SC4002EC-Hypnogram.edf"
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else base / "sample_report.html"

    stages, epoch_len, start_time, name = parse_edf(sig, hyp)
    metrics = compute_metrics(stages, epoch_len)
    cycles = detect_cycles(stages, epoch_len)
    spectra = stage_spectra(sig, stages, epoch_len)
    end_time = start_time.timestamp() + len(stages) * epoch_len

    from datetime import datetime, timedelta
    end_dt = start_time + timedelta(seconds=len(stages) * epoch_len)

    parts = [
        "<h1>🌙 睡眠品質報告</h1>",
        f'<p class="sub">{name} · {start_time:%Y-%m-%d %H:%M} – {end_dt:%H:%M} · '
        f"紀錄時長 {fmt_hm(metrics['tib'])}</p>",
        '<div class="cards">',
        card("總睡眠時間 TST", fmt_hm(metrics["tst"]), "", metrics["ratings"].get("tst")),
        card("睡眠效率 SE", f"{metrics['se']:.1f}", "%", metrics["ratings"].get("se")),
        card("入睡潛伏期 SOL", f"{metrics['sol']:.0f}", "min", metrics["ratings"].get("sol")),
        card("睡後醒來 WASO", f"{metrics['waso']:.0f}", "min", metrics["ratings"].get("waso")),
        card("覺醒次數", str(metrics["n_awakenings"]), "次", None),
        "</div>",
        "<h3>睡眠結構圖 Hypnogram</h3>",
        fig_html(draw_hypnogram(stages, epoch_len, start_time)),
        '<div class="row"><div><h3>睡眠分期比例</h3>',
        fig_html(draw_stage_pie(metrics)),
        "</div><div><h3>與正常值比較</h3>",
        fig_html(draw_norm_comparison(metrics)),
        "</div></div>",
    ]

    if cycles:
        fig_c = draw_cycles(cycles, epoch_len, start_time)
        parts += [f"<h3>睡眠週期分析(共 {len(cycles)} 個週期)</h3>", fig_html(fig_c)]

    if spectra[0] is not None and spectra[1]:
        fig_sp, ch = draw_spectra(spectra)
        parts += [f"<h3>各階段 EEG 功率頻譜({ch})</h3>", fig_html(fig_sp)]

    detail_rows = [
        ("紀錄時長 TIB", fmt_hm(metrics["tib"]), "—"),
        ("總睡眠時間 TST", fmt_hm(metrics["tst"]), "7h–9h"),
        ("睡眠效率 SE", f"{metrics['se']:.1f} %", "> 85 %"),
        ("入睡潛伏期 SOL", f"{metrics['sol']:.1f} min", "< 30 min"),
        ("睡後醒來 WASO", f"{metrics['waso']:.1f} min", "< 30 min"),
        ("覺醒次數", str(metrics["n_awakenings"]), "—"),
        ("REM 潛伏期", fmt_hm(metrics["rem_latency"]), "—"),
        ("睡眠週期數", str(len(cycles)), "4–6"),
        ("N1 佔比 (TST)", f"{metrics['stage_pct']['N1']:.1f} %", "2–5 %"),
        ("N2 佔比 (TST)", f"{metrics['stage_pct']['N2']:.1f} %", "45–55 %"),
        ("N3 佔比 (TST)", f"{metrics['stage_pct']['N3']:.1f} %", "13–23 %"),
        ("REM 佔比 (TST)", f"{metrics['stage_pct']['REM']:.1f} %", "20–25 %"),
    ]
    parts += ["<h3>詳細數據</h3>",
              '<table class="detail"><tr><th>指標</th><th>數值</th><th>正常範圍</th></tr>']
    parts += [f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in detail_rows]
    parts.append("</table>")

    html = (
        "<!DOCTYPE html><html lang='zh-Hant'><head><meta charset='utf-8'>"
        f"<title>睡眠報告 — {name}</title><style>{CSS}</style>"
        f"<script>{get_plotlyjs()}</script></head><body>"
        + "\n".join(parts) + "</body></html>"
    )
    out.write_text(html, encoding="utf-8")
    print(f"已輸出 {out}({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
