"""睡眠報告分析引擎 — 解析 EDF/CSV、計算睡眠指標、偵測睡眠週期。"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_Y = {"W": 5, "REM": 4, "N1": 3, "N2": 2, "N3": 1, "?": 0}
STAGE_ZH = {"W": "清醒", "N1": "N1 淺睡", "N2": "N2 淺睡", "N3": "N3 深睡", "REM": "REM 快速動眼"}
STAGE_COLORS = {
    "W": "#E07B39", "N1": "#63B3E8", "N2": "#3A82C4",
    "N3": "#1A4B8C", "REM": "#C6427F", "?": "#8A8F98",
}
NORMS = {
    "tst": (420, 540), "se": (85, 100), "sol": (0, 30), "waso": (0, 30),
    "n1_pct": (2, 5), "n2_pct": (45, 55), "n3_pct": (13, 23), "rem_pct": (20, 25),
}


def normalize_stage(desc: str) -> str:
    d = str(desc).strip().lower().replace("sleep stage", "").strip()
    if d.startswith("movement"):
        return "W"
    table = {
        "w": "W", "wake": "W", "0": "W",
        "1": "N1", "n1": "N1", "s1": "N1",
        "2": "N2", "n2": "N2", "s2": "N2",
        "3": "N3", "n3": "N3", "s3": "N3",
        "4": "N3", "n4": "N3", "s4": "N3",
        "r": "REM", "rem": "REM", "5": "REM",
    }
    return table.get(d, "?")


# ================================================================ EDF

def parse_edf(signal_path: str | Path, hypno_path: str | Path, epoch_len: float = 30.0):
    import mne
    mne.set_log_level("ERROR")
    signal_path, hypno_path = Path(signal_path), Path(hypno_path)

    raw = mne.io.read_raw_edf(str(signal_path), preload=False, verbose="ERROR")
    total_sec = raw.n_times / float(raw.info["sfreq"])
    n_epochs = max(1, int(np.ceil(total_sec / epoch_len)))

    meas_date = raw.info.get("meas_date")
    start_time = meas_date.replace(tzinfo=None) if meas_date else datetime(2000, 1, 1)

    annot = mne.read_annotations(str(hypno_path))
    shift = 0.0
    if annot.orig_time is not None and meas_date is not None:
        shift = (annot.orig_time - meas_date).total_seconds()

    onsets = np.asarray(annot.onset, dtype=float) + shift
    durations = np.asarray(annot.duration, dtype=float)
    labels = np.array([normalize_stage(d) for d in annot.description], dtype=object)
    order = np.argsort(onsets, kind="stable")
    onsets, durations, labels = onsets[order], durations[order], labels[order]

    centers = (np.arange(n_epochs) + 0.5) * epoch_len
    idx = np.searchsorted(onsets, centers, side="right") - 1
    stages = []
    for i in range(n_epochs):
        if idx[i] >= 0 and centers[i] < onsets[idx[i]] + durations[idx[i]]:
            stages.append(str(labels[idx[i]]))
        else:
            stages.append("?")
    return stages, epoch_len, start_time, signal_path.stem


def find_hypnogram(signal_path: Path) -> Path | None:
    stem = signal_path.stem
    for suffix in ("-PSG", "_PSG"):
        if stem.upper().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    d = signal_path.parent
    exact = d / f"{stem}-Hypnogram.edf"
    if exact.is_file():
        return exact
    best, best_score = None, 0
    for cand in sorted(d.glob("*.edf")):
        if "hypnogram" not in cand.name.lower():
            continue
        score = sum(1 for a, b in zip(stem.lower(), cand.stem.lower()) if a == b)
        if score > best_score:
            best, best_score = cand, score
    return best if best and best_score >= min(6, len(stem)) else None


def stage_spectra(
    signal_path: str | Path,
    stages: list[str],
    epoch_len: float = 30.0,
    max_epochs_per_stage: int = 40,
    fmax: float = 30.0,
):
    """各睡眠階段的 EEG 平均功率頻譜(Welch)。

    每個分期最多抽 max_epochs_per_stage 個 epoch(均勻取樣)平均,
    避免整夜資料全部載入。回傳 (freqs, {stage: psd}, channel_name);
    psd 單位為 µV²/Hz(原始單位是電壓時)。
    """
    import mne
    from scipy.signal import welch

    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_edf(str(signal_path), preload=False, verbose="ERROR")
    ch = next((c for c in raw.ch_names if "eeg" in c.lower()), raw.ch_names[0])
    pick = raw.ch_names.index(ch)
    sfreq = float(raw.info["sfreq"])
    samples_per_epoch = int(round(epoch_len * sfreq))

    # MNE 會把可辨識的電壓單位一律換算成伏特 → ×1e6 顯示成 µV;其他單位不動。
    orig = ((getattr(raw, "_orig_units", None) or {}).get(ch, "") or "").strip()
    key = orig.lower().replace("μ", "µ")
    voltage = key in ("µv", "uv", "microvolt", "microvolts",
                      "mv", "millivolt", "millivolts", "v", "volt", "volts")
    scale = 1e6 if voltage else 1.0

    arr = np.array(stages)
    nperseg = min(int(4 * sfreq), samples_per_epoch)
    freqs = None
    psds: dict[str, np.ndarray] = {}
    for stage in STAGES:
        idx = np.flatnonzero(arr == stage)
        idx = idx[(idx + 1) * samples_per_epoch <= raw.n_times]
        if idx.size == 0:
            continue
        if idx.size > max_epochs_per_stage:
            sel = np.linspace(0, idx.size - 1, max_epochs_per_stage).round().astype(int)
            idx = idx[np.unique(sel)]
        acc = []
        for e in idx:
            start = int(e) * samples_per_epoch
            y = raw.get_data(picks=[pick], start=start, stop=start + samples_per_epoch)[0] * scale
            f, p = welch(y, fs=sfreq, nperseg=nperseg)
            acc.append(p)
        keep = f <= fmax
        freqs = f[keep]
        psds[stage] = np.mean(np.stack(acc), axis=0)[keep]
    return freqs, psds, ch


# ================================================================ CSV

def parse_csv(content: str, epoch_len: float = 30.0):
    lines = content.strip().splitlines()
    if not lines:
        raise ValueError("CSV 檔案為空")
    first = lines[0].strip().lower()
    has_header = any(h in first for h in ("stage", "epoch", "time", "sleep", "phase"))
    data_lines = lines[1:] if has_header else lines

    stages, start_time = [], datetime(2000, 1, 1, 22, 0, 0)
    for li, line in enumerate(data_lines):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip().strip('"').strip("'") for p in re.split(r"[,\t;]", line)]
        if len(parts) == 1:
            stages.append(normalize_stage(parts[0]))
        else:
            found = False
            for p in reversed(parts):
                s = normalize_stage(p)
                if s != "?":
                    stages.append(s)
                    found = True
                    break
            if not found:
                stages.append("?")
            if li == 0:
                for p in parts:
                    try:
                        t = datetime.strptime(p, "%H:%M:%S")
                        start_time = start_time.replace(hour=t.hour, minute=t.minute, second=t.second)
                        break
                    except ValueError:
                        try:
                            start_time = datetime.fromisoformat(p)
                            break
                        except ValueError:
                            pass
    if not stages:
        raise ValueError("無法從 CSV 中解析出睡眠分期資料")
    return stages, epoch_len, start_time, "CSV 上傳"


# ================================================================ 指標計算

def compute_metrics(stages: list[str], epoch_len: float) -> dict:
    arr = np.array(stages)
    n = len(arr)
    epoch_min = epoch_len / 60.0
    counts = {s: int(np.sum(arr == s)) for s in STAGES}
    tib = n * epoch_min

    sleep_idx = np.where((arr != "W") & (arr != "?"))[0]
    if len(sleep_idx) == 0:
        return dict(
            tib=tib, tst=0, se=0, sol=tib, waso=0,
            n_awakenings=0, rem_latency=0,
            stage_min={s: 0.0 for s in STAGES},
            stage_pct={s: 0.0 for s in STAGES},
            ratings={},
        )

    onset, end = int(sleep_idx[0]), int(sleep_idx[-1]) + 1
    sol = onset * epoch_min
    sp = arr[onset:end]
    tst_epochs = int(np.sum((sp != "W") & (sp != "?")))
    tst = tst_epochs * epoch_min
    waso = float(np.sum(sp == "W")) * epoch_min
    se = (tst / tib * 100) if tib > 0 else 0

    n_awakenings = 0
    for i in range(1, len(sp)):
        if sp[i] == "W" and sp[i - 1] != "W":
            n_awakenings += 1

    rem_idx = np.where(arr[onset:] == "REM")[0]
    rem_latency = float(rem_idx[0]) * epoch_min if len(rem_idx) > 0 else 0

    stage_min = {s: counts[s] * epoch_min for s in STAGES}
    stage_pct = {}
    for s in STAGES:
        if s == "W":
            stage_pct[s] = (stage_min[s] / tib * 100) if tib > 0 else 0
        else:
            stage_pct[s] = (stage_min[s] / tst * 100) if tst > 0 else 0

    ratings = {}
    metric_map = {
        "tst": tst, "se": se, "sol": sol, "waso": waso,
        "n1_pct": stage_pct["N1"], "n2_pct": stage_pct["N2"],
        "n3_pct": stage_pct["N3"], "rem_pct": stage_pct["REM"],
    }
    higher_better = {"tst", "se", "n3_pct", "rem_pct"}
    for key, val in metric_map.items():
        lo, hi = NORMS[key]
        if lo <= val <= hi:
            ratings[key] = "normal"
        elif key in higher_better and val > hi:
            ratings[key] = "good"
        elif key not in higher_better and val < lo:
            ratings[key] = "good"
        else:
            ratings[key] = "warning" if abs(val - (lo + hi) / 2) < (hi - lo) else "alert"

    return dict(
        tib=tib, tst=tst, se=se, sol=sol, waso=waso,
        n_awakenings=n_awakenings, rem_latency=rem_latency,
        stage_min=stage_min, stage_pct=stage_pct, ratings=ratings,
    )


# ================================================================ 睡眠週期

def detect_cycles(stages: list[str], epoch_len: float) -> list[dict]:
    arr = np.array(stages)
    epoch_min = epoch_len / 60.0
    sleep_idx = np.where((arr != "W") & (arr != "?"))[0]
    if len(sleep_idx) == 0:
        return []

    onset, end = int(sleep_idx[0]), int(sleep_idx[-1]) + 1
    cycles, i, num = [], onset, 0

    while i < end:
        nrem_start = None
        while i < end:
            if arr[i] in ("N1", "N2", "N3"):
                nrem_start = i
                break
            i += 1
        if nrem_start is None:
            break

        j, nrem_end, wake_run = nrem_start, nrem_start, 0
        while j < end:
            if arr[j] in ("N1", "N2", "N3"):
                nrem_end = j + 1
                wake_run = 0
            elif arr[j] == "W":
                wake_run += 1
                if wake_run > 3:
                    break
            elif arr[j] == "REM":
                break
            j += 1

        rem_start, rem_end, wake_run = None, nrem_end, 0
        j = nrem_end
        while j < end:
            if arr[j] == "REM":
                if rem_start is None:
                    rem_start = j
                rem_end = j + 1
                wake_run = 0
            elif arr[j] == "W":
                wake_run += 1
                if wake_run > 3 and rem_start is not None:
                    break
            elif arr[j] in ("N1", "N2", "N3"):
                if rem_start is not None:
                    break
            j += 1

        nrem_dur = (nrem_end - nrem_start) * epoch_min
        rem_dur = ((rem_end - rem_start) * epoch_min) if rem_start else 0
        if nrem_dur >= 15:
            num += 1
            cycles.append(dict(
                number=num,
                start_epoch=nrem_start, end_epoch=rem_end if rem_start else nrem_end,
                nrem_min=nrem_dur, rem_min=rem_dur, total_min=nrem_dur + rem_dur,
            ))
        i = rem_end if rem_start else nrem_end
        if i <= nrem_start:
            i = nrem_start + 1
    return cycles


# ================================================================ Demo

def generate_demo_night(start_time: datetime | None = None, seed: int = 42):
    rng = np.random.RandomState(seed)
    if start_time is None:
        start_time = datetime(2024, 1, 15, 22, 30, 0)
    epoch_len = 30.0
    segments = [
        ("W", 15),
        ("N1", 5), ("N2", 20), ("N3", 35), ("N2", 10), ("N1", 3), ("REM", 12), ("W", 2),
        ("N1", 3), ("N2", 22), ("N3", 28), ("N2", 12), ("REM", 25), ("W", 2),
        ("N1", 3), ("N2", 25), ("N3", 15), ("N2", 12), ("REM", 30), ("W", 2),
        ("N2", 20), ("N3", 8), ("N2", 15), ("REM", 35), ("W", 1),
        ("N1", 4), ("N2", 25), ("REM", 30),
        ("N2", 10), ("N1", 5), ("W", 10),
    ]
    stages = []
    for stage, dur_min in segments:
        n = max(1, int(dur_min * 60 / epoch_len) + rng.randint(-1, 2))
        stages.extend([stage] * n)

    for _ in range(rng.randint(2, 5)):
        pos = rng.randint(30, len(stages) - 20)
        for j in range(rng.randint(1, 3)):
            if pos + j < len(stages):
                stages[pos + j] = "W"
    return stages, epoch_len, start_time, "Demo 範例資料"


def generate_multi_night(n_nights: int = 7):
    nights = []
    for i in range(n_nights):
        dt = datetime(2024, 1, 10 + i, 22, 15 + (i * 7) % 30)
        stages, el, st, name = generate_demo_night(start_time=dt, seed=42 + i)
        nights.append((stages, el, st, f"Night {i + 1}  ({dt:%m/%d})"))
    return nights


def fmt_hm(minutes: float) -> str:
    h, m = divmod(int(round(minutes)), 60)
    return f"{h}h {m:02d}m"
