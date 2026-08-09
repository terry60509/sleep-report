# 睡眠品質報告

用 Streamlit 從 PSG（EDF）或分期 CSV 自動產出睡眠品質報告：Hypnogram、分期比例圓餅圖、
睡眠效率 / 入睡潛伏期 / REM 潛伏期 / WASO / 覺醒次數、睡眠週期分析、各階段 EEG 功率頻譜，
以及多夜趨勢比較。可直接列印成 PDF。

## 啟動

```bash
pip install -r requirements.txt
python3 -m streamlit run app.py
```

或在 Finder 雙擊 `睡眠報告.command`。

## 資料來源

| 模式 | 說明 |
|---|---|
| Demo 範例 | 內建模擬整夜資料，不需任何檔案 |
| EDF 範例檔 | 讀取 `../EDF檢視器/` 下的 Sleep-EDF 檔（SC4002） |
| 上傳 EDF | 上傳 PSG 訊號檔 + Hypnogram 分期檔（只傳訊號檔會自動配對分期檔） |
| 上傳 CSV | 每行一個 epoch 的分期（W / N1 / N2 / N3 / REM），可多檔做多夜趨勢 |

測試資料可從 PhysioNet 的 [Sleep-EDF Database](https://physionet.org/content/sleep-edfx/1.0.0/) 下載：

```bash
wget https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4002E0-PSG.edf
wget https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4002EC-Hypnogram.edf
```

## 報告內容

- **指標卡**：TST、睡眠效率（SE）、入睡潛伏期（SOL）、WASO、覺醒次數，附正常值評級
- **Hypnogram**：整夜睡眠結構圖（依分期上色）
- **分期比例**：N1/N2/N3/REM 圓餅圖 + 與正常範圍比較圖
- **睡眠週期**：自動偵測 NREM–REM 週期，堆疊長條圖
- **各階段 EEG 功率頻譜**：Welch PSD，各分期每 stage 均勻抽最多 40 個 epoch 平均，
  對數座標並標示 δ/θ/α/σ/β 頻帶（僅 EDF 來源有原始訊號時顯示）
- **詳細數據表**：所有指標 + 正常範圍對照
- **多夜趨勢**：TST、SE、SOL、WASO、各期佔比逐夜變化
- **匯出**：頁尾「列印報告 / 匯出 PDF」按鈕（瀏覽器列印，已附 print CSS）

## 實作備註

- EDF 讀取用 `mne`（`preload=False`，功率頻譜只逐 epoch 抽讀，整夜大檔也不吃記憶體）
- 分期標註相容 R&K（S1–S4，S4 併入 N3）與 AASM（N1–N3）命名
- MNE 會把電壓單位換算成伏特，頻譜顯示前 ×1e6 還原成 µV²/Hz
