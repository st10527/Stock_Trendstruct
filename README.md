# TX期貨交易儀表板 - 進階版

一個功能完整的期貨交易分析儀表板，包含自動趨勢線檢測和突破點識別功能。

## 🚀 功能特色

- **📊 實時K線圖表**: 支援連續顯示模式，無時間間隙
- **📈 自動趨勢線檢測**: 智能識別支撐線和阻力線
- **🔍 突破點檢測**: 自動標記價格突破點
- **⚡ 搖擺點分析**: 識別重要的高點和低點
- **📋 詳細分析報告**: 完整的技術分析摘要
- **🎛️ 參數可調整**: 靈活的分析參數設定

## 📁 檔案結構

```
tx-futures-dashboard/
├── main_app.py              # 主應用程式
├── data_loader.py           # 資料載入模組
├── trendline_detector.py    # 趨勢線檢測模組
├── chart_visualizer.py      # 圖表視覺化模組
├── output/                  # 資料檔案目錄
│   └── kline_60min.txt     # 期貨資料檔案
└── README.md               # 說明文件
```

## 🛠️ 安裝需求

### Python套件依賴

```bash
pip install streamlit pandas numpy plotly
```

### 資料檔案格式

支援的資料格式（空格分隔的文字檔）：

```
日期                時間     開盤      最高      最低      收盤      成交量    成交值
2024/01/01         09:45:00 15496.000 15530.000 15490.000 15334.000 28339    ...
2024/01/01         10:45:00 15394.000 15532.000 15531.000 15565.000 16505    ...
```

## 🚀 快速開始

### 1. 下載檔案

將所有Python檔案下載到同一個目錄中：
- `main_app.py`
- `data_loader.py`
- `trendline_detector.py`  
- `chart_visualizer.py`

### 2. 準備資料

**選項A: 使用自己的資料**
- 將期貨資料檔案放在 `output/kline_60min.txt`
- 確保格式符合上述要求

**選項B: 使用測試資料**
- 應用程式內建測試資料生成功能
- 可直接使用進行功能測試

### 3. 運行應用程式

```bash
streamlit run main_app.py
```

### 4. 開始使用

1. 在左側邊欄選擇資料來源
2. 調整分析參數（可選）
3. 點擊「載入/重新整理資料」
4. 在主頁面查看分析結果

## 📊 模組說明

### 1. main_app.py - 主應用程式
- Streamlit用戶界面
- 整合所有功能模組
- 參數設定和結果展示

### 2. data_loader.py - 資料載入模組
負責：
- 多編碼格式檔案讀取
- 資料清理和驗證
- OHLCV格式轉換
- 基本指標計算

主要功能：
```python
from data_loader import DataLoader, calculate_basic_metrics

loader = DataLoader()
data = loader.load_from_text_file("output/kline_60min.txt")
metrics = calculate_basic_metrics(data)
```

### 3. trendline_detector.py - 趨勢線檢測模組
負責：
- 搖擺點識別
- 趨勢線構建
- 突破點檢測
- 強度評分計算

主要功能：
```python
from trendline_detector import TrendlineBreakoutDetector

detector = TrendlineBreakoutDetector(
    swing_window=3,
    min_touches=2,
    breakout_threshold=0.005,
    lookback_bars=100
)

analysis = detector.analyze(data)
```

### 4. chart_visualizer.py - 圖表視覺化模組
負責：
- K線圖創建
- 趨勢線繪製
- 突破點標記
- 搖擺點顯示

主要功能：
```python
from chart_visualizer import ChartVisualizer

visualizer = ChartVisualizer(theme='dark')
fig = visualizer.create_trendline_chart(data, analysis)
```

## 🎛️ 參數說明

### 趨勢線分析參數

| 參數名稱 | 預設值 | 說明 |
|---------|--------|------|
| swing_window | 3 | 搖擺點識別視窗大小 |
| min_touches | 2 | 趨勢線最少接觸點數 |
| breakout_threshold | 0.5% | 突破判定閥值 |
| lookback_bars | 100 | 分析的K棒數量 |

### 圖表顯示參數

| 參數名稱 | 預設值 | 說明 |
|---------|--------|------|
| max_trendlines | 3 | 最大顯示趨勢線數 |
| continuous_chart | True | 連續圖表模式 |
| theme | dark | 圖表主題色彩 |

## 🔧 自定義使用

### 單獨使用趨勢線檢測器

```python
from trendline_detector import TrendlineBreakoutDetector
import pandas as pd

# 準備資料 (必須包含: datetime, open, high, low, close, volume)
df = pd.read_csv("your_data.csv")

# 初始化檢測器
detector = TrendlineBreakoutDetector(
    swing_window=5,      # 更大的視窗
    min_touches=3,       # 更嚴格的要求
    breakout_threshold=0.01,  # 1%的突破閥值
    lookback_bars=200    # 分析更多資料
)

# 執行分析
results = detector.analyze(df)

# 獲取結果
support_lines = results['support_lines']
resistance_lines = results['resistance_lines']
breakouts = results['breakouts']
swing_points = results['swing_points']
```

### 批量處理多個檔案

```python
import os
from data_loader import DataLoader
from trendline_detector import TrendlineBreakoutDetector

loader = DataLoader()
detector = TrendlineBreakoutDetector()

# 處理多個檔案
data_dir = "data/"
results = {}

for filename in os.listdir(data_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(data_dir, filename)
        data = loader.load_from_text_file(filepath)
        if data is not None:
            analysis = detector.analyze(data)
            results[filename] = analysis

# 查看結果
for filename, analysis in results.items():
    print(f"{filename}: {len(analysis['breakouts'])} 個突破點")
```

## 🐛 故障排除

### 常見問題

1. **模組導入錯誤**
   - 確保所有Python檔案在同一目錄
   - 檢查檔案名稱是否正確

2. **資料讀取失敗**
   - 確認檔案路徑正確
   - 檢查資料格式是否符合要求
   - 嘗試不同的編碼格式

3. **圖表顯示問題**
   - 確認資料格式正確（OHLC邏輯）
   - 檢查是否有足夠的資料點
   - 嘗試調整參數設定

4. **趨勢線檢測無結果**
   - 增加lookback_bars數量
   - 降低min_touches要求
   - 調整swing_window大小

5. **效能緩慢**
   - 減少lookback_bars數量
   - 使用較少的資料點
   - 考慮資料取樣

### 除錯模式

在檔案開頭添加除錯代碼：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 在各模組中查看詳細日誌
detector = TrendlineBreakoutDetector()
results = detector.analyze(data)
print(f"分析摘要: {results['summary']}")
```

## 📈 進階用法

### 自定義指標添加

在 `data_loader.py` 中添加新的指標計算：

```python
def calculate_custom_indicators(df: pd.DataFrame) -> dict:
    """添加自定義技術指標"""
    
    # 移動平均線
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_50'] = df['close'].rolling(50).mean()
    
    # RSI指標
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta).where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df
```

### 警報系統集成

```python
def check_trading_signals(analysis: dict) -> list:
    """檢查交易信號"""
    signals = []
    
    for breakout in analysis['breakouts']:
        if breakout['strength'] >= 3:  # 強度評分
            signals.append({
                'type': 'strong_breakout',
                'direction': breakout['direction'],
                'price': breakout['price'],
                'confidence': breakout['strength']
            })
    
    return signals
```

### 回測功能

```python
def simple_backtest(data: pd.DataFrame, analysis: dict) -> dict:
    """簡單回測"""
    trades = []
    
    for breakout in analysis['breakouts']:
        entry_price = breakout['price']
        # 假設持有10根K棒
        exit_idx = min(len(data)-1, breakout_idx + 10)
        exit_price = data.iloc[exit_idx]['close']
        
        if breakout['direction'] == 'bullish_breakout':
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price
            
        trades.append({
            'entry': entry_price,
            'exit': exit_price,
            'pnl': pnl,
            'return': pnl / entry_price
        })
    
    return {
        'total_trades': len(trades),
        'total_pnl': sum(t['pnl'] for t in trades),
        'win_rate': len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0
    }
```

## 🔄 更新和維護

### 定期更新資料

建議設置定期任務更新資料：

```python
import schedule
import time

def update_analysis():
    """定期更新分析"""
    loader = DataLoader()
    detector = TrendlineBreakoutDetector()
    
    data = loader.load_from_text_file("output/kline_60min.txt")
    if data is not None:
        analysis = detector.analyze(data)
        # 保存結果或發送通知
        print(f"更新完成：{analysis['summary']['breakouts_count']} 個突破點")

# 每小時更新一次
schedule.every().hour.do(update_analysis)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### 版本控制建議

```bash
git init
git add *.py README.md
git commit -m "Initial commit: TX futures dashboard"
```

## 📝 API文檔

### TrendlineBreakoutDetector

#### 方法：
- `analyze(df)`: 主要分析方法
- `find_swing_points(df)`: 找尋搖擺點
- `find_trendlines(swing_points)`: 構建趨勢線
- `check_breakouts(df, support_lines, resistance_lines)`: 檢查突破

#### 返回格式：
```python
{
    'swing_points': {
        'highs': [(index, datetime, price), ...],
        'lows': [(index, datetime, price), ...]
    },
    'support_lines': [
        {
            'points': [...],
            'slope': float,
            'intercept': float,
            'touches': int,
            'strength_score': float
        }, ...
    ],
    'resistance_lines': [...],
    'breakouts': [
        {
            'datetime': datetime,
            'price': float,
            'direction': str,
            'strength': int,
            'breakout_magnitude': float
        }, ...
    ],
    'summary': {...}
}
```

## 🤝 貢獻指南

歡迎貢獻改進！請遵循以下步驟：

1. Fork這個專案
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交變更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟Pull Request

### 開發建議

- 添加單元測試
- 保持代碼註釋清晰
- 遵循PEP 8編碼規範
- 更新相關文檔

## 📄 許可證

本專案採用MIT許可證 - 詳見LICENSE文件

## 📧 聯絡方式

如有問題或建議，請聯絡：
- Email: your.email@example.com
- GitHub Issues: [專案Issues頁面]

## 🙏 致謝

感謝以下開源專案：
- [Streamlit](https://streamlit.io/) - Web應用框架
- [Plotly](https://plotly.com/) - 互動式圖表
- [Pandas](https://pandas.pydata.org/) - 資料處理
- [NumPy](https://numpy.org/) - 數值計算

---

**⚠️ 風險提示：本工具僅供教育和研究用途，不構成投資建議。交易有風險，投資需謹慎。**
