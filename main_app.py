"""
TX期貨交易儀表板主應用程式
Author: Your Name
Date: 2024

這是主要的Streamlit應用程式，整合所有模組功能
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# 添加模組路徑
sys.path.append(os.path.dirname(__file__))

# 導入自定義模組
try:
    from data_loader import DataLoader, calculate_basic_metrics, create_test_data
    from trendline_detector import TrendlineBreakoutDetector
    from chart_visualizer import ChartVisualizer, create_metric_cards_html
except ImportError as e:
    st.error(f"無法導入模組: {e}")
    st.error("請確保 data_loader.py, trendline_detector.py, 和 chart_visualizer.py 在同一目錄下")
    st.stop()


# 頁面配置
st.set_page_config(
    page_title="TX期貨交易儀表板 - 進階版",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .metric-container {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ffffff;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #888888;
    }
    .breakout-alert {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .bullish-breakout {
        background-color: #1a4d1a;
        color: #00ff00;
        border: 2px solid #00ff00;
    }
    .bearish-breakdown {
        background-color: #4d1a1a;
        color: #ff4444;
        border: 2px solid #ff4444;
    }
    .analysis-summary {
        background-color: #2d2d30;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
</style>
""", unsafe_allow_html=True)


class TradingDashboard:
    """交易儀表板主類別"""
    
    def __init__(self):
        """初始化儀表板"""
        self.data_loader = DataLoader()
        self.chart_visualizer = ChartVisualizer(theme='dark')
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """初始化session state"""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'trendline_analysis' not in st.session_state:
            st.session_state.trendline_analysis = None
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
    
    def render_sidebar(self):
        """渲染側邊欄"""
        with st.sidebar:
            st.markdown("## ⚙️ 設定")
            
            # 資料載入設定
            st.markdown("### 📊 資料設定")
            data_source = st.selectbox(
                "資料來源",
                ["本地檔案", "測試資料"],
                help="選擇資料來源"
            )
            
            if data_source == "本地檔案":
                file_path = st.text_input(
                    "檔案路徑",
                    value="output/kline_60min.txt",
                    help="輸入資料檔案的路徑"
                )
            else:
                file_path = None
            
            # 趨勢線分析設定
            st.markdown("### 📈 趨勢線分析設定")
            swing_window = st.slider(
                "搖擺點視窗",
                min_value=2, max_value=10, value=3,
                help="用於識別搖擺點的視窗大小"
            )
            
            min_touches = st.slider(
                "最少接觸點",
                min_value=2, max_value=5, value=2,
                help="趨勢線的最少接觸點數量"
            )
            
            breakout_threshold = st.slider(
                "突破閥值 (%)",
                min_value=0.1, max_value=2.0, value=0.5, step=0.1,
                help="突破判定的價格閥值百分比"
            ) / 100
            
            lookback_bars = st.slider(
                "分析K棒數量",
                min_value=50, max_value=500, value=100, step=10,
                help="用於分析的最近K棒數量"
            )
            
            # 圖表設定
            st.markdown("### 🎨 圖表設定")
            max_trendlines = st.slider(
                "最大趨勢線數",
                min_value=1, max_value=5, value=3,
                help="每種類型顯示的最大趨勢線數量"
            )
            
            continuous_chart = st.checkbox(
                "連續圖表",
                value=True,
                help="移除時間間隙，顯示連續的K線圖"
            )
            
            # 載入資料按鈕
            st.markdown("---")
            if st.button("🔄 載入/重新整理資料", type="primary"):
                self.load_data(data_source, file_path, swing_window, min_touches, 
                             breakout_threshold, lookback_bars)
            
            # 資料資訊
            if st.session_state.data is not None:
                st.markdown("### ℹ️ 資料資訊")
                data_info = self.data_loader.get_data_info(st.session_state.data)
                st.markdown(f"**總筆數:** {data_info['total_records']:,}")
                if data_info.get('date_range'):
                    st.markdown(f"**時間範圍:**")
                    st.markdown(f"從 {data_info['date_range']['start'].strftime('%Y-%m-%d %H:%M')}")
                    st.markdown(f"到 {data_info['date_range']['end'].strftime('%Y-%m-%d %H:%M')}")
            
            return {
                'data_source': data_source,
                'file_path': file_path,
                'swing_window': swing_window,
                'min_touches': min_touches,
                'breakout_threshold': breakout_threshold,
                'lookback_bars': lookback_bars,
                'max_trendlines': max_trendlines,
                'continuous_chart': continuous_chart
            }
    
    def load_data(self, data_source: str, file_path: str, swing_window: int,
                  min_touches: int, breakout_threshold: float, lookback_bars: int):
        """載入資料並執行分析"""
        try:
            with st.spinner("載入資料中..."):
                # 載入資料
                if data_source == "測試資料":
                    st.session_state.data = create_test_data(200, 15000, True)
                    st.success("測試資料載入成功！")
                else:
                    st.session_state.data = self.data_loader.load_from_text_file(file_path)
                
                if st.session_state.data is not None:
                    # 執行趨勢線分析
                    with st.spinner("執行趨勢線分析中..."):
                        detector = TrendlineBreakoutDetector(
                            swing_window=swing_window,
                            min_touches=min_touches,
                            breakout_threshold=breakout_threshold,
                            lookback_bars=lookback_bars
                        )
                        st.session_state.trendline_analysis = detector.analyze(st.session_state.data)
                        st.session_state.last_update = datetime.now()
                        st.success("分析完成！")
                
        except Exception as e:
            st.error(f"載入資料時發生錯誤: {str(e)}")
    
    def render_main_content(self, settings: dict):
        """渲染主要內容"""
        # 標題
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: #ffffff; margin: 0;">📈 TX期貨交易儀表板 - 進階版</h1>
            <p style="color: #888888; margin: 0.5rem 0;">含趨勢線分析與突破點檢測</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 檢查是否有資料
        if st.session_state.data is None or st.session_state.trendline_analysis is None:
            st.info("👈 請在左側設定區域載入資料以開始分析")
            return
        
        # 計算基本指標
        metrics = calculate_basic_metrics(st.session_state.data)
        
        # 顯示指標卡片
        st.markdown("## 📊 市場指標")
        metric_html = create_metric_cards_html(metrics)
        st.markdown(metric_html, unsafe_allow_html=True)
        
        # 顯示突破警報
        self.render_breakout_alerts()
        
        # 使用標籤頁組織內容
        tab1, tab2, tab3, tab4 = st.tabs(["📈 主圖表", "🔍 分析詳情", "📋 資料預覽", "⚙️ 設定說明"])
        
        with tab1:
            self.render_main_chart(settings)
        
        with tab2:
            self.render_analysis_details()
        
        with tab3:
            self.render_data_preview()
        
        with tab4:
            self.render_settings_help()
    
    def render_breakout_alerts(self):
        """渲染突破警報"""
        breakouts = st.session_state.trendline_analysis.get('breakouts', [])
        
        if breakouts:
            st.markdown("## 🚨 突破警報")
            for breakout in breakouts:
                alert_class = ("bullish-breakout" if breakout['direction'] == 'bullish_breakout' 
                              else "bearish-breakdown")
                
                direction_text = "看漲突破" if breakout['direction'] == 'bullish_breakout' else "看跌跌破"
                arrow = "⬆️" if breakout['direction'] == 'bullish_breakout' else "⬇️"
                
                st.markdown(f"""
                <div class="breakout-alert {alert_class}">
                    {arrow} <strong>{direction_text}</strong><br>
                    時間: {breakout['datetime'].strftime('%Y-%m-%d %H:%M')}<br>
                    價格: {breakout['price']:.0f}<br>
                    趨勢線強度: {breakout['strength']} 個接觸點<br>
                    突破幅度: {breakout['breakout_magnitude']*100:.2f}%
                </div>
                """, unsafe_allow_html=True)
    
    def render_main_chart(self, settings: dict):
        """渲染主圖表"""
        st.markdown("### 📈 價格圖表與趨勢線分析")
        
        # 創建圖表
        fig = self.chart_visualizer.create_trendline_chart(
            st.session_state.data,
            st.session_state.trendline_analysis,
            max_lines=settings['max_trendlines']
        )
        
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'displaylogo': False
            })
        else:
            st.error("無法創建圖表")
    
    def render_analysis_details(self):
        """渲染分析詳情"""
        st.markdown("### 🔍 趨勢線分析詳情")
        
        analysis = st.session_state.trendline_analysis
        summary = analysis.get('summary', {})
        
        # 分析摘要
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="analysis-summary">
                <h4>📊 搖擺點統計</h4>
                <p>搖擺高點: <strong>{}</strong></p>
                <p>搖擺低點: <strong>{}</strong></p>
                <p>分析視窗: <strong>{} 根K棒</strong></p>
            </div>
            """.format(
                summary.get('swing_highs_count', 0),
                summary.get('swing_lows_count', 0),
                summary.get('swing_window', 0)
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="analysis-summary">
                <h4>📈 趨勢線統計</h4>
                <p>支撐線: <strong>{}</strong></p>
                <p>阻力線: <strong>{}</strong></p>
                <p>突破點: <strong>{}</strong></p>
            </div>
            """.format(
                summary.get('support_lines_count', 0),
                summary.get('resistance_lines_count', 0),
                summary.get('breakouts_count', 0)
            ), unsafe_allow_html=True)
        
        # 趨勢線詳情
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🟢 支撐線詳情")
            support_lines = analysis.get('support_lines', [])
            if support_lines:
                for i, line in enumerate(support_lines[:3], 1):
                    st.markdown(f"""
                    **支撐線 {i}:**
                    - 接觸點: {line['touches']} 個
                    - 強度評分: {line.get('strength_score', 0):.2f}
                    - 斜率: {line['slope']:.6f}
                    """)
            else:
                st.info("未找到支撐線")
        
        with col2:
            st.markdown("#### 🔴 阻力線詳情")
            resistance_lines = analysis.get('resistance_lines', [])
            if resistance_lines:
                for i, line in enumerate(resistance_lines[:3], 1):
                    st.markdown(f"""
                    **阻力線 {i}:**
                    - 接觸點: {line['touches']} 個
                    - 強度評分: {line.get('strength_score', 0):.2f}
                    - 斜率: {line['slope']:.6f}
                    """)
            else:
                st.info("未找到阻力線")
    
    def render_data_preview(self):
        """渲染資料預覽"""
        st.markdown("### 📋 資料預覽")
        
        # 基本統計
        data_info = self.data_loader.get_data_info(st.session_state.data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 資料統計")
            st.markdown(f"""
            - **總筆數:** {data_info['total_records']:,}
            - **時間跨度:** {(data_info['date_range']['end'] - data_info['date_range']['start']).days} 天
            - **最新價格:** {data_info['price_range']['current_price']:.0f}
            - **價格範圍:** {data_info['price_range']['min_low']:.0f} - {data_info['price_range']['max_high']:.0f}
            """)
        
        with col2:
            st.markdown("#### 成交量統計")
            st.markdown(f"""
            - **總成交量:** {data_info['volume_stats']['total_volume']:,.0f}
            - **平均成交量:** {data_info['volume_stats']['avg_volume']:,.0f}
            - **最大成交量:** {data_info['volume_stats']['max_volume']:,.0f}
            """)
        
        # 資料表格預覽
        st.markdown("#### 最新資料 (前10筆)")
        st.dataframe(
            st.session_state.data.head(10)[['datetime', 'open', 'high', 'low', 'close', 'volume']],
            use_container_width=True
        )
        
        # 最新資料 (後10筆)
        st.markdown("#### 最新資料 (後10筆)")
        st.dataframe(
            st.session_state.data.tail(10)[['datetime', 'open', 'high', 'low', 'close', 'volume']],
            use_container_width=True
        )
    
    def render_settings_help(self):
        """渲染設定說明"""
        st.markdown("### ⚙️ 參數設定說明")
        
        st.markdown("""
        #### 🔧 趨勢線分析參數
        
        **搖擺點視窗 (Swing Window)**
        - 用於識別搖擺高點和低點的K棒數量
        - 較小的值會找到更多搖擺點，但可能包含雜訊
        - 較大的值會找到較少但更可靠的搖擺點
        - 建議值: 3-5
        
        **最少接觸點 (Min Touches)**
        - 形成有效趨勢線所需的最少接觸點數量
        - 更多接觸點意味著更強的趨勢線
        - 建議值: 2-3
        
        **突破閥值 (Breakout Threshold)**
        - 判定價格突破趨勢線的百分比閥值
        - 避免因小幅波動而誤判突破
        - 建議值: 0.3%-1.0%
        
        **分析K棒數量 (Lookback Bars)**
        - 用於分析的最近K棒數量
        - 較多的K棒提供更長期的趨勢視角
        - 較少的K棒focus在近期趨勢
        - 建議值: 100-200
        
        #### 📊 圖表參數
        
        **最大趨勢線數**
        - 在圖表上顯示的每種類型趨勢線的最大數量
        - 避免圖表過於混亂
        - 建議值: 2-3
        
        **連續圖表**
        - 移除時間間隙，讓K線緊密相連
        - 適合查看交易時段的價格走勢
        - 建議: 開啟
        """)
        
        st.markdown("#### 💡 使用建議")
        st.markdown("""
        1. **開始使用**: 建議先使用預設參數進行分析
        2. **調整參數**: 根據市場特性和個人偏好調整參數
        3. **驗證結果**: 檢查分析結果是否符合視覺觀察
        4. **定期更新**: 定期載入新資料以獲得最新分析
        """)
    
    def run(self):
        """運行儀表板"""
        # 渲染側邊欄並獲取設定
        settings = self.render_sidebar()
        
        # 渲染主要內容
        self.render_main_content(settings)
        
        # 頁腳資訊
        if st.session_state.last_update:
            st.markdown("---")
            st.markdown(f"🕒 最後更新: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """主函數"""
    try:
        dashboard = TradingDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"應用程式發生錯誤: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()