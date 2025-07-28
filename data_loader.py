"""
資料載入和處理模組
Author: Your Name
Date: 2024

這個模組負責載入和處理TX期貨的OHLCV資料
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, List
import streamlit as st


class DataLoader:
    """
    資料載入器類別，負責從各種來源載入和處理OHLCV資料
    """
    
    def __init__(self, file_path: str = "output/kline_60min.txt"):
        """
        初始化資料載入器
        
        Args:
            file_path: 預設的資料檔案路徑
        """
        self.file_path = file_path
        self.supported_encodings = ['utf-8', 'utf-8-sig', 'big5', 'gbk', 'cp950', 'latin1']
        
    def load_from_text_file(self, file_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        從文字檔載入資料
        
        Args:
            file_path: 檔案路徑，如果為None則使用預設路徑
            
        Returns:
            處理後的DataFrame，如果載入失敗則返回None
        """
        if file_path is None:
            file_path = self.file_path
            
        try:
            # 檢查檔案是否存在
            if not os.path.exists(file_path):
                st.error(f"資料檔案不存在: {file_path}")
                return None
            
            # 嘗試不同的編碼格式
            df = self._try_different_encodings(file_path)
            if df is None:
                st.error("無法使用任何支援的編碼格式讀取檔案")
                return None
            
            # 處理欄位名稱和格式
            df = self._process_columns(df)
            if df is None:
                return None
            
            # 資料清理和驗證
            df = self._clean_and_validate(df)
            if df is None or len(df) == 0:
                st.error("處理後沒有有效的資料")
                return None
            
            st.success(f"成功載入 {len(df)} 筆資料")
            return df
            
        except Exception as e:
            st.error(f"載入資料時發生錯誤: {str(e)}")
            return None
    
    def _try_different_encodings(self, file_path: str) -> Optional[pd.DataFrame]:
        """嘗試不同編碼格式讀取檔案"""
        for encoding in self.supported_encodings:
            try:
                df = pd.read_csv(file_path, sep='\s+', encoding=encoding, on_bad_lines='skip')
                return df
            except (UnicodeDecodeError, Exception):
                continue
        return None
    
    def _process_columns(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """處理欄位名稱和格式"""
        try:
            # 根據欄位數量判斷格式
            if len(df.columns) == 8:
                df.columns = ['日期', '時間', '開盤', '最高', '最低', '收盤', '成交量', '成交值']
            elif len(df.columns) == 7:
                df.columns = ['日期', '開盤', '最高', '最低', '收盤', '成交量', '成交值']
            elif len(df.columns) == 6:
                df.columns = ['日期', '開盤', '最高', '最低', '收盤', '成交量']
            else:
                st.error(f"不支援的欄位數量: {len(df.columns)}")
                return None
            
            # 處理日期時間欄位
            df = self._process_datetime(df)
            if df is None:
                return None
            
            # 轉換欄位名稱為英文
            column_mapping = {
                '完整時間': 'datetime',
                '日期': 'datetime',
                '時間': 'time_only',
                '開盤': 'open', 
                '最高': 'high',
                '最低': 'low',
                '收盤': 'close',
                '成交量': 'volume',
                '成交值': 'turnover'
            }
            
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            
            return df
            
        except Exception as e:
            st.error(f"處理欄位時發生錯誤: {str(e)}")
            return None
    
    def _process_datetime(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """處理日期時間欄位"""
        try:
            # 合併日期和時間欄位（如果分開的話）
            if '日期' in df.columns and '時間' in df.columns:
                df['完整時間'] = df['日期'].astype(str) + ' ' + df['時間'].astype(str)
                datetime_col = '完整時間'
            elif '日期' in df.columns:
                datetime_col = '日期'
            else:
                st.error("找不到日期欄位")
                return None
            
            # 轉換日期時間格式
            df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
            
            # 檢查轉換結果
            nan_count = df[datetime_col].isna().sum()
            if nan_count > 0:
                st.warning(f"發現 {nan_count} 個無效的日期時間項目")
            
            return df
            
        except Exception as e:
            st.error(f"處理日期時間時發生錯誤: {str(e)}")
            return None
    
    def _clean_and_validate(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """清理和驗證資料"""
        try:
            # 轉換數值欄位
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 移除包含NaN值的行
            essential_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            df = df.dropna(subset=[col for col in essential_columns if col in df.columns])
            
            # 驗證OHLC資料的合理性
            df = self._validate_ohlc(df)
            
            # 按時間排序
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # 檢查必要欄位
            required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"缺少必要欄位: {missing_cols}")
                return None
            
            return df
            
        except Exception as e:
            st.error(f"清理資料時發生錯誤: {str(e)}")
            return None
    
    def _validate_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """驗證OHLC資料的合理性"""
        original_length = len(df)
        
        # 移除不合理的OHLC資料
        df = df[
            (df['high'] >= df['low']) & 
            (df['high'] >= df['open']) & 
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) & 
            (df['low'] <= df['close']) &
            (df['open'] > 0) &
            (df['high'] > 0) &
            (df['low'] > 0) &
            (df['close'] > 0) &
            (df['volume'] >= 0)
        ]
        
        removed_count = original_length - len(df)
        if removed_count > 0:
            st.warning(f"移除了 {removed_count} 筆不合理的OHLC資料")
        
        return df
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        獲取資料基本資訊
        
        Args:
            df: 資料DataFrame
            
        Returns:
            包含資料資訊的字典
        """
        if df is None or len(df) == 0:
            return {}
        
        return {
            'total_records': len(df),
            'date_range': {
                'start': df['datetime'].min(),
                'end': df['datetime'].max()
            },
            'price_range': {
                'min_low': df['low'].min(),
                'max_high': df['high'].max(),
                'current_price': df['close'].iloc[-1]
            },
            'volume_stats': {
                'total_volume': df['volume'].sum(),
                'avg_volume': df['volume'].mean(),
                'max_volume': df['volume'].max()
            },
            'data_quality': {
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_timestamps': df['datetime'].duplicated().sum()
            }
        }
    
    def filter_data_by_date(self, df: pd.DataFrame, start_date: str = None, 
                           end_date: str = None) -> pd.DataFrame:
        """
        根據日期範圍過濾資料
        
        Args:
            df: 原始資料
            start_date: 開始日期 (格式: 'YYYY-MM-DD')
            end_date: 結束日期 (格式: 'YYYY-MM-DD')
            
        Returns:
            過濾後的資料
        """
        if df is None or len(df) == 0:
            return df
        
        filtered_df = df.copy()
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df['datetime'] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df['datetime'] <= end_dt]
        
        return filtered_df
    
    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        重新取樣資料到不同的時間週期
        
        Args:
            df: 原始資料
            timeframe: 時間週期 ('5min', '15min', '30min', '1H', '4H', '1D')
            
        Returns:
            重新取樣後的資料
        """
        if df is None or len(df) == 0:
            return df
        
        df_resampled = df.set_index('datetime').resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        
        return df_resampled


def calculate_basic_metrics(df: pd.DataFrame) -> dict:
    """
    計算基本的市場指標
    
    Args:
        df: OHLCV資料
        
    Returns:
        包含基本指標的字典
    """
    if df is None or len(df) < 2:
        return {}
    
    current_price = df['close'].iloc[-1]
    previous_price = df['close'].iloc[-2]
    
    # 價格變化
    price_change = current_price - previous_price
    price_change_pct = (price_change / previous_price) * 100
    
    # 期間統計
    period_high = df['high'].max()
    period_low = df['low'].min()
    period_range = period_high - period_low
    
    # 成交量統計
    total_volume = df['volume'].sum()
    avg_volume = df['volume'].mean()
    current_volume = df['volume'].iloc[-1]
    
    # 波動性指標 (簡單方法)
    price_changes = df['close'].pct_change().dropna()
    volatility = price_changes.std() * 100
    
    return {
        'current_price': current_price,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'period_high': period_high,
        'period_low': period_low,
        'period_range': period_range,
        'total_volume': total_volume,
        'avg_volume': avg_volume,
        'current_volume': current_volume,
        'volatility_pct': volatility,
        'data_points': len(df)
    }


def create_test_data(num_bars: int = 100, start_price: float = 15000.0, 
                    with_trend: bool = True) -> pd.DataFrame:
    """
    創建測試用的OHLCV資料
    
    Args:
        num_bars: 資料筆數
        start_price: 起始價格
        with_trend: 是否包含趨勢
        
    Returns:
        測試資料DataFrame
    """
    np.random.seed(42)
    dates = pd.date_range('2024-01-01 09:00:00', periods=num_bars, freq='H')
    
    # 基礎價格走勢
    if with_trend:
        trend = np.linspace(0, start_price * 0.1, num_bars)  # 10%的趨勢
    else:
        trend = np.zeros(num_bars)
    
    # 隨機波動
    random_walk = np.cumsum(np.random.randn(num_bars) * start_price * 0.005)
    base_prices = start_price + trend + random_walk
    
    # 生成OHLC
    opens = base_prices + np.random.randn(num_bars) * start_price * 0.002
    closes = opens + np.random.randn(num_bars) * start_price * 0.003
    
    # 確保high和low的合理性
    daily_range = np.abs(np.random.randn(num_bars)) * start_price * 0.01
    highs = np.maximum(opens, closes) + daily_range * 0.6
    lows = np.minimum(opens, closes) - daily_range * 0.4
    
    # 成交量（模擬交易時間的特徵）
    base_volume = 5000
    volume_variation = np.random.randint(-2000, 3000, num_bars)
    volumes = np.maximum(base_volume + volume_variation, 100)
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': opens.round(0),
        'high': highs.round(0),
        'low': lows.round(0),
        'close': closes.round(0),
        'volume': volumes
    })
    
    return df


if __name__ == "__main__":
    # 測試資料載入器
    print("=== 資料載入器測試 ===")
    
    # 創建測試資料
    test_data = create_test_data(200, 15000, True)
    print(f"創建測試資料: {len(test_data)} 筆")
    
    # 測試基本指標計算
    metrics = calculate_basic_metrics(test_data)
    print(f"\n基本指標:")
    print(f"- 當前價格: {metrics['current_price']:.0f}")
    print(f"- 價格變化: {metrics['price_change']:+.0f} ({metrics['price_change_pct']:+.2f}%)")
    print(f"- 期間高點: {metrics['period_high']:.0f}")
    print(f"- 期間低點: {metrics['period_low']:.0f}")
    print(f"- 總成交量: {metrics['total_volume']:,.0f}")
    print(f"- 波動率: {metrics['volatility_pct']:.2f}%")
    
    # 測試資料載入器（模擬）
    loader = DataLoader()
    data_info = loader.get_data_info(test_data)
    print(f"\n資料資訊:")
    print(f"- 總筆數: {data_info['total_records']}")
    print(f"- 時間範圍: {data_info['date_range']['start']} 到 {data_info['date_range']['end']}")
    print(f"- 價格範圍: {data_info['price_range']['min_low']:.0f} - {data_info['price_range']['max_high']:.0f}")
    
    # 測試資料重新取樣
    resampled_4h = loader.resample_data(test_data, '4H')
    print(f"\n重新取樣到4小時: {len(resampled_4h)} 筆資料")
    
    print("\n測試完成！")