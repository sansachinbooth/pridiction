#!/usr/bin/env python3
"""
crprid.py - Enhanced Crypto Prediction Tool with Improved Gain Prediction
Optimized for better gain prediction accuracy and market trend detection
"""

import os
import sys
import time
import math
import json
import argparse
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any, List

import requests
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

import warnings
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# color output
try:
    from colorama import init as colorama_init, Fore, Style, Back
    colorama_init(autoreset=True)
except Exception:
    # fallback no color
    class _C:
        def __getattr__(self, name): return ""
    Fore = Style = Back = _C()

# Optional imports for enhanced Excel reporting
try:
    import openpyxl
    from openpyxl.chart import LineChart, Reference
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl import load_workbook
    EXCEL_ENHANCED = True
except Exception:
    EXCEL_ENHANCED = False

# Optional imports
try:
    import ollama  # local ollama client if installed
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

# Constants
COINGECKO_API = "https://api.coingecko.com/api/v3"
BINANCE_API = "https://api.binance.com/api/v3"

# ----- Enhanced Gain Prediction Configuration -----

GAIN_PREDICTION_CONFIG = {
    'momentum_weight': 1.5,  # Increased weight for momentum features
    'volume_weight': 1.3,    # Increased weight for volume features
    'trend_weight': 1.4,     # Increased weight for trend features
    'min_gain_threshold': 0.02,  # Minimum gain to consider
    'confidence_boost_threshold': 0.05,  # Gain level that boosts confidence
    'bull_market_detection_period': 30,  # Days to detect bull market
}

# ----- Output formatting functions -----

def print_header(title: str, level: int = 1):
    if level == 1:
        print(Fore.CYAN + "=" * 70)
        print(Fore.CYAN + title.center(70))
        print(Fore.CYAN + "=" * 70)
    elif level == 2:
        print(Fore.GREEN + "─" * 50)
        print(Fore.GREEN + f"▶ {title}")
        print(Fore.GREEN + "─" * 50)
    else:
        print(Fore.YELLOW + f"● {title}")

def print_bullet(text: str, color=Fore.WHITE, indent: int = 2):
    indent_str = " " * indent
    print(f"{indent_str}{color}• {text}")

def print_status(text: str, status: str = "info"):
    colors = {
        "info": Fore.BLUE,
        "success": Fore.GREEN,
        "warning": Fore.YELLOW,
        "error": Fore.RED
    }
    color = colors.get(status, Fore.WHITE)
    print(f"{color}↳ {text}")

# ----- Enhanced Utilities and indicators -----

def ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average with proper handling"""
    if len(series) < period:
        return pd.Series([float('nan')] * len(series), index=series.index)
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI with proper error handling"""
    if len(series) < period + 1:
        return pd.Series([50.0] * len(series), index=series.index)
    
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    ma_up = up.rolling(window=period, min_periods=period).mean()
    ma_down = down.rolling(window=period, min_periods=period).mean()
    
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50.0)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD with proper validation"""
    if len(series) < slow:
        nan_series = pd.Series([float('nan')] * len(series), index=series.index)
        return nan_series, nan_series, nan_series
    
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    if len(df) < period:
        return pd.Series([float('nan')] * len(df), index=df.index)
    
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

def bollinger_bands(series: pd.Series, period: int = 20, std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    if len(series) < period:
        nan_series = pd.Series([float('nan')] * len(series), index=series.index)
        return nan_series, nan_series, nan_series
    
    sma = series.rolling(window=period, min_periods=1).mean()
    rolling_std = series.rolling(window=period, min_periods=1).std()
    upper_band = sma + (rolling_std * std)
    lower_band = sma - (rolling_std * std)
    return upper_band, sma, lower_band

def is_valid_value(x) -> bool:
    """Check if a value is valid (not None, NaN, or Inf)"""
    try:
        return x is not None and not (math.isnan(x) or math.isinf(x))
    except Exception:
        return False

# ----- Enhanced Feature Engineering for Gain Prediction -----
def add_gain_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Price momentum features
    df['price_change_1'] = df['close'].pct_change(1)
    df['price_change_3'] = df['close'].pct_change(3)
    df['price_change_5'] = df['close'].pct_change(5)
    
    # Volume-price divergence (crucial for gain prediction)
    if 'volume' in df.columns:
        df['volume_price_divergence'] = df['volume'] * df['price_change_1']
        df['volume_momentum'] = df['volume'] / df['volume'].rolling(10).mean()
    
    # Support/resistance breaks
    df['breakout_high_20'] = (df['close'] > df['high'].rolling(20).max()).astype(int)
    df['breakout_low_20'] = (df['close'] < df['low'].rolling(20).min()).astype(int)
    
    # Trend confirmation
    df['trend_confirmation'] = (
        (df['close'] > df['sma20']) & 
        (df['sma20'] > df['sma50']) & 
        (df['rsi14'] > 50)
    ).astype(int)
    
    return df


def train_simplified_model(df: pd.DataFrame, feature_cols: list):
    # Create binary target: 1 if price goes up, 0 if down
    df = df.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    
    if len(df) < 50:
        return None
    
    X = df[feature_cols]
    y = df['target']
    
    # Simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    # Use a single, well-tuned model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print_status(f"Direction Prediction Accuracy: {accuracy:.2%}", "success")
    
    return {
        'model': model,
        'accuracy': accuracy,
        'feature_importance': dict(zip(feature_cols, model.feature_importances_))
    }


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
     """Add advanced technical features for better prediction"""
     df = df.copy()
    
     n_rows = len(df)
     if n_rows < 10:
        return df
    
     try:
        # Price momentum features
        df['price_rate_of_change_5'] = df['close'].pct_change(5)
        df['price_rate_of_change_10'] = df['close'].pct_change(10)
        df['price_momentum'] = df['close'] - df['close'].shift(5)
        
        # Volatility features
        df['volatility_5'] = df['close'].pct_change().rolling(window=5).std()
        df['volatility_20'] = df['close'].pct_change().rolling(window=20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        
        # Volume-based features
        if 'volume' in df.columns:
            df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
            df['volume_price_trend'] = (df['volume'] * df['close'].pct_change()).rolling(window=10).mean()
        
        # Support and resistance levels
        for window in [10, 20]:
            df[f'resistance_{window}'] = df['high'].rolling(window=window).max()
            df[f'support_{window}'] = df['low'].rolling(window=window).min()
            df[f'price_vs_resistance_{window}'] = (df['close'] - df[f'resistance_{window}']) / df[f'resistance_{window}']
            df[f'price_vs_support_{window}'] = (df['close'] - df[f'support_{window}']) / df[f'support_{window}']
        
        # Trend strength indicators
        df['trend_strength'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()
        
        # Price position in Bollinger Bands
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Market regime detection
        df['above_ma_20'] = (df['close'] > df['close'].rolling(window=20).mean()).astype(int)
        df['above_ma_50'] = (df['close'] > df['close'].rolling(window=50).mean()).astype(int)
        
        # Time-based features (if timestamp available)
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Add gain prediction features
        df = add_gain_prediction_features(df)
        
        # Fill NaN values
        df = df.ffill().bfill().fillna(0)
        
     except Exception as e:
        print_status(f"Error in advanced features: {e}", "warning")
    
     return df

# ----- Data fetchers -----

def fetch_coin_gecko_market_chart(coin_id: str, vs_currency: str, days: float) -> pd.DataFrame:
    """Fetch coin market chart from CoinGecko"""
    url = f"{COINGECKO_API}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    prices = data.get("prices", [])
    df = pd.DataFrame(prices, columns=['timestamp_ms', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df = df[['timestamp', 'price']]
    return df

def coingecko_ohlcv_from_prices(df_prices: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Convert raw price points to OHLCV by resampling"""
    rule_map = {
        '10m': '10min', '20m': '20min', '30m': '30min',
        '1h': '1H', '4h': '4H', '1d': '1D', '1y': '1Y'
    }
    if timeframe not in rule_map:
        raise ValueError("Unsupported timeframe for resample")
    rule = rule_map[timeframe]
    df = df_prices.set_index('timestamp').resample(rule).agg({'price': ['first', 'max', 'min', 'last', 'count']})
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.dropna().reset_index()
    return df

def fetch_binance_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Fetch klines from Binance"""
    url = f"{BINANCE_API}/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=1500)
    r.raise_for_status()
    data = r.json()
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades',
            'taker_base_vol', 'taker_quote_vol', 'ignore']
    df = pd.DataFrame(data, columns=cols)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(float)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return df

# ----- Enhanced Feature engineering -----

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced version with advanced features"""
    if df.empty:
        print_status("Empty dataframe provided to technical indicators", "warning")
        return df
    
    df = df.copy()
    
    # Ensure numeric types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove any rows with invalid price data
    df = df.dropna(subset=['close', 'high', 'low', 'open'])
    
    if df.empty:
        print_status("No valid price data after cleaning", "error")
        return df
    
    n_rows = len(df)
    print_status(f"Processing {n_rows} rows for technical indicators", "info")
    
    try:
        # Basic moving averages
        df['sma10'] = df['close'].rolling(window=min(10, n_rows), min_periods=1).mean()
        df['sma20'] = df['close'].rolling(window=min(20, n_rows), min_periods=1).mean()
        df['sma50'] = df['close'].rolling(window=min(50, n_rows), min_periods=1).mean()
        
        # EMAs
        df['ema8'] = ema(df['close'], min(8, n_rows))
        df['ema21'] = ema(df['close'], min(21, n_rows))
        df['ema50'] = ema(df['close'], min(50, n_rows))
        
        # RSI with multiple periods
        df['rsi14'] = rsi(df['close'], min(14, n_rows))
        df['rsi7'] = rsi(df['close'], min(7, n_rows))
        
        # MACD
        macd_line, signal_line, hist_line = macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = hist_line
        
        # ATR
        df['atr14'] = atr(df, min(14, n_rows))
        
        # Bollinger Bands
        upper_bb, middle_bb, lower_bb = bollinger_bands(df['close'])
        df['bb_upper'] = upper_bb
        df['bb_middle'] = middle_bb
        df['bb_lower'] = lower_bb
        df['bb_width'] = (upper_bb - lower_bb) / middle_bb
        
        # Volume indicators
        if 'volume' in df.columns:
            df['volume_sma20'] = df['volume'].rolling(window=min(20, n_rows), min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma20']
            df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], np.nan)
        
        # Price changes
        for period in [1, 3, 5, 7, 10]:
            df[f'ret{period}'] = df['close'].pct_change(period)
        
        # Support/Resistance levels
        df['resistance_20'] = df['high'].rolling(window=min(20, n_rows), min_periods=1).max()
        df['support_20'] = df['low'].rolling(window=min(20, n_rows), min_periods=1).min()
        
        # Add advanced features
        df = add_advanced_features(df)
        
        # Fill NaN values
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                if 'rsi' in col:
                    df[col] = df[col].fillna(50.0)
                elif 'macd' in col or 'bb' in col:
                    df[col] = df[col].fillna(0.0)
                else:
                    df[col] = df[col].ffill().bfill().fillna(0.0)
        
        # Final cleanup
        df = df.dropna(thresh=len(df.columns) // 2)
        
        if not df.empty:
            print_status(f"Technical indicators calculated successfully. Final dataset: {len(df)} rows", "success")
        else:
            print_status("No valid data after technical indicator calculation", "warning")
            
        return df.reset_index(drop=True)
        
    except Exception as e:
        print_status(f"Error in technical indicator calculation: {str(e)}", "error")
        return df

# ----- Enhanced ML model with Ensemble for Gain Prediction -----

def train_direction_predictor(df: pd.DataFrame, feature_cols: list):
    """Simplified direction predictor"""
    df = df.copy()
    
    # Create direction target
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    
    if len(df) < 100:
        print_status("Not enough data for training", "warning")
        return None
    
    X = df[feature_cols].fillna(0)
    y = df['target']
    
    # Simple time-based split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    print_status(f"Train Accuracy: {train_acc:.2%}", "info")
    print_status(f"Test Accuracy: {test_acc:.2%}", "success")
    
    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print_status(f"Top features: {[f[0] for f in top_features]}", "info")
    
    return {
        'model': model,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'feature_importance': importance
    }


def enhanced_predict_future_prices(model, scaler, last_data: pd.Series, feature_cols: list, periods: int = 365) -> pd.DataFrame:
    """
    Enhanced future price prediction with gain-focused confidence estimation
    """
    predictions = []
    current_data = last_data.copy()
    
    # Calculate current trend strength
    current_trend = 0
    if 'trend_strength_20' in current_data:
        current_trend = current_data['trend_strength_20']
    
    for i in range(periods):
        try:
            # Prepare features with proper DataFrame structure
            features_data = current_data[feature_cols].values.reshape(1, -1)
            features_df = pd.DataFrame(features_data, columns=feature_cols)
            
            # Scale features if scaler is available - FIXED FEATURE NAMES
            if scaler is not None:
                # Ensure we're using the same features the scaler was trained on
                features_scaled = scaler.transform(features_df[feature_cols])
            else:
                features_scaled = features_df.values
            
            # Make prediction
            pred_price = float(model.predict(features_scaled)[0])
            
            # Rest of your function remains the same...

            # Enhanced confidence estimation for gains
            base_confidence = 0.7
            
            # Boost confidence for upward predictions
            if pred_price > current_data['close']:
                gain_pct = (pred_price - current_data['close']) / current_data['close'] * 100
                
                # Higher gains get confidence boost
                if gain_pct > GAIN_PREDICTION_CONFIG['confidence_boost_threshold'] * 100:
                    confidence_boost = min(0.3, gain_pct / 100)
                    base_confidence += confidence_boost
                
                # Strong trend boosts confidence
                if current_trend > 2:  # Strong uptrend
                    base_confidence += 0.1
                
                # Volume surge boosts confidence
                if 'volume_surge_ratio' in current_data and current_data['volume_surge_ratio'] > 1.5:
                    base_confidence += 0.05
            
            # Volatility adjustment
            if 'volatility_ratio_5_20' in current_data:
                vol_ratio = current_data['volatility_ratio_5_20']
                if vol_ratio > 1.2:  # High volatility
                    base_confidence *= 0.9
                elif vol_ratio < 0.8:  # Low volatility (compression)
                    base_confidence *= 1.05
            
            confidence = min(0.95, max(0.3, base_confidence))
            
            predictions.append({
                'period': i + 1,
                'predicted_price': pred_price,
                'confidence': confidence,
                'date': datetime.now() + timedelta(days=i),
                'expected_gain_pct': (pred_price - current_data['close']) / current_data['close'] * 100
            })
            
            # Update for next prediction
            if i < periods - 1:
                current_data['close'] = pred_price
                # Update technical indicators approximately
                current_data = update_technical_indicators(current_data, pred_price, feature_cols)
                
        except Exception as e:
            print_status(f"Prediction error at step {i}: {e}", "warning")
            fallback_price = predictions[-1]['predicted_price'] if predictions else current_data['close']
            predictions.append({
                'period': i + 1,
                'predicted_price': fallback_price,
                'confidence': 0.1,
                'date': datetime.now() + timedelta(days=i),
                'expected_gain_pct': 0
            })
    
    return pd.DataFrame(predictions)

def update_technical_indicators(current_data: pd.Series, new_price: float, feature_cols: list) -> pd.Series:
    """
    Approximate update of technical indicators for recursive forecasting
    """
    updated_data = current_data.copy()
    updated_data['close'] = new_price
    
    # Enhanced updates for gain prediction features
    if 'ema8' in feature_cols and 'ema8' in updated_data:
        updated_data['ema8'] = (updated_data['ema8'] * 7 + new_price) / 8
    
    if 'ema21' in feature_cols and 'ema21' in updated_data:
        updated_data['ema21'] = (updated_data['ema21'] * 20 + new_price) / 21
    
    if 'sma20' in feature_cols and 'sma20' in updated_data:
        updated_data['sma20'] = (updated_data['sma20'] * 19 + new_price) / 20
    
    # Update momentum indicators
    if 'momentum_5' in feature_cols:
        # Approximate momentum update
        old_price = current_data['close']
        if is_valid_value(old_price) and old_price > 0:
            momentum = (new_price - old_price) / old_price * 100
            updated_data['momentum_5'] = momentum
    
    # Update trend strength
    if 'trend_strength_20' in feature_cols and 'sma20' in updated_data:
        trend_strength = (new_price - updated_data['sma20']) / updated_data['sma20'] * 100
        updated_data['trend_strength_20'] = trend_strength
    
    return updated_data

# ----- Enhanced trading recommendation with Gain Focus -----

def enhanced_recommend_trade(last_price: float, predicted_price: float, atr_val: float, 
                           rsi_val: float, ml_confidence: float = 0.8, timeframe: str = '1h',
                           trend_strength: float = 0, volume_surge: float = 1.0) -> Dict[str, Any]:
    """
    Enhanced recommendation with ML confidence weighting and gain focus
    """
    if predicted_price is None or not is_valid_value(predicted_price):
        predicted_price = last_price
        ml_confidence = 0.1
    
    diff = predicted_price - last_price
    diff_pct = (diff / last_price) * 100.0
    
    # Multi-factor decision with confidence weighting
    factors = []
    
    # Enhanced price prediction factor with gain focus
    prediction_weight = ml_confidence * GAIN_PREDICTION_CONFIG['momentum_weight']
    
    if diff_pct > 5.0:
        factors.append(("price_prediction", "extremely_strong_bullish", 4, prediction_weight))
    elif diff_pct > 2.0:
        factors.append(("price_prediction", "very_strong_bullish", 3, prediction_weight))
    elif diff_pct > 1.0:
        factors.append(("price_prediction", "strong_bullish", 2, prediction_weight))
    elif diff_pct > GAIN_PREDICTION_CONFIG['min_gain_threshold'] * 100:
        factors.append(("price_prediction", "bullish", 1, prediction_weight * 0.9))
    elif diff_pct < -5.0:
        factors.append(("price_prediction", "extremely_strong_bearish", -4, prediction_weight))
    elif diff_pct < -2.0:
        factors.append(("price_prediction", "very_strong_bearish", -3, prediction_weight))
    elif diff_pct < -1.0:
        factors.append(("price_prediction", "strong_bearish", -2, prediction_weight))
    elif diff_pct < -GAIN_PREDICTION_CONFIG['min_gain_threshold'] * 100:
        factors.append(("price_prediction", "bearish", -1, prediction_weight * 0.9))
    else:
        factors.append(("price_prediction", "neutral", 0, prediction_weight * 0.5))
    
    # Enhanced RSI factor for gain detection
    rsi_weight = 0.8
    if rsi_val > 80:
        factors.append(("rsi", "extremely_overbought", -3, rsi_weight))
    elif rsi_val > 70:
        factors.append(("rsi", "overbought", -2, rsi_weight))
    elif rsi_val < 20:
        factors.append(("rsi", "extremely_oversold", 3, rsi_weight))
    elif rsi_val < 30:
        factors.append(("rsi", "oversold", 2, rsi_weight))
    elif 40 < rsi_val < 60:
        factors.append(("rsi", "optimal_bullish", 1, rsi_weight * 1.2))  # Boost for optimal RSI
    else:
        factors.append(("rsi", "neutral", 0, rsi_weight))
    
    # Trend strength factor
    trend_weight = GAIN_PREDICTION_CONFIG['trend_weight']
    if trend_strength > 5:
        factors.append(("trend", "very_strong_uptrend", 3, trend_weight))
    elif trend_strength > 2:
        factors.append(("trend", "strong_uptrend", 2, trend_weight))
    elif trend_strength > 0.5:
        factors.append(("trend", "uptrend", 1, trend_weight))
    elif trend_strength < -5:
        factors.append(("trend", "very_strong_downtrend", -3, trend_weight))
    elif trend_strength < -2:
        factors.append(("trend", "strong_downtrend", -2, trend_weight))
    elif trend_strength < -0.5:
        factors.append(("trend", "downtrend", -1, trend_weight))
    else:
        factors.append(("trend", "neutral", 0, trend_weight * 0.5))
    
    # Volume surge factor
    volume_weight = GAIN_PREDICTION_CONFIG['volume_weight']
    if volume_surge > 2.0:
        factors.append(("volume", "very_high_volume", 2, volume_weight))
    elif volume_surge > 1.5:
        factors.append(("volume", "high_volume", 1, volume_weight))
    elif volume_surge < 0.5:
        factors.append(("volume", "low_volume", -1, volume_weight))
    else:
        factors.append(("volume", "normal_volume", 0, volume_weight * 0.5))
    
    # Calculate weighted score
    total_score = 0
    total_weight = 0
    
    for _, _, score, weight in factors:
        total_score += score * weight
        total_weight += weight
    
    if total_weight > 0:
        normalized_score = total_score / total_weight
    else:
        normalized_score = 0
    
    # Enhanced advice determination with gain focus
    if normalized_score >= 2.5:
        advice = 'STRONG BUY'
        confidence_level = 'Very High'
        gain_outlook = 'High Gain Potential'
    elif normalized_score >= 1.5:
        advice = 'BUY'
        confidence_level = 'High'
        gain_outlook = 'Good Gain Potential'
    elif normalized_score >= 0.8:
        advice = 'WEAK BUY'
        confidence_level = 'Medium'
        gain_outlook = 'Moderate Gain Potential'
    elif normalized_score <= -2.5:
        advice = 'STRONG SELL'
        confidence_level = 'Very High'
        gain_outlook = 'High Loss Risk'
    elif normalized_score <= -1.5:
        advice = 'SELL'
        confidence_level = 'High'
        gain_outlook = 'Significant Loss Risk'
    elif normalized_score <= -0.8:
        advice = 'WEAK SELL'
        confidence_level = 'Medium'
        gain_outlook = 'Moderate Loss Risk'
    else:
        advice = 'HOLD'
        confidence_level = 'Low'
        gain_outlook = 'Neutral'
    
    # Enhanced risk management for gains
    volatility_pct = (atr_val / last_price) * 100 if atr_val > 0 else 2.0
    
    # Dynamic stop loss based on gain potential
    if 'BUY' in advice:
        # More aggressive stop loss for high gain potential
        stoploss_multiplier = 2.0 if diff_pct > 3 else 1.5 if diff_pct > 1 else 1.0
        stoploss = last_price - (stoploss_multiplier * atr_val) if is_valid_value(atr_val) and atr_val > 0 else last_price * 0.95
    else:
        stoploss_multiplier = 1.5
        stoploss = last_price - (stoploss_multiplier * atr_val) if is_valid_value(atr_val) and atr_val > 0 else last_price * 0.97
    
    # Enhanced take profit calculation
    if diff_pct > 0:
        # For bullish predictions, set take profit above predicted price
        take_profit = predicted_price * (1 + (diff_pct / 100) * 0.2)  # 20% above predicted
    else:
        take_profit = last_price * 1.02  # Conservative 2% gain
    
    risk_reward = (take_profit - last_price) / (last_price - stoploss) if stoploss and stoploss < last_price else None
    
    # Enhanced sale time prediction for gains
    sale_time = calculate_dynamic_sale_time(advice, timeframe, volatility_pct, diff_pct)
    
    return {
        'advice': advice,
        'confidence': confidence_level,
        'gain_outlook': gain_outlook,
        'diff_pct': diff_pct,
        'factors': factors,
        'normalized_score': normalized_score,
        'stoploss': stoploss,
        'take_profit': take_profit,
        'risk_reward_ratio': risk_reward,
        'sale_time': sale_time,
        'ml_confidence': ml_confidence,
        'expected_gain_pct': diff_pct
    }

def calculate_dynamic_sale_time(advice: str, timeframe: str, volatility_pct: float, gain_pct: float) -> datetime:
    """Calculate sale time based on advice, timeframe, volatility, and gain potential"""
    if advice not in ['BUY', 'STRONG BUY', 'WEAK BUY']:
        return None
    
    current_time = datetime.now()
    base_times = {
        '10m': timedelta(minutes=30),
        '20m': timedelta(minutes=60),
        '30m': timedelta(hours=2),
        '1h': timedelta(hours=4),
        '4h': timedelta(days=1),
        '1d': timedelta(days=3),
        '1y': timedelta(days=365)
    }
    
    base_time = base_times.get(timeframe, timedelta(hours=24))
    
    # Adjust for volatility
    volatility_factor = max(0.5, min(2.0, 1.0 - (volatility_pct - 2.0) / 10.0))
    adjusted_time = base_time * volatility_factor
    
    # Adjust for gain potential - higher gains get longer hold times
    if gain_pct > 5:
        gain_factor = 1.5
    elif gain_pct > 2:
        gain_factor = 1.3
    elif gain_pct > 1:
        gain_factor = 1.1
    else:
        gain_factor = 0.9
    
    adjusted_time *= gain_factor
    
    if 'STRONG' in advice:
        adjusted_time *= 1.3
    elif 'WEAK' in advice:
        adjusted_time *= 0.8
    
    return current_time + adjusted_time

# ----- Rest of the functions (unchanged but compatible) -----

def get_rsi_strength(rsi_value: float) -> Tuple[str, str]:
    """Get RSI strength description"""
    if not is_valid_value(rsi_value):
        return "UNKNOWN", Fore.WHITE
    if rsi_value >= 70:
        return "OVERBOUGHT", Fore.RED
    elif rsi_value >= 60:
        return "BULLISH", Fore.YELLOW
    elif rsi_value >= 40:
        return "NEUTRAL", Fore.WHITE
    elif rsi_value >= 30:
        return "BEARISH", Fore.YELLOW
    else:
        return "OVERSOLD", Fore.GREEN

# Replace TimeSeriesSplit with proper walk-forward validation
def time_series_train_test_split(X, y, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test

def get_trend_strength(ema_short: float, ema_long: float, price: float) -> Tuple[str, str]:
    """Determine trend strength based on EMAs and price"""
    if not all(is_valid_value(x) for x in [ema_short, ema_long, price]):
        return "UNKNOWN", Fore.WHITE
        
    if price > ema_short > ema_long:
        return "STRONG UPTREND", Fore.GREEN
    elif ema_short > price > ema_long:
        return "UPTREND", Fore.YELLOW
    elif price < ema_short < ema_long:
        return "STRONG DOWNTREND", Fore.RED
    elif ema_short < price < ema_long:
        return "DOWNTREND", Fore.YELLOW
    else:
        return "SIDEWAYS", Fore.WHITE

# ... (Keep all other functions like fetch_news_sentiment, fetch_binance_recent_trades, 
# analyze_whales_from_trades, generate_ollama_summary, create_simplified_export, 
# load_trading_signals, fetch_current_price, calculate_accuracy_metrics, 
# check_prediction_accuracy exactly as they were in your original code)


def generate_ollama_summary(prompt: str, host: str = 'http://localhost', port: int = 11500, model: str = 'gemma3:4b') -> Dict[str, Any]:
    """Generate summary using Ollama"""
    if not OLLAMA_AVAILABLE:
        return {'ok': False, 'error': 'Ollama not available'}
    
    try:
        client = ollama.Client(host=f"{host}:{port}")
        response = client.generate(model=model, prompt=prompt)
        return {'ok': True, 'text': response['response']}
    except Exception as e:
        return {'ok': False, 'error': str(e)}

# ----- Updated Feature Columns for Gain Prediction -----

SIMPLIFIED_FEATURE_COLS = [
    'rsi14', 'rsi7', 'macd', 'macd_hist', 'bb_width',
    'volume_ratio', 'price_change_1', 'price_change_3', 
    'breakout_high_20', 'breakout_low_20', 'trend_confirmation',
    'volume_price_divergence', 'volume_momentum'
]

GAIN_FEATURE_COLS = SIMPLIFIED_FEATURE_COLS 

def predict_direction(model, current_data, feature_cols):
    features = current_data[feature_cols].values.reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return {
        'direction': 'UP' if prediction == 1 else 'DOWN',
        'confidence': max(probability),
        'up_probability': probability[1],
        'down_probability': probability[0]
    }

# Add this test function
def test_direction_accuracy(df):
    df['actual_direction'] = (df['close'].shift(-1) > df['close']).astype(int)
    df['predicted_direction'] = (df['prediction_value'] > df['close']).astype(int)
    accuracy = (df['actual_direction'] == df['predicted_direction']).mean()
    return accuracy

###########################################################################################################33

def check_prediction_accuracy(file_path: str, vs_currency: str = 'usd') -> Dict[str, Any]:
    """Check prediction accuracy against live data with enhanced loss prediction tracking"""
    print_header("PREDICTION ACCURACY CHECK", 1)
    
    predictions_df = load_trading_signals(file_path)
    if predictions_df.empty:
        print_status("No data loaded from file", "error")
        return {'error': 'No data loaded'}
    
    current_time = datetime.now()
    
    # Filter predictions that are ready for evaluation (timestamp in the past)
    predictions_df = predictions_df[predictions_df['timestamp'] <= current_time]
    
    if predictions_df.empty:
        print_status("No predictions found that have reached their evaluation time", "warning")
        return {'error': 'No predictions to evaluate'}
    
    print_status(f"Evaluating {len(predictions_df)} predictions...", "info")
    
    results = []
    successful_fetches = 0
    
    for idx, row in predictions_df.iterrows():
        coin = row.get('crypto_name', '').lower() or row.get('symbol', '').lower()
        symbol = row.get('symbol', '')
        
        if not coin:
            continue
            
        current_price = fetch_current_price(coin, symbol, vs_currency)
        if current_price is None:
            print_status(f"Could not fetch current price for {coin}", "warning")
            continue
            
        result = row.to_dict()
        result['actual_price'] = current_price
        result['check_timestamp'] = current_time
        results.append(result)
        successful_fetches += 1
        
        # Progress indicator
        if successful_fetches % 10 == 0:
            print_status(f"Fetched prices for {successful_fetches} predictions...", "info")
    
    if not results:
        print_status("No successful price fetches for accuracy calculation", "error")
        return {'error': 'Price fetch failed'}
    
    results_df = pd.DataFrame(results)
    
    # Check if we have valid data
    if results_df.empty:
        print_status("No valid results after processing", "error")
        return {'error': 'No valid results'}
    
    print_status(f"Processing {len(results_df)} valid predictions for metrics...", "info")
    
    metrics = calculate_accuracy_metrics(results_df)
    
    print_header("ACCURACY RESULTS", 2)
    
    if metrics:
        print_bullet(f"Total Predictions Evaluated: {metrics['total_predictions']}", Fore.WHITE)
        print_bullet(f"Predictions with Valid Data: {len(results_df)}", Fore.WHITE)
        print_bullet(f"Mean Absolute Error: ${metrics['mae']:.6f}", Fore.WHITE)
        print_bullet(f"Root Mean Square Error: ${metrics['rmse']:.6f}", Fore.WHITE)
        print_bullet(f"Mean Absolute Percentage Error: {metrics['mape']:.2f}%", Fore.WHITE)
        
        # Enhanced accuracy metrics
	

    # NEW: Overall Prediction Accuracy
    overall_accuracy = calculate_overall_prediction_accuracy(metrics)
    accuracy_color = Fore.GREEN if overall_accuracy > 70 else Fore.YELLOW if overall_accuracy > 50 else Fore.RED
    accuracy_rating = "EXCELLENT" if overall_accuracy > 70 else "GOOD" if overall_accuracy > 50 else "NEEDS IMPROVEMENT"

    print_header("OVERALL PREDICTION ACCURACY", 2)
    print_bullet(f"Overall Accuracy: {accuracy_color}{overall_accuracy:.1f}% - {accuracy_rating}", Fore.WHITE)
    print_bullet(f"Combined metrics: Direction, Gain/Loss Prediction, Signal Profitability", Fore.CYAN)

    # Add example explanation
    if 'example_prediction' in metrics:
        example = metrics['example_prediction']
        direction_text = "Direction Correct" if example['direction_correct'] else "Direction Incorrect"
        print_bullet(f"Example: Current=${example['current_price']:.2f}, Predicted=${example['predicted_price']:.2f}, Actual=${example['actual_price']:.2f} → {direction_text}", 
                    Fore.CYAN)
        
        # Rest of your function remains the same...
        # ... [keep the existing code for SPECIFIC PREDICTION ACCURACY, DETAILED ANALYSIS, etc.]
        
    else:
        print_status("Could not calculate accuracy metrics", "error")
        return {'error': 'Metric calculation failed'}
    
    # ... [rest of your existing function code]

################################################################################################################33        
           # Enhanced accuracy metrics
    print_header("ENHANCED ACCURACY METRICS", 2)

    print_bullet(f"Direction Accuracy: {metrics['direction_accuracy']:.1f}%", 
                Fore.GREEN if metrics['direction_accuracy'] > 60 else Fore.YELLOW if metrics['direction_accuracy'] > 50 else Fore.RED)

    print_bullet(f"Prediction Accuracy (Direction-based): {metrics['prediction_accuracy']:.1f}%", 
                Fore.GREEN if metrics['prediction_accuracy'] > 60 else Fore.YELLOW if metrics['prediction_accuracy'] > 50 else Fore.RED)

    print_bullet(f"Signal Profitability: {metrics['signal_accuracy']:.1f}%", 
                Fore.GREEN if metrics['signal_accuracy'] > 60 else Fore.YELLOW if metrics['signal_accuracy'] > 50 else Fore.RED)

    print_bullet(f"Average Profit/Loss: {metrics['average_profit_loss_pct']:.2f}%", 
                Fore.GREEN if metrics['average_profit_loss_pct'] > 0 else Fore.RED)

    # Add example explanation
    if 'example_prediction' in metrics:
        example = metrics['example_prediction']
        print_bullet(f"Example: Current=${example['current_price']:.2f}, Predicted=${example['predicted_price']:.2f}, Actual=${example['actual_price']:.2f} → Direction Correct", 
                    Fore.CYAN)
        
        # NEW: Loss and Gain prediction accuracy
        print_header("SPECIFIC PREDICTION ACCURACY", 2)
        
        if metrics['loss_predictions_total'] > 0:
            print_bullet(f"Loss Predictions Accuracy: {metrics['loss_prediction_accuracy']:.1f}% ({metrics['loss_predictions_correct']}/{metrics['loss_predictions_total']})", 
                        Fore.GREEN if metrics['loss_prediction_accuracy'] > 60 else Fore.YELLOW if metrics['loss_prediction_accuracy'] > 50 else Fore.RED)
        else:
            print_bullet("Loss Predictions: No data available", Fore.YELLOW)
            
        if metrics['gain_predictions_total'] > 0:
            print_bullet(f"Gain Predictions Accuracy: {metrics['gain_prediction_accuracy']:.1f}% ({metrics['gain_predictions_correct']}/{metrics['gain_predictions_total']})", 
                        Fore.GREEN if metrics['gain_prediction_accuracy'] > 60 else Fore.YELLOW if metrics['gain_prediction_accuracy'] > 50 else Fore.RED)
        else:
            print_bullet("Gain Predictions: No data available", Fore.YELLOW)
        
        print_header("DETAILED ANALYSIS", 2)
        
        # Find best and worst predictions
        if not results_df.empty:
            best_pred = results_df.loc[results_df['abs_pct_diff'].idxmin()]
            worst_pred = results_df.loc[results_df['abs_pct_diff'].idxmax()]
            
            print_bullet(f"Best Prediction: {best_pred.get('crypto_name', 'N/A')} - Error: {best_pred['abs_pct_diff']:.2f}%", Fore.GREEN)
            print_bullet(f"Worst Prediction: {worst_pred.get('crypto_name', 'N/A')} - Error: {worst_pred['abs_pct_diff']:.2f}%", Fore.RED)
            
            # Analyze by timeframe if available
            if 'timeframe' in results_df.columns:
                timeframe_stats = results_df.groupby('timeframe').agg({
                    'abs_pct_diff': 'mean',
                    'direction_correct': 'mean',
                    'signal_profitable': 'mean',
                    'prediction_accurate': 'mean'
                }).round(4)
                
                print_header("ACCURACY BY TIMEFRAME", 2)
                for timeframe, stats in timeframe_stats.iterrows():
                    accuracy_color = Fore.GREEN if stats['prediction_accurate'] > 0.7 else Fore.YELLOW if stats['prediction_accurate'] > 0.5 else Fore.RED
                    print_bullet(f"{timeframe}: {accuracy_color}{stats['prediction_accurate']*100:.1f}% accurate, {stats['abs_pct_diff']:.2f}% avg error", Fore.WHITE)
        
        # Save detailed report
        output_file = file_path.replace('.xlsx', '_accuracy_report.xlsx').replace('.csv', '_accuracy_report.csv')
        
        if EXCEL_ENHANCED and output_file.endswith('.xlsx'):
            try:
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    results_df.to_excel(writer, sheet_name='Detailed Results', index=False)
                    
                    # Create summary sheet
                    summary_data = {
                        'Metric': list(metrics.keys()),
                        'Value': list(metrics.values())
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Create performance analysis sheet
                    performance_data = {
                        'Category': ['Overall Accuracy', 'Loss Prediction Accuracy', 'Gain Prediction Accuracy', 'Signal Profitability'],
                        'Percentage': [
                            metrics['prediction_accuracy'],
                            metrics['loss_prediction_accuracy'],
                            metrics['gain_prediction_accuracy'],
                            metrics['signal_accuracy']
                        ],
                        'Rating': [
                            'Excellent' if metrics['prediction_accuracy'] > 70 else 'Good' if metrics['prediction_accuracy'] > 50 else 'Needs Improvement',
                            'Excellent' if metrics['loss_prediction_accuracy'] > 70 else 'Good' if metrics['loss_prediction_accuracy'] > 50 else 'Needs Improvement',
                            'Excellent' if metrics['gain_prediction_accuracy'] > 70 else 'Good' if metrics['gain_prediction_accuracy'] > 50 else 'Needs Improvement',
                            'Excellent' if metrics['signal_accuracy'] > 70 else 'Good' if metrics['signal_accuracy'] > 50 else 'Needs Improvement'
                        ]
                    }
                    pd.DataFrame(performance_data).to_excel(writer, sheet_name='Performance Analysis', index=False)
                    
                print_status(f"Detailed accuracy report saved to: {output_file}", "success")
            except Exception as e:
                print_status(f"Excel export failed: {e}, saving as CSV", "warning")
                results_df.to_csv(output_file, index=False)
        else:
            results_df.to_csv(output_file, index=False)
            print_status(f"Accuracy report saved to: {output_file}", "success")
        
                # Calculate overall accuracy for the summary
        overall_accuracy = calculate_overall_prediction_accuracy(metrics)
        
        return {
            'metrics': metrics,
            'detailed_results': results_df,
            'overall_accuracy': overall_accuracy,
            'summary': f"Overall Accuracy: {overall_accuracy:.1f}% | Direction: {metrics['direction_accuracy']:.1f}% | Gain Prediction: {metrics['gain_prediction_accuracy']:.1f}% | Profitability: {metrics['signal_accuracy']:.1f}%"
        }
    else:
        print_status("Could not calculate accuracy metrics", "error")
        return {'error': 'Metric calculation failed'}


def load_trading_signals(file_path: str) -> pd.DataFrame:
    """Load trading signals from Excel or CSV file"""
    try:
        if file_path.endswith('.xlsx') and EXCEL_ENHANCED:
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
        
        for col in ['timestamp', 'buy_time', 'sale_time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        numeric_cols = ['current_price', 'prediction_value', 'stoploss', 'rsi']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print_status(f"Loaded {len(df)} trading signals from {file_path}", "success")
        return df
    except Exception as e:
        print_status(f"Error loading trading signals: {e}", "error")
        return pd.DataFrame()


def fetch_current_price(coin: str, symbol: str, vs_currency: str = 'usd') -> Optional[float]:
    """Fetch current price from CoinGecko or Binance"""
    try:
        if symbol:
            try:
                url = f"https://api.binance.com/api/v3/ticker/price"
                params = {"symbol": symbol.upper()}
                r = requests.get(url, params=params, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    return float(data['price'])
            except:
                pass
        
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": coin.lower(), "vs_currencies": vs_currency}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data[coin.lower()][vs_currency]
    except Exception as e:
        print_status(f"Error fetching current price: {e}", "error")
        return None

#######################################################################################3

def calculate_accuracy_metrics(predictions_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate various accuracy metrics for predictions with direction-based accuracy"""
    if predictions_df.empty:
        print_status("No data in predictions_df", "error")
        return {}
    
    metrics = {}
    
    try:
        # Debug: Check what columns we have
        print_status(f"Available columns: {list(predictions_df.columns)}", "info")
        
        # Check if we have the required columns
        required_cols = ['current_price', 'prediction_value', 'actual_price']
        missing_cols = [col for col in required_cols if col not in predictions_df.columns]
        if missing_cols:
            print_status(f"Missing required columns: {missing_cols}", "error")
            return {}
        
        # Debug: Show sample data
        print_status(f"Sample data - Current: {predictions_df['current_price'].iloc[0]}, "
                    f"Predicted: {predictions_df['prediction_value'].iloc[0]}, "
                    f"Actual: {predictions_df['actual_price'].iloc[0]}", "info")
        
        # Calculate price differences
        predictions_df['price_diff'] = predictions_df['actual_price'] - predictions_df['prediction_value']
        predictions_df['abs_diff'] = predictions_df['price_diff'].abs()
        predictions_df['pct_diff'] = (predictions_df['price_diff'] / predictions_df['prediction_value']) * 100
        predictions_df['abs_pct_diff'] = predictions_df['pct_diff'].abs()
        
        # ENHANCED: Calculate direction accuracy (up/down prediction)
        # A prediction is directionally correct if both predicted and actual prices are above/below the reference price
        predictions_df['direction_correct'] = (
            (predictions_df['actual_price'] > predictions_df['current_price']) == 
            (predictions_df['prediction_value'] > predictions_df['current_price'])
        )
        
        # Calculate signal profitability (for BUY signals)
        predictions_df['signal_profitable'] = False
        
        # Check if 'signal' column exists, if not create a default
        if 'signal' not in predictions_df.columns:
            print_status("No 'signal' column found, creating default signals based on prediction", "warning")
            # Create default signals based on prediction direction
            predictions_df['signal'] = 'BUY'
            predictions_df.loc[predictions_df['prediction_value'] <= predictions_df['current_price'], 'signal'] = 'SELL'
        
        buy_signals = predictions_df['signal'].str.contains('BUY', case=False, na=False)
        sell_signals = predictions_df['signal'].str.contains('SELL', case=False, na=False)
        
        # For BUY signals: profitable if actual price > current price (price went up)
        predictions_df.loc[buy_signals, 'signal_profitable'] = (
            predictions_df.loc[buy_signals, 'actual_price'] > predictions_df.loc[buy_signals, 'current_price']
        )
        
        # For SELL signals: profitable if actual price < current price (price went down)
        predictions_df.loc[sell_signals, 'signal_profitable'] = (
            predictions_df.loc[sell_signals, 'actual_price'] < predictions_df.loc[sell_signals, 'current_price']
        )
        
        # ENHANCED: Calculate prediction accuracy based on direction, not exact price
        # Consider a prediction "accurate" if the direction (up/down) is correct
        predictions_df['prediction_accurate'] = predictions_df['direction_correct']
        
        # Calculate profit/loss percentage
        predictions_df['profit_loss_pct'] = (
            (predictions_df['actual_price'] - predictions_df['current_price']) / predictions_df['current_price'] * 100
        )
        
        # Calculate prediction error percentage (how close prediction was to actual)
        predictions_df['prediction_error_pct'] = predictions_df['abs_pct_diff']
        
        # Debug: Show some calculated values
        print_status(f"Direction correct count: {predictions_df['direction_correct'].sum()}/{len(predictions_df)}", "info")
        print_status(f"Signal profitable count: {predictions_df['signal_profitable'].sum()}/{len(predictions_df)}", "info")
        
        # Basic metrics
        metrics['total_predictions'] = len(predictions_df)
        metrics['mae'] = predictions_df['abs_diff'].mean()
        metrics['mse'] = mean_squared_error(predictions_df['actual_price'], predictions_df['prediction_value'])
        metrics['rmse'] = math.sqrt(metrics['mse'])
        metrics['mape'] = predictions_df['abs_pct_diff'].mean()
        
        # ENHANCED: Accuracy metrics based on direction
        metrics['direction_accuracy'] = predictions_df['direction_correct'].mean() * 100
        metrics['signal_accuracy'] = predictions_df['signal_profitable'].mean() * 100
        metrics['prediction_accuracy'] = predictions_df['prediction_accurate'].mean() * 100
        
        # Profit/loss metrics
        metrics['average_profit_loss_pct'] = predictions_df['profit_loss_pct'].mean()
        metrics['total_profitable_signals'] = predictions_df['signal_profitable'].sum()
        metrics['profitability_rate'] = (metrics['total_profitable_signals'] / len(predictions_df)) * 100
        
        # ENHANCED: Loss prediction accuracy based on direction
        loss_predictions = predictions_df[predictions_df['prediction_value'] < predictions_df['current_price']]
        if not loss_predictions.empty:
            metrics['loss_predictions_total'] = len(loss_predictions)
            # Count as correct if actual price also went down
            metrics['loss_predictions_correct'] = (loss_predictions['actual_price'] < loss_predictions['current_price']).sum()
            if metrics['loss_predictions_total'] > 0:
                metrics['loss_prediction_accuracy'] = (metrics['loss_predictions_correct'] / metrics['loss_predictions_total']) * 100
            else:
                metrics['loss_prediction_accuracy'] = 0
        else:
            metrics['loss_predictions_total'] = 0
            metrics['loss_predictions_correct'] = 0
            metrics['loss_prediction_accuracy'] = 0
        
        # ENHANCED: Gain prediction accuracy based on direction
        gain_predictions = predictions_df[predictions_df['prediction_value'] > predictions_df['current_price']]
        if not gain_predictions.empty:
            metrics['gain_predictions_total'] = len(gain_predictions)
            # Count as correct if actual price also went up
            metrics['gain_predictions_correct'] = (gain_predictions['actual_price'] > gain_predictions['current_price']).sum()
            if metrics['gain_predictions_total'] > 0:
                metrics['gain_prediction_accuracy'] = (metrics['gain_predictions_correct'] / metrics['gain_predictions_total']) * 100
            else:
                metrics['gain_prediction_accuracy'] = 0
        else:
            metrics['gain_predictions_total'] = 0
            metrics['gain_predictions_correct'] = 0
            metrics['gain_prediction_accuracy'] = 0
        
        # NEW: Add detailed examples for clarity
        if not predictions_df.empty:
            # Find examples of correct direction predictions
            correct_direction = predictions_df[predictions_df['direction_correct'] == True]
            if not correct_direction.empty:
                example = correct_direction.iloc[0]
                metrics['example_prediction'] = {
                    'current_price': example['current_price'],
                    'predicted_price': example['prediction_value'],
                    'actual_price': example['actual_price'],
                    'direction_correct': True
                }
            else:
                # If no correct predictions, show first example anyway
                example = predictions_df.iloc[0]
                metrics['example_prediction'] = {
                    'current_price': example['current_price'],
                    'predicted_price': example['prediction_value'],
                    'actual_price': example['actual_price'],
                    'direction_correct': False
                }
        
        print_status(f"Calculated metrics - Direction Accuracy: {metrics['direction_accuracy']:.1f}%", "success")
        
    except Exception as e:
        print_status(f"Error in calculate_accuracy_metrics: {str(e)}", "error")
        import traceback
        print_status(f"Traceback: {traceback.format_exc()}", "error")
        return {}
    
    return metrics

def calculate_overall_prediction_accuracy(metrics: Dict[str, Any]) -> float:
    """
    Calculate overall prediction accuracy percentage combining all parameters
    with weighted importance for gain-focused prediction
    """
    if not metrics:
        return 0.0
    
    try:
        # Weights for different accuracy metrics (gain-focused)
        weights = {
            'direction_accuracy': 0.25,        # Direction is important
            'prediction_accuracy': 0.20,       # Overall prediction accuracy
            'gain_prediction_accuracy': 0.25,  # Gain prediction is crucial
            'loss_prediction_accuracy': 0.15,  # Loss prediction important for risk management
            'signal_accuracy': 0.15            # Signal profitability
        }
        
        # Calculate weighted average
        total_weight = 0
        weighted_sum = 0
        
        for metric, weight in weights.items():
            if metric in metrics and metrics[metric] > 0:
                weighted_sum += metrics[metric] * weight
                total_weight += weight
        
        # If some metrics are missing, adjust the calculation
        if total_weight > 0:
            overall_accuracy = weighted_sum / total_weight
        else:
            # Fallback to simple average if weights don't work
            relevant_metrics = [
                metrics.get('direction_accuracy', 0),
                metrics.get('prediction_accuracy', 0),
                metrics.get('gain_prediction_accuracy', 0),
                metrics.get('loss_prediction_accuracy', 0),
                metrics.get('signal_accuracy', 0)
            ]
            valid_metrics = [m for m in relevant_metrics if m > 0]
            overall_accuracy = sum(valid_metrics) / len(valid_metrics) if valid_metrics else 0
        
        return max(0, min(100, overall_accuracy))  # Ensure between 0-100%
        
    except Exception as e:
        print_status(f"Error calculating overall accuracy: {e}", "warning")
        return 0.0

################################################################################3
# ----- Main CLI flow with Gain Prediction Enhancements -----

def parse_args():
    p = argparse.ArgumentParser(description="crprid.py - Enhanced Crypto Prediction Tool with Improved Gain Prediction")
    p.add_argument('coin', nargs='?', help='Coin id (coingecko) or name, e.g. bitcoin, ethereum')
    p.add_argument('--tf', '--timeframe', dest='timeframe', default='1h',
                   choices=['10m', '20m', '30m', '1h', '4h', '1d', '1y'], help='timeframe to analyze')
    p.add_argument('--symbol', help='Optional exchange symbol for minute-level data (e.g. BTCUSDT) to use Binance klines')
    p.add_argument('--vs', default='usd', help='Quote currency for CoinGecko (default usd)')
    p.add_argument('--limit', type=int, default=500, help='Number of candles to fetch (where applicable)')
    p.add_argument('--news', action='store_true', help='Fetch news sentiment (requires NEWSAPI_KEY env var)')
    p.add_argument('--ollama', action='store_true', help='Generate Ollama summary (requires local Ollama running)')
    p.add_argument('--ollama-host', default=os.environ.get('OLLAMA_HOST', 'http://localhost'), help='Ollama host')
    p.add_argument('--ollama-port', type=int, default=int(os.environ.get('OLLAMA_PORT', 11500)), help='Ollama port')
    p.add_argument('--model', default='gemma3:4b', help='Ollama model name')
    p.add_argument('--out', default='crprid_output', help='Output file prefix (CSV/XLSX)')
    p.add_argument('--simple', action='store_true', help='Simplified output for beginners')
    p.add_argument('--year-prediction', action='store_true', help='Generate 1-year price prediction')
    p.add_argument('--append', action='store_true', default=True, help='Append to existing Excel file (default: True)')
    p.add_argument('--check-accuracy', metavar='FILE', help='Check prediction accuracy against live data using FILE (e.g., trade.xlsx)')
    p.add_argument('--gain-focus', action='store_true', default=True, help='Focus on gain prediction (default: True)')
    
    return p.parse_args()

def create_simplified_export(coin: str, symbol: str, timeframe: str, last_price: float, 
                           predicted_price: float, recommendation: Dict, rsi_value: float, 
                           output_path: str, timestamp_str: str = None, append: bool = True):
    """Create simplified Excel export with append support"""
    if timestamp_str is None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    current_time = datetime.now()
    
    # Use predicted_price if available, otherwise use last_price
    final_predicted_price = predicted_price if predicted_price else last_price
    
    data = {
        'Sl No': [1],
        'Symbol': [symbol.upper() if symbol else coin.upper()],
        'Crypto Name': [coin.upper()],
        'Timestamp': [current_time],
        'Timeframe': [timeframe],
        'Current Price': [last_price],
        'Prediction Value': [final_predicted_price],
        'Signal': [recommendation.get('advice', 'HOLD')],
        'Stoploss': [recommendation.get('stoploss', 0)],
        'RSI': [rsi_value],
        'Buy Time': [current_time if recommendation.get('advice') in ['BUY', 'STRONG BUY'] else None],
        'Sale Time': [recommendation.get('sale_time')]
    }
    
    df = pd.DataFrame(data)
    
    try:
        if EXCEL_ENHANCED and (output_path.endswith('.xlsx') or not output_path.endswith('.csv')):
            # Ensure .xlsx extension
            if not output_path.endswith('.xlsx'):
                output_path = output_path.replace('.csv', '.xlsx').replace('.xlxs', '.xlsx')
                if not output_path.endswith('.xlsx'):
                    output_path += '.xlsx'
            
            # APPEND FUNCTIONALITY - FIXED: Don't delete old data
            if append and os.path.exists(output_path):
                try:
                    # Read existing data
                    existing_df = pd.read_excel(output_path, sheet_name='Trading Signals')
                    
                    # Append new data to existing data
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    
                    # Update serial numbers
                    combined_df['Sl No'] = range(1, len(combined_df) + 1)
                    
                    # Write combined data back to file
                    combined_df.to_excel(output_path, sheet_name='Trading Signals', index=False)
                    
                    print_status(f"Appended to existing file: {output_path} (Total records: {len(combined_df)})", "success")
                    return {'csv': output_path, 'xlsx': output_path, 'enhanced': True, 'appended': True, 'total_records': len(combined_df)}
                    
                except Exception as e:
                    print_status(f"Append failed: {e}, creating new file", "warning")
                    df.to_excel(output_path, sheet_name='Trading Signals', index=False)
                    return {'csv': output_path, 'xlsx': output_path, 'enhanced': True, 'appended': False, 'total_records': len(df)}
            else:
                # Create new file
                df.to_excel(output_path, sheet_name='Trading Signals', index=False)
                print_status(f"Created new file: {output_path}", "success")
                return {'csv': output_path, 'xlsx': output_path, 'enhanced': True, 'appended': False, 'total_records': len(df)}
        else:
            # CSV fallback with append
            if output_path.endswith('.xlsx'):
                output_path = output_path.replace('.xlsx', '.csv')
            
            if append and os.path.exists(output_path):
                try:
                    existing_df = pd.read_csv(output_path)
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df['Sl No'] = range(1, len(combined_df) + 1)
                    combined_df.to_csv(output_path, index=False)
                    print_status(f"Appended to CSV: {output_path} (Total records: {len(combined_df)})", "success")
                    return {'csv': output_path, 'xlsx': None, 'enhanced': False, 'appended': True, 'total_records': len(combined_df)}
                except Exception as e:
                    print_status(f"CSV append failed: {e}, creating new", "warning")
                    df.to_csv(output_path, index=False)
                    return {'csv': output_path, 'xlsx': None, 'enhanced': False, 'appended': False, 'total_records': len(df)}
            else:
                df.to_csv(output_path, index=False)
                print_status(f"Created new CSV: {output_path}", "success")
                return {'csv': output_path, 'xlsx': None, 'enhanced': False, 'appended': False, 'total_records': len(df)}
            
    except Exception as e:
        print_status(f"Export failed: {e}, using CSV fallback", "warning")
        # CSV fallback
        csv_path = output_path.replace('.xlsx', '.csv').replace('.xlxs', '.csv')
        df.to_csv(csv_path, index=False)
        return {'csv': csv_path, 'xlsx': None, 'enhanced': False, 'appended': False, 'total_records': len(df)}

def fetch_binance_recent_trades(symbol: str, limit: int = 100) -> pd.DataFrame:
    """Fetch recent trades from Binance"""
    try:
        url = f"{BINANCE_API}/aggTrades"
        params = {"symbol": symbol.upper(), "limit": limit}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        trades = r.json()
        df = pd.DataFrame(trades)
        if df.empty:
            return df
        df['price'] = df['p'].astype(float)
        df['qty'] = df['q'].astype(float)
        df['timestamp'] = pd.to_datetime(df['T'], unit='ms')
        return df[['timestamp', 'price', 'qty']]
    except Exception as e:
        print(f"Error fetching Binance trades: {e}")
        return pd.DataFrame()

def analyze_whales_from_trades(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze whale transactions from trades data"""
    if trades_df is None or trades_df.empty:
        return {'available': False, 'reason': 'no_trades'}
    
    try:
        q = trades_df['qty']
        median = q.median()
        threshold = median * 10
        whales = trades_df[trades_df['qty'] >= threshold]
        
        return {
            'available': True, 
            'median_qty': float(median), 
            'threshold': float(threshold), 
            'whales_count': len(whales),
            'whales': whales.to_dict(orient='records') if not whales.empty else []
        }
    except Exception as e:
        return {'available': False, 'reason': f'analysis_error: {str(e)}'}


def fetch_news_sentiment(query: str, api_key: Optional[str], page_size: int = 5) -> Dict[str, Any]:
    """Fetch news sentiment"""
    if not api_key:
        return {'available': False, 'reason': 'No NEWSAPI_KEY provided'}
    
    url = "https://newsapi.org/v2/everything"
    params = {'q': query, 'language': 'en', 'pageSize': page_size, 'sortBy': 'publishedAt'}
    headers = {'Authorization': api_key}
    
    try:
        r = requests.get(url, params=params, headers=headers, timeout=1500)
        if r.status_code != 200:
            return {'available': False, 'reason': f"newsapi error {r.status_code}: {r.text}"}
        data = r.json()
        articles = data.get('articles', [])
        
        # Simple sentiment analysis
        pos_words = {'gain', 'surge', 'bull', 'rise', 'record', 'beat', 'positive', 'up'}
        neg_words = {'drop', 'crash', 'bear', 'decline', 'fall', 'hack', 'negative', 'down', 'dump'}
        
        scores = []
        for a in articles:
            text = (a.get('title') or '') + ' ' + (a.get('description') or '')
            t = text.lower()
            score = sum(1 for w in pos_words if w in t) - sum(1 for w in neg_words if w in t)
            scores.append({
                'title': a.get('title'), 
                'publishedAt': a.get('publishedAt'), 
                'score': score, 
                'url': a.get('url')
            })
        
        avg = float(np.mean([s['score'] for s in scores])) if scores else 0.0
        return {'available': True, 'avg_score': avg, 'articles': scores}
        
    except Exception as e:
        return {'available': False, 'reason': f"Exception: {str(e)}"}

def main():
    args = parse_args()

    # Initialize all variables at the start
    last_price = 0
    predicted_next = None
    future_predictions = None
    recommendation = None
    ml_out = None
    export_result = {'csv': 'not_exported', 'xlsx': None, 'enhanced': False}
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.check_accuracy:
        if not os.path.exists(args.check_accuracy):
            print_status(f"File not found: {args.check_accuracy}", "error")
            sys.exit(1)
        
        result = check_prediction_accuracy(args.check_accuracy, args.vs)
        
        if 'error' in result:
            print_status(f"Accuracy check failed: {result['error']}", "error")
            sys.exit(1)
        else:
            print_header("ACCURACY CHECK COMPLETE ✅", 1)
            return result
    
    if not args.coin:
        print_status("Error: Coin argument required for prediction mode", "error")
        print_status("Use --check-accuracy FILE for accuracy checking mode", "info")
        sys.exit(1)
    
    coin = args.coin.lower()
    timeframe = args.timeframe
    vs = args.vs.lower()
    limit = args.limit
    symbol = args.symbol
    news_flag = args.news
    ollama_flag = args.ollama
    simple_mode = args.simple
    year_prediction_flag = args.year_prediction
    append_mode = args.append
    gain_focus = args.gain_focus

    print_header(f"CRPRID - GAIN-OPTIMIZED CRYPTO ANALYSIS TOOL", 1)
    print_bullet(f"Coin: {Fore.CYAN}{coin.upper()}{Fore.WHITE} | Timeframe: {Fore.CYAN}{timeframe}{Fore.WHITE} | Currency: {Fore.CYAN}{vs.upper()}", Fore.WHITE, 0)
    if year_prediction_flag:
        print_bullet(f"1-Year Prediction: {Fore.GREEN}ENABLED", Fore.WHITE, 0)
    if gain_focus:
        print_bullet(f"Gain Prediction Focus: {Fore.GREEN}ENABLED", Fore.WHITE, 0)
    if append_mode and EXCEL_ENHANCED:
        print_bullet(f"Excel Append Mode: {Fore.GREEN}ENABLED", Fore.WHITE, 0)

    try:
        # 1) Fetch price data
        print_header("DATA COLLECTION", 2)
        df_ohlcv = None
        try:
            if symbol and timeframe in ['10m', '20m', '30m', '1h', '4h']:
                tf_map = {'10m': '10m', '20m': '20m', '30m': '30m', '1h': '1h', '4h': '4h'}
                interval = tf_map[timeframe]
                print_status(f"Fetching Binance data for {symbol} ({interval})...", "info")
                df_ohlcv = fetch_binance_klines(symbol=symbol, interval=interval, limit=limit)
            else:
                if timeframe == '1y':
                    days = 365 * 2
                elif timeframe == '1d':
                    days = 365
                elif timeframe == '4h':
                    days = 60
                elif timeframe == '1h':
                    days = 30
                elif timeframe == '30m':
                    days = 7
                elif timeframe == '20m':
                    days = 5
                elif timeframe == '10m':
                    days = 3
                else:
                    days = 30
                
                print_status(f"Fetching CoinGecko data for {coin} (last {days} days)...", "info")
                raw = fetch_coin_gecko_market_chart(coin_id=coin, vs_currency=vs, days=days)
                df_ohlcv = coingecko_ohlcv_from_prices(raw, timeframe)
        except Exception as e:
            print_status(f"Error fetching price data: {e}", "error")
            sys.exit(1)

        if df_ohlcv is None or df_ohlcv.empty:
            print_status("No OHLCV data available.", "error")
            sys.exit(1)

        print_status(f"Fetched {len(df_ohlcv)} price records", "success")

        # 2) Add enhanced technical indicators with gain focus
        print_header("TECHNICAL ANALYSIS", 2)
        print_status("Calculating gain-optimized technical indicators...", "info")
        df = add_technical_indicators(df_ohlcv)

        if df.empty:
            print_status("Not enough data for analysis after calculating technical indicators.", "error")
            sys.exit(1)

        # 3) Enhanced ML Prediction with gain optimization
                # 3) Enhanced ML Prediction with gain optimization
        available_features = [col for col in SIMPLIFIED_FEATURE_COLS if col in df.columns]
        df_features = df.dropna(subset=available_features + ['close']).reset_index(drop=True)
        
        ml_out = None
        future_predictions = None
        ml_confidence = 0.5

        if not df_features.empty and len(df_features) >= 100:
            try:
                print_status("Training gain-optimized machine learning model...", "info")
                ml_out = train_direction_predictor(df_features, available_features)
                
                if ml_out and ml_out.get('test_accuracy', 0) > 0.5:  # Use accuracy instead of R²
                    ml_confidence = ml_out['test_accuracy']
                    print_status(f"ML model trained successfully - Accuracy: {ml_confidence:.2%}", "success")
                else:
                    print_status("ML model quality too poor for predictions", "warning")
                    ml_out = None
                    
            except Exception as e:
                print_status(f"ML training failed: {e}", "warning")
                ml_out = None
        else:
            print_status("Insufficient data for ML prediction", "warning")

        # 4) Get market data - ENSURED TO RUN
        try:
            if ml_out and ml_out.get('last_row') is not None:
                last_row = ml_out['last_row']
            elif not df_features.empty:
                last_row = df_features.iloc[-1]
            elif not df.empty:
                last_row = df.iloc[-1]
            else:
                last_row = df_ohlcv.iloc[-1]
                
            last_price = float(last_row['close'])
            last_atr = float(last_row['atr14']) if 'atr14' in last_row and is_valid_value(last_row['atr14']) else 0.02 * last_price
            last_rsi = float(last_row['rsi14']) if 'rsi14' in last_row and is_valid_value(last_row['rsi14']) else 50.0
            ema8_val = float(last_row['ema8']) if 'ema8' in last_row and is_valid_value(last_row['ema8']) else last_price
            ema21_val = float(last_row['ema21']) if 'ema21' in last_row and is_valid_value(last_row['ema21']) else last_price
            
            trend_strength = float(last_row['trend_strength_20']) if 'trend_strength_20' in last_row and is_valid_value(last_row['trend_strength_20']) else 0
            volume_surge = float(last_row['volume_surge_ratio']) if 'volume_surge_ratio' in last_row and is_valid_value(last_row['volume_surge_ratio']) else 1.0
            
        except Exception as e:
            print_status(f"Error accessing market data: {e}", "error")
            last_price = float(df_ohlcv['close'].iloc[-1]) if not df_ohlcv.empty else 0
            last_atr = 0.02 * last_price
            last_rsi = 50.0
            ema8_val = last_price
            ema21_val = last_price
            trend_strength = 0
            volume_surge = 1.0

        # 5) Prediction with fallback
        predicted_next = None
        if ml_out and ml_out.get('model') is not None:
            try:
                # Use the new direction prediction
                direction_result = predict_direction(ml_out['model'], last_row, available_features)
                
                # Convert direction to price prediction
                if direction_result['direction'] == 'UP':
                    # Predict a small gain (you can adjust this)
                    predicted_next = last_price * 1.01  # 1% gain
                else:
                    # Predict a small loss
                    predicted_next = last_price * 0.99  # 1% loss
                    
                ml_confidence = direction_result['confidence']
                
            except Exception as e:
                print_status(f"ML prediction failed: {e}", "warning")
                predicted_next = None

        # Fallback prediction
        if predicted_next is None and last_price > 0:
            # Simple momentum prediction
            if len(df) > 5:
                momentum = df['close'].pct_change(3).mean()
                predicted_next = last_price * (1 + momentum * 0.3)
                ml_confidence = 0.3
            else:
                predicted_next = last_price
                ml_confidence = 0.1
 ################################################################################################

                # 5) Prediction with fallback
        predicted_next = None
        if ml_out and ml_out.get('model') is not None:
            try:
                # Use the new direction prediction
                direction_result = predict_direction(ml_out['model'], last_row, available_features)
                
                # Convert direction to price prediction
                if direction_result['direction'] == 'UP':
                    # Predict a small gain (you can adjust this)
                    predicted_next = last_price * 1.01  # 1% gain
                else:
                    # Predict a small loss
                    predicted_next = last_price * 0.99  # 1% loss
                    
                ml_confidence = direction_result['confidence']
                
            except Exception as e:
                print_status(f"ML prediction failed: {e}", "warning")
                predicted_next = None

        # Fallback prediction
        if predicted_next is None and last_price > 0:
            # Simple momentum prediction
            if len(df) > 5:
                momentum = df['close'].pct_change(3).mean()
                predicted_next = last_price * (1 + momentum * 0.3)
                ml_confidence = 0.3
            else:
                predicted_next = last_price
                ml_confidence = 0.1

        # 6) Generate recommendation - ENSURED TO RUN
        try:
            recommendation = enhanced_recommend_trade(
                last_price, predicted_next, last_atr, last_rsi, ml_confidence, timeframe,
                trend_strength, volume_surge
            )
            print_status("Trading recommendation generated", "success")
        except Exception as e:
            print_status(f"Recommendation failed: {e}, using fallback", "error")
            # Fallback recommendation
            recommendation = {
                'advice': 'HOLD',
                'confidence': 'Low',
                'gain_outlook': 'Technical Issues',
                'diff_pct': 0,
                'stoploss': last_price * 0.95 if last_price > 0 else 0,
                'take_profit': last_price * 1.05 if last_price > 0 else 0,
                'risk_reward_ratio': 1.0,
                'ml_confidence': 0.1
            }

        # 7) Display results
        print_header("📊 MARKET ANALYSIS RESULTS", 1)
        
        # Price info
        print_header("Price Information", 2)
        print_bullet(f"Current Price: {Fore.MAGENTA}${last_price:.2f}", Fore.WHITE)
        if predicted_next:
            change_pct = ((predicted_next - last_price) / last_price) * 100
            change_color = Fore.GREEN if change_pct > 0 else Fore.RED
            print_bullet(f"Predicted Price: {change_color}${predicted_next:.2f}", Fore.WHITE)
            print_bullet(f"Expected Change: {change_color}{change_pct:.2f}%", Fore.WHITE)

        # Technical indicators
        print_header("Technical Indicators", 2)
        rsi_strength, rsi_color = get_rsi_strength(last_rsi)
        trend_strength_text, trend_color = get_trend_strength(ema8_val, ema21_val, last_price)
        print_bullet(f"RSI (14): {rsi_color}{last_rsi:.1f} - {rsi_strength}", Fore.WHITE)
        print_bullet(f"Trend: {trend_color}{trend_strength_text}", Fore.WHITE)

        # Trading recommendation
        print_header("TRADING RECOMMENDATION", 2)
        if recommendation:
            advice_color = Fore.GREEN if 'BUY' in recommendation['advice'] else Fore.RED if 'SELL' in recommendation['advice'] else Fore.YELLOW
            print_bullet(f"Action: {advice_color}{recommendation['advice']}", Fore.WHITE)
            print_bullet(f"Confidence: {recommendation['confidence']}", Fore.WHITE)
            print_bullet(f"Stop Loss: ${recommendation.get('stoploss', 0):.2f}", Fore.WHITE)
            print_bullet(f"Take Profit: ${recommendation.get('take_profit', 0):.2f}", Fore.WHITE)

        # 8) News sentiment analysis with output
        news_result = None
        if news_flag:
            key = os.environ.get('NEWSAPI_KEY')
            print_status("Analyzing news sentiment...", "info")
            news_result = fetch_news_sentiment(coin, key)
            
            # Display news results immediately
            if news_result and news_result.get('available'):
                print_header("NEWS SENTIMENT ANALYSIS", 2)
                avg_score = news_result.get('avg_score', 0)
                sentiment = "POSITIVE" if avg_score > 0.5 else "NEGATIVE" if avg_score < -0.5 else "NEUTRAL"
                sentiment_color = Fore.GREEN if sentiment == "POSITIVE" else Fore.RED if sentiment == "NEGATIVE" else Fore.YELLOW
                
                print_bullet(f"Overall Sentiment: {sentiment_color}{sentiment}", Fore.WHITE)
                print_bullet(f"Average Score: {avg_score:.2f}", Fore.WHITE)
                print_bullet(f"Articles Analyzed: {len(news_result.get('articles', []))}", Fore.WHITE)
                
                # Show top articles
                articles = sorted(news_result.get('articles', []), key=lambda x: abs(x.get('score', 0)), reverse=True)[:3]
                if articles:
                    print_bullet("Top Articles:", Fore.WHITE)
                    for i, article in enumerate(articles, 1):
                        score_color = Fore.GREEN if article.get('score', 0) > 0 else Fore.RED if article.get('score', 0) < 0 else Fore.YELLOW
                        title = article.get('title', 'No title')[:60] + "..." if len(article.get('title', '')) > 60 else article.get('title', 'No title')
                        print_bullet(f"  {i}. {score_color}{article.get('score', 0):+d} {Fore.WHITE}- {title}", Fore.WHITE, 4)
            else:
                print_header("NEWS SENTIMENT", 2)
                reason = news_result.get('reason', 'Unknown reason') if news_result else 'No API key or other issue'
                print_bullet(f"News analysis unavailable: {reason}", Fore.YELLOW)

        # 9) Whale analysis with output
        whale_res = None
        if symbol:
            try:
                print_status("Analyzing whale transactions...", "info")
                trades = fetch_binance_recent_trades(symbol=symbol, limit=200)
                whale_res = analyze_whales_from_trades(trades)
                
                # Display whale results immediately
                if whale_res and whale_res.get('available'):
                    print_header("WHALE TRANSACTION ANALYSIS", 2)
                    whales_count = whale_res.get('whales_count', 0)
                    median_qty = whale_res.get('median_qty', 0)
                    threshold = whale_res.get('threshold', 0)
                    
                    whale_activity = "HIGH" if whales_count > 10 else "MODERATE" if whales_count > 5 else "LOW"
                    whale_color = Fore.GREEN if whale_activity == "HIGH" else Fore.YELLOW if whale_activity == "MODERATE" else Fore.RED
                    
                    print_bullet(f"Whale Activity Level: {whale_color}{whale_activity}", Fore.WHITE)
                    print_bullet(f"Large Transactions: {whales_count}", Fore.WHITE)
                    print_bullet(f"Median Trade Size: {median_qty:.4f}", Fore.WHITE)
                    print_bullet(f"Whale Threshold: {threshold:.4f}", Fore.WHITE)
                    
                    # Show recent whale trades
                    whales = whale_res.get('whales', [])
                    if whales:
                        recent_whales = sorted(whales, key=lambda x: x.get('timestamp', ''), reverse=True)[:3]
                        print_bullet("Recent Whale Trades:", Fore.WHITE)
                        for i, whale in enumerate(recent_whales, 1):
                            timestamp = whale.get('timestamp', '')
                            if isinstance(timestamp, str):
                                time_str = timestamp
                            else:
                                time_str = timestamp.strftime('%H:%M:%S') if hasattr(timestamp, 'strftime') else str(timestamp)
                            
                            price = whale.get('price', 0)
                            qty = whale.get('qty', 0)
                            print_bullet(f"  {i}. {time_str} - {qty:.4f} @ ${price:.2f}", Fore.WHITE, 4)
                else:
                    print_header("WHALE ANALYSIS", 2)
                    reason = whale_res.get('reason', 'Unknown reason') if whale_res else 'No data'
                    print_bullet(f"Whale analysis unavailable: {reason}", Fore.YELLOW)
                    
            except Exception as e:
                whale_res = {'available': False, 'reason': str(e)}
                print_header("WHALE ANALYSIS", 2)
                print_bullet(f"Whale analysis failed: {e}", Fore.RED)
        else:
            print_header("WHALE ANALYSIS", 2)
            print_bullet("Whale analysis skipped (no symbol provided)", Fore.YELLOW)

        # 10) Ollama summary
        ollama_summary = None
        if ollama_flag:
            print_status("Generating AI analysis...", "info")
            prompt = f"Provide a concise trading summary for {coin} at ${last_price:.2f} based on {timeframe} timeframe. Key indicators: RSI={last_rsi:.1f}, Trend analysis needed. Give simple advice."
            try:
                out = generate_ollama_summary(prompt, host=args.ollama_host, port=args.ollama_port, model=args.model)
                if out.get('ok'):
                    ollama_summary = out.get('text')
                else:
                    ollama_summary = f"AI analysis unavailable: {out.get('error')}"
            except Exception as e:
                ollama_summary = f"AI analysis failed: {e}"

        # 11) Data export
        print_header("DATA EXPORT", 2)
                     # FIX: Handle custom --out parameter properly for append mode
        if args.out == 'crprid_output':
                       output_file = f"{args.out}_trading_signals.xlsx"
        else:
                     # If custom --out is provided, use it directly
                      if args.out.endswith(('.xlsx', '.csv')):
                       output_file = args.out
                      else:
                       output_file = f"{args.out}.xlsx"

        try:
                      export_result = create_simplified_export(
                      coin=coin,
                      symbol=symbol,
                      timeframe=timeframe,
                      last_price=last_price,
                      predicted_price=predicted_next,
                      recommendation=recommendation,
                      rsi_value=last_rsi,
                      output_path=output_file,
                      timestamp_str=timestamp_str,
		      append=append_mode
    )
    
                      if export_result['enhanced']:
                       print_status(f"Simplified trading data exported to: {export_result['xlsx']}", "success")
                      else:
                       print_status(f"Simplified trading data exported to: {export_result['csv']}", "success")
        
        except Exception as e:
                      print_status(f"Export failed: {e}", "warning")

        # 12) Display 1-year prediction if available
        if future_predictions is not None and not future_predictions.empty:
            print_header("1-YEAR PRICE PREDICTION", 2)
            current_price = last_price
            predicted_1y = future_predictions['predicted_price'].iloc[-1]
            change_pct = ((predicted_1y - current_price) / current_price) * 100
            avg_confidence = future_predictions['confidence'].mean()
            
            print_bullet(f"Current Price: ${current_price:.8f}", Fore.WHITE)
            print_bullet(f"Predicted 1-Year Price: ${predicted_1y:.8f}", Fore.GREEN if change_pct > 0 else Fore.RED)
            print_bullet(f"Expected Change: {change_pct:.2f}%", Fore.GREEN if change_pct > 0 else Fore.RED)
            print_bullet(f"Average Confidence: {avg_confidence:.1%}", Fore.WHITE)
            
            pred_csv = f"{args.out}_1y_prediction.csv"
            future_predictions.to_csv(pred_csv, index=False)
            print_status(f"1-year prediction saved to: {pred_csv}", "success")

        # 13) Simple Mode Summary
        if simple_mode:
            print_header("🚀 QUICK SUMMARY", 1)
            advice_color = Fore.GREEN if 'BUY' in recommendation['advice'] else Fore.RED if 'SELL' in recommendation['advice'] else Fore.YELLOW
            print_bullet(f"Action: {advice_color}{recommendation['advice']}", Fore.WHITE, 0)
            print_bullet(f"Price: ${last_price:.2f} | Trend: {trend_strength_text}", Fore.WHITE, 0)
            print_bullet(f"Confidence: {recommendation['confidence']}", Fore.WHITE, 0)
            if recommendation.get('sale_time'):
                print_bullet(f"Sale Time: {recommendation['sale_time'].strftime('%Y-%m-%d %H:%M')}", Fore.WHITE, 0)
        
        # 14) Ollama AI Summary
        if ollama_summary and not simple_mode:
            print_header("AI ANALYSIS", 2)
            print(f"{Fore.CYAN}{ollama_summary}")
        
        # 15) Final Disclaimer
        print_header("⚠️  DISCLAIMER", 2)
        print_bullet("This analysis is for educational purposes only", Fore.YELLOW)
        print_bullet("Always do your own research before trading", Fore.YELLOW)
        print_bullet("Never invest more than you can afford to lose", Fore.YELLOW)

    except Exception as e:
        print_status(f"Critical error in main analysis: {e}", "error")
        # Ensure we have basic values even after critical error
        if last_price == 0:
            last_price = 1  # Fallback value
        if recommendation is None:
            recommendation = {
                'advice': 'ERROR',
                'confidence': 'None',
                'gain_outlook': 'Analysis Failed'
            }

    print_header("ENHANCED ANALYSIS COMPLETE ✅", 1)

    return {
        'last_price': last_price,
        'predicted_next': predicted_next,
        'future_predictions': future_predictions,
        'recommendation': recommendation,
        'export_paths': export_result,
        'ml_performance': ml_out,
        'timestamp': timestamp_str

    }


if __name__ == "__main__":
    res = main()