#!/usr/bin/env python3
"""
pridict.py - Enhanced Crypto Prediction Tool
Features implemented:
- Direct current price fetching from CoinGecko API
- Multi-timeframe support (10m, 20m, 30m, 1h, 4h, 1d)
- ML prediction with technical indicators (RandomForest)
- Optional local Ollama AI (gemma3:4b) integration (configurable)
- News sentiment analysis via NewsAPI (optional)
- Whale transaction analysis via Binance recent trades (basic)
- Trading recommendations and risk/reward analysis (simple)
- Enhanced visual output with colors and formatting (colorama)
- Excel export functionality (.xlsx and .csv)
"""

import os
import sys
import time
import math
import json
import argparse
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any

import requests
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# color output
try:
    from colorama import init as colorama_init, Fore, Style, Back
    colorama_init(autoreset=True)
except Exception:
    # fallback no color
    class _C:
        def __getattr__(self, name): return ""
    Fore = Style = Back = _C()

# Optional imports
try:
    import ollama  # local ollama client if installed
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

# Constants
COINGECKO_API = "https://api.coingecko.com/api/v3"
BINANCE_API = "https://api.binance.com/api/v3"

# ----- Utilities and indicators -----

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=period).mean()
    ma_down = down.rolling(window=period, min_periods=period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

def is_valid_value(x) -> bool:
    """Check if a value is valid (not None, NaN, or Inf)"""
    try:
        return x is not None and not (math.isnan(x) or math.isinf(x))
    except Exception:
        return False

# ----- Data fetchers -----

def fetch_coin_gecko_market_chart(coin_id: str, vs_currency: str, days: float) -> pd.DataFrame:
    """
    Fetch coin market chart (prices, market_caps, total_volumes).
    - days can be fractional (e.g., 0.0208 for 30 minutes) but CoinGecko may restrict minute-resolution to 1 day.
    Returns DataFrame with columns: timestamp, price, market_cap, total_volume
    """
    url = f"{COINGECKO_API}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    prices = data.get("prices", [])
    # prices: [ [timestamp_ms, price], ... ]
    df = pd.DataFrame(prices, columns=['timestamp_ms', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df = df[['timestamp', 'price']]
    return df

def coingecko_ohlcv_from_prices(df_prices: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Convert raw price points to OHLCV by resampling.
    timeframe examples: '10m', '20m', '30m', '1h', '4h', '1d'
    """
    rule_map = {
        '10m': '10min', '20m': '20min', '30m': '30min',
        '1h': '1H', '4h': '4H', '1d': '1D'
    }
    if timeframe not in rule_map:
        raise ValueError("Unsupported timeframe for resample")
    rule = rule_map[timeframe]
    df = df_prices.set_index('timestamp').resample(rule).agg({'price': ['first', 'max', 'min', 'last', 'count']})
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.dropna().reset_index()
    return df

def fetch_binance_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """
    Fetch klines from Binance: returns DataFrame with timestamp, open, high, low, close, volume
    interval examples: '1m', '5m', '15m', '1h', '4h', '1d'
    """
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

# ----- Feature engineering -----

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)

    df['ema8'] = ema(df['close'], 8)
    df['ema21'] = ema(df['close'], 21)
    df['rsi14'] = rsi(df['close'], 14)
    macd_line, signal_line, hist = macd(df['close'])
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = hist
    df['atr14'] = atr(df, 14)
    # returns
    df['ret1'] = df['close'].pct_change(1)
    df['ret3'] = df['close'].pct_change(3)
    df = df.dropna().reset_index(drop=True)
    return df

# ----- ML model -----

def train_ml_predictor(df: pd.DataFrame, feature_cols: list, target_col: str = 'close', test_size: float = 0.15):
    """
    Train a RandomForestRegressor to predict next candle close.
    Returns (model, X_test, y_test, preds)
    """
    df = df.copy()
    # target is next candle close
    df['target'] = df[target_col].shift(-1)
    df = df.dropna()
    
    if df.empty:
        return None
        
    X = df[feature_cols]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'preds': preds,
        'mae': mae,
        'last_row': df.iloc[-1] if not df.empty else None
    }

# ----- News sentiment (simple) -----

def fetch_news_sentiment(query: str, api_key: Optional[str], page_size: int = 5) -> Dict[str, Any]:
    """
    Uses NewsAPI if api_key provided. Returns simple sentiment (pos/neg/neutral) based on naive word list.
    """
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
        # very naive sentiment
        pos_words = {'gain', 'surge', 'bull', 'rise', 'record', 'beat', 'positive', 'up'}
        neg_words = {'drop', 'crash', 'bear', 'decline', 'fall', 'hack', 'negative', 'down', 'dump'}
        scores = []
        for a in articles:
            text = (a.get('title') or '') + ' ' + (a.get('description') or '')
            t = text.lower()
            score = sum(1 for w in pos_words if w in t) - sum(1 for w in neg_words if w in t)
            scores.append({'title': a.get('title'), 'publishedAt': a.get('publishedAt'), 'score': score, 'url': a.get('url')})
        avg = float(np.mean([s['score'] for s in scores])) if scores else 0.0
        return {'available': True, 'avg_score': avg, 'articles': scores}
    except Exception as e:
        return {'available': False, 'reason': f"Exception: {str(e)}"}

# ----- Whale check (basic using Binance recent trades) -----

def fetch_binance_recent_trades(symbol: str, limit: int = 100) -> pd.DataFrame:
    try:
        url = f"{BINANCE_API}/aggTrades"
        params = {"symbol": symbol.upper(), "limit": limit}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        trades = r.json()
        # trades: [{a,b,p,q,f,l,T,m,...}]
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
    if trades_df is None or trades_df.empty:
        return {'available': False, 'reason': 'no_trades'}
    q = trades_df['qty']
    median = q.median()
    threshold = median * 10  # heuristic: 10x median quantity is considered a whale
    whales = trades_df[trades_df['qty'] >= threshold]
    return {'available': True, 'median_qty': float(median), 'threshold': float(threshold), 'whales': whales.to_dict(orient='records')}

# ----- Ollama integration (optional) -----

def generate_ollama_summary(prompt: str, host: str = "http://localhost", port: int = 11500, model: str = "gemma3:4b") -> Dict[str, Any]:
    """Fixed URL construction for Ollama API"""
    try:
        # Fix URL construction - remove duplicate port
        base_url = host
        if not base_url.startswith(('http://', 'https://')):
            base_url = f"http://{base_url}"
        
        # Clean up port formatting
        if ':' in base_url and base_url.rfind(':') > 4:  # If port already in host
            url = f"{base_url}/api/generate"
        else:
            url = f"{base_url}:{port}/api/generate"
        
        payload = {"model": model, "prompt": prompt, "stream": False}
        r = requests.post(url, json=payload, timeout=3000)
        r.raise_for_status()
        resp = r.json()
        return {'ok': True, 'text': resp.get('response', '')}
    except requests.exceptions.ConnectionError:
        return {'ok': False, 'error': 'Ollama not running - start with: ollama serve'}
    except Exception as e:
        return {'ok': False, 'error': f'Ollama error: {str(e)}'}

# ----- Helpers and output -----

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

def recommend_trade(last_price: float, predicted_price: float, atr_val: float, rsi_val: float) -> Dict[str, Any]:
    """
    Enhanced recommendation with RSI consideration
    """
    diff = predicted_price - last_price if predicted_price is not None else 0
    diff_pct = (diff / last_price) * 100.0 if predicted_price is not None else 0
    
    # Multi-factor decision
    factors = []
    
    # Price prediction factor
    if diff_pct > 1.0:
        factors.append(("price_prediction", "strong_bullish", 2))
    elif diff_pct > 0.3:
        factors.append(("price_prediction", "bullish", 1))
    elif diff_pct < -1.0:
        factors.append(("price_prediction", "strong_bearish", -2))
    elif diff_pct < -0.3:
        factors.append(("price_prediction", "bearish", -1))
    else:
        factors.append(("price_prediction", "neutral", 0))
    
    # RSI factor
    if rsi_val > 70:
        factors.append(("rsi", "overbought", -1))
    elif rsi_val < 30:
        factors.append(("rsi", "oversold", 1))
    else:
        factors.append(("rsi", "neutral", 0))
    
    # Calculate weighted score
    total_score = sum(score for _, _, score in factors)
    
    # Determine advice
    if total_score >= 2:
        advice = 'STRONG BUY'
        confidence = 'High'
    elif total_score >= 1:
        advice = 'BUY'
        confidence = 'Medium'
    elif total_score <= -2:
        advice = 'STRONG SELL'
        confidence = 'High'
    elif total_score <= -1:
        advice = 'SELL'
        confidence = 'Medium'
    else:
        advice = 'HOLD'
        confidence = 'Low'
    
    # Risk management
    stoploss = last_price - (2 * atr_val) if is_valid_value(atr_val) and atr_val > 0 else last_price * 0.98  # 2% stoploss fallback
    take_profit = predicted_price if predicted_price else last_price * 1.02  # 2% target if no prediction
    
    return {
        'advice': advice,
        'confidence': confidence,
        'diff_pct': diff_pct,
        'factors': factors,
        'total_score': total_score,
        'stoploss': stoploss,
        'take_profit': take_profit,
        'risk_reward_ratio': (take_profit - last_price) / (last_price - stoploss) if stoploss and stoploss < last_price else None
    }

def export_results(df: pd.DataFrame, output_prefix: str):
    csv_path = f"{output_prefix}.csv"
    xlsx_path = f"{output_prefix}.xlsx"
    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception:
        # openpyxl may be missing, but CSV at least is created
        pass
    return {'csv': csv_path, 'xlsx': xlsx_path}

def get_rsi_strength(rsi_value: float) -> Tuple[str, str]:
    """Get RSI strength description"""
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

def get_trend_strength(ema_short: float, ema_long: float, price: float) -> Tuple[str, str]:
    """Determine trend strength based on EMAs and price"""
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

# ----- Main CLI flow -----

def parse_args():
    p = argparse.ArgumentParser(description="pridict.py - Enhanced Crypto Prediction Tool")
    p.add_argument('coin', help='Coin id (coingecko) or name, e.g. bitcoin, ethereum')
    p.add_argument('--tf', '--timeframe', dest='timeframe', default='1h',
                   choices=['10m', '20m', '30m', '1h', '4h', '1d'], help='timeframe to analyze')
    p.add_argument('--symbol', help='Optional exchange symbol for minute-level data (e.g. BTCUSDT) to use Binance klines')
    p.add_argument('--vs', default='usd', help='Quote currency for CoinGecko (default usd)')
    p.add_argument('--limit', type=int, default=500, help='Number of candles to fetch (where applicable)')
    p.add_argument('--news', action='store_true', help='Fetch news sentiment (requires NEWSAPI_KEY env var)')
    p.add_argument('--ollama', action='store_true', help='Generate Ollama summary (requires local Ollama running)')
    p.add_argument('--ollama-host', default=os.environ.get('OLLAMA_HOST', 'http://localhost'), help='Ollama host')
    p.add_argument('--ollama-port', type=int, default=int(os.environ.get('OLLAMA_PORT', 11500)), help='Ollama port')  # Fixed default port
    p.add_argument('--model', default='gemma3:4b', help='Ollama model name')
    p.add_argument('--out', default='pridict_output', help='Output file prefix (CSV/XLSX)')
    p.add_argument('--simple', action='store_true', help='Simplified output for beginners')
    return p.parse_args()

def main():
    args = parse_args()
    coin = args.coin.lower()
    timeframe = args.timeframe
    vs = args.vs.lower()
    limit = args.limit
    symbol = args.symbol
    news_flag = args.news
    ollama_flag = args.ollama
    simple_mode = args.simple

    print_header(f"PRIDICT - CRYPTO ANALYSIS TOOL", 1)
    print_bullet(f"Coin: {Fore.CYAN}{coin.upper()}{Fore.WHITE} | Timeframe: {Fore.CYAN}{timeframe}{Fore.WHITE} | Currency: {Fore.CYAN}{vs.upper()}", Fore.WHITE, 0)

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
            if timeframe == '1d':
                days = 30
            elif timeframe == '4h':
                days = 7
            elif timeframe == '1h':
                days = 2
            else:
                days = 1.5
            print_status(f"Fetching CoinGecko data for {coin} (last {days} days)...", "info")
            raw = fetch_coin_gecko_market_chart(coin_id=coin, vs_currency=vs, days=days)
            df_ohlcv = coingecko_ohlcv_from_prices(raw, timeframe)
    except Exception as e:
        print_status(f"Error fetching price data: {e}", "error")
        sys.exit(1)

    if df_ohlcv is None or df_ohlcv.empty:
        print_status("No OHLCV data available.", "error")
        sys.exit(1)

    # Ensure columns are present
    for c in ['open', 'high', 'low', 'close', 'volume', 'timestamp']:
        if c not in df_ohlcv.columns:
            print_status(f"Missing column {c} in OHLCV", "error")
            sys.exit(1)

    print_status(f"Successfully loaded {len(df_ohlcv)} candles", "success")
    print_bullet(f"Date range: {df_ohlcv['timestamp'].iloc[0]} to {df_ohlcv['timestamp'].iloc[-1]}", Fore.WHITE)

    # 2) Add indicators
    print_header("TECHNICAL ANALYSIS", 2)
    print_status("Calculating technical indicators...", "info")
    df = add_technical_indicators(df_ohlcv)

    # Check if we have enough data after technical indicator calculation
    if df.empty:
        print_status("Not enough data for analysis after calculating technical indicators.", "error")
        sys.exit(1)

    # 3) Whale analysis
    whale_res = None
    if symbol:
        try:
            print_status("Analyzing whale transactions...", "info")
            trades = fetch_binance_recent_trades(symbol=symbol, limit=200)
            whale_res = analyze_whales_from_trades(trades)
        except Exception as e:
            whale_res = {'available': False, 'reason': str(e)}

    # 4) Train ML predictor
    feature_cols = ['ema8', 'ema21', 'rsi14', 'macd', 'macd_signal', 'macd_hist', 'atr14', 'ret1', 'ret3', 'volume']
    df_features = df.dropna(subset=feature_cols + ['close']).reset_index(drop=True)
    
    # Check if we have enough features data
    if df_features.empty:
        print_status("Not enough complete data for ML prediction after feature engineering.", "warning")
        ml_out = None
    elif len(df_features) < 50:
        print_status(f"Only {len(df_features)} complete data points available (minimum 50 recommended for ML).", "warning")
        ml_out = None
    else:
        try:
            print_status("Training machine learning model...", "info")
            ml_out = train_ml_predictor(df_features, feature_cols)
            if ml_out is None:
                print_status("ML training returned no results.", "warning")
            else:
                print_status(f"Model trained successfully (MAE: {ml_out['mae']:.6f})", "success")
        except Exception as e:
            print_status(f"ML training failed: {e}", "warning")
            ml_out = None

    # Get current market state - SAFE ACCESS
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
        
        # Safely get other indicators with fallbacks
        last_atr = float(last_row['atr14']) if 'atr14' in last_row and is_valid_value(last_row['atr14']) else 0.02 * last_price  # 2% fallback
        last_rsi = float(last_row['rsi14']) if 'rsi14' in last_row and is_valid_value(last_row['rsi14']) else 50.0
        ema8_val = float(last_row['ema8']) if 'ema8' in last_row and is_valid_value(last_row['ema8']) else last_price
        ema21_val = float(last_row['ema21']) if 'ema21' in last_row and is_valid_value(last_row['ema21']) else last_price
        
    except (IndexError, KeyError, ValueError) as e:
        print_status(f"Error accessing market data: {e}", "error")
        # Use basic fallback values
        last_price = float(df_ohlcv['close'].iloc[-1])
        last_atr = 0.02 * last_price
        last_rsi = 50.0
        ema8_val = last_price
        ema21_val = last_price

    # Prediction with safety check
    predicted_next = None
    if ml_out and ml_out.get('model') is not None:
        try:
            if ml_out.get('last_row') is not None:
                recent_feat = ml_out['last_row'][feature_cols].values.reshape(1, -1)
                predicted_next = float(ml_out['model'].predict(recent_feat)[0])
        except Exception as e:
            print_status(f"Prediction failed: {e}", "warning")

    # 5) News sentiment
    news_result = None
    if news_flag:
        key = os.environ.get('NEWSAPI_KEY')
        print_status("Analyzing news sentiment...", "info")
        news_result = fetch_news_sentiment(coin, key)
        if not news_result.get('available'):
            print_status(f"News unavailable: {news_result.get('reason')}", "warning")

    # 6) Ollama summary
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

    # 7) Enhanced recommendation
    recommendation = recommend_trade(last_price, predicted_next, last_atr, last_rsi)
    rsi_strength, rsi_color = get_rsi_strength(last_rsi)
    trend_strength, trend_color = get_trend_strength(ema8_val, ema21_val, last_price)

    # 8) OUTPUT - Enhanced for beginners
    print_header("📊 MARKET ANALYSIS RESULTS", 1)
    
    # Price Section
    print_header("Price Information", 2)
    print_bullet(f"Current Price: {Fore.MAGENTA}${last_price:.8f}", Fore.WHITE)
    if predicted_next:
        change_color = Fore.GREEN if predicted_next > last_price else Fore.RED
        change_icon = "📈" if predicted_next > last_price else "📉"
        print_bullet(f"Predicted Price: {change_color}${predicted_next:.8f} {change_icon}", Fore.WHITE)
        print_bullet(f"Expected Change: {change_color}{recommendation['diff_pct']:.3f}%", Fore.WHITE)
    
    # Technical Indicators
    print_header("Technical Indicators", 2)
    print_bullet(f"RSI (14): {rsi_color}{last_rsi:.1f} - {rsi_strength}", Fore.WHITE)
    print_bullet(f"Trend: {trend_color}{trend_strength}", Fore.WHITE)
    print_bullet(f"Volatility (ATR): ${last_atr:.8f}", Fore.WHITE)
    print_bullet(f"EMA8: ${ema8_val:.8f} | EMA21: ${ema21_val:.8f}", Fore.WHITE)
    
    # Trading Recommendation
    print_header("TRADING RECOMMENDATION", 2)
    advice_color = Fore.GREEN if 'BUY' in recommendation['advice'] else Fore.RED if 'SELL' in recommendation['advice'] else Fore.YELLOW
    print_bullet(f"Action: {advice_color}{recommendation['advice']} {Style.BRIGHT}", Fore.WHITE)
    print_bullet(f"Confidence: {recommendation['confidence']}", Fore.WHITE)
    
    # Risk Management
    if recommendation['stoploss']:
        print_header("Risk Management", 2)
        print_bullet(f"Stop Loss: ${recommendation['stoploss']:.8f}", Fore.WHITE)
        print_bullet(f"Take Profit: ${recommendation['take_profit']:.8f}", Fore.WHITE)
        if recommendation['risk_reward_ratio']:
            rr_color = Fore.GREEN if recommendation['risk_reward_ratio'] > 2 else Fore.YELLOW if recommendation['risk_reward_ratio'] > 1 else Fore.RED
            print_bullet(f"Risk/Reward Ratio: {rr_color}{recommendation['risk_reward_ratio']:.2f}", Fore.WHITE)
    
    # Additional Factors - FIXED SECTION
    print_header("Market Factors", 2)
    
    # Whale Activity - FIXED
    if whale_res and whale_res.get('available'):
        whales = whale_res.get('whales', [])
        if whales:
            whale_count = len(whales)
            if whale_count > 10:
                whale_color = Fore.RED
                whale_text = f"Very High ({whale_count} large trades detected)"
            elif whale_count > 5:
                whale_color = Fore.YELLOW
                whale_text = f"High ({whale_count} large trades detected)"
            else:
                whale_color = Fore.GREEN
                whale_text = f"Moderate ({whale_count} large trades detected)"
            print_bullet(f"Whale Activity: {whale_color}{whale_text}", Fore.WHITE)
        else:
            print_bullet("Whale Activity: Low (no significant large trades)", Fore.WHITE)
    elif symbol:
        print_bullet("Whale Activity: Data unavailable (symbol required)", Fore.WHITE)
    else:
        print_bullet("Whale Activity: Analysis skipped (no symbol provided)", Fore.WHITE)
    
    # News Sentiment - FIXED
    if news_result and news_result.get('available'):
        sentiment_score = news_result.get('avg_score', 0)
        sentiment_color = Fore.GREEN if sentiment_score > 0.5 else Fore.RED if sentiment_score < -0.5 else Fore.YELLOW
        sentiment_text = "Positive" if sentiment_score > 0.5 else "Negative" if sentiment_score < -0.5 else "Neutral"
        print_bullet(f"News Sentiment: {sentiment_color}{sentiment_text} (score: {sentiment_score:.2f})", Fore.WHITE)
    elif news_flag:
        print_bullet("News Sentiment: Analysis unavailable (check NEWSAPI_KEY)", Fore.WHITE)
    else:
        print_bullet("News Sentiment: Analysis skipped (use --news to enable)", Fore.WHITE)
    
    # ML Model Performance
    if ml_out and ml_out.get('mae'):
        mae = ml_out['mae']
        mae_pct = (mae / last_price) * 100
        mae_color = Fore.GREEN if mae_pct < 0.5 else Fore.YELLOW if mae_pct < 1.0 else Fore.RED
        print_bullet(f"ML Model Accuracy: {mae_color}MAE {mae_pct:.3f}%", Fore.WHITE)
    
    # Simple Mode Summary
    if simple_mode:
        print_header("🚀 QUICK SUMMARY", 1)
        print_bullet(f"Action: {advice_color}{recommendation['advice']}", Fore.WHITE, 0)
        print_bullet(f"Price: ${last_price:.2f} | Trend: {trend_strength}", Fore.WHITE, 0)
        print_bullet(f"Confidence: {recommendation['confidence']}", Fore.WHITE, 0)
    
    # Ollama AI Summary
    if ollama_summary and not simple_mode:
        print_header("AI ANALYSIS", 2)
        print(f"{Fore.CYAN}{ollama_summary}")
    
    # Export Results
    print_header("DATA EXPORT", 2)
    out_paths = export_results(df, args.out)
    print_status(f"Data exported to: {out_paths['csv']}", "success")
    
    # Final Disclaimer
    print_header("⚠️  DISCLAIMER", 2)
    print_bullet("This analysis is for educational purposes only", Fore.YELLOW)
    print_bullet("Always do your own research before trading", Fore.YELLOW)
    print_bullet("Never invest more than you can afford to lose", Fore.YELLOW)
    
    print_header("ANALYSIS COMPLETE ✅", 1)
    
    return {
        'last_price': last_price,
        'predicted_next': predicted_next,
        'recommendation': recommendation,
        'export_paths': out_paths,
        'ml_mae': (ml_out['mae'] if ml_out else None)
    }

if __name__ == "__main__":
    res = main()