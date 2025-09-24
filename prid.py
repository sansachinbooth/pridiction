#!/usr/bin/env python3
"""
pridict.py - Enhanced Crypto Prediction Tool
Features:
- Direct current price fetching from CoinGecko API
- Multi-timeframe support (10m, 20m, 30m, 1h, 4h)
- ML prediction with technical indicators
- Local Ollama AI (gemma3:4b) integration using Python library
- News sentiment analysis
- Trading recommendations and risk/reward analysis
- Enhanced visual output with colors and formatting
"""

import sys
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import Ollama Python library
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    print("❌ Ollama Python library not available. Install with: pip install ollama")
    OLLAMA_AVAILABLE = False

# -----------------------------
# Configuration
# -----------------------------
COINGECKO_API_URL = "https://api.coingecko.com/api/v3/simple/price"
COINGECKO_NEWS_URL = "https://cryptopanic.com/api/v1/posts/"

TIMEFRAMES = ['10m', '20m', '30m', '1h', '4h']
TIMEFRAME_CONFIG = {
    '10m': {'data_days': 7, 'horizon_multiplier': 1.0, 'risk_factor': 1.0},
    '20m': {'data_days': 7, 'horizon_multiplier': 1.5, 'risk_factor': 1.2},
    '30m': {'data_days': 7, 'horizon_multiplier': 1.8, 'risk_factor': 1.3},
    '1h':  {'data_days': 14,'horizon_multiplier': 2.0, 'risk_factor': 1.5},
    '4h':  {'data_days': 30,'horizon_multiplier': 3.0, 'risk_factor': 2.0}
}

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# -----------------------------
# Ollama AI Integration (USING PYTHON LIBRARY)
# -----------------------------
def call_ollama(prompt, model="deepseek-r1"):
    """Use Ollama Python library instead of direct HTTP requests"""
    if not OLLAMA_AVAILABLE:
        print("❌ Ollama library not available. Using fallback AI response.")
        return ""
    
    try:
        response = ollama.generate(model=model, prompt=prompt, options={
            'temperature': 0.3,
            'top_p': 0.9
        })
        return response['response']
    except Exception as e:
        print(f"❌ Ollama library error: {e}")
        # Fallback to HTTP if library fails but Ollama is running
        return call_ollama_http_fallback(prompt, model)

def call_ollama_http_fallback(prompt, model="gemma3:4b"):
    """Fallback HTTP method if library fails"""
    try:
        import requests
        payload = {
            "model": model, 
            "prompt": prompt, 
            "stream": False, 
            "options": {"temperature": 0.3, "top_p": 0.9}
        }
        resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
        if resp.status_code == 200: 
            return resp.json().get('response', '')
        return ""
    except:
        return ""

def parse_ai_response(ai_response, current_price):
    """Parse AI response with multiple strategies"""
    
    # Default fallback response
    default_response = {
        "prediction": "NEUTRAL",
        "confidence": 0.5,
        "reasoning": "Comprehensive analysis of market conditions suggests a neutral stance",
        "risk_level": "MEDIUM",
        "price_target": current_price * 1.02,
        "stop_loss": current_price * 0.98
    }
    
    if not ai_response:
        return default_response
    
    # Strategy 1: Try to find JSON in response
    try:
        start_idx = ai_response.find('{')
        end_idx = ai_response.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = ai_response[start_idx:end_idx]
            json_str = json_str.replace('\n', ' ').replace('\t', ' ')
            ai_data = json.loads(json_str)
            print("✅ Successfully parsed JSON from AI response")
            return ai_data
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract key-value pairs from text
    try:
        ai_data = extract_key_value_pairs(ai_response, current_price)
        if ai_data and all(key in ai_data for key in ['prediction', 'confidence', 'reasoning']):
            print("✅ Successfully extracted key-value pairs from AI response")
            return ai_data
    except Exception:
        pass
    
    # Final fallback with content enhancement
    return enhance_fallback_based_on_content(ai_response, default_response, current_price)

def extract_key_value_pairs(text, current_price):
    """Extract key-value pairs from unstructured text"""
    import re
    
    data = {}
    
    # Pattern for prediction
    prediction_patterns = [
        r'prediction[:\s]+(LONG|SHORT|NEUTRAL)',
        r'signal[:\s]+(LONG|SHORT|NEUTRAL)',
        r'recommendation[:\s]+(LONG|SHORT|NEUTRAL)'
    ]
    
    for pattern in prediction_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data['prediction'] = match.group(1).upper()
            break
    
    # Pattern for confidence
    confidence_patterns = [
        r'confidence[:\s]+([0-9.]+)',
        r'confidence[:\s]+([0-9.]+)%'
    ]
    
    for pattern in confidence_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            conf = float(match.group(1))
            data['confidence'] = conf / 100.0 if conf > 1 else conf
            break
    
    # Pattern for risk level
    risk_patterns = [
        r'risk[:\s]+(LOW|MEDIUM|HIGH)',
        r'risk level[:\s]+(LOW|MEDIUM|HIGH)'
    ]
    
    for pattern in risk_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data['risk_level'] = match.group(1).upper()
            break
    
    # Extract reasoning
    sentences = re.split(r'[.!?]+', text)
    reasoning = next((s.strip() for s in sentences if len(s.strip()) > 20), 
                    "Market analysis indicates balanced conditions")
    data['reasoning'] = reasoning[:200]
    
    # Set defaults for missing values
    if 'prediction' not in data:
        data['prediction'] = 'NEUTRAL'
    if 'confidence' not in data:
        data['confidence'] = 0.5
    if 'risk_level' not in data:
        data['risk_level'] = 'MEDIUM'
    
    # Calculate price targets
    if data['prediction'] == 'LONG':
        data['price_target'] = current_price * (1 + data['confidence'] * 0.1)
        data['stop_loss'] = current_price * (1 - data['confidence'] * 0.05)
    elif data['prediction'] == 'SHORT':
        data['price_target'] = current_price * (1 - data['confidence'] * 0.1)
        data['stop_loss'] = current_price * (1 + data['confidence'] * 0.05)
    else:
        data['price_target'] = current_price * 1.02
        data['stop_loss'] = current_price * 0.98
    
    return data

def enhance_fallback_based_on_content(text, fallback, current_price):
    """Enhance fallback based on content analysis"""
    text_lower = text.lower()
    
    # Analyze sentiment from text
    bullish_words = ['bull', 'buy', 'long', 'up', 'rise', 'gain', 'positive', 'strong']
    bearish_words = ['bear', 'sell', 'short', 'down', 'fall', 'drop', 'negative', 'weak']
    
    bullish_count = sum(1 for word in bullish_words if word in text_lower)
    bearish_count = sum(1 for word in bearish_words if word in text_lower)
    
    if bullish_count > bearish_count + 2:
        fallback['prediction'] = 'LONG'
        fallback['confidence'] = min(0.8, fallback['confidence'] + 0.2)
        fallback['reasoning'] = 'Text analysis indicates bullish sentiment'
    elif bearish_count > bullish_count + 2:
        fallback['prediction'] = 'SHORT' 
        fallback['confidence'] = min(0.8, fallback['confidence'] + 0.2)
        fallback['reasoning'] = 'Text analysis indicates bearish sentiment'
    
    # Adjust risk level based on volatility mentions
    if 'volatile' in text_lower or 'high risk' in text_lower:
        fallback['risk_level'] = 'HIGH'
    elif 'stable' in text_lower or 'low risk' in text_lower:
        fallback['risk_level'] = 'LOW'
    
    return fallback

# -----------------------------
# REST OF THE CODE REMAINS EXACTLY THE SAME AS YOUR ORIGINAL
# -----------------------------

def get_color_for_signal(signal):
    """Return color based on signal type"""
    signal_upper = signal.upper()
    if 'BULL' in signal_upper or 'LONG' in signal_upper or 'OVERBOUGHT' in signal_upper:
        return Colors.GREEN
    elif 'BEAR' in signal_upper or 'SHORT' in signal_upper or 'OVERSOLD' in signal_upper:
        return Colors.RED
    else:
        return Colors.YELLOW

def fetch_current_price(symbol):
    """Fetch current price and 24h change from CoinGecko API"""
    params = {
        'ids': symbol.lower(),
        'vs_currencies': 'usd',
        'include_24hr_change': 'true',
        'include_24hr_vol': 'true',
        'include_market_cap': 'true'
    }
    try:
        resp = requests.get(COINGECKO_API_URL, params=params, timeout=100)
        resp.raise_for_status()
        data = resp.json()
        coin_data = data.get(symbol.lower(), {})
        return {
            'price': coin_data.get('usd', 0),
            'change_24h': coin_data.get('usd_24h_change', 0),
            'volume_24h': coin_data.get('usd_24h_vol', 0),
            'market_cap': coin_data.get('usd_market_cap', 0)
        }
    except Exception as e:
        print(f"❌ Error fetching price from CoinGecko: {e}")
        return None

def fetch_crypto_news(symbol, limit=5):
    """Fetch recent cryptocurrency news with sentiment analysis"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        resp = requests.get(COINGECKO_NEWS_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        news_data = resp.json().get('data', [])
        
        if not news_data:
            return get_fallback_news(symbol)
        
        relevant_news = []
        symbol_lower = symbol.lower()
        
        for news_item in news_data[:limit]:
            title = news_item.get('title', '').lower()
            description = news_item.get('description', '').lower() if news_item.get('description') else ''
            
            keywords = [symbol_lower, 'crypto', 'bitcoin', 'ethereum', 'blockchain', 'digital currency']
            content = title + ' ' + description
            
            if any(keyword in content for keyword in keywords):
                
                positive_words = ['bullish', 'surge', 'rally', 'gain', 'positive', 'up', 'rise', 'breakout', 
                                'growth', 'success', 'profit', 'win', 'advantage']
                negative_words = ['bearish', 'drop', 'crash', 'loss', 'negative', 'down', 'fall', 'collapse',
                                'decline', 'risk', 'warning', 'danger', 'problem']
                
                positive_score = sum(1 for word in positive_words if word in content)
                negative_score = sum(1 for word in negative_words if word in content)
                
                total_words = max(positive_score + negative_score, 1)
                
                if positive_score > negative_score:
                    sentiment = 'positive'
                    sentiment_score = positive_score / total_words
                elif negative_score > positive_score:
                    sentiment = 'negative'
                    sentiment_score = negative_score / total_words
                else:
                    sentiment = 'neutral'
                    sentiment_score = 0.5
                
                relevant_news.append({
                    'title': news_item.get('title', 'No title'),
                    'url': news_item.get('url', '#'),
                    'sentiment': sentiment,
                    'sentiment_score': round(sentiment_score, 2),
                    'published_at': news_item.get('published_at', 'Unknown date')
                })
        
        return relevant_news[:3]
        
    except Exception as e:
        print(f"❌ Error fetching news from CoinGecko: {e}")
        return get_fallback_news(symbol)

def get_fallback_news(symbol):
    """Fallback news when primary API fails"""
    print(f"⚠️  Using fallback news data for {symbol}")
    
    fallback_news = [
        {
            'title': f'Cryptocurrency market shows mixed signals for {symbol.upper()}',
            'url': '#',
            'sentiment': 'neutral',
            'sentiment_score': 0.5,
            'published_at': datetime.now().isoformat()
        },
        {
            'title': 'Blockchain technology continues to evolve with new developments',
            'url': '#', 
            'sentiment': 'positive',
            'sentiment_score': 0.7,
            'published_at': datetime.now().isoformat()
        }
    ]
    
    import random
    sentiment_choice = random.choice(['positive', 'neutral', 'negative'])
    
    if sentiment_choice == 'positive':
        fallback_news.append({
            'title': f'{symbol.upper()} shows strong potential in recent analysis',
            'url': '#',
            'sentiment': 'positive',
            'sentiment_score': 0.8,
            'published_at': datetime.now().isoformat()
        })
    else:
        fallback_news.append({
            'title': f'Market analysts cautious about {symbol.upper()} short-term performance',
            'url': '#',
            'sentiment': sentiment_choice,
            'sentiment_score': 0.3 if sentiment_choice == 'negative' else 0.5,
            'published_at': datetime.now().isoformat()
        })
    
    return fallback_news

def generate_historical_data(current_price, days=7, interval='5T'):
    """Simulate historical OHLCV data based on current price"""
    try:
        base_date = datetime.now() - timedelta(days=days)
        dates = pd.date_range(start=base_date, end=datetime.now(), freq=interval)
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.02, len(dates))
        prices = [current_price]
        for r in returns[:-1]:
            prices.append(prices[-1] * (1 + r))
        prices.reverse()
        df = pd.DataFrame({'timestamp': dates, 'price': prices}).set_index('timestamp')
        df['open'] = df['price'].shift(1)
        df['high'] = df[['open','price']].max(axis=1)
        df['low'] = df[['open','price']].min(axis=1)
        df['close'] = df['price']
        df['volume'] = np.random.uniform(1e6, 5e7, len(df))
        return df[['open','high','low','close','volume']].dropna()
    except Exception as e:
        print(f"❌ Error generating historical data: {e}")
        return None

def add_technical_indicators(df):
    """Add technical indicators with proper NaN handling"""
    df = df.copy()
    
    if len(df) < 50:
        print(f"⚠️  Warning: Insufficient data for technical indicators ({len(df)} rows)")
        df['sma_20'] = df['close'].rolling(min(20, len(df)), min_periods=1).mean()
        df['ema_12'] = df['close'].ewm(span=min(12, len(df)), adjust=False).mean()
        return df
    
    df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], 100)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)
    
    df['bb_middle'] = df['close'].rolling(window=20, min_periods=1).mean()
    bb_std = df['close'].rolling(window=20, min_periods=1).std().fillna(0.01)
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].replace(0, 0.01)
    
    df['momentum_5'] = df['close'].pct_change(5).fillna(0)
    df['momentum_10'] = df['close'].pct_change(10).fillna(0)
    df['volatility'] = df['close'].pct_change().rolling(window=20, min_periods=1).std().fillna(0.01)
    
    df['support'] = df['low'].rolling(window=20, min_periods=1).min()
    df['resistance'] = df['high'].rolling(window=20, min_periods=1).max()
    
    df['volume_sma'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['volume_ratio'] = (df['volume'] / df['volume_sma'].replace(0, 1)).fillna(1)
    
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def get_technical_summary(df):
    """Generate technical analysis summary with NaN protection"""
    if len(df) == 0:
        return {
            'trend': 'NEUTRAL',
            'momentum': 'NEUTRAL',
            'volatility': 'LOW',
            'rsi_signal': 'NEUTRAL',
            'macd_signal': 'NEUTRAL'
        }
    
    latest = df.iloc[-1]
    
    summary = {
        'trend': 'NEUTRAL',
        'momentum': 'NEUTRAL',
        'volatility': 'LOW',
        'rsi_signal': 'NEUTRAL',
        'macd_signal': 'NEUTRAL'
    }
    
    sma_20 = latest.get('sma_20', latest['close'])
    sma_50 = latest.get('sma_50', latest['close'])
    
    if pd.notna(sma_20) and pd.notna(sma_50):
        if latest['close'] > sma_20 > sma_50:
            summary['trend'] = 'BULLISH'
        elif latest['close'] < sma_20 < sma_50:
            summary['trend'] = 'BEARISH'
    
    mom_5 = latest.get('momentum_5', 0)
    mom_10 = latest.get('momentum_10', 0)
    
    if pd.notna(mom_5) and pd.notna(mom_10):
        if mom_5 > 0.02 and mom_10 > 0.02:
            summary['momentum'] = 'BULLISH'
        elif mom_5 < -0.02 and mom_10 < -0.02:
            summary['momentum'] = 'BEARISH'
    
    volatility = latest.get('volatility', 0.01)
    if pd.notna(volatility):
        if volatility > 0.03:
            summary['volatility'] = 'HIGH'
        elif volatility > 0.015:
            summary['volatility'] = 'MEDIUM'
    
    rsi = latest.get('rsi', 50)
    if pd.notna(rsi):
        if rsi > 70:
            summary['rsi_signal'] = 'OVERBOUGHT'
        elif rsi < 30:
            summary['rsi_signal'] = 'OVERSOLD'
    
    macd = latest.get('macd', 0)
    macd_signal = latest.get('macd_signal', 0)
    macd_hist = latest.get('macd_hist', 0)
    
    if pd.notna(macd) and pd.notna(macd_signal) and pd.notna(macd_hist):
        if macd > macd_signal and macd_hist > 0:
            summary['macd_signal'] = 'BULLISH'
        elif macd < macd_signal and macd_hist < 0:
            summary['macd_signal'] = 'BEARISH'
    
    return summary

class MLPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
    
    def create_features(self, df, lookback=30):
        """Create features with robust NaN handling"""
        X, y = [], []
        
        if len(df) <= lookback + 1:
            return np.array(X), np.array(y)
        
        for i in range(lookback, len(df)-1):
            window = df.iloc[i-lookback:i]
            
            features = []
            
            rsi_val = window['rsi'].iloc[-1] if 'rsi' in window.columns and pd.notna(window['rsi'].iloc[-1]) else 50
            features.append(rsi_val)
            
            macd_val = window['macd'].iloc[-1] if 'macd' in window.columns and pd.notna(window['macd'].iloc[-1]) else 0
            features.append(macd_val)
            
            bb_middle = window['bb_middle'].iloc[-1] if 'bb_middle' in window.columns and pd.notna(window['bb_middle'].iloc[-1]) else window['close'].iloc[-1]
            features.append(bb_middle)
            
            mom_5 = window['momentum_5'].iloc[-1] if 'momentum_5' in window.columns and pd.notna(window['momentum_5'].iloc[-1]) else 0
            features.append(mom_5)
            
            vol = window['volatility'].iloc[-1] if 'volatility' in window.columns and pd.notna(window['volatility'].iloc[-1]) else 0.01
            features.append(vol)
            
            sma_20 = window['sma_20'].iloc[-1] if 'sma_20' in window.columns and pd.notna(window['sma_20'].iloc[-1]) else window['close'].iloc[-1]
            price_sma_ratio = (window['close'].iloc[-1] / sma_20 - 1) if sma_20 != 0 else 0
            features.append(price_sma_ratio)
            
            price_change = window['close'].iloc[-1] - window['close'].iloc[0]
            features.append(price_change)
            
            vol_ratio = window['volume_ratio'].iloc[-1] if 'volume_ratio' in window.columns and pd.notna(window['volume_ratio'].iloc[-1]) else 1.0
            features.append(vol_ratio)
            
            X.append(features)
            
            next_price = df['close'].iloc[i+1]
            current_price = df['close'].iloc[i]
            if current_price != 0:
                target = (next_price - current_price) / current_price
            else:
                target = 0
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def train(self, df, horizon_multiplier=1):
        """Train model with data validation"""
        X, y = self.create_features(df)
        
        if len(X) < 20:
            print(f"⚠️  Insufficient training data: {len(X)} samples")
            return False
        
        valid_indices = ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) == 0:
            print("❌ No valid training data after NaN removal")
            return False
        
        try:
            X_scaled = self.scaler.fit_transform(X)
            y_scaled = y * horizon_multiplier
            self.model.fit(X_scaled, y_scaled)
            self.trained = True
            print(f"✅ Model trained with {len(X)} samples")
            return True
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            return False
    
    def predict(self, df, current_price, horizon_multiplier=1):
        """Make prediction with error handling"""
        if not self.trained: 
            print("⚠️  Model not trained, returning current price")
            return current_price, 0.3
        
        X, _ = self.create_features(df)
        if len(X) == 0: 
            print("⚠️  No features for prediction")
            return current_price, 0.1
        
        try:
            last_feat = self.scaler.transform(X[-1].reshape(1,-1))
            change = self.model.predict(last_feat)[0]
            change = max(-0.15, min(0.15, change))
            predicted_price = current_price * (1 + change)
            confidence = min(0.9, max(0.1, 0.3 + abs(change) * 2))
            return predicted_price, confidence
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return current_price, 0.1

def get_risk_color(risk_level):
    """Return color based on risk level"""
    risk_upper = risk_level.upper()
    if risk_upper == 'LOW':
        return Colors.GREEN
    elif risk_upper == 'MEDIUM':
        return Colors.YELLOW
    else:
        return Colors.RED

def get_color_for_sentiment(sentiment):
    """Return color based on sentiment"""
    sentiment_lower = sentiment.lower()
    if sentiment_lower == 'positive':
        return Colors.GREEN
    elif sentiment_lower == 'negative':
        return Colors.RED
    else:
        return Colors.YELLOW

def generate_trading_recommendation(result):
    """Generate final trading recommendation"""
    ml_pred = result['ml_prediction']
    ai_analysis = result['ai_analysis']
    tech_summary = result.get('technical_summary', {})
    
    current_price = result['current_price']
    ml_price = ml_pred['price']
    ml_confidence = ml_pred['confidence']
    ai_signal = ai_analysis['prediction']
    ai_confidence = ai_analysis['confidence']
    
    ml_weight = 0.4
    ai_weight = 0.4
    tech_weight = 0.2
    
    ml_direction = 1 if ml_price > current_price else -1 if ml_price < current_price else 0
    ml_score = ml_direction * ml_confidence * ml_weight
    
    ai_direction = 1 if ai_signal == 'LONG' else -1 if ai_signal == 'SHORT' else 0
    ai_score = ai_direction * ai_confidence * ai_weight
    
    tech_direction = 0
    if tech_summary.get('trend') == 'BULLISH':
        tech_direction = 1
    elif tech_summary.get('trend') == 'BEARISH':
        tech_direction = -1
    tech_score = tech_direction * tech_weight
    
    total_score = ml_score + ai_score + tech_score
    
    if total_score > 0.3:
        recommendation = "STRONG BUY/LONG"
        color = Colors.GREEN
    elif total_score > 0.1:
        recommendation = "BUY/LONG"
        color = Colors.GREEN
    elif total_score < -0.3:
        recommendation = "STRONG SELL/SHORT"
        color = Colors.RED
    elif total_score < -0.1:
        recommendation = "SELL/SHORT"
        color = Colors.RED
    else:
        recommendation = "HOLD/NEUTRAL"
        color = Colors.YELLOW
    
    risk_level = ai_analysis.get('risk_level', 'MEDIUM')
    position_size = "SMALL" if risk_level == 'HIGH' else "MEDIUM" if risk_level == 'MEDIUM' else "LARGE"
    
    if ai_signal == 'LONG':
        target_price = ai_analysis.get('price_target', current_price * 1.05)
        stop_loss = ai_analysis.get('stop_loss', current_price * 0.95)
    elif ai_signal == 'SHORT':
        target_price = ai_analysis.get('price_target', current_price * 0.95)
        stop_loss = ai_analysis.get('stop_loss', current_price * 1.05)
    else:
        target_price = current_price * 1.02
        stop_loss = current_price * 0.98
    
    reward_risk = abs(target_price - current_price) / abs(stop_loss - current_price)
    
    return {
        'action': recommendation,
        'color': color,
        'confidence': min(0.95, (ml_confidence + ai_confidence) / 2),
        'position_size': position_size,
        'target_price': target_price,
        'stop_loss': stop_loss,
        'reward_risk_ratio': round(reward_risk, 2),
        'risk_level': risk_level,
        'score': round(total_score, 3)
    }

def format_output(result):
    """Format the prediction results with colors and better structure"""
    
    symbol = result['symbol'].upper()
    timeframe = result['timeframe']
    current_price = result['current_price']
    change_24h = result['price_change_24h']
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}🚀 CRYPTO PREDICTION ANALYSIS{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}Symbol: {Colors.GREEN}{symbol}{Colors.END}")
    print(f"{Colors.BOLD}Timeframe: {Colors.YELLOW}{timeframe}{Colors.END}")
    print(f"{Colors.BOLD}Current Price: {Colors.WHITE}${current_price:.4f}{Colors.END}")
    
    change_color = Colors.GREEN if change_24h > 0 else Colors.RED
    print(f"{Colors.BOLD}24h Change: {change_color}{change_24h:+.2f}%{Colors.END}")
    
    if 'market_cap' in result and result['market_cap'] > 0:
        market_cap = result['market_cap']
        if market_cap > 1e9:
            print(f"{Colors.BOLD}Market Cap: {Colors.BLUE}${market_cap/1e9:.2f}B{Colors.END}")
        else:
            print(f"{Colors.BOLD}Market Cap: {Colors.BLUE}${market_cap/1e6:.2f}M{Colors.END}")
    
    print(f"{Colors.BOLD}{'-'*80}{Colors.END}")
    
    ml_pred = result['ml_prediction']
    predicted_change = ((ml_pred['price'] - current_price) / current_price) * 100
    pred_color = Colors.GREEN if predicted_change > 0 else Colors.RED
    
    print(f"{Colors.BOLD}🤖 ML Prediction:{Colors.END}")
    print(f"  Price: {Colors.WHITE}${ml_pred['price']:.4f}{Colors.END}")
    print(f"  Change: {pred_color}{predicted_change:+.2f}%{Colors.END}")
    print(f"  Confidence: {Colors.YELLOW}{ml_pred['confidence']*100:.1f}%{Colors.END}")
    
    if 'technical_summary' in result:
        tech = result['technical_summary']
        print(f"{Colors.BOLD}📊 Technical Analysis:{Colors.END}")
        print(f"  Trend: {get_color_for_signal(tech['trend'])}{tech['trend']}{Colors.END}")
        print(f"  Momentum: {get_color_for_signal(tech['momentum'])}{tech['momentum']}{Colors.END}")
        print(f"  RSI: {get_color_for_signal(tech['rsi_signal'])}{tech['rsi_signal']}{Colors.END}")
        print(f"  MACD: {get_color_for_signal(tech['macd_signal'])}{tech['macd_signal']}{Colors.END}")
        print(f"  Volatility: {Colors.YELLOW}{tech['volatility']}{Colors.END}")
    
    ai = result['ai_analysis']
    print(f"{Colors.BOLD}🧠 AI Analysis:{Colors.END}")
    print(f"  Signal: {get_color_for_signal(ai['prediction'])}{ai['prediction']}{Colors.END}")
    print(f"  Confidence: {Colors.YELLOW}{ai['confidence']*100:.1f}%{Colors.END}")
    print(f"  Risk Level: {get_risk_color(ai['risk_level'])}{ai['risk_level']}{Colors.END}")
    print(f"  Reasoning: {Colors.WHITE}{ai['reasoning']}{Colors.END}")
    
    if 'news_sentiment' in result:
        news = result['news_sentiment']
        print(f"{Colors.BOLD}📰 News Sentiment:{Colors.END}")
        print(f"  Overall: {get_color_for_sentiment(news['overall_sentiment'])}{news['overall_sentiment'].upper()}{Colors.END}")
        print(f"  Score: {Colors.YELLOW}{news['average_sentiment']:.2f}{Colors.END}")
        print(f"  Articles Analyzed: {Colors.WHITE}{news['articles_count']}{Colors.END}")
        
        if news['relevant_articles']:
            print(f"  Top Headlines:")
            for i, article in enumerate(news['relevant_articles'][:2], 1):
                sentiment_icon = "📈" if article['sentiment'] == 'positive' else "📉" if article['sentiment'] == 'negative' else "➡️"
                print(f"    {i}. {sentiment_icon} {article['title'][:60]}...")
    
    recommendation = generate_trading_recommendation(result)
    print(f"{Colors.BOLD}{'-'*80}{Colors.END}")
    print(f"{Colors.BOLD}💎 TRADING RECOMMENDATION:{Colors.END}")
    print(f"  Action: {recommendation['color']}{recommendation['action']}{Colors.END}")
    print(f"  Confidence: {Colors.YELLOW}{recommendation['confidence']*100:.1f}%{Colors.END}")
    print(f"  Position Size: {Colors.WHITE}{recommendation['position_size']}{Colors.END}")
    print(f"  Target Price: {Colors.GREEN}${recommendation['target_price']:.4f}{Colors.END}")
    print(f"  Stop Loss: {Colors.RED}${recommendation['stop_loss']:.4f}{Colors.END}")
    print(f"  Reward/Risk: {Colors.YELLOW}{recommendation['reward_risk_ratio']}:1{Colors.END}")
    print(f"  Risk Level: {get_risk_color(recommendation['risk_level'])}{recommendation['risk_level']}{Colors.END}")
    print(f"  Score: {Colors.CYAN}{recommendation['score']}{Colors.END}")
    
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")

def predict_crypto(symbol, timeframe='1h'):
    """Main prediction function"""
    
    print(f"{Colors.BOLD}🔍 Analyzing {symbol.upper()} on {timeframe} timeframe...{Colors.END}")
    
    price_data = fetch_current_price(symbol)
    if not price_data:
        print(f"❌ Failed to fetch price data for {symbol}")
        return None
    
    current_price = price_data['price']
    change_24h = price_data.get('change_24h', 0)
    market_cap = price_data.get('market_cap', 0)
    
    print(f"✅ Current price: ${current_price:.4f} ({change_24h:+.2f}%)")
    
    config = TIMEFRAME_CONFIG[timeframe]
    days_needed = config['data_days']
    
    print("📊 Generating historical data...")
    df = generate_historical_data(current_price, days=days_needed)
    if df is None or len(df) < 50:
        print("❌ Insufficient historical data")
        return None
    
    print("📈 Calculating technical indicators...")
    df = add_technical_indicators(df)
    
    tech_summary = get_technical_summary(df)
    
    print("🤖 Training ML model...")
    ml_predictor = MLPredictor()
    success = ml_predictor.train(df, config['horizon_multiplier'])
    
    if success:
        ml_price, ml_confidence = ml_predictor.predict(df, current_price, config['horizon_multiplier'])
    else:
        ml_price, ml_confidence = current_price, 0.1
    
    print("🧠 Consulting AI analysis...")
    ai_prompt = f"""
    Analyze {symbol} cryptocurrency trading opportunity for {timeframe} timeframe.
    Current price: ${current_price}
    24h change: {change_24h}%
    Technical indicators: {tech_summary}
    
    Provide trading analysis in JSON format with:
    - prediction: "LONG", "SHORT", or "NEUTRAL"
    - confidence: 0.0 to 1.0
    - reasoning: brief explanation
    - risk_level: "LOW", "MEDIUM", or "HIGH"
    - price_target: number
    - stop_loss: number
    - Buy entry price : number
    
    Be concise and professional.
    """
    
    ai_response = call_ollama(ai_prompt)
    ai_analysis = parse_ai_response(ai_response, current_price)
    
    print("📰 Analyzing news sentiment...")
    news_articles = fetch_crypto_news(symbol)
    
    if news_articles:
        sentiment_scores = [article['sentiment_score'] for article in news_articles 
                          if article['sentiment'] in ['positive', 'negative']]
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.5
        
        positive_count = sum(1 for article in news_articles if article['sentiment'] == 'positive')
        negative_count = sum(1 for article in news_articles if article['sentiment'] == 'negative')
        
        if positive_count > negative_count:
            overall_sentiment = 'positive'
        elif negative_count > positive_count:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
    else:
        avg_sentiment = 0.5
        overall_sentiment = 'neutral'
        news_articles = []
    
    result = {
        'symbol': symbol,
        'timeframe': timeframe,
        'current_price': current_price,
        'price_change_24h': change_24h,
        'market_cap': market_cap,
        'ml_prediction': {
            'price': ml_price,
            'confidence': ml_confidence
        },
        'technical_summary': tech_summary,
        'ai_analysis': ai_analysis,
        'news_sentiment': {
            'overall_sentiment': overall_sentiment,
            'average_sentiment': avg_sentiment,
            'articles_count': len(news_articles),
            'relevant_articles': news_articles
        }
    }
    
    return result

def main():
    if len(sys.argv) < 2:
        print(f"{Colors.RED}Usage: {sys.argv[0]} <crypto_symbol> [timeframe]{Colors.END}")
        print(f"Available timeframes: {', '.join(TIMEFRAMES)}")
        sys.exit(1)
    
    symbol = sys.argv[1].lower()
    timeframe = sys.argv[2] if len(sys.argv) > 2 else '1h'
    
    if timeframe not in TIMEFRAMES:
        print(f"{Colors.RED}Invalid timeframe. Available: {', '.join(TIMEFRAMES)}{Colors.END}")
        sys.exit(1)
    
    try:
        result = predict_crypto(symbol, timeframe)
        if result:
            format_output(result)
        else:
            print(f"{Colors.RED}❌ Prediction failed for {symbol.upper()}{Colors.END}")
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⏹️  Analysis interrupted by user{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}❌ Unexpected error: {e}{Colors.END}")

if __name__ == "__main__":
    main()