import requests
import json
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import time
import threading
from collections import deque
from bs4 import BeautifulSoup
import yfinance as yf
import re
import argparse
import sys
import os

class GrowwAIPredictor:
    def __init__(self, base_url="http://localhost:11500"):
        self.ollama_url = base_url
        self.model = "gemma3:4b"
        self.realtime_data = deque(maxlen=100)
        self.is_monitoring = False
        self.prediction_history = []
        self.performance_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy': 0.0
        }
        
    def query_ollama(self, prompt, max_tokens=1200):
        """Query Ollama AI model for trading predictions"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.8,
                "max_tokens": max_tokens
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", 
                                   json=payload, timeout=9000)
            response.raise_for_status()
            result = response.json()
            return result["response"]
        except Exception as e:
            print(f"Ollama API Error: {e}")
            return None

    def get_groww_stock_data(self, symbol, is_indian_stock=True, period="2d", interval="5m"):
        """Get comprehensive stock data with multiple timeframes"""
        try:
            if is_indian_stock:
                yf_symbol = f"{symbol}.NS"
            else:
                yf_symbol = symbol
                
            ticker = yf.Ticker(yf_symbol)
            
            # Get multiple timeframes for better analysis
            data_5m = ticker.history(period=period, interval=interval)
            data_1d = ticker.history(period="1mo", interval="1d")
            
            if not data_5m.empty and len(data_5m) > 1:
                current_price = float(data_5m['Close'].iloc[-1])
                prev_close = float(data_5m['Close'].iloc[-2])
                
                # Calculate additional metrics
                if not data_1d.empty:
                    # 52-week high/low
                    week_52_high = float(data_1d['High'].max())
                    week_52_low = float(data_1d['Low'].min())
                    
                    # Volume analysis
                    avg_volume_30d = int(data_1d['Volume'].tail(30).mean())
                    current_volume = int(data_5m['Volume'].iloc[-1])
                    volume_ratio = current_volume / avg_volume_30d if avg_volume_30d > 0 else 1
                else:
                    week_52_high = current_price * 1.2
                    week_52_low = current_price * 0.8
                    avg_volume_30d = current_volume = volume_ratio = 0
                
                stock_data = {
                    'symbol': symbol,
                    'price': current_price,
                    'open': float(data_5m['Open'].iloc[-1]),
                    'high': float(data_5m['High'].iloc[-1]),
                    'low': float(data_5m['Low'].iloc[-1]),
                    'volume': current_volume,
                    'prev_close': prev_close,
                    'change': current_price - prev_close,
                    'change_percent': ((current_price - prev_close) / prev_close) * 100,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'week_52_high': week_52_high,
                    'week_52_low': week_52_low,
                    'avg_volume_30d': avg_volume_30d,
                    'volume_ratio': volume_ratio,
                    'distance_from_52_high': ((current_price - week_52_high) / week_52_high) * 100,
                    'distance_from_52_low': ((current_price - week_52_low) / week_52_low) * 100
                }
                
                self.realtime_data.append(stock_data)
                return stock_data
            else:
                print(f"Insufficient data for {symbol}")
                return None
                
        except Exception as e:
            print(f"Data fetch error for {symbol}: {e}")
        return None

    def calculate_technical_indicators(self):
        """Calculate comprehensive technical indicators"""
        if len(self.realtime_data) < 5:
            return {}
            
        prices = [d['price'] for d in self.realtime_data]
        closes = np.array(prices)
        highs = np.array([d['high'] for d in self.realtime_data])
        lows = np.array([d['low'] for d in self.realtime_data])
        volumes = np.array([d['volume'] for d in self.realtime_data])
        
        indicators = {}
        available_data = len(closes)
        
        try:
            # Trend Indicators
            if available_data >= 5:
                indicators['sma_5'] = talib.SMA(closes, timeperiod=min(5, available_data))[-1]
            if available_data >= 10:
                indicators['sma_10'] = talib.SMA(closes, timeperiod=min(10, available_data))[-1]
            if available_data >= 20:
                indicators['sma_20'] = talib.SMA(closes, timeperiod=min(20, available_data))[-1]
            if available_data >= 50:
                indicators['sma_50'] = talib.SMA(closes, timeperiod=min(50, available_data))[-1]
            
            # Exponential Moving Averages
            if available_data >= 12:
                indicators['ema_12'] = talib.EMA(closes, timeperiod=min(12, available_data))[-1]
            if available_data >= 26:
                indicators['ema_26'] = talib.EMA(closes, timeperiod=min(26, available_data))[-1]
            
            # Momentum Indicators
            if available_data >= 14:
                indicators['rsi'] = talib.RSI(closes, timeperiod=min(14, available_data))[-1]
            if available_data >= 14:
                indicators['williams_r'] = talib.WILLR(highs, lows, closes, timeperiod=min(14, available_data))[-1]
            if available_data >= 14:
                indicators['mfi'] = talib.MFI(highs, lows, closes, volumes, timeperiod=min(14, available_data))[-1]
            
            # MACD
            if available_data >= 26:
                macd, macd_signal, macd_hist = talib.MACD(closes)
                indicators['macd'] = macd[-1] if macd is not None else 0
                indicators['macd_signal'] = macd_signal[-1] if macd_signal is not None else 0
                indicators['macd_histogram'] = macd_hist[-1] if macd_hist is not None else 0
            
            # Stochastic
            if available_data >= 14:
                stoch_k, stoch_d = talib.STOCH(highs, lows, closes)
                indicators['stoch_k'] = stoch_k[-1] if stoch_k is not None else 50
                indicators['stoch_d'] = stoch_d[-1] if stoch_d is not None else 50
            
            # Bollinger Bands
            if available_data >= 20:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(closes, timeperiod=min(20, available_data))
                indicators['bb_upper'] = bb_upper[-1] if bb_upper is not None else closes[-1] * 1.1
                indicators['bb_middle'] = bb_middle[-1] if bb_middle is not None else closes[-1]
                indicators['bb_lower'] = bb_lower[-1] if bb_lower is not None else closes[-1] * 0.9
                indicators['bb_position'] = (closes[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) * 100
            
            # ATR for volatility
            if available_data >= 14:
                atr = talib.ATR(highs, lows, closes, timeperiod=min(14, available_data))
                indicators['atr'] = atr[-1] if atr is not None else 0
                indicators['atr_percent'] = (atr[-1] / closes[-1]) * 100 if atr is not None else 0
            
            # OBV
            if available_data >= 1:
                obv = talib.OBV(closes, volumes)
                indicators['obv'] = obv[-1] if obv is not None else 0
            
        except Exception as e:
            print(f"Indicator calculation error: {e}")
            
        return indicators

    def get_market_sentiment(self, symbol):
        """Get basic market sentiment indicators"""
        try:
            # Simple sentiment based on price movement and volume
            if len(self.realtime_data) < 2:
                return "NEUTRAL"
            
            current = self.realtime_data[-1]
            prev = self.realtime_data[-2] if len(self.realtime_data) > 1 else current
            
            price_change = current['change_percent']
            volume_ratio = current.get('volume_ratio', 1)
            
            if price_change > 1 and volume_ratio > 1.2:
                return "BULLISH"
            elif price_change < -1 and volume_ratio > 1.2:
                return "BEARISH"
            elif price_change > 0.5 and volume_ratio > 1:
                return "SLIGHTLY_BULLISH"
            elif price_change < -0.5 and volume_ratio > 1:
                return "SLIGHTLY_BEARISH"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return "NEUTRAL"

    def generate_grow_prediction(self, symbol, current_data, indicators):
        """Generate AI-powered grow prediction with enhanced analysis"""
        
        sentiment = self.get_market_sentiment(symbol)
        
        prompt = f"""
        Analyze this stock comprehensively and provide ONLY JSON output:

        STOCK ANALYSIS REQUEST:
        - Symbol: {symbol}
        - Current Price: ₹{current_data['price']:.2f}
        - Previous Close: ₹{current_data['prev_close']:.2f}
        - Change: {current_data['change']:+.2f} ({current_data['change_percent']:+.2f}%)
        - Volume: {current_data['volume']:,.0f} (Ratio: {current_data.get('volume_ratio', 1):.2f})
        - 52-week Range: ₹{current_data.get('week_52_low', 0):.2f} - ₹{current_data.get('week_52_high', 0):.2f}
        - Distance from 52W High: {current_data.get('distance_from_52_high', 0):.2f}%
        - Market Sentiment: {sentiment}

        TECHNICAL INDICATORS:
        - RSI (14): {indicators.get('rsi', 50):.1f}
        - MACD: {indicators.get('macd', 0):.3f} | Signal: {indicators.get('macd_signal', 0):.3f}
        - Stochastic K: {indicators.get('stoch_k', 50):.1f}% | D: {indicators.get('stoch_d', 50):.1f}%
        - Bollinger Band Position: {indicators.get('bb_position', 50):.1f}%
        - SMA (20): ₹{indicators.get('sma_20', current_data['price']):.2f}
        - SMA (50): ₹{indicators.get('sma_50', current_data['price']):.2f}
        - ATR: {indicators.get('atr_percent', 0):.2f}%
        - Williams %R: {indicators.get('williams_r', 0):.1f}

        Provide detailed analysis in this exact JSON format:

        {{
            "symbol": "{symbol}",
            "timestamp": "{datetime.now().isoformat()}",
            "overall_recommendation": "STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL",
            "confidence_score": 0.85,
            "time_horizon": "SHORT_TERM/MEDIUM_TERM/LONG_TERM",
            "risk_level": "LOW/MEDIUM/HIGH",
            "expected_return_percent": 5.5,
            "key_technical_factors": ["RSI neutral", "MACD bullish", "Above SMA20"],
            "fundamental_analysis": {{
                "current_valuation": "UNDERVALUED/FAIRLY_VALUED/OVERVALUED",
                "growth_potential": "LOW/MEDIUM/HIGH",
                "momentum": "BULLISH/BEARISH/NEUTRAL"
            }},
            "trading_recommendations": {{
                "immediate_action": "BUY/SELL/HOLD/WAIT_FOR_DIP",
                "entry_strategy": "IMMEDIATE/WAIT_FOR_PULLBACK/SCALE_IN",
                "position_sizing": "LIGHT/MODERATE/AGGRESSIVE",
                "price_targets": {{
                    "short_term_target": {current_data['price'] * 1.05:.2f},
                    "medium_term_target": {current_data['price'] * 1.10:.2f},
                    "long_term_target": {current_data['price'] * 1.20:.2f}
                }},
                "risk_management": {{
                    "stop_loss": {current_data['price'] * 0.95:.2f},
                    "stop_loss_percent": 5.0,
                    "risk_reward_ratio": 2.0
                }}
            }},
            "market_context": {{
                "sector_outlook": "POSITIVE/NEUTRAL/NEGATIVE",
                "market_timing": "GOOD/AVERAGE/POOR",
                "volume_analysis": "STRONG/WEAK/AVERAGE"
            }},
            "cautions": ["High volatility expected", "Watch market sentiment"],
            "alternative_scenarios": {{
                "bull_case": {{
                    "probability": 0.3,
                    "target": {current_data['price'] * 1.15:.2f},
                    "catalyst": "Positive earnings"
                }},
                "base_case": {{
                    "probability": 0.5,
                    "target": {current_data['price'] * 1.08:.2f},
                    "catalyst": "Steady growth"
                }},
                "bear_case": {{
                    "probability": 0.2,
                    "target": {current_data['price'] * 0.92:.2f},
                    "catalyst": "Market correction"
                }}
            }}
        }}
        """

        print("🔄 Sending comprehensive analysis request to Ollama...")
        response = self.query_ollama(prompt, max_tokens=1500)
        
        if response:
            print("📊 Raw AI Response (first 500 chars):")
            print(response[:500] + "..." if len(response) > 500 else response)
            
        return self.parse_ai_response(response, current_data)

    def parse_ai_response(self, response, current_data):
        """Parse AI response into structured data with enhanced error handling"""
        if not response or "Error" in response:
            print("❌ No valid response from AI")
            return self.create_fallback_prediction(current_data)
        
        try:
            # Clean the response and extract JSON
            cleaned_response = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', response)
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                prediction = json.loads(json_str)
                
                # Validate and enhance prediction
                required_fields = ['symbol', 'overall_recommendation', 'confidence_score']
                if all(field in prediction for field in required_fields):
                    prediction['timestamp'] = datetime.now().isoformat()
                    prediction['analysis_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Add performance tracking
                    self.performance_stats['total_predictions'] += 1
                    
                    self.prediction_history.append(prediction)
                    return prediction
                else:
                    print("⚠️ Missing required fields in AI response")
                    return self.create_fallback_prediction(current_data)
            else:
                print("❌ No JSON found in AI response")
                return self.create_fallback_prediction(current_data)
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            print("Response content:", response[:200])
            return self.create_fallback_prediction(current_data)
        except Exception as e:
            print(f"❌ Unexpected parsing error: {e}")
            return self.create_fallback_prediction(current_data)

    def create_fallback_prediction(self, current_data):
        """Create comprehensive fallback prediction"""
        current_price = current_data['price']
        sentiment = self.get_market_sentiment(current_data['symbol'])
        
        return {
            "symbol": current_data['symbol'],
            "timestamp": datetime.now().isoformat(),
            "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "overall_recommendation": "HOLD",
            "confidence_score": 0.5,
            "time_horizon": "SHORT_TERM",
            "risk_level": "MEDIUM",
            "expected_return_percent": 3.0,
            "key_technical_factors": ["Data insufficient for full analysis"],
            "fundamental_analysis": {
                "current_valuation": "FAIRLY_VALUED",
                "growth_potential": "MEDIUM",
                "momentum": sentiment
            },
            "trading_recommendations": {
                "immediate_action": "HOLD",
                "entry_strategy": "WAIT_FOR_CONFIRMATION",
                "position_sizing": "LIGHT",
                "price_targets": {
                    "short_term_target": round(current_price * 1.03, 2),
                    "medium_term_target": round(current_price * 1.08, 2),
                    "long_term_target": round(current_price * 1.15, 2)
                },
                "risk_management": {
                    "stop_loss": round(current_price * 0.94, 2),
                    "stop_loss_percent": 6.0,
                    "risk_reward_ratio": 1.5
                }
            },
            "market_context": {
                "sector_outlook": "NEUTRAL",
                "market_timing": "AVERAGE",
                "volume_analysis": "AVERAGE"
            },
            "cautions": ["AI analysis unavailable - using fallback data"],
            "alternative_scenarios": {
                "bull_case": {
                    "probability": 0.4,
                    "target": round(current_price * 1.10, 2),
                    "catalyst": "Market recovery"
                },
                "base_case": {
                    "probability": 0.4,
                    "target": round(current_price * 1.05, 2),
                    "catalyst": "Steady performance"
                },
                "bear_case": {
                    "probability": 0.2,
                    "target": round(current_price * 0.95, 2),
                    "catalyst": "Market volatility"
                }
            },
            "fallback_analysis": True
        }

    def test_ollama_connection(self):
        """Test if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=9000)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                print("📋 Available models:", available_models)
                
                if any(self.model in model for model in available_models):
                    print(f"✅ Model {self.model} is available")
                    return True
                else:
                    print(f"❌ Model {self.model} not found.")
                    if available_models:
                        self.model = available_models[0].split(':')[0] + ':4b'
                        print(f"🔄 Using model: {self.model}")
                        return True
            return False
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            print("💡 Make sure Ollama is running: ollama serve")
            return False

    def display_prediction(self, current_data, prediction):
        """Display enhanced prediction results"""
        print(f"\n{'='*80}")
        print(f"🎯AI STOCK PREDICTION ANALYSIS")
        print(f"{'='*80}")
        print(f"📈 SYMBOL: {current_data['symbol']}")
        print(f"⏰ ANALYSIS TIME: {prediction.get('analysis_date', 'N/A')}")
        print(f"💰 CURRENT PRICE: ₹{current_data['price']:.2f}")
        print(f"📊 PREVIOUS CLOSE: ₹{current_data['prev_close']:.2f}")
        print(f"🔄 CHANGE: {current_data['change']:+.2f} ({current_data['change_percent']:+.2f}%)")
        print(f"📈 52-WEEK RANGE: ₹{current_data.get('week_52_low', 0):.2f} - ₹{current_data.get('week_52_high', 0):.2f}")
        
        print(f"\n🎯 OVERALL RECOMMENDATION: {prediction.get('overall_recommendation', 'HOLD')}")
        print(f"✅ CONFIDENCE LEVEL: {prediction.get('confidence_score', 0)*100:.1f}%")
        print(f"⏳ TIME HORIZON: {prediction.get('time_horizon', 'N/A')}")
        print(f"⚠️  RISK LEVEL: {prediction.get('risk_level', 'MEDIUM')}")
        print(f"📈 EXPECTED RETURN: {prediction.get('expected_return_percent', 0):.1f}%")
        
        # Technical Factors
        factors = prediction.get('key_technical_factors', [])
        if factors:
            print(f"\n🔧 KEY TECHNICAL FACTORS:")
            for factor in factors[:5]:  # Show first 5 factors
                print(f"   • {factor}")
        
        # Trading Recommendations
        trading_rec = prediction.get('trading_recommendations', {})
        if trading_rec:
            print(f"\n💡 TRADING RECOMMENDATIONS:")
            print(f"   📋 Immediate Action: {trading_rec.get('immediate_action', 'HOLD')}")
            print(f"   🎯 Entry Strategy: {trading_rec.get('entry_strategy', 'N/A')}")
            print(f"   ⚖️  Position Sizing: {trading_rec.get('position_sizing', 'N/A')}")
            
            price_targets = trading_rec.get('price_targets', {})
            if price_targets:
                print(f"   🎯 Price Targets:")
                print(f"      • Short-term: ₹{price_targets.get('short_term_target', 0):.2f} ({((price_targets.get('short_term_target', 0)/current_data['price'])-1)*100:+.1f}%)")
                print(f"      • Medium-term: ₹{price_targets.get('medium_term_target', 0):.2f} ({((price_targets.get('medium_term_target', 0)/current_data['price'])-1)*100:+.1f}%)")
                print(f"      • Long-term: ₹{price_targets.get('long_term_target', 0):.2f} ({((price_targets.get('long_term_target', 0)/current_data['price'])-1)*100:+.1f}%)")
            
            risk_mgmt = trading_rec.get('risk_management', {})
            if risk_mgmt:
                print(f"   🛡️  Risk Management:")
                print(f"      • Stop Loss: ₹{risk_mgmt.get('stop_loss', 0):.2f} ({risk_mgmt.get('stop_loss_percent', 0):.1f}%)")
                print(f"      • Risk/Reward Ratio: {risk_mgmt.get('risk_reward_ratio', 0):.1f}")
        
        # Alternative Scenarios
        scenarios = prediction.get('alternative_scenarios', {})
        if scenarios:
            print(f"\n🔮 ALTERNATIVE SCENARIOS:")
            for scenario, details in scenarios.items():
                print(f"   📊 {scenario.replace('_', ' ').title()}:")
                print(f"      • Probability: {details.get('probability', 0)*100:.0f}%")
                print(f"      • Target: ₹{details.get('target', 0):.2f}")
                print(f"      • Catalyst: {details.get('catalyst', 'N/A')}")
        
        # Market Context
        market_ctx = prediction.get('market_context', {})
        if market_ctx:
            print(f"\n🌐 MARKET CONTEXT:")
            print(f"   📈 Sector Outlook: {market_ctx.get('sector_outlook', 'N/A')}")
            print(f"   ⏰ Market Timing: {market_ctx.get('market_timing', 'N/A')}")
            print(f"   📊 Volume Analysis: {market_ctx.get('volume_analysis', 'N/A')}")
        
        # Cautions
        cautions = prediction.get('cautions', [])
        if cautions:
            print(f"\n⚠️  IMPORTANT CAUTIONS:")
            for caution in cautions:
                print(f"   • {caution}")
        
        print(f"{'='*80}")

    def start_realtime_monitoring(self, symbol, interval=300, is_indian_stock=True):
        """Start enhanced real-time monitoring"""
        if not self.test_ollama_connection():
            print("❌ Cannot start monitoring without Ollama connection")
            return
            
        self.is_monitoring = True
        print(f"🔮 Starting real-time monitoring for {symbol}")
        print(f"⏰ Update interval: {interval} seconds")
        print("="*80)
        
        def monitor_loop():
            prediction_count = 0
            while self.is_monitoring:
                try:
                    prediction_count += 1
                    print(f"\n📊 PREDICTION CYCLE #{prediction_count}")
                    print("-"*40)
                    
                    # Get current data
                    current_data = self.get_groww_stock_data(symbol, is_indian_stock)
                    if not current_data:
                        print(f"❌ No data for {symbol}, retrying...")
                        time.sleep(interval)
                        continue
                    
                    # Calculate indicators
                    indicators = self.calculate_technical_indicators()
                    print(f"✅ Calculated {len(indicators)} technical indicators")
                    
                    # Generate prediction
                    prediction = self.generate_grow_prediction(symbol, current_data, indicators)
                    
                    # Display results
                    self.display_prediction(current_data, prediction)
                    
                    # Save prediction to file
                    self.save_prediction_to_file(prediction)
                    
                    print(f"⏳ Next update in {interval} seconds...")
                    time.sleep(interval)
                    
                except KeyboardInterrupt:
                    print("\n🛑 Monitoring stopped by user")
                    self.is_monitoring = False
                    break
                except Exception as e:
                    print(f"❌ Monitoring error: {e}")
                    time.sleep(interval)
        
        monitor_thread = threading.Thread(target=monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()

    def save_prediction_to_file(self, prediction):
        """Save prediction to JSON file for tracking"""
        try:
            symbol = prediction.get('symbol', 'unknown')
            filename = f"groww_predictions_{symbol}.json"
            
            # Read existing predictions
            existing_data = []
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    existing_data = json.load(f)
            
            # Add new prediction
            existing_data.append(prediction)
            
            # Keep only last 100 predictions
            if len(existing_data) > 100:
                existing_data = existing_data[-100:]
            
            # Save back to file
            with open(filename, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
        except Exception as e:
            print(f"❌ Error saving prediction: {e}")

def main():
    parser = argparse.ArgumentParser(description='GROWW AI Stock Predictor')
    parser.add_argument('symbol', nargs='?', help='Stock symbol (e.g., RELIANCE, TCS)')
    parser.add_argument('--monitor', '-m', action='store_true', help='Enable real-time monitoring')
    parser.add_argument('--interval', '-i', type=int, default=300, help='Monitoring interval in seconds (default: 300)')
    parser.add_argument('--international', '-int', action='store_true', help='International stock (default: Indian)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = GrowwAIPredictor()
    
    # Test Ollama connection first
    if not predictor.test_ollama_connection():
        print("❌ Please start Ollama first: ollama serve")
        return
    
    # If no symbol provided, show available stocks
    if not args.symbol:
        indian_stocks = [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
            "SBIN", "BHARTIARTL", "ITC", "LT", "HINDUNILVR", 
            "SUZLON", "TATAGOLD", "JSWINFRA", "JSWHL", "TATAMOTORS",
            "BAJFINANCE", "KOTAKBANK", "AXISBANK", "ASIANPAINT", "MARUTI"
        ]
        
        print("🇮🇳 GROWW AI STOCK PREDICTOR")
        print("📋 Available Indian Stocks:")
        for i, stock in enumerate(indian_stocks, 1):
            print(f"   {i:2d}. {stock}")
        
        print(f"\n💻 Usage: python {sys.argv[0]} RELIANCE")
        print("   python groww.py RELIANCE --monitor")
        print("   python groww.py AAPL --international")
        return
    
    selected_stock = args.symbol.upper()
    is_indian_stock = not args.international
    
    try:
        # Get single prediction first
        print(f"🔄 Fetching data for {selected_stock}...")
        current_data = predictor.get_groww_stock_data(selected_stock, is_indian_stock=is_indian_stock)
        
        if current_data:
            print(f"✅ Current Price: ₹{current_data['price']:.2f}")
            print(f"📊 Change: {current_data['change']:+.2f} ({current_data['change_percent']:+.2f}%)")
            
            indicators = predictor.calculate_technical_indicators()
            print(f"✅ Calculated {len(indicators)} technical indicators")
            
            prediction = predictor.generate_grow_prediction(selected_stock, current_data, indicators)
            predictor.display_prediction(current_data, prediction)
            
            # Start real-time monitoring if requested
            if args.monitor:
                print(f"\n🔮 Starting real-time monitoring for {selected_stock}...")
                predictor.start_realtime_monitoring(
                    selected_stock, 
                    interval=args.interval, 
                    is_indian_stock=is_indian_stock
                )
                
                try:
                    while predictor.is_monitoring:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n👋 Stopping monitoring...")
                    predictor.is_monitoring = False
            else:
                print(f"\n💡 Tip: Use 'python {sys.argv[0]} {selected_stock} --monitor' for real-time updates")
                    
        else:
            print(f"❌ Could not fetch data for {selected_stock}")
            if is_indian_stock:
                print("💡 Try adding .NS for Indian stocks or use --international for global stocks")
                
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()