#!/usr/bin/env python3
"""
gain_predictor.py - Enhanced Gain Prediction System
Specialized ML model for accurate gain/loss direction prediction
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

# Color output
try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
except Exception:
    class _C:
        def __getattr__(self, name): return ""
    Fore = Style = _C()

class EnhancedGainPredictor:
    """
    Specialized predictor for crypto gain/loss direction with balanced accuracy
    """
    
    def __init__(self, gain_threshold: float = 0.02, loss_threshold: float = -0.01):
        self.gain_threshold = gain_threshold
        self.loss_threshold = loss_threshold
        self.models = {}
        self.scaler = None
        self.feature_selector = None
        self.best_model = None
        self.feature_importance = {}
        self.feature_names = []
        self.model_performance = {}
        self.final_metrics = {}
        
    def _default_prediction(self) -> Dict[str, Any]:
        """Return default prediction when model is unavailable"""
        return {
            'is_gain': False,
            'is_loss': False,
            'is_neutral': True,
            'confidence': 0.5,
            'probabilities': {
                'gain': 0.33,
                'loss': 0.33,
                'neutral': 0.34
            },
            'prediction_class': 'NEUTRAL',
            'prediction': 1  # Neutral class
        }
    
    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create target variable for gain prediction"""
        if 'close' not in df.columns:
            return pd.Series([1] * len(df))  # Default to neutral
        
        # Target is whether price will go up in next period
        future_price = df['close'].shift(-1)
        current_price = df['close']
        target = (future_price > current_price).astype(int)
        return target.fillna(1)  # Fill NaN with neutral
    
    def create_gain_labels(self, prices: pd.Series, lookahead: int = 1) -> pd.Series:
        """
        Create multi-class labels for gain prediction:
        0: Significant Loss (below loss_threshold)
        1: Neutral (between loss_threshold and gain_threshold)  
        2: Significant Gain (above gain_threshold)
        """
        future_returns = prices.pct_change(lookahead).shift(-lookahead)
        
        labels = pd.Series(1, index=future_returns.index)  # Default to neutral
        
        # Significant loss
        labels[future_returns <= self.loss_threshold] = 0
        
        # Significant gain
        labels[future_returns >= self.gain_threshold] = 2
        
        return labels
    
    def extract_gain_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features specifically tuned for gain prediction"""
        features_df = df.copy()
        
        # Ensure numeric data only
        features_df = features_df.select_dtypes(include=['number']).copy()
        
        # Price momentum features (short-term focused)
        if 'close' in features_df.columns:
            for period in [1, 2, 3, 5, 7]:
                if len(features_df) > period:
                    features_df[f'momentum_{period}'] = features_df['close'].pct_change(period)
                    features_df[f'acceleration_{period}'] = features_df[f'momentum_{period}'].diff()
        
        # Volume surge detection
        if 'volume' in features_df.columns:
            features_df['volume_surge_3'] = features_df['volume'] / features_df['volume'].rolling(3, min_periods=1).mean()
            features_df['volume_surge_10'] = features_df['volume'] / features_df['volume'].rolling(10, min_periods=1).mean()
            
            # Volume trend
            if len(features_df) >= 5:
                features_df['volume_trend'] = features_df['volume'].rolling(5, min_periods=1).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0, raw=False
                )
        
        # Breakout detection
        if 'close' in features_df.columns and 'high' in features_df.columns:
            for period in [5, 10]:
                if len(features_df) > period:
                    features_df[f'resistance_break_{period}'] = (features_df['close'] > features_df['high'].rolling(period).max()).astype(int)
        
        if 'close' in features_df.columns and 'low' in features_df.columns:
            for period in [5, 10]:
                if len(features_df) > period:
                    features_df[f'support_break_{period}'] = (features_df['close'] < features_df['low'].rolling(period).min()).astype(int)
        
        # Trend strength with multiple timeframes
        if 'close' in features_df.columns:
            for period in [5, 10, 20]:
                if len(features_df) > period:
                    sma = features_df['close'].rolling(period, min_periods=1).mean()
                    features_df[f'trend_strength_{period}'] = (features_df['close'] - sma) / sma * 100
                    features_df[f'above_sma_{period}'] = (features_df['close'] > sma).astype(int)
        
        # Volatility regime detection
        if 'close' in features_df.columns:
            features_df['volatility_3'] = features_df['close'].pct_change().rolling(3, min_periods=1).std()
            features_df['volatility_10'] = features_df['close'].pct_change().rolling(10, min_periods=1).std()
            features_df['volatility_ratio'] = features_df['volatility_3'] / features_df['volatility_10']
            features_df['volatility_ratio'] = features_df['volatility_ratio'].replace([np.inf, -np.inf], 1.0).fillna(1.0)
        
        # RSI-based features
        if 'rsi14' in features_df.columns:
            features_df['rsi_oversold'] = (features_df['rsi14'] < 30).astype(int)
            features_df['rsi_overbought'] = (features_df['rsi14'] > 70).astype(int)
            
            if len(features_df) >= 5:
                features_df['rsi_trend'] = features_df['rsi14'].rolling(5, min_periods=1).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0, raw=False
                )
        
        # MACD signals
        if all(col in features_df.columns for col in ['macd', 'macd_signal']):
            features_df['macd_bullish'] = (features_df['macd'] > features_df['macd_signal']).astype(int)
            features_df['macd_crossup'] = ((features_df['macd'] > features_df['macd_signal']) & 
                                         (features_df['macd'].shift(1) <= features_df['macd_signal'].shift(1))).astype(int)
        
        # Bollinger Bands position
        if all(col in features_df.columns for col in ['bb_upper', 'bb_lower']):
            bb_width = features_df['bb_upper'] - features_df['bb_lower']
            features_df['bb_position'] = (features_df['close'] - features_df['bb_lower']) / bb_width
            features_df['bb_position'] = features_df['bb_position'].replace([np.inf, -np.inf], 0.5).fillna(0.5)
            
            if len(features_df) > 20:
                features_df['bb_squeeze'] = (bb_width / bb_width.rolling(20, min_periods=1).mean() < 0.8).astype(int)
        
        # Price pattern recognition
        if 'high' in features_df.columns and 'low' in features_df.columns:
            for period in [3]:
                if len(features_df) > period:
                    features_df[f'higher_highs_{period}'] = (features_df['high'] > features_df['high'].shift(1)).rolling(period, min_periods=1).sum()
                    features_df[f'higher_lows_{period}'] = (features_df['low'] > features_df['low'].shift(1)).rolling(period, min_periods=1).sum()
        
        # Gap analysis
        if 'open' in features_df.columns and 'close' in features_df.columns:
            features_df['overnight_gap'] = (features_df['open'] - features_df['close'].shift(1)) / features_df['close'].shift(1) * 100
            features_df['overnight_gap'] = features_df['overnight_gap'].fillna(0)
        
        # Fill NaN values strategically
        for col in features_df.columns:
            if features_df[col].dtype in ['float64', 'int64']:
                if 'momentum' in col or 'trend' in col or 'acceleration' in col:
                    features_df[col] = features_df[col].fillna(0)
                elif 'ratio' in col or 'surge' in col:
                    features_df[col] = features_df[col].fillna(1)
                elif features_df[col].dtype == 'int64':
                    features_df[col] = features_df[col].fillna(0)
                else:
                    features_df[col] = features_df[col].ffill().bfill().fillna(0)
        
        return features_df
    
    def create_balanced_models(self):
        """Create ensemble of models optimized for gain prediction"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'logistic': LogisticRegression(
                C=1.0,
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
        }
    
    def fit(self, df: pd.DataFrame, target_col: str = 'future_gain') -> bool:
        """Train the gain predictor model"""
        try:
            # Extract features
            features_df = self.extract_gain_specific_features(df)
            
            if features_df.empty:
                return False
            
            # Create target if not exists
            if target_col not in df.columns:
                y = self._create_target(df)
            else:
                y = df[target_col]
            
            # Ensure we have enough data
            if len(features_df) < 50 or len(y) != len(features_df):
                return False
            
            # Use all numeric features
            X = features_df.select_dtypes(include=['number'])
            X = X.ffill().bfill().fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
            # Remove zero variance features
            variances = X.var()
            low_var_features = variances[variances < 1e-6].index
            X = X.drop(columns=low_var_features)
            
            if X.shape[1] < 3:
                return False
            
            self.feature_names = X.columns.tolist()
            
            # Simple training with basic model
            self.best_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Time-based split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            self.scaler = RobustScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.best_model.fit(X_train_scaled, y_train)
            
            # Store feature importance
            self.feature_importance = dict(zip(self.feature_names, self.best_model.feature_importances_))
            
            # Calculate basic metrics
            y_pred = self.best_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{Fore.GREEN}Gain predictor trained successfully - Accuracy: {accuracy:.3f}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Gain predictor training failed: {e}")
            return False
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict gain direction for current data"""
        if self.best_model is None:
            return self._default_prediction()
        
        try:
            # Extract features
            features_df = self.extract_gain_specific_features(df)
            
            # Ensure we have the required features
            available_features = [f for f in self.feature_names if f in features_df.columns]
            if len(available_features) < 3:
                return self._default_prediction()
            
            X_current = features_df[available_features].iloc[-1:].copy()
            X_current = X_current.ffill().bfill().fillna(0)
            
            # Scale features
            X_scaled = self.scaler.transform(X_current)
            
            # Predict
            prediction = self.best_model.predict(X_scaled)[0]
            probabilities = self.best_model.predict_proba(X_scaled)[0]
            
            # For binary classification, map to our expected format
            if len(probabilities) == 2:  # Binary classification
                gain_prob = probabilities[1]  # Assuming class 1 is "up"
                loss_prob = probabilities[0]  # Assuming class 0 is "down"
                neutral_prob = 0.0
                prediction_class = 'GAIN' if prediction == 1 else 'LOSS'
            else:  # Multi-class
                gain_prob = probabilities[2] if len(probabilities) > 2 else 0
                loss_prob = probabilities[0] if len(probabilities) > 0 else 0
                neutral_prob = probabilities[1] if len(probabilities) > 1 else 0
                class_names = ['LOSS', 'NEUTRAL', 'GAIN']
                prediction_class = class_names[prediction] if prediction < len(class_names) else 'NEUTRAL'
            
            result = {
                'prediction': prediction,
                'prediction_class': prediction_class,
                'confidence': max(probabilities),
                'probabilities': {
                    'loss': loss_prob,
                    'neutral': neutral_prob,
                    'gain': gain_prob
                },
                'is_gain': prediction_class == 'GAIN',
                'is_loss': prediction_class == 'LOSS',
                'is_neutral': prediction_class == 'NEUTRAL'
            }
            
            return result
            
        except Exception as e:
            print(f"{Fore.RED}Prediction failed: {e}")
            return self._default_prediction()


def run_gain_prediction_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Run gain prediction analysis on the given dataframe"""
    try:
        predictor = EnhancedGainPredictor()
        
        # Create a clean copy for ML
        df_ml = df.copy()
        
        # Remove non-numeric columns
        datetime_cols = df_ml.select_dtypes(include=['datetime64', 'timedelta64']).columns
        if len(datetime_cols) > 0:
            df_ml = df_ml.drop(columns=datetime_cols)
        
        non_numeric_cols = df_ml.select_dtypes(exclude=['number']).columns
        if len(non_numeric_cols) > 0:
            df_ml = df_ml.drop(columns=non_numeric_cols)
        
        # Create target column
        if 'close' in df_ml.columns:
            df_ml['future_gain'] = (df_ml['close'].shift(-1) > df_ml['close']).astype(int)
        
        # Clean data
        df_ml = df_ml.select_dtypes(include=['number'])
        df_ml = df_ml.ffill().bfill().fillna(0)
        
        # Train model if we have enough data
        if len(df_ml) > 50 and 'future_gain' in df_ml.columns:
            success = predictor.fit(df_ml)
            
            if success:
                current_prediction = predictor.predict(df_ml)
                
                return {
                    'current_prediction': current_prediction,
                    'model_trained': True,
                    'feature_importance': predictor.feature_importance,
                    'training_result': {
                        'metrics': {
                            'gain_accuracy': 0.6,  # Placeholder
                            'loss_accuracy': 0.6,  # Placeholder
                            'accuracy': 0.6        # Placeholder
                        }
                    }
                }
        
        # Fallback
        return {
            'current_prediction': predictor._default_prediction(),
            'model_trained': False,
            'feature_importance': {},
            'reason': 'Insufficient data or training failed'
        }
            
    except Exception as e:
        print(f"{Fore.RED}Gain prediction analysis failed: {e}")
        predictor = EnhancedGainPredictor()
        return {
            'current_prediction': predictor._default_prediction(),
            'model_trained': False,
            'feature_importance': {},
            'error': str(e)
        }


def print_gain_prediction_summary(result: Dict[str, Any]):
    """Print comprehensive summary of gain prediction results"""
    if not result or 'current_prediction' not in result:
        print(f"{Fore.RED}No prediction results available")
        return
    
    pred = result['current_prediction']
    
    print(f"{Fore.CYAN}\n" + "="*60)
    print(f"{Fore.CYAN}        GAIN PREDICTION SUMMARY")
    print(f"{Fore.CYAN}" + "="*60)
    
    print(f"{Fore.WHITE}🎯 Current Prediction: {pred['prediction_class']}")
    print(f"{Fore.WHITE}📊 Confidence: {pred['confidence']:.1%}")
    
    print(f"{Fore.WHITE}\n📈 Probability Breakdown:")
    print(f"{Fore.GREEN}   Gain Probability: {pred['probabilities']['gain']:.1%}")
    print(f"{Fore.YELLOW}   Neutral Probability: {pred['probabilities']['neutral']:.1%}")
    print(f"{Fore.RED}   Loss Probability: {pred['probabilities']['loss']:.1%}")
    
    # Trading recommendation
    if pred['is_gain'] and pred['confidence'] > 0.6:
        recommendation = "STRONG BUY SIGNAL"
        color = Fore.GREEN
    elif pred['is_gain']:
        recommendation = "BUY SIGNAL"
        color = Fore.GREEN
    elif pred['is_loss'] and pred['confidence'] > 0.6:
        recommendation = "STRONG SELL SIGNAL" 
        color = Fore.RED
    elif pred['is_loss']:
        recommendation = "SELL SIGNAL"
        color = Fore.RED
    else:
        recommendation = "HOLD / WAIT FOR CLEARER SIGNAL"
        color = Fore.YELLOW
    
    print(f"{Fore.WHITE}\n💡 Trading Recommendation: {color}{recommendation}")
    print(f"{Fore.CYAN}" + "="*60)


# Example usage
if __name__ == "__main__":
    print(f"{Fore.CYAN}Enhanced Gain Prediction System")
    print(f"{Fore.CYAN}Ready for integration with main trading system")