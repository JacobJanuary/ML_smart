"""
Smart ML Validator - Model Validation and Backtesting
======================================================
Валидация моделей на исторических данных с учетом:
- Walk-forward validation
- Stability analysis при смене режимов
- A/B тестирование
"""

import pandas as pd
import numpy as np
import joblib
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from scipy import stats
from dotenv import load_dotenv
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SmartValidator:
    """Comprehensive validator for market-adaptive models."""

    def __init__(self):
        """Initialize validator with database connection."""
        self.conn_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }

        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        self.feature_columns = {}
        self.model_configs = {}

        self._load_all_models()

    def _load_all_models(self):
        """Load all trained models for validation."""
        model_names = [
            'BULL_BUY', 'BULL_SELL',
            'NEUTRAL_BUY', 'NEUTRAL_SELL',
            'BEAR_BUY', 'BEAR_SELL'
        ]

        logger.info(f"Loading models from: models/smart_ml/")
        loaded_count = 0

        for model_name in model_names:
            model_path = f'models/smart_ml/{model_name.lower()}_model.pkl'

            if os.path.exists(model_path):
                try:
                    logger.info(f"Loading {model_name} from {model_path}...")
                    model_data = joblib.load(model_path)

                    # Validate model data structure
                    required_keys = ['model', 'scaler', 'threshold', 'feature_columns']
                    missing_keys = [k for k in required_keys if k not in model_data]

                    if missing_keys:
                        logger.error(f"Model {model_name} missing keys: {missing_keys}")
                        continue

                    self.models[model_name] = model_data['model']
                    self.scalers[model_name] = model_data['scaler']
                    self.thresholds[model_name] = model_data['threshold']
                    self.feature_columns[model_name] = model_data['feature_columns']
                    self.model_configs[model_name] = model_data.get('config', {})
                    logger.info(f"✅ Loaded {model_name} model (features: {len(self.feature_columns[model_name])})")
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            else:
                logger.warning(f"Model file not found: {model_path}")

        logger.info(f"Loaded {loaded_count}/{len(model_names)} models")

        if loaded_count == 0:
            logger.error("No models loaded! Please train models first.")
            logger.info("Run: python smart_ml_orchestrator.py train")

    def load_test_data(self, model_name: str, start_days_ago: int = 30,
                      end_days_ago: int = 2) -> pd.DataFrame:
        """Load test data for specific model."""
        market_regime, signal_type = model_name.rsplit('_', 1)

        logger.info(f"Loading test data for {model_name} ({start_days_ago} to {end_days_ago} days ago)...")

        query = f"""
        SELECT *
        FROM fas.mv_ml_training_data_simplified
        WHERE market_regime = '{market_regime}'
            AND signal_type = '{signal_type}'
            AND target IS NOT NULL
            AND timestamp >= NOW() - INTERVAL '{start_days_ago} days'
            AND timestamp < NOW() - INTERVAL '{end_days_ago} days'
        ORDER BY timestamp
        """

        try:
            with psycopg2.connect(**self.conn_params) as conn:
                df = pd.read_sql(query, conn)

            logger.info(f"📊 Loaded {len(df)} test samples for {model_name}")

            if len(df) > 0:
                logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                logger.info(f"   Base win rate: {df['target'].mean():.1%}")
            else:
                logger.warning(f"No test data found for {model_name} in the specified period")

            return df

        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return pd.DataFrame()

    def prepare_features(self, df: pd.DataFrame, model_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for validation (same as training)."""
        df_proc = df.copy()
        config = self.model_configs.get(model_name, {})

        # Remove unnecessary columns
        remove_cols = ['id', 'trading_pair_id', 'timestamp', 'pair_symbol',
                      'signal_type', 'signal_strength', 'patterns_details',
                      'combinations_details', 'created_at']
        remove_cols += [col for col in df_proc.columns if col.startswith('_meta_')]

        for col in remove_cols:
            if col in df_proc.columns:
                df_proc = df_proc.drop(columns=[col])

        # Process extreme values
        for col in ['poc_volume_7d', 'poc_volume_24h']:
            if col in df_proc.columns:
                q99 = df_proc[col].quantile(0.99) if len(df_proc) > 100 else df_proc[col].max()
                df_proc[col] = df_proc[col].clip(upper=q99)
                df_proc[f'{col}_log'] = np.log1p(df_proc[col])

        # Time features
        df_proc['hour'] = pd.to_datetime(df.index).hour
        df_proc['day_of_week'] = pd.to_datetime(df.index).dayofweek
        df_proc['is_weekend'] = df_proc['day_of_week'].isin([5, 6]).astype(int)

        # Model-specific features
        if 'BULL' in model_name:
            df_proc['momentum_strength'] = df_proc.get('rs_momentum', 0) * df_proc.get('macd_histogram', 0)
            df_proc['volume_momentum'] = df_proc.get('volume_zscore', 0) * df_proc.get('buy_ratio_weighted', 0)
            df_proc['trend_strength'] = (df_proc.get('macd_line', 0) - df_proc.get('macd_signal', 0)).abs()

            if 'SELL' in model_name:
                df_proc['price_rsi_divergence'] = (df_proc.get('price_change_pct', 0) * df_proc.get('rsi', 50)) / 100
                df_proc['cvd_price_ratio'] = df_proc.get('cvd_delta', 0) / (df_proc.get('close_price', 1) + 1)

        elif 'NEUTRAL' in model_name:
            df_proc['poc_deviation'] = (df_proc.get('price_to_poc_7d_pct', 0).abs() +
                                       df_proc.get('price_to_poc_24h_pct', 0).abs()) / 2
            df_proc['imbalance_smoothed'] = df_proc.get('normalized_imbalance', 0) * df_proc.get('smoothed_imbalance', 0)
            df_proc['volatility_adjusted_atr'] = df_proc.get('atr_pct', 0) / (df_proc.get('volume_zscore', 0).abs() + 1)

        elif 'BEAR' in model_name:
            df_proc['oversold_strength'] = (30 - df_proc.get('rsi', 50).clip(upper=30)) / 30
            df_proc['fear_index'] = df_proc.get('funding_rate_avg', 0) * df_proc.get('oi_delta_pct', 0)

            if 'SELL' in model_name:
                df_proc['bearish_continuation'] = df_proc.get('has_momentum_exhaustion', 0) * df_proc.get('cvd_delta', 0).clip(upper=0).abs()

        # Add weighted focus features
        for feat in config.get('focus_features', []):
            if feat in df_proc.columns:
                df_proc[f'{feat}_weighted'] = df_proc[feat] * 1.5

        # Pattern features
        pattern_cols = [col for col in df_proc.columns if 'pattern_' in col and 'confidence' in col]
        if pattern_cols:
            df_proc['max_pattern_confidence'] = df_proc[pattern_cols].max(axis=1)
            df_proc['avg_pattern_confidence'] = df_proc[pattern_cols].mean(axis=1)

        # Combo features
        combo_cols = [col for col in df_proc.columns if 'combo_' in col and 'score' in col]
        if combo_cols:
            df_proc['total_combo_score'] = df_proc[combo_cols].sum(axis=1)

        # Handle categorical
        categorical_cols = df_proc.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'target':
                df_proc[col] = pd.Categorical(df_proc[col].fillna('unknown')).codes

        # Add missing features
        for feat in self.feature_columns[model_name]:
            if feat not in df_proc.columns:
                df_proc[feat] = 0

        X = df_proc[self.feature_columns[model_name]].fillna(0)
        y = df['target'].astype(int)

        return X, y

    def walk_forward_validation(self, model_name: str, n_splits: int = 5) -> Dict:
        """Perform walk-forward validation."""
        logger.info(f"\n🚶 Walk-Forward Validation for {model_name}")

        # Load data for last 30 days with timeout
        try:
            with psycopg2.connect(**self.conn_params) as conn:
                # Set timeout
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL statement_timeout = '10s';")

                df = self.load_test_data(model_name, start_days_ago=30, end_days_ago=2)

        except psycopg2.OperationalError as e:
            logger.error(f"Database timeout in walk-forward validation: {e}")
            return {}

        if len(df) < 100:
            logger.warning(f"Insufficient data for walk-forward validation: {len(df)}")
            return {}

        # Limit data size for performance
        if len(df) > 10000:
            logger.info(f"Limiting data from {len(df)} to 10000 samples for performance")
            df = df.tail(10000)

        # Prepare features
        try:
            X, y = self.prepare_features(df, model_name)
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return {}

        # Scale features
        X_scaled = pd.DataFrame(
            self.scalers[model_name].transform(X),
            columns=X.columns,
            index=X.index
        )

        # Walk-forward splits
        split_size = len(X) // n_splits
        results = []

        logger.info(f"   Running {n_splits-1} forward validation splits...")

        for i in range(1, n_splits):
            test_start = i * split_size
            test_end = min((i + 1) * split_size, len(X))

            X_test = X_scaled.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]

            if len(X_test) == 0:
                continue

            # Predict
            try:
                y_pred_proba = self.models[model_name].predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba >= self.thresholds[model_name]).astype(int)

                # Calculate metrics
                if y_pred.sum() > 0:
                    win_rate = y_test[y_pred == 1].mean()
                    signals_pct = y_pred.sum() / len(y_pred)

                    results.append({
                        'split': i,
                        'win_rate': float(win_rate),
                        'signals_pct': float(signals_pct),
                        'n_signals': int(y_pred.sum()),
                        'auc': float(roc_auc_score(y_test, y_pred_proba)) if y_test.nunique() > 1 else 0
                    })

                    logger.info(f"   Split {i}: WR={win_rate:.1%}, Signals={signals_pct:.1%}")

            except Exception as e:
                logger.error(f"Error in split {i}: {e}")
                continue

        if results:
            avg_metrics = {
                'avg_win_rate': np.mean([r['win_rate'] for r in results]),
                'avg_signals_pct': np.mean([r['signals_pct'] for r in results]),
                'avg_auc': np.mean([r['auc'] for r in results]),
                'stability': 1 - np.std([r['win_rate'] for r in results]) if len(results) > 1 else 1.0,
                'splits': results
            }

            logger.info(f"   Average Win Rate: {avg_metrics['avg_win_rate']:.1%}")
            logger.info(f"   Average Signals: {avg_metrics['avg_signals_pct']:.1%}")
            logger.info(f"   Stability: {avg_metrics['stability']:.3f}")

            return avg_metrics

        logger.warning("   No valid splits in walk-forward validation")
        return {}

    def test_regime_transitions(self, model_name: str) -> Dict:
        """Test model performance during regime transitions."""
        logger.info(f"\n🔄 Testing Regime Transitions for {model_name}")

        market_regime, signal_type = model_name.rsplit('_', 1)

        # Simplified and optimized query
        query = f"""
        WITH data_with_lag AS (
            SELECT 
                *,
                LAG(market_regime) OVER (ORDER BY timestamp) as prev_regime,
                LEAD(market_regime) OVER (ORDER BY timestamp) as next_regime
            FROM fas.mv_ml_training_data_simplified
            WHERE signal_type = '{signal_type}'
                AND target IS NOT NULL
                AND timestamp >= NOW() - INTERVAL '14 days'
                AND timestamp < NOW() - INTERVAL '2 days'
            ORDER BY timestamp
            LIMIT 10000
        )
        SELECT 
            *,
            CASE 
                WHEN market_regime != COALESCE(prev_regime, market_regime) THEN true
                WHEN market_regime != COALESCE(next_regime, market_regime) THEN true
                ELSE false
            END as near_transition
        FROM data_with_lag
        ORDER BY timestamp
        """

        try:
            with psycopg2.connect(**self.conn_params) as conn:
                # Set timeout for this query
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL statement_timeout = '5s';")

                logger.info(f"   Loading transition data for {model_name}...")
                df = pd.read_sql(query, conn)

            if len(df) == 0:
                logger.warning(f"   No data found for regime transitions test")
                return {}

            logger.info(f"   Loaded {len(df)} samples for transition analysis")

            results = {
                'transitions_found': 0,
                'performance_during_transitions': None,
                'performance_stable_periods': None,
                'stability_ratio': None
            }

            # Split data into transition and stable periods
            transition_data = df[df['near_transition'] == True]
            stable_data = df[(df['near_transition'] == False) & (df['market_regime'] == market_regime)]

            results['transitions_found'] = len(transition_data)
            logger.info(f"   Found {len(transition_data)} samples near transitions")
            logger.info(f"   Found {len(stable_data)} stable period samples")

            # Test on transition periods
            if len(transition_data) >= 20:
                logger.info(f"   Testing performance during transitions...")
                X_trans, y_trans = self.prepare_features(transition_data, model_name)

                if len(X_trans) > 0:
                    X_trans_scaled = pd.DataFrame(
                        self.scalers[model_name].transform(X_trans),
                        columns=X_trans.columns,
                        index=X_trans.index
                    )

                    y_pred_proba = self.models[model_name].predict_proba(X_trans_scaled)[:, 1]
                    y_pred = (y_pred_proba >= self.thresholds[model_name]).astype(int)

                    if y_pred.sum() > 0:
                        transition_win_rate = y_trans[y_pred == 1].mean()
                        results['performance_during_transitions'] = float(transition_win_rate)
                        logger.info(f"   Transition win rate: {transition_win_rate:.1%} ({y_pred.sum()} trades)")

            # Test on stable periods
            if len(stable_data) >= 20:
                logger.info(f"   Testing performance during stable periods...")
                X_stable, y_stable = self.prepare_features(stable_data.iloc[:min(1000, len(stable_data))], model_name)

                if len(X_stable) > 0:
                    X_stable_scaled = pd.DataFrame(
                        self.scalers[model_name].transform(X_stable),
                        columns=X_stable.columns,
                        index=X_stable.index
                    )

                    y_pred_proba = self.models[model_name].predict_proba(X_stable_scaled)[:, 1]
                    y_pred = (y_pred_proba >= self.thresholds[model_name]).astype(int)

                    if y_pred.sum() > 0:
                        stable_win_rate = y_stable[y_pred == 1].mean()
                        results['performance_stable_periods'] = float(stable_win_rate)
                        logger.info(f"   Stable win rate: {stable_win_rate:.1%} ({y_pred.sum()} trades)")

            # Calculate stability ratio
            if results['performance_during_transitions'] is not None and results['performance_stable_periods'] is not None:
                if results['performance_stable_periods'] > 0:
                    results['stability_ratio'] = results['performance_during_transitions'] / results['performance_stable_periods']
                    logger.info(f"   Stability ratio: {results['stability_ratio']:.3f}")

                    if results['stability_ratio'] > 0.8:
                        logger.info(f"   ✅ Model is stable across regime transitions")
                    else:
                        logger.warning(f"   ⚠️ Model performance degrades during transitions")

            return results

        except psycopg2.OperationalError as e:
            if "statement timeout" in str(e):
                logger.warning(f"   Query timeout - skipping regime transition test")
            else:
                logger.error(f"   Database error during regime transition test: {e}")
            return {}
        except Exception as e:
            logger.error(f"   Error testing regime transitions: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {}

    def backtest_model(self, model_name: str, days: int = 30) -> Dict:
        """Backtest model on historical data."""
        logger.info(f"\n📈 Backtesting {model_name} for {days} days")

        # Load test data
        df = self.load_test_data(model_name, start_days_ago=days, end_days_ago=2)

        if len(df) == 0:
            return {}

        # Prepare and scale features
        X, y = self.prepare_features(df, model_name)
        X_scaled = pd.DataFrame(
            self.scalers[model_name].transform(X),
            columns=X.columns,
            index=X.index
        )

        # Make predictions
        model = self.models[model_name]
        threshold = self.thresholds[model_name]

        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate metrics
        trades_taken = y_pred.sum()
        total_signals = len(y_pred)
        trades_pct = trades_taken / total_signals if total_signals > 0 else 0

        results = {
            'total_signals': total_signals,
            'trades_taken': trades_taken,
            'trades_pct': trades_pct,
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'win_rate': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'expected_profit': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }

        if trades_taken > 0:
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

            results.update({
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'win_rate': tp / trades_taken,
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1_score': f1_score(y, y_pred)
            })

            # Calculate profit metrics (assuming 3% TP/SL)
            profits = []
            cumulative_profit = 0
            max_profit = 0
            drawdowns = []

            for i, (pred, actual) in enumerate(zip(y_pred, y)):
                if pred == 1:
                    profit = 0.03 if actual == 1 else -0.03
                    profits.append(profit)
                    cumulative_profit += profit
                    max_profit = max(max_profit, cumulative_profit)

                    if max_profit > 0:
                        drawdown = (max_profit - cumulative_profit) / max_profit
                        drawdowns.append(drawdown)

            if profits:
                results['expected_profit'] = np.mean(profits)
                results['max_drawdown'] = max(drawdowns) if drawdowns else 0

                # Sharpe ratio (annualized)
                if len(profits) > 1:
                    daily_returns = pd.Series(profits)
                    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
                    results['sharpe_ratio'] = sharpe

        # Log results
        logger.info(f"   Total signals: {results['total_signals']}")
        logger.info(f"   Trades taken: {results['trades_taken']} ({results['trades_pct']:.1%})")

        if results['trades_taken'] > 0:
            logger.info(f"   Win rate: {results['win_rate']:.1%}")
            logger.info(f"   Expected profit per trade: {results['expected_profit']:.1%}")
            logger.info(f"   Max drawdown: {results['max_drawdown']:.1%}")
            logger.info(f"   Sharpe ratio: {results['sharpe_ratio']:.2f}")

        return results

    def ab_test_models(self, model_a: str, model_b: str, days: int = 14) -> Dict:
        """A/B test two models on the same data."""
        logger.info(f"\n🆚 A/B Testing: {model_a} vs {model_b}")

        # Extract common parameters
        _, signal_type_a = model_a.rsplit('_', 1)
        _, signal_type_b = model_b.rsplit('_', 1)

        if signal_type_a != signal_type_b:
            logger.warning("Can only A/B test models with same signal type")
            return {}

        # Load data for both regimes
        query = f"""
        SELECT *
        FROM fas.mv_ml_training_data_simplified
        WHERE signal_type = '{signal_type_a}'
            AND target IS NOT NULL
            AND timestamp >= NOW() - INTERVAL '{days} days'
            AND timestamp < NOW() - INTERVAL '2 days'
        ORDER BY timestamp
        """

        with psycopg2.connect(**self.conn_params) as conn:
            df = pd.read_sql(query, conn)

        results = {
            'model_a': model_a,
            'model_b': model_b,
            'test_samples': len(df),
            'performance_a': {},
            'performance_b': {},
            'statistical_significance': {}
        }

        # Test Model A
        if model_a in self.models:
            X_a, y_a = self.prepare_features(df, model_a)
            X_a_scaled = pd.DataFrame(
                self.scalers[model_a].transform(X_a),
                columns=X_a.columns
            )

            y_pred_a = (self.models[model_a].predict_proba(X_a_scaled)[:, 1] >=
                       self.thresholds[model_a]).astype(int)

            if y_pred_a.sum() > 0:
                results['performance_a'] = {
                    'trades': y_pred_a.sum(),
                    'win_rate': y_a[y_pred_a == 1].mean(),
                    'signals_pct': y_pred_a.sum() / len(y_pred_a)
                }

        # Test Model B
        if model_b in self.models:
            X_b, y_b = self.prepare_features(df, model_b)
            X_b_scaled = pd.DataFrame(
                self.scalers[model_b].transform(X_b),
                columns=X_b.columns
            )

            y_pred_b = (self.models[model_b].predict_proba(X_b_scaled)[:, 1] >=
                       self.thresholds[model_b]).astype(int)

            if y_pred_b.sum() > 0:
                results['performance_b'] = {
                    'trades': y_pred_b.sum(),
                    'win_rate': y_b[y_pred_b == 1].mean(),
                    'signals_pct': y_pred_b.sum() / len(y_pred_b)
                }

        # Statistical significance test
        if results['performance_a'] and results['performance_b']:
            # McNemar's test for paired binary outcomes
            if 'y_pred_a' in locals() and 'y_pred_b' in locals():
                # Create contingency table
                both_correct = ((y_pred_a == y_a) & (y_pred_b == y_b)).sum()
                a_correct_b_wrong = ((y_pred_a == y_a) & (y_pred_b != y_b)).sum()
                a_wrong_b_correct = ((y_pred_a != y_a) & (y_pred_b == y_b)).sum()
                both_wrong = ((y_pred_a != y_a) & (y_pred_b != y_b)).sum()

                # Chi-square test
                if a_correct_b_wrong + a_wrong_b_correct > 0:
                    chi2 = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2 / (a_correct_b_wrong + a_wrong_b_correct)
                    p_value = 1 - stats.chi2.cdf(chi2, 1)

                    results['statistical_significance'] = {
                        'chi2': chi2,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'winner': model_a if a_correct_b_wrong > a_wrong_b_correct else model_b
                    }

        # Log results
        logger.info(f"   Test samples: {results['test_samples']}")

        if results['performance_a']:
            logger.info(f"   {model_a}: WR={results['performance_a']['win_rate']:.1%}, "
                       f"Signals={results['performance_a']['signals_pct']:.1%}")

        if results['performance_b']:
            logger.info(f"   {model_b}: WR={results['performance_b']['win_rate']:.1%}, "
                       f"Signals={results['performance_b']['signals_pct']:.1%}")

        if results['statistical_significance']:
            sig = results['statistical_significance']
            logger.info(f"   Statistical significance: p={sig['p_value']:.4f}")
            if sig['significant']:
                logger.info(f"   ✅ {sig['winner']} is significantly better")
            else:
                logger.info(f"   ⭕ No significant difference")

        return results

    def save_validation_results(self, model_name: str, results: Dict):
        """Save validation results to database."""
        if not results:
            return

        query = """
        INSERT INTO smart_ml.model_performance (
            model_name, evaluation_date, total_predictions,
            true_positives, false_positives, true_negatives, false_negatives,
            win_rate, precision_score, recall_score,
            profit_factor, max_drawdown, regime_stability
        ) VALUES (
            %(model_name)s, %(evaluation_date)s, %(total_predictions)s,
            %(true_positives)s, %(false_positives)s, %(true_negatives)s, %(false_negatives)s,
            %(win_rate)s, %(precision_score)s, %(recall_score)s,
            %(profit_factor)s, %(max_drawdown)s, %(regime_stability)s
        )
        """

        params = {
            'model_name': model_name,
            'evaluation_date': datetime.now().date(),
            'total_predictions': int(results.get('trades_taken', 0)),
            'true_positives': int(results.get('true_positives', 0)),
            'false_positives': int(results.get('false_positives', 0)),
            'true_negatives': int(results.get('true_negatives', 0)),
            'false_negatives': int(results.get('false_negatives', 0)),
            'win_rate': float(results.get('win_rate', 0)),
            'precision_score': float(results.get('precision', 0)),
            'recall_score': float(results.get('recall', 0)),
            'profit_factor': float(results.get('expected_profit', 0) * 100),
            'max_drawdown': float(results.get('max_drawdown', 0)),
            'regime_stability': float(results.get('stability', 0.5))
        }

        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                conn.commit()
            logger.info(f"💾 Saved validation results for {model_name}")
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")

    def validate_all_models(self):
        """Validate all loaded models."""
        logger.info("="*60)
        logger.info("SMART ML VALIDATOR - COMPREHENSIVE MODEL VALIDATION")
        logger.info(f"Started at: {datetime.now()}")
        logger.info("="*60)

        if not self.models:
            logger.error("No models loaded for validation!")
            logger.info("Please train models first: python smart_ml_orchestrator.py train")
            return {}

        validation_results = {}

        logger.info(f"\nValidating {len(self.models)} models...")

        for model_name in self.models.keys():
            logger.info(f"\n{'='*40}")
            logger.info(f"Validating {model_name}")
            logger.info(f"{'='*40}")

            try:
                results = {
                    'model_name': model_name,
                    'walk_forward': None,
                    'regime_transitions': None,
                    'backtest': None
                }

                # Walk-forward validation
                logger.info("Running walk-forward validation...")
                results['walk_forward'] = self.walk_forward_validation(model_name)

                # Regime transitions test
                logger.info("Testing regime transitions...")
                results['regime_transitions'] = self.test_regime_transitions(model_name)

                # Backtest
                logger.info("Running backtest...")
                results['backtest'] = self.backtest_model(model_name)

                validation_results[model_name] = results

                # Save to database
                if results['backtest']:
                    self.save_validation_results(model_name, results['backtest'])

            except Exception as e:
                logger.error(f"Error validating {model_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        # A/B Tests
        logger.info("\n" + "="*40)
        logger.info("A/B TESTING")
        logger.info("="*40)

        # Test BULL vs NEUTRAL for BUY
        if 'BULL_BUY' in self.models and 'NEUTRAL_BUY' in self.models:
            self.ab_test_models('BULL_BUY', 'NEUTRAL_BUY')

        # Test BEAR vs NEUTRAL for SELL
        if 'BEAR_SELL' in self.models and 'NEUTRAL_SELL' in self.models:
            self.ab_test_models('BEAR_SELL', 'NEUTRAL_SELL')

        # Summary
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)

        for model_name, results in validation_results.items():
            config = self.model_configs.get(model_name, {})
            backtest = results.get('backtest', {})

            logger.info(f"\n{model_name}:")

            if backtest.get('trades_taken', 0) > 0:
                win_rate = backtest.get('win_rate', 0)
                signals_pct = backtest.get('trades_pct', 0)
                target_wr = config.get('target_win_rate', 0.6)
                target_signals = config.get('target_signals_pct', 0.1)

                logger.info(f"  Win Rate: {win_rate:.1%} (target: {target_wr:.1%})")
                logger.info(f"  Signals: {signals_pct:.1%} (target: {target_signals:.1%})")
                logger.info(f"  Sharpe Ratio: {backtest.get('sharpe_ratio', 0):.2f}")

                # Assessment
                if win_rate >= target_wr * 0.9 and \
                   target_signals * 0.5 <= signals_pct <= target_signals * 1.5:
                    logger.info(f"  ✅ MEETS TARGETS")
                else:
                    logger.info(f"  ⚠️ NEEDS OPTIMIZATION")
            else:
                logger.info(f"  ❌ NO TRADES TAKEN")

        logger.info(f"\n✅ Validation complete at {datetime.now()}")

        return validation_results


def main():
    """Run comprehensive validation."""
    validator = SmartValidator()
    results = validator.validate_all_models()

    # Generate alert if any model needs retraining
    alerts = []
    for model_name, result in results.items():
        backtest = result.get('backtest', {})
        config = validator.model_configs.get(model_name, {})

        if backtest.get('trades_taken', 0) > 0:
            win_rate = backtest.get('win_rate', 0)
            target_wr = config.get('target_win_rate', 0.6)

            if win_rate < target_wr * 0.8:  # 20% below target
                alerts.append(f"🚨 {model_name}: Win rate {win_rate:.1%} is critically low!")

    if alerts:
        logger.info("\n⚠️ ALERTS:")
        for alert in alerts:
            logger.info(alert)
        logger.info("Consider retraining these models.")


if __name__ == "__main__":
    main()