#!/usr/bin/env python3
"""
Smart ML Out-of-Sample Backtesting
Tests models on data from 2-4 weeks ago (not used in training)
"""

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta
import logging
import joblib
import os
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')

# Import ML libraries for model loading
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class SmartMLBacktest:
    """Out-of-sample backtesting for Smart ML models."""

    def __init__(self):
        self.conn_params = {
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }
        self.models = {}
        self.load_models()

    def load_models(self):
        """Load all trained models."""
        model_names = ['BULL_BUY', 'BULL_SELL', 'NEUTRAL_BUY',
                       'NEUTRAL_SELL', 'BEAR_BUY', 'BEAR_SELL']

        for model_name in model_names:
            path = f'models/smart_ml/{model_name.lower().replace("_", "_")}_model.pkl'
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    self.models[model_name] = joblib.load(f)
                logger.info(f"✅ Loaded {model_name}")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features that models expect."""
        df = df.copy()

        # Add log features for POC volumes (handle nulls)
        df['poc_volume_24h_log'] = np.log1p(df['poc_volume_24h'].fillna(0))
        df['poc_volume_7d_log'] = np.log1p(df['poc_volume_7d'].fillna(0))

        # Momentum features
        df['momentum_strength'] = df['rsi'] - 50
        df['volume_momentum'] = df['volume_zscore'] * df['buy_ratio']
        df['trend_strength'] = df['macd_line'] / (df['atr'] + 1e-8)

        # Weighted features
        df['rs_momentum_weighted'] = df['rs_value'] * df.get('rs_momentum', 0).fillna(0)
        df['macd_histogram_weighted'] = df['macd_histogram'] * df['volume_zscore']
        df['volume_zscore_weighted'] = df['volume_zscore'] * abs(df['normalized_imbalance'])
        df['buy_ratio_weighted_weighted'] = df['buy_ratio_weighted'] * df['volume_zscore']

        # Pattern aggregations
        pattern_cols = ['pattern_1_confidence', 'pattern_2_confidence', 'pattern_3_confidence']
        df['max_pattern_confidence'] = df[pattern_cols].max(axis=1).fillna(0)
        df['avg_pattern_confidence'] = df[pattern_cols].mean(axis=1).fillna(0)

        # Combo score
        combo_cols = ['combo_1_score', 'combo_2_score']
        df['total_combo_score'] = df[combo_cols].sum(axis=1).fillna(0)

        return df

    def get_test_data(self, weeks_ago_start: int = 4, weeks_ago_end: int = 2) -> pd.DataFrame:
        """Get test data from specified weeks ago."""
        query = f"""
        SELECT *
        FROM fas.mv_ml_training_data_simplified
        WHERE timestamp >= NOW() - INTERVAL '{weeks_ago_start} weeks'
          AND timestamp < NOW() - INTERVAL '{weeks_ago_end} weeks'
          AND target IS NOT NULL
          AND _meta_outcome_type IS NOT NULL
        ORDER BY timestamp
        """

        with psycopg2.connect(**self.conn_params) as conn:
            df = pd.read_sql(query, conn)

        logger.info(f"📊 Loaded {len(df)} test samples")
        logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Add engineered features
        df = self.engineer_features(df)

        # Group by regime and signal type
        for regime in df['market_regime'].unique():
            for signal_type in df['signal_type'].unique():
                subset = df[(df['market_regime'] == regime) & (df['signal_type'] == signal_type)]
                if len(subset) > 0:
                    win_rate = subset['target'].mean()
                    logger.info(f"   {regime}_{signal_type}: {len(subset)} samples, "
                                f"WR: {win_rate:.1%}")

        return df

    def test_model(self, model_name: str, df: pd.DataFrame) -> Dict:
        """Test single model on out-of-sample data."""
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found")
            return {}

        model_data = self.models[model_name]
        model = model_data['model']
        scaler = model_data['scaler']
        threshold = model_data['threshold']
        features = model_data['feature_columns']

        # Filter data for this model
        regime, signal_type = model_name.split('_')
        test_df = df[(df['market_regime'] == regime) &
                     (df['signal_type'] == signal_type)].copy()

        if len(test_df) == 0:
            return {'error': 'No test data'}

        # Check for missing features and add defaults
        missing_features = set(features) - set(test_df.columns)
        if missing_features:
            logger.warning(f"Missing features for {model_name}: {missing_features}")
            for feat in missing_features:
                test_df[feat] = 0

        # Prepare features
        X = test_df[features].fillna(0)
        X_scaled = scaler.transform(X)
        y_true = test_df['target'].astype(int).values

        # Make predictions
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate metrics
        trades_taken = y_pred.sum()
        if trades_taken > 0:
            win_rate = (y_true[y_pred == 1] == 1).mean()

            # Calculate returns based on actual outcomes
            returns = []
            for idx in range(len(test_df)):
                if y_pred[idx]:
                    row = test_df.iloc[idx]
                    if row['target']:  # Win
                        # Use favorable move or default 2%
                        ret = abs(row.get('_meta_max_favorable_move', 2.0))
                    else:  # Loss
                        # Use adverse move or default -2%
                        ret = -abs(row.get('_meta_max_adverse_move', 2.0))
                    returns.append(ret)

            returns = np.array(returns)

            # Calculate performance metrics
            avg_return = returns.mean() if len(returns) > 0 else 0
            std_return = returns.std() if len(returns) > 1 else 0
            sharpe = (avg_return / std_return * np.sqrt(252 / 2)) if std_return > 0 else 0

            # Calculate max drawdown
            cumsum = np.cumsum(returns)
            if len(cumsum) > 0:
                running_max = np.maximum.accumulate(cumsum)
                drawdown = (cumsum - running_max)
                max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
            else:
                max_dd = 0

            # Probability distribution
            if trades_taken > 0:
                prob_bins = pd.cut(y_pred_proba[y_pred == 1],
                                   bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0],
                                   labels=['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'])
                prob_dist = prob_bins.value_counts().to_dict()
            else:
                prob_dist = {}

        else:
            win_rate = 0
            avg_return = 0
            sharpe = 0
            max_dd = 0
            prob_dist = {}

        return {
            'total_signals': len(test_df),
            'trades_taken': trades_taken,
            'trade_frequency': trades_taken / len(test_df) * 100 if len(test_df) > 0 else 0,
            'win_rate': win_rate * 100,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'threshold': threshold,
            'avg_probability': y_pred_proba[y_pred == 1].mean() if trades_taken > 0 else 0,
            'probability_distribution': prob_dist,
            'date_range': f"{test_df['timestamp'].min()} to {test_df['timestamp'].max()}"
        }

    def compare_with_training_results(self, model_name: str, test_results: Dict):
        """Compare out-of-sample results with training metrics."""
        query = """
        SELECT 
            win_rate * 100 as training_wr,
            precision_score * 100 as training_precision,
            total_predictions
        FROM smart_ml.model_performance
        WHERE model_name = %s
        ORDER BY evaluation_date DESC
        LIMIT 1
        """

        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (model_name,))
                    training = cur.fetchone()

            if training and test_results.get('trades_taken', 0) > 0:
                degradation = test_results['win_rate'] - training[0]

                logger.info(f"\n📊 Performance Comparison for {model_name}:")
                logger.info(f"   Training WR: {training[0]:.1f}%")
                logger.info(f"   Out-of-Sample WR: {test_results['win_rate']:.1f}%")
                logger.info(f"   Degradation: {degradation:+.1f}%")

                if abs(degradation) > 10:
                    logger.warning(f"   ⚠️ Significant {'degradation' if degradation < 0 else 'improvement'}!")
                else:
                    logger.info(f"   ✅ Performance stable")
        except Exception as e:
            logger.warning(f"Could not compare with training results: {e}")

    def run_full_backtest(self):
        """Run complete out-of-sample backtest."""
        logger.info("=" * 60)
        logger.info("SMART ML OUT-OF-SAMPLE BACKTEST")
        logger.info(f"Testing on data from 2-4 weeks ago")
        logger.info("=" * 60)

        # Get test data
        df = self.get_test_data(weeks_ago_start=4, weeks_ago_end=2)

        if len(df) == 0:
            logger.error("No test data available")
            return

        results = {}

        # Test each model
        for model_name in self.models.keys():
            logger.info(f"\nTesting {model_name}...")
            results[model_name] = self.test_model(model_name, df)

            if 'error' not in results[model_name] and results[model_name].get('trades_taken', 0) > 0:
                logger.info(f"   Trades: {results[model_name]['trades_taken']}/{results[model_name]['total_signals']} "
                            f"({results[model_name]['trade_frequency']:.1f}%)")
                logger.info(f"   Win Rate: {results[model_name]['win_rate']:.1f}%")
                logger.info(f"   Avg Return: {results[model_name]['avg_return']:.2f}%")
                logger.info(f"   Sharpe: {results[model_name]['sharpe_ratio']:.2f}")
                logger.info(f"   Max DD: {results[model_name]['max_drawdown']:.1f}%")

                # Compare with training
                self.compare_with_training_results(model_name, results[model_name])
            else:
                logger.warning(f"   No trades generated or no data available")

        # Summary statistics
        self.print_summary(results)

        return results

    def print_summary(self, results: Dict):
        """Print summary of backtest results."""
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 60)

        # Best performing models
        valid_results = {k: v for k, v in results.items()
                         if 'error' not in v and v.get('trades_taken', 0) > 0}

        if valid_results:
            by_wr = sorted([(k, v['win_rate']) for k, v in valid_results.items()],
                           key=lambda x: x[1], reverse=True)

            logger.info("\n🏆 Top Models by Win Rate:")
            for model, wr in by_wr[:3]:
                trades = valid_results[model]['trades_taken']
                logger.info(f"   {model}: {wr:.1f}% ({trades} trades)")

            # Models with most trades
            by_trades = sorted([(k, v['trades_taken']) for k, v in valid_results.items()],
                               key=lambda x: x[1], reverse=True)

            logger.info("\n📊 Most Active Models:")
            for model, trades in by_trades[:3]:
                wr = valid_results[model]['win_rate']
                logger.info(f"   {model}: {trades} trades (WR: {wr:.1f}%)")

            # Overall statistics
            total_trades = sum(r['trades_taken'] for r in valid_results.values())
            if total_trades > 0:
                weighted_wr = sum(r['win_rate'] * r['trades_taken']
                                  for r in valid_results.values()) / total_trades

                logger.info(f"\n📈 Overall Statistics:")
                logger.info(f"   Total Trades: {total_trades}")
                logger.info(f"   Weighted Win Rate: {weighted_wr:.1f}%")


def main():
    backtest = SmartMLBacktest()
    results = backtest.run_full_backtest()

    # Save results if any
    if results:
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            df_results = pd.DataFrame(valid_results).T
            filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_results.to_csv(filename)
            logger.info(f"\n💾 Results saved to {filename}")


if __name__ == "__main__":
    main()