"""
Smart ML Training System - Market-Adaptive Models
===================================================
Обучает 6 специализированных моделей для каждой комбинации:
- Market Regime: BULL, NEUTRAL, BEAR
- Signal Type: BUY, SELL
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.ensemble import VotingClassifier
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import joblib
import json
import warnings
import logging
import os
from typing import Dict, Tuple, Optional, List
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SmartMLTrainer:
    """Market-adaptive ML trainer with specialized models for each regime."""

    # Конфигурация для каждой модели
    MODEL_CONFIGS = {
        'BULL_BUY': {
            'window_days': 14,
            'min_samples': 3000,
            'target_signals_pct': 0.15,  # 15% сигналов
            'target_win_rate': 0.78,
            'focus_features': ['rs_momentum', 'macd_histogram', 'volume_zscore', 'buy_ratio_weighted'],
            'ensemble': False
        },
        'BULL_SELL': {
            'window_days': 14,
            'min_samples': 2000,
            'target_signals_pct': 0.02,  # 1-2% сигналов
            'target_win_rate': 0.81,
            'focus_features': ['has_cvd_divergence', 'pattern_1_impact', 'has_distribution', 'rsi_zone'],
            'ensemble': True  # Критичная модель - используем ensemble
        },
        'NEUTRAL_BUY': {
            'window_days': 14,
            'min_samples': 3000,
            'target_signals_pct': 0.02,
            'target_win_rate': 0.65,
            'focus_features': ['price_to_poc_7d_pct', 'normalized_imbalance', 'has_accumulation', 'atr_pct'],
            'ensemble': True
        },
        'NEUTRAL_SELL': {
            'window_days': 14,
            'min_samples': 3000,
            'target_signals_pct': 0.02,
            'target_win_rate': 0.80,
            'focus_features': ['price_to_poc_7d_pct', 'has_distribution', 'combo_1_score', 'oi_delta_pct'],
            'ensemble': True
        },
        'BEAR_BUY': {
            'window_days': 14,
            'min_samples': 2000,
            'target_signals_pct': 0.02,
            'target_win_rate': 0.65,
            'focus_features': ['rsi', 'has_squeeze_ignition', 'funding_rate_avg', 'pattern_2_confidence'],
            'ensemble': True
        },
        'BEAR_SELL': {
            'window_days': 14,
            'min_samples': 3000,
            'target_signals_pct': 0.12,  # 10-15% сигналов
            'target_win_rate': 0.70,
            'focus_features': ['has_momentum_exhaustion', 'cvd_delta', 'pattern_1_impact', 'volume_zscore'],
            'ensemble': False
        }
    }

    def __init__(self):
        """Initialize trainer with database connection."""
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

        # Создаем схему и таблицы при инициализации
        self._setup_database()

    def _setup_database(self):
        """Create smart_ml schema and tables if not exists."""
        setup_sql = """
        -- Создание схемы
        CREATE SCHEMA IF NOT EXISTS smart_ml;

        -- История обучения моделей
        CREATE TABLE IF NOT EXISTS smart_ml.training_history (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(50),
            market_regime VARCHAR(20),
            signal_type VARCHAR(10),
            training_window_days INT,
            samples_count INT,
            train_win_rate DECIMAL(5,4),
            val_win_rate DECIMAL(5,4),
            threshold DECIMAL(5,4),
            signals_percentage DECIMAL(5,4),
            expected_profit DECIMAL(7,4),
            feature_importance JSONB,
            model_params JSONB,
            model_version VARCHAR(100),
            created_at TIMESTAMP DEFAULT NOW()
        );

        -- Предсказания
        CREATE TABLE IF NOT EXISTS smart_ml.predictions (
            id SERIAL PRIMARY KEY,
            signal_id BIGINT,
            model_name VARCHAR(50),
            market_regime VARCHAR(20),
            signal_type VARCHAR(10),
            prediction_proba DECIMAL(5,4),
            prediction BOOLEAN,
            confidence_level VARCHAR(20),
            features_hash VARCHAR(32),
            created_at TIMESTAMP DEFAULT NOW()
        );

        -- Производительность моделей
        CREATE TABLE IF NOT EXISTS smart_ml.model_performance (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(50),
            evaluation_date DATE,
            total_predictions INT,
            true_positives INT,
            false_positives INT,
            true_negatives INT,
            false_negatives INT,
            win_rate DECIMAL(5,4),
            precision_score DECIMAL(5,4),
            recall_score DECIMAL(5,4),
            profit_factor DECIMAL(7,4),
            max_drawdown DECIMAL(5,4),
            regime_stability DECIMAL(5,4),
            created_at TIMESTAMP DEFAULT NOW()
        );

        -- Дрифт моделей
        CREATE TABLE IF NOT EXISTS smart_ml.model_drift (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(50),
            check_timestamp TIMESTAMP,
            kl_divergence DECIMAL(7,6),
            psi_score DECIMAL(7,6),
            feature_drift JSONB,
            target_drift DECIMAL(5,4),
            needs_retrain BOOLEAN,
            drift_severity VARCHAR(20),
            created_at TIMESTAMP DEFAULT NOW()
        );

        -- Индексы для производительности
        CREATE INDEX IF NOT EXISTS idx_training_history_model_name ON smart_ml.training_history(model_name);
        CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON smart_ml.predictions(created_at);
        CREATE INDEX IF NOT EXISTS idx_performance_model_date ON smart_ml.model_performance(model_name, evaluation_date);
        CREATE INDEX IF NOT EXISTS idx_drift_model_timestamp ON smart_ml.model_drift(model_name, check_timestamp);
        """

        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(setup_sql)
                    conn.commit()
            logger.info("✅ Database schema smart_ml created/verified successfully")
        except Exception as e:
            logger.error(f"❌ Failed to setup database: {e}")
            raise

    def load_regime_data(self, market_regime: str, signal_type: str, window_days: int) -> pd.DataFrame:
        """Load data for specific market regime and signal type."""
        query = f"""
        WITH regime_data AS (
            SELECT *
            FROM fas.mv_ml_training_data_simplified
            WHERE market_regime = '{market_regime}'
                AND signal_type = '{signal_type}'
                AND target IS NOT NULL
                AND timestamp >= NOW() - INTERVAL '{window_days} days'
                AND timestamp < NOW() - INTERVAL '48 hours'
            ORDER BY timestamp
        )
        SELECT *
        FROM regime_data
        WHERE (SELECT COUNT(*) FROM regime_data) >= {self.MODEL_CONFIGS[f'{market_regime}_{signal_type}']['min_samples']}
        """

        with psycopg2.connect(**self.conn_params) as conn:
            df = pd.read_sql(query, conn)

        if len(df) > 0:
            logger.info(f"📊 Loaded {len(df)} samples for {market_regime}_{signal_type}")
            logger.info(f"   Win rate: {df['target'].mean():.1%}")
            logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        else:
            logger.warning(f"⚠️ Insufficient data for {market_regime}_{signal_type}")

        return df

    def engineer_features(self, df: pd.DataFrame, model_name: str, is_training: bool = True) -> Tuple[
        pd.DataFrame, Optional[pd.Series]]:
        """Feature engineering specific to each model."""
        df = df.copy()
        config = self.MODEL_CONFIGS[model_name]

        # Удаляем ненужные колонки
        remove_cols = ['id', 'trading_pair_id', 'timestamp', 'pair_symbol',
                       'signal_type', 'signal_strength', 'patterns_details',
                       'combinations_details', 'created_at']
        remove_cols += [col for col in df.columns if col.startswith('_meta_')]

        for col in remove_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Обработка экстремальных значений
        for col in ['poc_volume_7d', 'poc_volume_24h']:
            if col in df.columns:
                q99 = df[col].quantile(0.99) if len(df) > 100 else df[col].max()
                df[col] = df[col].clip(upper=q99)
                df[f'{col}_log'] = np.log1p(df[col])

        # Временные признаки
        df['hour'] = pd.to_datetime(df.index).hour
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Model-specific features based on focus areas
        if 'BULL' in model_name:
            # Momentum features для Bull market
            df['momentum_strength'] = df['rs_momentum'] * df['macd_histogram']
            df['volume_momentum'] = df['volume_zscore'] * df['buy_ratio_weighted']
            df['trend_strength'] = (df['macd_line'] - df['macd_signal']).abs()

            if 'SELL' in model_name:
                # Divergence features для BULL_SELL
                df['price_rsi_divergence'] = (df['price_change_pct'] * df['rsi']) / 100
                df['cvd_price_ratio'] = df['cvd_delta'] / (df['close_price'] + 1)

        elif 'NEUTRAL' in model_name:
            # Mean reversion features
            df['poc_deviation'] = (df['price_to_poc_7d_pct'].abs() +
                                   df['price_to_poc_24h_pct'].abs()) / 2
            df['imbalance_smoothed'] = df['normalized_imbalance'] * df['smoothed_imbalance']
            df['volatility_adjusted_atr'] = df['atr_pct'] / (df['volume_zscore'].abs() + 1)

        elif 'BEAR' in model_name:
            # Oversold/continuation features
            df['oversold_strength'] = (30 - df['rsi'].clip(upper=30)) / 30
            df['fear_index'] = df['funding_rate_avg'] * df['oi_delta_pct']

            if 'SELL' in model_name:
                # Continuation patterns
                df['bearish_continuation'] = df['has_momentum_exhaustion'] * df['cvd_delta'].clip(upper=0).abs()

        # Добавляем focus features с повышенным весом
        for feat in config['focus_features']:
            if feat in df.columns:
                df[f'{feat}_weighted'] = df[feat] * 1.5  # Увеличиваем важность

        # Pattern confidence features
        pattern_cols = [col for col in df.columns if 'pattern_' in col and 'confidence' in col]
        if pattern_cols:
            df['max_pattern_confidence'] = df[pattern_cols].max(axis=1)
            df['avg_pattern_confidence'] = df[pattern_cols].mean(axis=1)

        # Combo features
        combo_cols = [col for col in df.columns if 'combo_' in col and 'score' in col]
        if combo_cols:
            df['total_combo_score'] = df[combo_cols].sum(axis=1)

        # Категориальные переменные
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'target':
                df[col] = pd.Categorical(df[col].fillna('unknown')).codes

        # Убираем константные признаки
        if is_training:
            constant_features = []
            for col in df.columns:
                if col != 'target' and df[col].nunique() <= 1:
                    constant_features.append(col)

            if constant_features:
                df = df.drop(columns=constant_features)
                logger.info(f"   Removed {len(constant_features)} constant features")

            self.feature_columns[model_name] = [col for col in df.columns if col != 'target']
            logger.info(f"   Total features: {len(self.feature_columns[model_name])}")

        # Подготовка X и y
        X = df[self.feature_columns[model_name]].fillna(0)
        y = df['target'].astype(int) if 'target' in df.columns else None

        return X, y

    def find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                               target_signals_pct: float, target_win_rate: float) -> float:
        """Find threshold that achieves target signals percentage and win rate."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

        best_threshold = 0.5
        best_score = 0

        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            signals_pct = y_pred.sum() / len(y_pred)

            # Проверяем попадание в целевой диапазон сигналов
            if target_signals_pct * 0.5 <= signals_pct <= target_signals_pct * 1.5:
                if y_pred.sum() > 0:
                    win_rate = y_true[y_pred == 1].mean()

                    # Scoring function: баланс между win rate и количеством сигналов
                    score = win_rate * min(signals_pct / target_signals_pct, 1.0)

                    # Бонус за приближение к целевому win rate
                    if win_rate >= target_win_rate * 0.9:
                        score *= 1.2

                    if score > best_score:
                        best_score = score
                        best_threshold = thresh

        return best_threshold

    def train_single_model(self, X: pd.DataFrame, y: pd.Series, model_name: str) -> Tuple[object, object]:
        """Train a single XGBoost or ensemble model."""
        config = self.MODEL_CONFIGS[model_name]

        # Scaler selection based on regime
        if 'BEAR' in model_name or 'NEUTRAL' in model_name:
            scaler = RobustScaler()  # Более устойчив к выбросам
        else:
            scaler = StandardScaler()

        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # Base parameters
        base_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'verbosity': 0
        }

        # Model-specific parameters
        if 'BULL_BUY' in model_name:
            # Агрессивная модель для bull market
            xgb_params = {
                **base_params,
                'learning_rate': 0.15,
                'max_depth': 5,
                'n_estimators': 150,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'min_child_weight': 5,
                'gamma': 0.1
            }
        elif 'SELL' in model_name and config['ensemble']:
            # Консервативная модель для редких сигналов
            xgb_params = {
                **base_params,
                'learning_rate': 0.05,
                'max_depth': 3,
                'n_estimators': 300,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_weight': 30,
                'gamma': 0.3,
                'scale_pos_weight': (y == 0).sum() / (y == 1).sum()  # Балансировка классов
            }
        else:
            # Стандартные параметры
            xgb_params = {
                **base_params,
                'learning_rate': 0.1,
                'max_depth': 4,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 10,
                'gamma': 0.2
            }

        if config['ensemble']:
            # Создаем ensemble из XGBoost и LightGBM
            xgb_model = xgb.XGBClassifier(**xgb_params)

            lgb_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': xgb_params['learning_rate'],
                'feature_fraction': xgb_params['colsample_bytree'],
                'bagging_fraction': xgb_params['subsample'],
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
                'n_estimators': xgb_params['n_estimators']
            }

            lgb_model = lgb.LGBMClassifier(**lgb_params)

            # Voting classifier with soft voting
            model = VotingClassifier(
                estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
                voting='soft',
                weights=[0.6, 0.4]  # XGBoost имеет больший вес
            )
        else:
            model = xgb.XGBClassifier(**xgb_params)

        # Train model
        model.fit(X_scaled, y)

        return model, scaler

    def train_model(self, model_name: str) -> Dict:
        """Train a specific market regime model."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training {model_name} model")
        logger.info(f"{'=' * 60}")

        # Parse model name
        market_regime, signal_type = model_name.rsplit('_', 1)
        config = self.MODEL_CONFIGS[model_name]

        # Load data
        df = self.load_regime_data(market_regime, signal_type, config['window_days'])

        if len(df) < config['min_samples']:
            logger.warning(f"❌ Not enough data for {model_name}: {len(df)} < {config['min_samples']}")
            return None

        # Feature engineering
        X, y = self.engineer_features(df, model_name, is_training=True)

        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        val_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train model
            model, scaler = self.train_single_model(X_train, y_train, model_name)

            # Validate
            X_val_scaled = pd.DataFrame(
                scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )

            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            val_scores.append(roc_auc_score(y_val, y_pred_proba))

        logger.info(f"📊 Validation AUC scores: {[f'{s:.3f}' for s in val_scores]}")
        logger.info(f"   Mean AUC: {np.mean(val_scores):.3f}")

        # Train final model on all data
        final_model, final_scaler = self.train_single_model(X, y, model_name)

        # Scale all data for threshold optimization
        X_scaled = pd.DataFrame(
            final_scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

        # Get predictions
        y_pred_proba = final_model.predict_proba(X_scaled)[:, 1]

        # Find optimal threshold
        threshold = self.find_optimal_threshold(
            y, y_pred_proba,
            config['target_signals_pct'],
            config['target_win_rate']
        )

        # Apply threshold
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate metrics
        signals_pct = y_pred.sum() / len(y_pred)
        win_rate = y[y_pred == 1].mean() if y_pred.sum() > 0 else 0

        logger.info(f"\n🎯 Training Results:")
        logger.info(f"   Threshold: {threshold:.3f}")
        logger.info(f"   Signals: {signals_pct:.1%} (target: {config['target_signals_pct']:.1%})")
        logger.info(f"   Win rate: {win_rate:.1%} (target: {config['target_win_rate']:.1%})")

        # Feature importance
        feature_importance = {}
        if hasattr(final_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': final_model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)

            feature_importance = importance_df.set_index('feature')['importance'].to_dict()

            logger.info("\n📈 Top 10 Features:")
            for feat, imp in list(feature_importance.items())[:10]:
                logger.info(f"   {feat}: {imp:.3f}")

        # Save to database
        self._save_training_history(
            model_name=model_name,
            market_regime=market_regime,
            signal_type=signal_type,
            config=config,
            train_win_rate=y.mean(),
            val_win_rate=win_rate,
            threshold=threshold,
            signals_pct=signals_pct,
            feature_importance=feature_importance,
            samples_count=len(df)
        )

        # Store model components
        self.models[model_name] = final_model
        self.scalers[model_name] = final_scaler
        self.thresholds[model_name] = threshold

        return {
            'model': final_model,
            'scaler': final_scaler,
            'threshold': threshold,
            'feature_columns': self.feature_columns[model_name],
            'metrics': {
                'win_rate': win_rate,
                'signals_pct': signals_pct,
                'val_auc': np.mean(val_scores)
            }
        }

    def _save_training_history(self, **kwargs):
        """Save training history to database."""
        query = """
        INSERT INTO smart_ml.training_history (
            model_name, market_regime, signal_type, training_window_days,
            samples_count, train_win_rate, val_win_rate, threshold,
            signals_percentage, feature_importance, model_params, model_version
        ) VALUES (
            %(model_name)s, %(market_regime)s, %(signal_type)s, %(window_days)s,
            %(samples_count)s, %(train_win_rate)s, %(val_win_rate)s, %(threshold)s,
            %(signals_pct)s, %(feature_importance)s, %(model_params)s, %(model_version)s
        )
        """

        params = {
            'model_name': kwargs['model_name'],
            'market_regime': kwargs['market_regime'],
            'signal_type': kwargs['signal_type'],
            'window_days': kwargs['config']['window_days'],
            'samples_count': kwargs['samples_count'],
            'train_win_rate': float(kwargs['train_win_rate']),
            'val_win_rate': float(kwargs['val_win_rate']),
            'threshold': float(kwargs['threshold']),
            'signals_pct': float(kwargs['signals_pct']),
            'feature_importance': json.dumps(kwargs['feature_importance']),
            'model_params': json.dumps(kwargs['config']),
            'model_version': f"v{datetime.now().strftime('%Y%m%d_%H%M')}"
        }

        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save training history: {e}")

    def train_all_models(self):
        """Train all 6 market-specific models."""
        logger.info("=" * 60)
        logger.info("SMART ML TRAINING - MARKET ADAPTIVE MODELS")
        logger.info(f"Started at: {datetime.now()}")
        logger.info("=" * 60)

        results = {}

        for model_name in self.MODEL_CONFIGS.keys():
            try:
                result = self.train_model(model_name)
                if result:
                    results[model_name] = result

                    # Save model to disk
                    self.save_model(model_name)
            except Exception as e:
                logger.error(f"❌ Failed to train {model_name}: {e}")
                continue

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)

        for model_name, result in results.items():
            metrics = result['metrics']
            logger.info(f"\n{model_name}:")
            logger.info(f"  Win Rate: {metrics['win_rate']:.1%}")
            logger.info(f"  Signals: {metrics['signals_pct']:.1%}")
            logger.info(f"  Val AUC: {metrics['val_auc']:.3f}")

        logger.info(f"\n✅ Successfully trained {len(results)}/{len(self.MODEL_CONFIGS)} models")
        logger.info(f"Completed at: {datetime.now()}")

        return results

    def save_model(self, model_name: str):
        """Save model to disk."""
        os.makedirs('models/smart_ml', exist_ok=True)

        model_data = {
            'model': self.models[model_name],
            'scaler': self.scalers[model_name],
            'threshold': self.thresholds[model_name],
            'feature_columns': self.feature_columns[model_name],
            'config': self.MODEL_CONFIGS[model_name],
            'timestamp': datetime.now().isoformat(),
            'version': f"v{datetime.now().strftime('%Y%m%d_%H%M')}"
        }

        filename = f'models/smart_ml/{model_name.lower()}_model.pkl'
        joblib.dump(model_data, filename)
        logger.info(f"💾 Model saved to {filename}")


def main():
    """Main execution."""
    trainer = SmartMLTrainer()
    results = trainer.train_all_models()

    if results:
        logger.info("\n🎯 TARGET ASSESSMENT:")
        for model_name, result in results.items():
            config = trainer.MODEL_CONFIGS[model_name]
            metrics = result['metrics']

            target_achieved = (
                    metrics['win_rate'] >= config['target_win_rate'] * 0.9 and
                    config['target_signals_pct'] * 0.5 <= metrics['signals_pct'] <= config['target_signals_pct'] * 1.5
            )

            status = "✅ TARGET ACHIEVED" if target_achieved else "⚠️ NEEDS OPTIMIZATION"
            logger.info(f"{model_name}: {status}")


if __name__ == "__main__":
    main()