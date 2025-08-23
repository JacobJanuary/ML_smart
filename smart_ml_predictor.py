"""
Smart ML Predictor - Production Predictions with Auto Model Selection
======================================================================
Автоматически выбирает и применяет правильную модель на основе:
- Текущего market regime
- Типа сигнала (BUY/SELL)
"""

import pandas as pd
import numpy as np
import joblib
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import logging
import os
import hashlib
import json
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SmartPredictor:
    """Production predictor with dynamic model selection based on market regime."""

    def __init__(self):
        """Initialize predictor with database connection and load all models."""
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

        # Загружаем все модели при инициализации
        self._load_all_models()

        # Fallback модель (если специализированная недоступна)
        self.fallback_models = {
            'BUY': None,
            'SELL': None
        }
        self._load_fallback_models()

    def _load_all_models(self):
        """Load all trained market-specific models."""
        model_names = [
            'BULL_BUY', 'BULL_SELL',
            'NEUTRAL_BUY', 'NEUTRAL_SELL',
            'BEAR_BUY', 'BEAR_SELL'
        ]

        models_loaded = 0

        for model_name in model_names:
            model_path = f'models/smart_ml/{model_name.lower()}_model.pkl'

            if os.path.exists(model_path):
                try:
                    model_data = joblib.load(model_path)
                    self.models[model_name] = model_data['model']
                    self.scalers[model_name] = model_data['scaler']
                    self.thresholds[model_name] = model_data['threshold']
                    self.feature_columns[model_name] = model_data['feature_columns']
                    self.model_configs[model_name] = model_data.get('config', {})

                    logger.info(f"✅ Loaded {model_name} model (v{model_data.get('version', 'unknown')})")
                    models_loaded += 1
                except Exception as e:
                    logger.error(f"❌ Failed to load {model_name} model: {e}")
            else:
                logger.warning(f"⚠️ Model file not found: {model_path}")

        logger.info(f"📊 Loaded {models_loaded}/{len(model_names)} market-specific models")

        if models_loaded == 0:
            raise Exception("No models loaded! Please run smart_ml_training.py first.")

    def _load_fallback_models(self):
        """Load fallback models for graceful degradation."""
        # Пытаемся загрузить общие модели из старой системы
        for signal_type in ['BUY', 'SELL']:
            fallback_path = f'models/{signal_type.lower()}_adaptive_model.pkl'

            if os.path.exists(fallback_path):
                try:
                    model_data = joblib.load(fallback_path)
                    self.fallback_models[signal_type] = {
                        'model': model_data['model'],
                        'scaler': model_data['scaler'],
                        'threshold': model_data['threshold'],
                        'feature_columns': model_data['feature_columns']
                    }
                    logger.info(f"📦 Loaded fallback {signal_type} model")
                except Exception as e:
                    logger.warning(f"Could not load fallback {signal_type} model: {e}")

    def get_current_market_regime(self) -> str:
        """Get current market regime from database."""
        query = """
        SELECT regime
        FROM fas.market_regime
        WHERE timeframe = '4h'::fas.timeframe_enum
        ORDER BY timestamp DESC
        LIMIT 1
        """

        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query)
                    result = cur.fetchone()

                    if result:
                        regime = result['regime']
                        logger.info(f"📈 Current market regime: {regime}")
                        return regime
                    else:
                        logger.warning("No market regime found, defaulting to NEUTRAL")
                        return 'NEUTRAL'
        except Exception as e:
            logger.error(f"Failed to get market regime: {e}")
            return 'NEUTRAL'

    def get_active_signals(self) -> pd.DataFrame:
        """Fetch active signals from scoring_history."""
        query = """
        WITH active_signals AS (
            SELECT
                sh.id,
                sh.timestamp,
                sh.trading_pair_id,
                sh.pair_symbol,
                sh.indicator_score,
                sh.pattern_score,
                sh.combination_score,
                sh.total_score,
                sh.patterns_details,
                sh.combinations_details,
                sh.created_at,
                CASE
                    WHEN sh.total_score > 0 THEN 'BUY'
                    ELSE 'SELL'
                END AS signal_type,
                CASE
                    WHEN abs(sh.total_score) >= 100 THEN 'VERY_STRONG'
                    WHEN abs(sh.total_score) >= 50 THEN 'STRONG'
                    WHEN abs(sh.total_score) >= 20 THEN 'MODERATE'
                    ELSE 'WEAK'
                END AS signal_strength,
                public.is_meme_coin(sh.trading_pair_id) AS is_meme,
                ind.*,
                poc.poc_24h,
                poc.poc_7d,
                poc.poc_30d,
                poc.volume_24h AS poc_volume_24h,
                poc.volume_7d AS poc_volume_7d
            FROM (SELECT * FROM fas.scoring_history WHERE is_active = true) sh
            LEFT JOIN LATERAL (
                SELECT *
                FROM fas.indicators i
                WHERE i.trading_pair_id = sh.trading_pair_id
                  AND i.timeframe = '15m'::fas.timeframe_enum
                  AND i.timestamp <= sh.timestamp
                ORDER BY i.timestamp DESC
                LIMIT 1
            ) ind ON true
            LEFT JOIN LATERAL (
                SELECT *
                FROM fas.poc_levels p
                WHERE p.trading_pair_id = sh.trading_pair_id
                  AND p.calculated_at = date_trunc('hour', sh.timestamp)
                ORDER BY p.calculated_at DESC
                LIMIT 1
            ) poc ON true
            WHERE NOT public.is_stablecoin_pair(sh.trading_pair_id)
        ),
        patterns_expanded AS (
            SELECT
                id,
                (patterns_details -> 0) ->> 'pattern' AS pattern_1_name,
                ((patterns_details -> 0) ->> 'impact')::numeric AS pattern_1_impact,
                ((patterns_details -> 0) ->> 'confidence')::numeric AS pattern_1_confidence,
                (patterns_details -> 1) ->> 'pattern' AS pattern_2_name,
                ((patterns_details -> 1) ->> 'impact')::numeric AS pattern_2_impact,
                ((patterns_details -> 1) ->> 'confidence')::numeric AS pattern_2_confidence,
                (patterns_details -> 2) ->> 'pattern' AS pattern_3_name,
                ((patterns_details -> 2) ->> 'impact')::numeric AS pattern_3_impact,
                ((patterns_details -> 2) ->> 'confidence')::numeric AS pattern_3_confidence,
                jsonb_array_length(patterns_details) AS pattern_count,
                CASE WHEN patterns_details::text LIKE '%DISTRIBUTION%' THEN 1 ELSE 0 END AS has_distribution,
                CASE WHEN patterns_details::text LIKE '%ACCUMULATION%' THEN 1 ELSE 0 END AS has_accumulation,
                CASE WHEN patterns_details::text LIKE '%VOLUME_ANOMALY%' THEN 1 ELSE 0 END AS has_volume_anomaly,
                CASE WHEN patterns_details::text LIKE '%MOMENTUM_EXHAUSTION%' THEN 1 ELSE 0 END AS has_momentum_exhaustion,
                CASE WHEN patterns_details::text LIKE '%OI_EXPLOSION%' THEN 1 ELSE 0 END AS has_oi_explosion,
                CASE WHEN patterns_details::text LIKE '%SQUEEZE_IGNITION%' THEN 1 ELSE 0 END AS has_squeeze_ignition,
                CASE WHEN patterns_details::text LIKE '%CVD_PRICE_DIVERGENCE%' THEN 1 ELSE 0 END AS has_cvd_divergence
            FROM active_signals
        ),
        combinations_expanded AS (
            SELECT
                id,
                (combinations_details -> 0) ->> 'combination_name' AS combo_1_name,
                ((combinations_details -> 0) ->> 'score')::numeric AS combo_1_score,
                ((combinations_details -> 0) ->> 'confidence')::numeric AS combo_1_confidence,
                (combinations_details -> 1) ->> 'combination_name' AS combo_2_name,
                ((combinations_details -> 1) ->> 'score')::numeric AS combo_2_score,
                ((combinations_details -> 1) ->> 'confidence')::numeric AS combo_2_confidence,
                jsonb_array_length(combinations_details) AS combo_count,
                CASE WHEN combinations_details::text LIKE '%VOLUME_DISTRIBUTION%' THEN 1 ELSE 0 END AS has_volume_distribution,
                CASE WHEN combinations_details::text LIKE '%VOLUME_ACCUMULATION%' THEN 1 ELSE 0 END AS has_volume_accumulation,
                CASE WHEN combinations_details::text LIKE '%INSTITUTIONAL_SURGE%' THEN 1 ELSE 0 END AS has_institutional_surge,
                CASE WHEN combinations_details::text LIKE '%SQUEEZE_MOMENTUM%' THEN 1 ELSE 0 END AS has_squeeze_momentum,
                CASE WHEN combinations_details::text LIKE '%SMART_ACCUMULATION%' THEN 1 ELSE 0 END AS has_smart_accumulation
            FROM active_signals
        )
        SELECT
            so.*,
            CASE so.signal_strength
                WHEN 'VERY_STRONG' THEN 4
                WHEN 'STRONG' THEN 3
                WHEN 'MODERATE' THEN 2
                WHEN 'WEAK' THEN 1
                ELSE 0
            END AS strength_numeric,
            CASE WHEN so.poc_24h > 0 THEN (so.close_price - so.poc_24h) / so.poc_24h * 100 ELSE NULL END AS price_to_poc_24h_pct,
            CASE WHEN so.poc_7d > 0 THEN (so.close_price - so.poc_7d) / so.poc_7d * 100 ELSE NULL END AS price_to_poc_7d_pct,
            CASE WHEN so.poc_30d > 0 THEN (so.close_price - so.poc_30d) / so.poc_30d * 100 ELSE NULL END AS price_to_poc_30d_pct,
            CASE
                WHEN so.rsi > 70 THEN 1
                WHEN so.rsi < 30 THEN -1
                ELSE 0
            END AS rsi_zone,
            so.atr / NULLIF(so.close_price, 0) * 100 AS atr_pct,
            pe.pattern_1_name, pe.pattern_1_impact, pe.pattern_1_confidence,
            pe.pattern_2_name, pe.pattern_2_impact, pe.pattern_2_confidence,
            pe.pattern_3_name, pe.pattern_3_impact, pe.pattern_3_confidence,
            pe.pattern_count, pe.has_distribution, pe.has_accumulation,
            pe.has_volume_anomaly, pe.has_momentum_exhaustion,
            pe.has_oi_explosion, pe.has_squeeze_ignition, pe.has_cvd_divergence,
            ce.combo_1_name, ce.combo_1_score, ce.combo_1_confidence,
            ce.combo_2_name, ce.combo_2_score, ce.combo_2_confidence,
            ce.combo_count, ce.has_volume_distribution, ce.has_volume_accumulation,
            ce.has_institutional_surge, ce.has_squeeze_momentum, ce.has_smart_accumulation
        FROM active_signals so
        LEFT JOIN patterns_expanded pe ON so.id = pe.id
        LEFT JOIN combinations_expanded ce ON so.id = ce.id
        """

        try:
            with psycopg2.connect(**self.conn_params) as conn:
                df = pd.read_sql(query, conn)

            logger.info(f"📥 Fetched {len(df)} active signals")
            if len(df) > 0:
                signal_types = df['signal_type'].value_counts()
                logger.info(f"   BUY: {signal_types.get('BUY', 0)}, SELL: {signal_types.get('SELL', 0)}")

            return df
        except Exception as e:
            logger.error(f"Failed to fetch active signals: {e}")
            return pd.DataFrame()

    def prepare_features(self, df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Prepare features for specific model."""
        df_proc = df.copy()

        # Удаляем ненужные колонки
        remove_cols = ['id', 'trading_pair_id', 'timestamp', 'pair_symbol',
                       'signal_type', 'signal_strength', 'patterns_details',
                       'combinations_details', 'created_at']

        for col in remove_cols:
            if col in df_proc.columns:
                df_proc = df_proc.drop(columns=[col])

        # Обработка POC volumes
        for col in ['poc_volume_7d', 'poc_volume_24h']:
            if col in df_proc.columns:
                q99 = df_proc[col].quantile(0.99) if len(df_proc) > 100 else df_proc[col].max()
                df_proc[col] = df_proc[col].clip(upper=q99)
                df_proc[f'{col}_log'] = np.log1p(df_proc[col])

        # Временные признаки
        df_proc['hour'] = datetime.now().hour
        df_proc['day_of_week'] = datetime.now().weekday()
        df_proc['is_weekend'] = 1 if datetime.now().weekday() in [5, 6] else 0

        # Model-specific features (аналогично training)
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
            df_proc['imbalance_smoothed'] = df_proc.get('normalized_imbalance', 0) * df_proc.get('smoothed_imbalance',
                                                                                                 0)
            df_proc['volatility_adjusted_atr'] = df_proc.get('atr_pct', 0) / (df_proc.get('volume_zscore', 0).abs() + 1)

        elif 'BEAR' in model_name:
            df_proc['oversold_strength'] = (30 - df_proc.get('rsi', 50).clip(upper=30)) / 30
            df_proc['fear_index'] = df_proc.get('funding_rate_avg', 0) * df_proc.get('oi_delta_pct', 0)

            if 'SELL' in model_name:
                df_proc['bearish_continuation'] = df_proc.get('has_momentum_exhaustion', 0) * df_proc.get('cvd_delta',
                                                                                                          0).clip(
                    upper=0).abs()

        # Добавляем weighted features
        config = self.model_configs.get(model_name, {})
        for feat in config.get('focus_features', []):
            if feat in df_proc.columns:
                df_proc[f'{feat}_weighted'] = df_proc[feat] * 1.5

        # Pattern confidence features
        pattern_cols = [col for col in df_proc.columns if 'pattern_' in col and 'confidence' in col]
        if pattern_cols:
            df_proc['max_pattern_confidence'] = df_proc[pattern_cols].max(axis=1)
            df_proc['avg_pattern_confidence'] = df_proc[pattern_cols].mean(axis=1)

        # Combo features
        combo_cols = [col for col in df_proc.columns if 'combo_' in col and 'score' in col]
        if combo_cols:
            df_proc['total_combo_score'] = df_proc[combo_cols].sum(axis=1)

        # Категориальные переменные
        categorical_cols = df_proc.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_proc[col] = pd.Categorical(df_proc[col].fillna('unknown')).codes

        # Выбираем только нужные features для модели
        feature_cols = self.feature_columns[model_name]
        missing_features = set(feature_cols) - set(df_proc.columns)

        if missing_features:
            logger.warning(f"Missing features for {model_name}: {missing_features}")
            for feat in missing_features:
                df_proc[feat] = 0

        return df_proc[feature_cols].fillna(0)

    def predict_with_model(self, df: pd.DataFrame, model_name: str) -> List[Dict]:
        """Make predictions using specific model."""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not loaded!")
            return []

        # Prepare features
        X = self.prepare_features(df, model_name)

        # Scale features
        X_scaled = pd.DataFrame(
            self.scalers[model_name].transform(X),
            columns=X.columns,
            index=X.index
        )

        # Make predictions
        model = self.models[model_name]
        threshold = self.thresholds[model_name]

        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(bool)

        # Calculate confidence levels
        confidence_levels = []
        for prob in y_pred_proba:
            if prob > threshold + 0.2:
                confidence_levels.append('HIGH')
            elif prob > threshold + 0.1:
                confidence_levels.append('MEDIUM')
            else:
                confidence_levels.append('LOW')

        # Create prediction records
        predictions = []
        processed_ids = []
        for idx, signal in enumerate(df.itertuples(index=False)):
            processed_ids.append(int(signal.id))
            if y_pred[idx]:  # Only keep positive predictions
                features_hash = hashlib.md5(
                    json.dumps(X_scaled.iloc[idx].to_dict(), sort_keys=True).encode()
                ).hexdigest()[:16]

                predictions.append({
                    'signal_id': int(signal.id),
                    'signal_timestamp': signal.timestamp,
                    'trading_pair_id': int(signal.trading_pair_id),
                    'pair_symbol': signal.pair_symbol,
                    'signal_type': signal.signal_type,
                    'total_score': float(signal.total_score),
                    'model_name': model_name,
                    'prediction_proba': float(y_pred_proba[idx]),
                    'threshold': float(threshold),
                    'prediction': bool(y_pred[idx]),
                    'confidence_level': confidence_levels[idx],
                    'features_hash': features_hash
                })

        return predictions, processed_ids

    def make_predictions(self, df: pd.DataFrame, market_regime: str) -> List[Dict]:
        """Make predictions for all signals using appropriate models."""
        all_predictions = []
        all_processed_ids = []

        # Group signals by type
        for signal_type in ['BUY', 'SELL']:
            type_signals = df[df['signal_type'] == signal_type]

            if len(type_signals) == 0:
                continue

            # Select appropriate model
            model_name = f'{market_regime}_{signal_type}'

            logger.info(f"🔮 Processing {len(type_signals)} {signal_type} signals with {model_name} model")

            # Try specialized model first
            if model_name in self.models:
                predictions, processed_ids = self.predict_with_model(type_signals, model_name)
                all_processed_ids.extend(processed_ids)

                if predictions:
                    all_predictions.extend(predictions)
                    logger.info(f"   ✅ {len(predictions)} positive predictions from {model_name}")
                else:
                    logger.info(f"   ⭕ No signals passed {model_name} threshold")

            # Fallback to general model if specialized fails
            elif self.fallback_models[signal_type]:
                logger.warning(f"   ⚠️ Using fallback model for {signal_type}")
                predictions, processed_ids = self._predict_with_fallback(type_signals, signal_type)
                all_processed_ids.extend(processed_ids)
                all_predictions.extend(predictions)
            else:
                logger.error(f"   ❌ No model available for {model_name}")

        return all_predictions, all_processed_ids

    def _predict_with_fallback(self, df: pd.DataFrame, signal_type: str) -> List[Dict]:
        """Use fallback model for predictions."""
        fallback = self.fallback_models[signal_type]
        if not fallback:
            return [], []

        # Simplified feature preparation for fallback
        X = df[fallback['feature_columns']].fillna(0)
        X_scaled = fallback['scaler'].transform(X)

        y_pred_proba = fallback['model'].predict_proba(X_scaled)[:, 1]
        y_pred = (y_pred_proba >= fallback['threshold']).astype(bool)

        predictions = []
        processed_ids = []
        for idx, signal in enumerate(df.itertuples(index=False)):
            processed_ids.append(int(signal.id))  # Track ALL processed signals
            if y_pred[idx]:
                predictions.append({
                    'signal_id': int(signal.id),
                    'signal_timestamp': signal.timestamp,
                    'trading_pair_id': int(signal.trading_pair_id),
                    'pair_symbol': signal.pair_symbol,
                    'signal_type': signal_type,
                    'total_score': float(signal.total_score),
                    'model_name': f'FALLBACK_{signal_type}',
                    'prediction_proba': float(y_pred_proba[idx]),
                    'threshold': float(fallback['threshold']),
                    'prediction': bool(y_pred[idx]),
                    'confidence_level': 'LOW',  # Always low for fallback
                    'features_hash': ''
                })

        return predictions, processed_ids

    def save_predictions(self, predictions: List[Dict], market_regime: str):
        """Save predictions to database."""
        if not predictions:
            logger.warning("No predictions to save")
            return []

        saved_ids = []

        with psycopg2.connect(**self.conn_params) as conn:
            with conn.cursor() as cur:
                for pred in predictions:
                    try:
                        cur.execute("""
                            INSERT INTO smart_ml.predictions (
                                signal_id, model_name, market_regime, signal_type,
                                prediction_proba, prediction, confidence_level, features_hash
                            ) VALUES (
                                %(signal_id)s, %(model_name)s, %(market_regime)s, %(signal_type)s,
                                %(prediction_proba)s, %(prediction)s, %(confidence_level)s, %(features_hash)s
                            )
                            ON CONFLICT (signal_id) DO UPDATE SET
                                prediction_proba = EXCLUDED.prediction_proba,
                                prediction = EXCLUDED.prediction,
                                created_at = NOW()
                            RETURNING id, signal_id
                        """, {
                            **pred,
                            'market_regime': market_regime
                        })

                        result = cur.fetchone()
                        if result:
                            saved_ids.append(result[1])

                    except Exception as e:
                        logger.error(f"Failed to save prediction for signal {pred['signal_id']}: {e}")
                        conn.rollback()
                        continue

                conn.commit()

        logger.info(f"💾 Saved {len(saved_ids)} predictions to database")
        return saved_ids

    def mark_signals_processed(self, signal_ids: List[int]):
        """Mark signals as processed."""
        if not signal_ids:
            return

        with psycopg2.connect(**self.conn_params) as conn:
            with conn.cursor() as cur:
                ids_tuple = tuple(signal_ids)
                cur.execute("""
                    UPDATE fas.scoring_history
                    SET is_active = false
                    WHERE id IN %s
                """, (ids_tuple,))

                updated = cur.rowcount
                conn.commit()

        logger.info(f"✅ Marked {updated} signals as processed")

    def get_prediction_summary(self, predictions: List[Dict], market_regime: str):
        """Generate summary of predictions."""
        if not predictions:
            logger.info("No predictions to summarize")
            return

        df = pd.DataFrame(predictions)

        logger.info("\n" + "=" * 60)
        logger.info(f"PREDICTION SUMMARY - Market Regime: {market_regime}")
        logger.info("=" * 60)

        # Overall stats
        logger.info(f"\nTotal signals to trade: {len(df)}")

        # By model
        model_counts = df['model_name'].value_counts()
        logger.info("\nBy Model:")
        for model, count in model_counts.items():
            avg_prob = df[df['model_name'] == model]['prediction_proba'].mean()
            logger.info(f"  {model}: {count} signals (avg prob: {avg_prob:.3f})")

        # By confidence
        confidence_counts = df['confidence_level'].value_counts()
        logger.info("\nBy Confidence:")
        for conf, count in confidence_counts.items():
            logger.info(f"  {conf}: {count} signals")

        # Top signals
        logger.info("\nTop 10 Signals:")
        top_signals = df.nlargest(10, 'prediction_proba')
        for _, sig in top_signals.iterrows():
            logger.info(f"  {sig['signal_type']} {sig['pair_symbol']}: "
                        f"{sig['prediction_proba']:.3f} ({sig['confidence_level']}) "
                        f"[{sig['model_name']}]")

    def run(self):
        """Main execution flow."""
        logger.info("\n" + "=" * 60)
        logger.info("SMART ML PREDICTOR - MARKET ADAPTIVE PREDICTIONS")
        logger.info(f"Started at: {datetime.now()}")
        logger.info("=" * 60)

        # 1. Get current market regime
        market_regime = self.get_current_market_regime()

        # 2. Get active signals
        active_signals = self.get_active_signals()

        if len(active_signals) == 0:
            logger.info("✅ No active signals to process")
            return []

        # 3. Make predictions
        predictions, all_processed_ids = self.make_predictions(active_signals, market_regime)

        # 4. Show summary
        self.get_prediction_summary(predictions, market_regime)

        # 5. Save predictions
        if predictions:
            saved_signal_ids = self.save_predictions(predictions, market_regime)

            # 6. Mark ALL processed signals (not just saved ones)
        if all_processed_ids:
            self.mark_signals_processed(all_processed_ids)

            # 7. Export for review
            #df_pred = pd.DataFrame(predictions)
            #csv_file = f"smart_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            #df_pred.to_csv(csv_file, index=False)
            #logger.info(f"📄 Predictions exported to {csv_file}")

        logger.info(f"\n✅ Processing complete at {datetime.now()}")

        return predictions


def main():
    """Run the smart predictor."""
    predictor = SmartPredictor()
    predictions = predictor.run()

    # Show actionable signals
    if predictions:
        high_confidence = [p for p in predictions if p['confidence_level'] == 'HIGH']

        if high_confidence:
            logger.info(f"\n🎯 {len(high_confidence)} HIGH CONFIDENCE SIGNALS:")
            for signal in sorted(high_confidence, key=lambda x: x['prediction_proba'], reverse=True)[:5]:
                logger.info(f"  {signal['signal_type']} {signal['pair_symbol']}: "
                            f"prob={signal['prediction_proba']:.3f} "
                            f"[{signal['model_name']}]")


if __name__ == "__main__":
    main()