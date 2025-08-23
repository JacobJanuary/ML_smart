#!/usr/bin/env python3
"""
Smart ML Outcome Tracker
=========================
Отслеживает результаты предсказаний и обновляет smart_ml.prediction_outcomes.
Использует данные из fas.mv_ml_training_data_simplified для определения outcomes.
"""

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OutcomeTracker:
    """Track and update prediction outcomes."""

    def __init__(self):
        """Initialize outcome tracker."""
        self.conn_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }

    def track_pending_predictions(self):
        """Find predictions that need outcome tracking."""
        query = """
        -- Найти предсказания без outcomes
        WITH pending_predictions AS (
            SELECT 
                p.id as prediction_id,
                p.signal_id,
                p.model_name,
                p.created_at as predicted_at,
                sh.timestamp as signal_timestamp,
                sh.trading_pair_id,
                sh.pair_symbol
            FROM smart_ml.predictions p
            INNER JOIN fas.scoring_history sh ON p.signal_id = sh.id
            LEFT JOIN smart_ml.prediction_outcomes po 
                ON p.signal_id = po.signal_id 
                AND p.model_name = po.model_name
            WHERE po.id IS NULL
                AND p.prediction = true  -- Только положительные предсказания
                AND p.created_at >= NOW() - INTERVAL '7 days'
        )
        SELECT * FROM pending_predictions
        ORDER BY predicted_at DESC
        LIMIT 1000;
        """

        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query)
                    pending = cur.fetchall()

                    logger.info(f"Found {len(pending)} predictions pending outcome tracking")

                    # Создаем записи для отслеживания
                    for pred in pending:
                        self._create_outcome_record(conn, pred)

                    conn.commit()
                    return len(pending)

        except Exception as e:
            logger.error(f"Error tracking pending predictions: {e}")
            return 0

    def _create_outcome_record(self, conn, prediction):
        """Create initial outcome record."""
        query = """
        INSERT INTO smart_ml.prediction_outcomes (
            prediction_id, signal_id, model_name, predicted_at, outcome_type
        ) VALUES (
            %(prediction_id)s, %(signal_id)s, %(model_name)s, %(predicted_at)s, 'PENDING'
        )
        ON CONFLICT (signal_id, model_name) DO NOTHING;
        """

        try:
            with conn.cursor() as cur:
                cur.execute(query, {
                    'prediction_id': prediction['prediction_id'],
                    'signal_id': prediction['signal_id'],
                    'model_name': prediction['model_name'],
                    'predicted_at': prediction['predicted_at']
                })
        except Exception as e:
            logger.error(f"Error creating outcome record: {e}")

    def update_outcomes(self):
        """Update outcomes for pending predictions."""
        query = """
        -- Обновить outcomes используя данные из fas.mv_ml_training_data_simplified
        WITH outcomes_to_update AS (
            SELECT 
                po.id,
                po.signal_id,
                po.model_name,
                po.predicted_at,
                td.target as outcome_achieved,
                td._meta_time_to_outcome_hours as time_to_outcome,
                td._meta_max_favorable_move as max_favorable,
                td._meta_max_adverse_move as max_adverse,
                td._meta_outcome_type as outcome_type_raw,
                CASE 
                    WHEN td.target = true AND td._meta_outcome_type = 'TP_HIT' THEN 'TP_HIT'
                    WHEN td.target = false AND td._meta_outcome_type = 'SL_HIT' THEN 'SL_HIT'
                    WHEN td._meta_outcome_type = 'NO_OUTCOME' THEN 'TIMEOUT'
                    ELSE COALESCE(td._meta_outcome_type, 'UNKNOWN')
                END as outcome_type
            FROM smart_ml.prediction_outcomes po
            INNER JOIN fas.scoring_history sh ON po.signal_id = sh.id
            INNER JOIN fas.mv_ml_training_data_simplified td 
                ON sh.trading_pair_id = td.trading_pair_id
                AND sh.timestamp = td.timestamp
            WHERE po.outcome_type = 'PENDING'
                AND td.target IS NOT NULL
                AND (NOW() - po.predicted_at) > INTERVAL '48 hours'
        )
        UPDATE smart_ml.prediction_outcomes po
        SET 
            outcome_achieved = ou.outcome_achieved,
            time_to_outcome_hours = ou.time_to_outcome,
            max_favorable_move_pct = ou.max_favorable,
            max_adverse_move_pct = ou.max_adverse,
            outcome_type = ou.outcome_type,
            outcome_timestamp = NOW(),
            updated_at = NOW()
        FROM outcomes_to_update ou
        WHERE po.id = ou.id;
        """

        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    updated = cur.rowcount
                    conn.commit()

                    logger.info(f"Updated {updated} prediction outcomes")
                    return updated

        except Exception as e:
            logger.error(f"Error updating outcomes: {e}")
            return 0

    def get_outcome_statistics(self, hours: int = 24):
        """Get statistics on recent outcomes."""
        query = """
        SELECT 
            model_name,
            COUNT(*) as total_predictions,
            COUNT(outcome_achieved) as evaluated,
            COUNT(*) FILTER (WHERE outcome_type = 'PENDING') as pending,
            COUNT(*) FILTER (WHERE outcome_achieved = true) as successful,
            COUNT(*) FILTER (WHERE outcome_achieved = false) as failed,
            AVG(CASE WHEN outcome_achieved = true THEN 1.0 ELSE 0.0 END) as win_rate,
            AVG(time_to_outcome_hours) as avg_time_to_outcome,
            AVG(max_favorable_move_pct) as avg_max_favorable,
            AVG(max_adverse_move_pct) as avg_max_adverse
        FROM smart_ml.prediction_outcomes
        WHERE predicted_at >= NOW() - INTERVAL '%(hours)s HOUR'
        GROUP BY model_name
        ORDER BY model_name;
        """

        try:
            with psycopg2.connect(**self.conn_params) as conn:
                df = pd.read_sql(query, conn, params={'hours': hours})

                if not df.empty:
                    logger.info(f"\nOutcome Statistics (last {hours} hours):")
                    logger.info("=" * 60)

                    for _, row in df.iterrows():
                        logger.info(f"\n{row['model_name']}:")
                        logger.info(f"  Total: {row['total_predictions']}")
                        logger.info(f"  Evaluated: {row['evaluated']}")
                        logger.info(f"  Pending: {row['pending']}")

                        if row['evaluated'] > 0:
                            logger.info(f"  Win Rate: {row['win_rate']:.1%}")
                            logger.info(f"  Avg Time to Outcome: {row['avg_time_to_outcome']:.1f} hours")
                            logger.info(f"  Avg Max Favorable: {row['avg_max_favorable']:.2%}")
                            logger.info(f"  Avg Max Adverse: {row['avg_max_adverse']:.2%}")

                return df

        except Exception as e:
            logger.error(f"Error getting outcome statistics: {e}")
            return pd.DataFrame()

    def cleanup_old_pending(self, days: int = 7):
        """Clean up old pending predictions that will never resolve."""
        query = """
        UPDATE smart_ml.prediction_outcomes
        SET 
            outcome_type = 'EXPIRED',
            updated_at = NOW()
        WHERE outcome_type = 'PENDING'
            AND predicted_at < NOW() - INTERVAL '%(days)s DAY';
        """

        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, {'days': days})
                    cleaned = cur.rowcount
                    conn.commit()

                    if cleaned > 0:
                        logger.info(f"Marked {cleaned} old pending predictions as EXPIRED")

                    return cleaned

        except Exception as e:
            logger.error(f"Error cleaning up old pending: {e}")
            return 0

    def run_full_update(self):
        """Run complete outcome tracking update."""
        logger.info("=" * 60)
        logger.info("SMART ML OUTCOME TRACKER")
        logger.info(f"Started at: {datetime.now()}")
        logger.info("=" * 60)

        # 1. Track new predictions
        new_tracked = self.track_pending_predictions()

        # 2. Update outcomes for pending predictions
        updated = self.update_outcomes()

        # 3. Clean up old pending
        cleaned = self.cleanup_old_pending()

        # 4. Get statistics
        stats = self.get_outcome_statistics(hours=72)

        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"New predictions tracked: {new_tracked}")
        logger.info(f"Outcomes updated: {updated}")
        logger.info(f"Old pending cleaned: {cleaned}")
        logger.info(f"Completed at: {datetime.now()}")

        return {
            'new_tracked': new_tracked,
            'updated': updated,
            'cleaned': cleaned,
            'stats': stats
        }


def main():
    """Main execution."""
    tracker = OutcomeTracker()

    # Run full update
    results = tracker.run_full_update()

    # Check if any models have poor performance
    if not results['stats'].empty:
        for _, row in results['stats'].iterrows():
            if row['evaluated'] >= 10 and row['win_rate'] < 0.4:
                logger.warning(f"⚠️ {row['model_name']} has low win rate: {row['win_rate']:.1%}")


if __name__ == "__main__":
    main()