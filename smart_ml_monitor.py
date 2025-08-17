"""
Smart ML Monitor - Real-time Monitoring and Auto-Retraining
=============================================================
Мониторинг производительности моделей и автоматическое переобучение при деградации.
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import logging
import os
import json
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import entropy
import schedule
import time
import requests
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SmartMonitor:
    """Real-time monitor for market-adaptive ML models."""

    # Пороги для переобучения
    RETRAIN_THRESHOLDS = {
        'BULL_BUY': {'win_rate_drop': 0.15, 'drift_threshold': 0.3},
        'BULL_SELL': {'win_rate_drop': 0.20, 'drift_threshold': 0.25},
        'NEUTRAL_BUY': {'win_rate_drop': 0.15, 'drift_threshold': 0.25},
        'NEUTRAL_SELL': {'win_rate_drop': 0.20, 'drift_threshold': 0.25},
        'BEAR_BUY': {'win_rate_drop': 0.15, 'drift_threshold': 0.25},
        'BEAR_SELL': {'win_rate_drop': 0.15, 'drift_threshold': 0.3}
    }

    def __init__(self, telegram_bot_token: Optional[str] = None, telegram_chat_id: Optional[str] = None):
        """Initialize monitor with database connection and alerting."""
        self.conn_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }

        # Telegram alerts (optional)
        self.telegram_bot_token = telegram_bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = telegram_chat_id or os.getenv('TELEGRAM_CHAT_ID')

        # Model baselines (загружаются при первом запуске)
        self.model_baselines = {}
        self._load_model_baselines()

    def _load_model_baselines(self):
        """Load baseline performance metrics for each model."""
        query = """
        SELECT 
            model_name,
            val_win_rate as baseline_win_rate,
            signals_percentage as baseline_signals_pct,
            threshold as baseline_threshold,
            created_at
        FROM smart_ml.training_history
        WHERE (model_name, created_at) IN (
            SELECT model_name, MAX(created_at)
            FROM smart_ml.training_history
            GROUP BY model_name
        )
        """

        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query)
                    results = cur.fetchall()

                    for row in results:
                        self.model_baselines[row['model_name']] = {
                            'win_rate': float(row['baseline_win_rate']),
                            'signals_pct': float(row['baseline_signals_pct']),
                            'threshold': float(row['baseline_threshold']),
                            'trained_at': row['created_at']
                        }

            logger.info(f"📊 Loaded baselines for {len(self.model_baselines)} models")
        except Exception as e:
            logger.error(f"Failed to load baselines: {e}")

    def send_telegram_alert(self, message: str, priority: str = 'INFO'):
        """Send alert to Telegram."""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return

        # Add emoji based on priority
        emoji_map = {
            'CRITICAL': '🚨',
            'WARNING': '⚠️',
            'INFO': 'ℹ️',
            'SUCCESS': '✅'
        }

        emoji = emoji_map.get(priority, '📢')
        full_message = f"{emoji} *Smart ML Monitor*\n\n{message}"

        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': full_message,
                'parse_mode': 'Markdown'
            }

            response = requests.post(url, json=payload)
            if response.status_code == 200:
                logger.info(f"📱 Telegram alert sent: {priority}")
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")

    def calculate_model_drift(self, model_name: str, hours: int = 24) -> Dict:
        """Calculate drift metrics for a model."""
        logger.info(f"🔍 Calculating drift for {model_name}")

        # Get recent predictions
        query = """
        WITH recent_predictions AS (
            SELECT 
                p.prediction_proba,
                p.prediction,
                p.created_at,
                CASE 
                    WHEN po.outcome_achieved THEN 1 
                    ELSE 0 
                END as actual_outcome
            FROM smart_ml.predictions p
            LEFT JOIN smart_ml.prediction_outcomes po ON p.signal_id = po.signal_id
            WHERE p.model_name = %(model_name)s
                AND p.created_at >= NOW() - INTERVAL '%(hours)s HOUR'
        ),
        baseline_predictions AS (
            SELECT 
                p.prediction_proba,
                p.prediction
            FROM smart_ml.predictions p
            WHERE p.model_name = %(model_name)s
                AND p.created_at >= NOW() - INTERVAL '7 days'
                AND p.created_at < NOW() - INTERVAL '2 days'
        )
        SELECT 
            (SELECT array_agg(prediction_proba) FROM recent_predictions) as recent_probs,
            (SELECT array_agg(prediction_proba) FROM baseline_predictions) as baseline_probs,
            (SELECT COUNT(*) FILTER (WHERE prediction = true AND actual_outcome = 1) 
             FROM recent_predictions WHERE actual_outcome IS NOT NULL) as true_positives,
            (SELECT COUNT(*) FILTER (WHERE prediction = true) 
             FROM recent_predictions WHERE actual_outcome IS NOT NULL) as total_predictions
        """

        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, {'model_name': model_name, 'hours': hours})
                    result = cur.fetchone()

                    if not result or not result['recent_probs'] or not result['baseline_probs']:
                        return {}

                    recent_probs = np.array(result['recent_probs'])
                    baseline_probs = np.array(result['baseline_probs'])

                    # Calculate KL Divergence
                    hist_recent, bins = np.histogram(recent_probs, bins=20, range=(0, 1))
                    hist_baseline, _ = np.histogram(baseline_probs, bins=bins)

                    # Add small epsilon to avoid division by zero
                    hist_recent = hist_recent + 1e-10
                    hist_baseline = hist_baseline + 1e-10

                    # Normalize
                    hist_recent = hist_recent / hist_recent.sum()
                    hist_baseline = hist_baseline / hist_baseline.sum()

                    kl_divergence = entropy(hist_recent, hist_baseline)

                    # Calculate PSI (Population Stability Index)
                    psi = np.sum((hist_recent - hist_baseline) * np.log(hist_recent / hist_baseline))

                    # Recent win rate
                    recent_win_rate = 0
                    if result['total_predictions'] > 0:
                        recent_win_rate = result['true_positives'] / result['total_predictions']

                    drift_metrics = {
                        'kl_divergence': float(kl_divergence),
                        'psi_score': float(psi),
                        'recent_win_rate': float(recent_win_rate),
                        'n_recent': len(recent_probs),
                        'n_baseline': len(baseline_probs),
                        'prob_shift': float(np.mean(recent_probs) - np.mean(baseline_probs))
                    }

                    # Determine drift severity
                    if kl_divergence > 0.5 or abs(psi) > 0.25:
                        drift_metrics['severity'] = 'HIGH'
                    elif kl_divergence > 0.25 or abs(psi) > 0.1:
                        drift_metrics['severity'] = 'MEDIUM'
                    else:
                        drift_metrics['severity'] = 'LOW'

                    logger.info(f"   KL Divergence: {drift_metrics['kl_divergence']:.4f}")
                    logger.info(f"   PSI Score: {drift_metrics['psi_score']:.4f}")
                    logger.info(f"   Drift Severity: {drift_metrics['severity']}")

                    return drift_metrics

        except Exception as e:
            logger.error(f"Failed to calculate drift: {e}")
            return {}

    def check_model_performance(self, model_name: str, hours: int = 24) -> Dict:
        """Check recent performance of a model."""
        query = """
        WITH recent_predictions AS (
            SELECT 
                p.signal_id,
                p.prediction,
                p.prediction_proba,
                p.created_at,
                po.outcome_achieved,
                po.time_to_outcome_hours,
                po.max_favorable_move_pct
            FROM smart_ml.predictions p
            LEFT JOIN smart_ml.prediction_outcomes po ON p.signal_id = po.signal_id
            WHERE p.model_name = %(model_name)s
                AND p.created_at >= NOW() - INTERVAL '%(hours)s HOUR'
                AND p.prediction = true
        )
        SELECT 
            COUNT(*) as total_predictions,
            COUNT(outcome_achieved) as evaluated_predictions,
            COUNT(*) FILTER (WHERE outcome_achieved = true) as successful_predictions,
            AVG(CASE WHEN outcome_achieved = true THEN 1.0 ELSE 0.0 END) as win_rate,
            AVG(prediction_proba) as avg_confidence,
            AVG(time_to_outcome_hours) FILTER (WHERE outcome_achieved = true) as avg_time_to_win,
            AVG(max_favorable_move_pct) as avg_max_move
        FROM recent_predictions
        WHERE outcome_achieved IS NOT NULL
        """

        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, {'model_name': model_name, 'hours': hours})
                    result = cur.fetchone()

                    if not result or result['evaluated_predictions'] == 0:
                        return {
                            'total_predictions': 0,
                            'evaluated_predictions': 0,
                            'status': 'NO_DATA'
                        }

                    performance = {
                        'total_predictions': result['total_predictions'],
                        'evaluated_predictions': result['evaluated_predictions'],
                        'successful_predictions': result['successful_predictions'],
                        'win_rate': float(result['win_rate']) if result['win_rate'] else 0,
                        'avg_confidence': float(result['avg_confidence']) if result['avg_confidence'] else 0,
                        'avg_time_to_win': float(result['avg_time_to_win']) if result['avg_time_to_win'] else 0,
                        'avg_max_move': float(result['avg_max_move']) if result['avg_max_move'] else 0,
                        'status': 'OK'
                    }

                    # Check against baseline
                    if model_name in self.model_baselines:
                        baseline = self.model_baselines[model_name]
                        performance['baseline_win_rate'] = baseline['win_rate']
                        performance['win_rate_change'] = performance['win_rate'] - baseline['win_rate']

                        # Determine if performance degraded
                        threshold = self.RETRAIN_THRESHOLDS[model_name]['win_rate_drop']
                        if performance['win_rate_change'] < -threshold:
                            performance['status'] = 'DEGRADED'
                        elif performance['win_rate_change'] > 0.1:
                            performance['status'] = 'IMPROVED'

                    return performance

        except Exception as e:
            logger.error(f"Failed to check performance: {e}")
            return {'status': 'ERROR'}

    def save_drift_metrics(self, model_name: str, drift_metrics: Dict):
        """Save drift metrics to database."""
        if not drift_metrics:
            return

        query = """
        INSERT INTO smart_ml.model_drift (
            model_name, check_timestamp, kl_divergence, psi_score,
            feature_drift, target_drift, needs_retrain, drift_severity
        ) VALUES (
            %(model_name)s, %(check_timestamp)s, %(kl_divergence)s, %(psi_score)s,
            %(feature_drift)s, %(target_drift)s, %(needs_retrain)s, %(drift_severity)s
        )
        """

        # Determine if retraining needed
        threshold = self.RETRAIN_THRESHOLDS[model_name]['drift_threshold']
        needs_retrain = (
            drift_metrics.get('kl_divergence', 0) > threshold or
            abs(drift_metrics.get('psi_score', 0)) > threshold
        )

        params = {
            'model_name': model_name,
            'check_timestamp': datetime.now(),
            'kl_divergence': drift_metrics.get('kl_divergence', 0),
            'psi_score': drift_metrics.get('psi_score', 0),
            'feature_drift': json.dumps({'prob_shift': drift_metrics.get('prob_shift', 0)}),
            'target_drift': drift_metrics.get('recent_win_rate', 0) -
                           self.model_baselines.get(model_name, {}).get('win_rate', 0.5),
            'needs_retrain': needs_retrain,
            'drift_severity': drift_metrics.get('severity', 'UNKNOWN')
        }

        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                conn.commit()
            logger.info(f"💾 Saved drift metrics for {model_name}")
        except Exception as e:
            logger.error(f"Failed to save drift metrics: {e}")

    def trigger_retraining(self, model_name: str, reason: str):
        """Trigger model retraining."""
        logger.info(f"🔄 Triggering retraining for {model_name}")
        logger.info(f"   Reason: {reason}")

        # Send alert
        alert_message = f"*Model Retraining Triggered*\n\n"
        alert_message += f"Model: `{model_name}`\n"
        alert_message += f"Reason: {reason}\n"
        alert_message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        self.send_telegram_alert(alert_message, 'WARNING')

        # Execute retraining (в production это может быть отдельный процесс)
        try:
            from smart_ml_training import SmartMLTrainer

            trainer = SmartMLTrainer()
            result = trainer.train_model(model_name)

            if result:
                success_message = f"✅ Successfully retrained {model_name}\n"
                success_message += f"New win rate: {result['metrics']['win_rate']:.1%}"
                self.send_telegram_alert(success_message, 'SUCCESS')

                # Update baseline
                self.model_baselines[model_name] = {
                    'win_rate': result['metrics']['win_rate'],
                    'signals_pct': result['metrics']['signals_pct'],
                    'threshold': result['threshold'],
                    'trained_at': datetime.now()
                }
            else:
                self.send_telegram_alert(f"Failed to retrain {model_name}", 'CRITICAL')

        except Exception as e:
            logger.error(f"Failed to trigger retraining: {e}")
            self.send_telegram_alert(f"Retraining failed: {str(e)}", 'CRITICAL')

    def monitor_all_models(self):
        """Monitor all models and trigger retraining if needed."""
        logger.info("\n" + "="*60)
        logger.info("SMART ML MONITOR - CHECKING ALL MODELS")
        logger.info(f"Time: {datetime.now()}")
        logger.info("="*60)

        models_to_retrain = []

        for model_name in self.model_baselines.keys():
            logger.info(f"\n📊 Monitoring {model_name}")

            # Check performance
            performance = self.check_model_performance(model_name, hours=24)

            if performance['status'] == 'NO_DATA':
                logger.info("   No recent predictions to evaluate")
                continue

            logger.info(f"   Recent predictions: {performance['total_predictions']}")
            logger.info(f"   Evaluated: {performance['evaluated_predictions']}")

            if performance['evaluated_predictions'] >= 10:  # Need minimum samples
                logger.info(f"   Win rate: {performance['win_rate']:.1%}")

                if 'win_rate_change' in performance:
                    logger.info(f"   Change from baseline: {performance['win_rate_change']:+.1%}")

                # Check drift
                drift = self.calculate_model_drift(model_name, hours=24)

                if drift:
                    self.save_drift_metrics(model_name, drift)

                    # Determine if retraining needed
                    retrain_reasons = []

                    # Check performance degradation
                    if performance['status'] == 'DEGRADED':
                        retrain_reasons.append(f"Win rate dropped by {abs(performance['win_rate_change']):.1%}")

                    # Check drift
                    if drift['severity'] == 'HIGH':
                        retrain_reasons.append(f"High drift detected (KL={drift['kl_divergence']:.3f})")

                    # Check age
                    baseline = self.model_baselines[model_name]
                    model_age_days = (datetime.now() - baseline['trained_at']).days

                    if model_age_days > 7:  # Retrain weekly
                        retrain_reasons.append(f"Model is {model_age_days} days old")

                    if retrain_reasons:
                        models_to_retrain.append({
                            'model_name': model_name,
                            'reasons': ', '.join(retrain_reasons),
                            'priority': 'HIGH' if performance['status'] == 'DEGRADED' else 'MEDIUM'
                        })

        # Process retraining queue
        if models_to_retrain:
            logger.info(f"\n⚠️ {len(models_to_retrain)} models need retraining")

            # Sort by priority
            models_to_retrain.sort(key=lambda x: x['priority'] == 'HIGH', reverse=True)

            for model_info in models_to_retrain[:2]:  # Retrain max 2 models at a time
                self.trigger_retraining(model_info['model_name'], model_info['reasons'])
        else:
            logger.info("\n✅ All models performing within acceptable parameters")

    def generate_performance_report(self) -> str:
        """Generate daily performance report."""
        logger.info("📊 Generating performance report")

        query = """
        WITH daily_stats AS (
            SELECT 
                p.model_name,
                DATE(p.created_at) as prediction_date,
                COUNT(*) as total_predictions,
                COUNT(po.outcome_achieved) as evaluated,
                COUNT(*) FILTER (WHERE po.outcome_achieved = true) as successful,
                AVG(CASE WHEN po.outcome_achieved = true THEN 1.0 ELSE 0.0 END) as win_rate
            FROM smart_ml.predictions p
            LEFT JOIN smart_ml.prediction_outcomes po ON p.signal_id = po.signal_id
            WHERE p.created_at >= NOW() - INTERVAL '7 days'
                AND p.prediction = true
            GROUP BY p.model_name, DATE(p.created_at)
        ),
        model_summary AS (
            SELECT 
                model_name,
                SUM(total_predictions) as week_total,
                SUM(successful) as week_wins,
                AVG(win_rate) as avg_win_rate
            FROM daily_stats
            WHERE evaluated > 0
            GROUP BY model_name
        )
        SELECT * FROM model_summary
        ORDER BY model_name
        """

        try:
            with psycopg2.connect(**self.conn_params) as conn:
                df = pd.read_sql(query, conn)

            report = "📊 *WEEKLY PERFORMANCE REPORT*\n"
            report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"

            for _, row in df.iterrows():
                model_name = row['model_name']
                baseline = self.model_baselines.get(model_name, {})

                report += f"*{model_name}*\n"
                report += f"  Predictions: {int(row['week_total'])}\n"
                report += f"  Win Rate: {row['avg_win_rate']:.1%}"

                if baseline:
                    diff = row['avg_win_rate'] - baseline.get('win_rate', 0.5)
                    report += f" ({diff:+.1%} vs baseline)\n"
                else:
                    report += "\n"

                report += "\n"

            # Check for any critical issues
            query_issues = """
            SELECT model_name, COUNT(*) as drift_checks
            FROM smart_ml.model_drift
            WHERE created_at >= NOW() - INTERVAL '24 hours'
                AND needs_retrain = true
            GROUP BY model_name
            """

            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query_issues)
                    issues = cur.fetchall()

            if issues:
                report += "⚠️ *MODELS REQUIRING ATTENTION*\n"
                for issue in issues:
                    report += f"  • {issue['model_name']}\n"
            else:
                report += "✅ All models operating normally\n"

            return report

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return "Failed to generate report"

    def run_scheduled_monitoring(self):
        """Run scheduled monitoring tasks."""
        # Schedule tasks
        schedule.every(1).hours.do(self.monitor_all_models)
        schedule.every().day.at("09:00").do(lambda: self.send_telegram_alert(
            self.generate_performance_report(), 'INFO'
        ))
        schedule.every().sunday.at("10:00").do(self.trigger_weekly_retraining)

        logger.info("🚀 Starting scheduled monitoring")
        logger.info("   Monitoring: Every 1 hour")
        logger.info("   Daily report: 09:00")
        logger.info("   Weekly retraining: Sunday 10:00")

        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in scheduled monitoring: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def trigger_weekly_retraining(self):
        """Trigger weekly retraining for all models."""
        logger.info("🔄 Starting weekly retraining")

        from smart_ml_training import SmartMLTrainer

        trainer = SmartMLTrainer()
        results = trainer.train_all_models()

        if results:
            report = "*Weekly Retraining Complete*\n\n"
            for model_name, result in results.items():
                metrics = result['metrics']
                report += f"*{model_name}*\n"
                report += f"  Win Rate: {metrics['win_rate']:.1%}\n"
                report += f"  Signals: {metrics['signals_pct']:.1%}\n\n"

            self.send_telegram_alert(report, 'SUCCESS')

            # Update baselines
            self._load_model_baselines()


def main():
    """Run the smart monitor."""
    monitor = SmartMonitor()

    # Check if running in scheduled mode or one-time check
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--scheduled':
        # Run scheduled monitoring
        monitor.run_scheduled_monitoring()
    else:
        # One-time check
        monitor.monitor_all_models()

        # Generate and print report
        report = monitor.generate_performance_report()
        print("\n" + report)

        # Send report if Telegram configured
        if monitor.telegram_bot_token:
            monitor.send_telegram_alert(report, 'INFO')


if __name__ == "__main__":
    main()