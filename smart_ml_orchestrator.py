"""
Smart ML Orchestrator - Main System Coordinator
================================================
Координирует работу всех компонентов Market-Adaptive ML Trading System.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
from dotenv import load_dotenv
import subprocess
import time
from pathlib import Path

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/smart_ml_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SmartMLOrchestrator:
    """Main orchestrator for the Smart ML Trading System."""

    def __init__(self):
        """Initialize orchestrator."""
        self.conn_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }

        # Create logs directory if not exists
        os.makedirs('logs', exist_ok=True)
        os.makedirs('models/smart_ml', exist_ok=True)

        self.system_status = self._check_system_status()

    def _check_system_status(self) -> Dict:
        """Check the status of all system components."""
        status = {
            'database': False,
            'models_trained': {},
            'last_prediction': None,
            'last_validation': None,
            'active_signals': 0,
            'market_regime': None
        }

        try:
            # Check database connection
            with psycopg2.connect(**self.conn_params) as conn:
                status['database'] = True

                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Check models
                    cur.execute("""
                        SELECT model_name, MAX(created_at) as last_trained
                        FROM smart_ml.training_history
                        GROUP BY model_name
                    """)

                    for row in cur.fetchall():
                        status['models_trained'][row['model_name']] = row['last_trained']

                    # Check last prediction
                    cur.execute("""
                        SELECT MAX(created_at) as last_prediction
                        FROM smart_ml.predictions
                    """)
                    result = cur.fetchone()
                    if result:
                        status['last_prediction'] = result['last_prediction']

                    # Check active signals
                    cur.execute("""
                        SELECT COUNT(*) as active_count
                        FROM fas.scoring_history
                        WHERE is_active = true
                    """)
                    result = cur.fetchone()
                    if result:
                        status['active_signals'] = result['active_count']

                    # Get current market regime
                    cur.execute("""
                        SELECT regime
                        FROM fas.market_regime
                        WHERE timeframe = '4h'
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """)
                    result = cur.fetchone()
                    if result:
                        status['market_regime'] = result['regime']

        except Exception as e:
            logger.error(f"Failed to check system status: {e}")

        return status

    def print_status(self):
        """Print system status."""
        print("\n" + "="*60)
        print("SMART ML SYSTEM STATUS")
        print("="*60)

        print(f"\n📊 Database: {'✅ Connected' if self.system_status['database'] else '❌ Disconnected'}")
        print(f"📈 Market Regime: {self.system_status['market_regime'] or 'Unknown'}")
        print(f"📥 Active Signals: {self.system_status['active_signals']}")

        if self.system_status['last_prediction']:
            time_since = datetime.now() - self.system_status['last_prediction'].replace(tzinfo=None)
            print(f"🔮 Last Prediction: {time_since.total_seconds()/60:.1f} minutes ago")
        else:
            print("🔮 Last Prediction: Never")

        print(f"\n🤖 Trained Models ({len(self.system_status['models_trained'])}/):")
        for model_name, trained_at in self.system_status['models_trained'].items():
            age_days = (datetime.now() - trained_at.replace(tzinfo=None)).days
            status_emoji = "✅" if age_days < 7 else "⚠️" if age_days < 14 else "❌"
            print(f"  {status_emoji} {model_name}: {age_days} days old")

    def train_models(self, models: Optional[List[str]] = None, force: bool = False):
        """Train specified models or all models."""
        logger.info("="*60)
        logger.info("TRAINING MODELS")
        logger.info("="*60)

        from smart_ml_training import SmartMLTrainer

        trainer = SmartMLTrainer()

        if models:
            # Train specific models
            for model_name in models:
                logger.info(f"Training {model_name}...")

                # Check if recently trained
                if not force and model_name in self.system_status['models_trained']:
                    age_days = (datetime.now() -
                              self.system_status['models_trained'][model_name].replace(tzinfo=None)).days
                    if age_days < 1:
                        logger.info(f"  Skipping {model_name} - trained {age_days} days ago")
                        continue

                try:
                    result = trainer.train_model(model_name)
                    if result:
                        logger.info(f"  ✅ Successfully trained {model_name}")
                    else:
                        logger.error(f"  ❌ Failed to train {model_name}")
                except Exception as e:
                    logger.error(f"  ❌ Error training {model_name}: {e}")
        else:
            # Train all models
            logger.info("Training all models...")
            results = trainer.train_all_models()
            logger.info(f"✅ Trained {len(results)} models")

    def validate_models(self, models: Optional[List[str]] = None):
        """Validate specified models or all models."""
        logger.info("="*60)
        logger.info("VALIDATING MODELS")
        logger.info("="*60)

        # Check if any models exist
        model_dir = Path('models/smart_ml')
        if not model_dir.exists() or not list(model_dir.glob('*.pkl')):
            logger.error("No model files found!")
            logger.info("Please train models first: python smart_ml_orchestrator.py train")
            return

        try:
            from smart_ml_validator import SmartValidator

            validator = SmartValidator()

            # Check if models were loaded
            if not validator.models:
                logger.error("Failed to load models for validation")
                logger.info("Run diagnostic: python check_models.py")
                return

            if models:
                for model_name in models:
                    if model_name not in validator.models:
                        logger.warning(f"Model {model_name} not found, skipping...")
                        continue

                    logger.info(f"Validating {model_name}...")

                    try:
                        # Walk-forward validation
                        validator.walk_forward_validation(model_name)

                        # Backtest
                        validator.backtest_model(model_name)

                        # Test regime transitions
                        validator.test_regime_transitions(model_name)
                    except Exception as e:
                        logger.error(f"Error validating {model_name}: {e}")
            else:
                # Validate all models
                results = validator.validate_all_models()
                if results:
                    logger.info(f"✅ Validated {len(results)} models")
                else:
                    logger.warning("No models were validated")

        except ImportError as e:
            logger.error(f"Failed to import validator: {e}")
        except Exception as e:
            logger.error(f"Validation error: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def run_predictions(self):
        """Run predictions on active signals."""
        logger.info("="*60)
        logger.info("RUNNING PREDICTIONS")
        logger.info("="*60)

        from smart_ml_predictor import SmartPredictor

        predictor = SmartPredictor()
        predictions = predictor.run()

        if predictions:
            logger.info(f"✅ Generated {len(predictions)} predictions")

            # Summary by model
            model_counts = {}
            for pred in predictions:
                model = pred['model_name']
                model_counts[model] = model_counts.get(model, 0) + 1

            logger.info("Predictions by model:")
            for model, count in model_counts.items():
                logger.info(f"  {model}: {count}")
        else:
            logger.info("No predictions generated")

        return predictions

    def monitor_performance(self):
        """Monitor model performance."""
        logger.info("="*60)
        logger.info("MONITORING PERFORMANCE")
        logger.info("="*60)

        from smart_ml_monitor import SmartMonitor

        monitor = SmartMonitor()
        monitor.monitor_all_models()

        # Generate report
        report = monitor.generate_performance_report()
        print("\n" + report)

    def run_pipeline(self, skip_training: bool = False, skip_validation: bool = False):
        """Run the complete ML pipeline."""
        logger.info("="*60)
        logger.info("SMART ML PIPELINE - FULL EXECUTION")
        logger.info(f"Started at: {datetime.now()}")
        logger.info("="*60)

        start_time = time.time()

        # Step 1: Check system status
        logger.info("\n📊 Step 1: Checking system status...")
        self.system_status = self._check_system_status()
        self.print_status()

        # Step 2: Train models if needed
        if not skip_training:
            logger.info("\n🤖 Step 2: Training models...")

            models_to_train = []
            for model_name in ['BULL_BUY', 'BULL_SELL', 'NEUTRAL_BUY',
                              'NEUTRAL_SELL', 'BEAR_BUY', 'BEAR_SELL']:
                if model_name not in self.system_status['models_trained']:
                    models_to_train.append(model_name)
                else:
                    age_days = (datetime.now() -
                              self.system_status['models_trained'][model_name].replace(tzinfo=None)).days
                    if age_days > 7:
                        models_to_train.append(model_name)

            if models_to_train:
                logger.info(f"Training {len(models_to_train)} models: {models_to_train}")
                self.train_models(models_to_train)
            else:
                logger.info("All models are up to date")

        # Step 3: Validate models
        if not skip_validation:
            logger.info("\n✅ Step 3: Validating models...")
            self.validate_models()

        # Step 4: Run predictions
        logger.info("\n🔮 Step 4: Running predictions...")
        predictions = self.run_predictions()

        # Step 5: Monitor performance
        logger.info("\n📈 Step 5: Monitoring performance...")
        self.monitor_performance()

        # Summary
        elapsed_time = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Total time: {elapsed_time/60:.1f} minutes")
        logger.info(f"Predictions generated: {len(predictions) if predictions else 0}")
        logger.info("="*60)

    def run_continuous(self, interval_minutes: int = 15):
        """Run continuous monitoring and predictions."""
        logger.info("🚀 Starting continuous operation")
        logger.info(f"   Prediction interval: {interval_minutes} minutes")
        logger.info("   Press Ctrl+C to stop")

        while True:
            try:
                # Check for active signals
                self.system_status = self._check_system_status()

                if self.system_status['active_signals'] > 0:
                    logger.info(f"\n📥 Found {self.system_status['active_signals']} active signals")

                    # Run predictions
                    predictions = self.run_predictions()

                    if predictions:
                        # Log high confidence signals
                        high_conf = [p for p in predictions if p['confidence_level'] == 'HIGH']
                        if high_conf:
                            logger.info(f"🎯 {len(high_conf)} HIGH CONFIDENCE signals found!")
                            for signal in high_conf[:5]:
                                logger.info(f"  {signal['signal_type']} {signal['pair_symbol']}: "
                                          f"{signal['prediction_proba']:.3f}")
                else:
                    logger.info(f"No active signals at {datetime.now().strftime('%H:%M:%S')}")

                # Sleep until next check
                logger.info(f"Next check in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                logger.info("\n🛑 Stopping continuous operation")
                break
            except Exception as e:
                logger.error(f"Error in continuous operation: {e}")
                time.sleep(60)  # Wait 1 minute on error

    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old prediction and monitoring data."""
        logger.info(f"🧹 Cleaning up data older than {days_to_keep} days")

        queries = [
            f"""
            DELETE FROM smart_ml.predictions 
            WHERE created_at < NOW() - INTERVAL '{days_to_keep} days'
            """,
            f"""
            DELETE FROM smart_ml.model_drift 
            WHERE created_at < NOW() - INTERVAL '{days_to_keep} days'
            """,
            f"""
            DELETE FROM smart_ml.model_performance 
            WHERE created_at < NOW() - INTERVAL '{days_to_keep * 3} days'
            """
        ]

        try:
            with psycopg2.connect(**self.conn_params) as conn:
                with conn.cursor() as cur:
                    for query in queries:
                        cur.execute(query)
                        logger.info(f"  Deleted {cur.rowcount} rows")
                conn.commit()

            logger.info("✅ Cleanup complete")
        except Exception as e:
            logger.error(f"Failed to cleanup: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Smart ML Trading System Orchestrator')

    parser.add_argument('command', choices=[
        'status', 'train', 'validate', 'predict',
        'monitor', 'pipeline', 'continuous', 'cleanup'
    ], help='Command to execute')

    parser.add_argument('--models', nargs='+',
                       help='Specific models to train/validate')
    parser.add_argument('--force', action='store_true',
                       help='Force training even if recently trained')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training in pipeline')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip validation in pipeline')
    parser.add_argument('--interval', type=int, default=15,
                       help='Interval for continuous operation (minutes)')
    parser.add_argument('--days', type=int, default=30,
                       help='Days to keep for cleanup')

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = SmartMLOrchestrator()

    # Execute command
    if args.command == 'status':
        orchestrator.print_status()

    elif args.command == 'train':
        orchestrator.train_models(args.models, args.force)

    elif args.command == 'validate':
        orchestrator.validate_models(args.models)

    elif args.command == 'predict':
        orchestrator.run_predictions()

    elif args.command == 'monitor':
        orchestrator.monitor_performance()

    elif args.command == 'pipeline':
        orchestrator.run_pipeline(args.skip_training, args.skip_validation)

    elif args.command == 'continuous':
        orchestrator.run_continuous(args.interval)

    elif args.command == 'cleanup':
        orchestrator.cleanup_old_data(args.days)


if __name__ == "__main__":
    main()