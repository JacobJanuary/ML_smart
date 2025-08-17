#!/usr/bin/env python3
"""
Check Models - Diagnostic script for Smart ML models
"""

import os
import sys
import joblib
import pandas as pd
import psycopg2
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# Colors for output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_status(msg):
    print(f"{Colors.GREEN}[✓]{Colors.RESET} {msg}")


def print_error(msg):
    print(f"{Colors.RED}[✗]{Colors.RESET} {msg}")


def print_warning(msg):
    print(f"{Colors.YELLOW}[!]{Colors.RESET} {msg}")


def print_info(msg):
    print(f"{Colors.BLUE}[i]{Colors.RESET} {msg}")


def check_model_files():
    """Check if model files exist."""
    print(f"\n{Colors.BOLD}Checking Model Files:{Colors.RESET}")
    print("-" * 40)

    model_dir = Path('models/smart_ml')

    if not model_dir.exists():
        print_error(f"Model directory not found: {model_dir}")
        return []

    model_names = [
        'BULL_BUY', 'BULL_SELL',
        'NEUTRAL_BUY', 'NEUTRAL_SELL',
        'BEAR_BUY', 'BEAR_SELL'
    ]

    found_models = []

    for model_name in model_names:
        model_path = model_dir / f"{model_name.lower()}_model.pkl"

        if model_path.exists():
            size = model_path.stat().st_size / 1024  # KB
            mod_time = datetime.fromtimestamp(model_path.stat().st_mtime)
            age_hours = (datetime.now() - mod_time).total_seconds() / 3600

            print_status(f"{model_name}: {size:.1f}KB, {age_hours:.1f} hours old")
            found_models.append(model_name)

            # Try to load model
            try:
                model_data = joblib.load(model_path)
                keys = model_data.keys() if isinstance(model_data, dict) else []
                print(f"  └─ Keys: {', '.join(keys)}")

                # Check model components
                if 'model' in model_data:
                    model_type = type(model_data['model']).__name__
                    print(f"     └─ Model type: {model_type}")
                if 'feature_columns' in model_data:
                    n_features = len(model_data['feature_columns'])
                    print(f"     └─ Features: {n_features}")
                if 'threshold' in model_data:
                    threshold = model_data['threshold']
                    print(f"     └─ Threshold: {threshold:.3f}")

            except Exception as e:
                print_error(f"  └─ Failed to load: {e}")
        else:
            print_error(f"{model_name}: Not found")

    return found_models


def check_database_records():
    """Check training history in database."""
    print(f"\n{Colors.BOLD}Checking Database Records:{Colors.RESET}")
    print("-" * 40)

    conn_params = {
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }

    try:
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()

        # Check training history
        cur.execute("""
            SELECT 
                model_name,
                val_win_rate,
                signals_percentage,
                samples_count,
                created_at
            FROM smart_ml.training_history
            WHERE (model_name, created_at) IN (
                SELECT model_name, MAX(created_at)
                FROM smart_ml.training_history
                GROUP BY model_name
            )
            ORDER BY model_name
        """)

        records = cur.fetchall()

        if records:
            for record in records:
                model, wr, signals, samples, created = record
                age_hours = (datetime.now() - created.replace(tzinfo=None)).total_seconds() / 3600
                print_status(f"{model}: WR={wr:.1%}, Signals={signals:.1%}, Samples={samples}, Age={age_hours:.1f}h")
        else:
            print_warning("No training records found in database")

        # Check for test data availability
        print(f"\n{Colors.BOLD}Checking Test Data Availability:{Colors.RESET}")
        print("-" * 40)

        cur.execute("""
            SELECT 
                market_regime,
                signal_type,
                COUNT(*) as count,
                MIN(timestamp) as earliest,
                MAX(timestamp) as latest
            FROM fas.ml_training_data_direct
            WHERE timestamp >= NOW() - INTERVAL '7 days'
                AND timestamp < NOW() - INTERVAL '48 hours'
                AND target IS NOT NULL
            GROUP BY market_regime, signal_type
            ORDER BY market_regime, signal_type
        """)

        data_records = cur.fetchall()

        if data_records:
            for record in data_records:
                regime, signal, count, earliest, latest = record
                print_info(
                    f"{regime}_{signal}: {count} samples ({earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')})")
        else:
            print_error("No test data available in the last 7 days")

        cur.close()
        conn.close()

    except Exception as e:
        print_error(f"Database error: {e}")


def test_model_loading(model_name):
    """Test loading a specific model."""
    print(f"\n{Colors.BOLD}Testing Model Loading: {model_name}{Colors.RESET}")
    print("-" * 40)

    model_path = f'models/smart_ml/{model_name.lower()}_model.pkl'

    try:
        print_info(f"Loading {model_path}...")
        model_data = joblib.load(model_path)

        print_status("Model loaded successfully")

        # Test components
        if 'model' in model_data:
            model = model_data['model']
            print_status(f"Model object: {type(model).__name__}")

            # Check if it's an ensemble
            if hasattr(model, 'estimators_'):
                print_info(f"  Ensemble with {len(model.estimators_)} estimators")

        if 'scaler' in model_data:
            scaler = model_data['scaler']
            print_status(f"Scaler: {type(scaler).__name__}")

            if hasattr(scaler, 'mean_'):
                print_info(f"  Features scaled: {len(scaler.mean_)}")

        if 'feature_columns' in model_data:
            features = model_data['feature_columns']
            print_status(f"Features: {len(features)}")
            print_info(f"  First 5: {features[:5]}")

        if 'threshold' in model_data:
            print_status(f"Threshold: {model_data['threshold']:.3f}")

        if 'config' in model_data:
            config = model_data['config']
            print_status(
                f"Config: target_win_rate={config.get('target_win_rate', 'N/A')}, target_signals={config.get('target_signals_pct', 'N/A')}")

        return True

    except FileNotFoundError:
        print_error(f"Model file not found: {model_path}")
        return False
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction(model_name):
    """Test making a prediction with a model."""
    print(f"\n{Colors.BOLD}Testing Prediction: {model_name}{Colors.RESET}")
    print("-" * 40)

    try:
        # Load model
        model_path = f'models/smart_ml/{model_name.lower()}_model.pkl'
        model_data = joblib.load(model_path)

        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['feature_columns']
        threshold = model_data['threshold']

        # Create dummy data
        import numpy as np
        dummy_data = pd.DataFrame(
            np.random.randn(5, len(features)),
            columns=features
        )

        # Scale
        dummy_scaled = scaler.transform(dummy_data)

        # Predict
        proba = model.predict_proba(dummy_scaled)[:, 1]
        pred = (proba >= threshold).astype(int)

        print_status(f"Prediction successful!")
        print_info(f"  Probabilities: {proba}")
        print_info(f"  Predictions: {pred}")
        print_info(f"  Signals: {pred.sum()}/{len(pred)}")

        return True

    except Exception as e:
        print_error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main diagnostic routine."""
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}    Smart ML Models Diagnostic{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 60}{Colors.RESET}")

    # 1. Check model files
    found_models = check_model_files()

    # 2. Check database records
    check_database_records()

    # 3. Test loading first available model
    if found_models:
        test_model = found_models[0]
        if test_model_loading(test_model):
            test_prediction(test_model)
    else:
        print_error("\nNo models found to test!")
        print_info("Please train models first:")
        print("  python smart_ml_orchestrator.py train")

    # Summary
    print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
    print("-" * 40)

    if len(found_models) == 6:
        print_status(f"All 6 models found")
    else:
        print_warning(f"Only {len(found_models)}/6 models found")
        missing = set(['BULL_BUY', 'BULL_SELL', 'NEUTRAL_BUY', 'NEUTRAL_SELL', 'BEAR_BUY', 'BEAR_SELL']) - set(
            found_models)
        if missing:
            print_info(f"Missing: {', '.join(missing)}")

    print("\nIf validation is hanging, possible causes:")
    print("  1. Not enough test data (need data from 48+ hours ago)")
    print("  2. Model files corrupted")
    print("  3. Database connection timeout")
    print("  4. Memory issues with large models")


if __name__ == "__main__":
    main()