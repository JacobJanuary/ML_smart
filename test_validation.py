#!/usr/bin/env python3
"""
Test validation directly with verbose output
"""

import os
import sys
import logging
from pathlib import Path

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_validation():
    """Test validation with detailed output."""

    print("=" * 60)
    print("TESTING VALIDATION")
    print("=" * 60)

    # 1. Check model files
    print("\n1. Checking model files...")
    model_dir = Path('models/smart_ml')

    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        return

    model_files = list(model_dir.glob('*.pkl'))
    print(f"Found {len(model_files)} model files:")
    for f in model_files:
        print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    if not model_files:
        print("ERROR: No model files found!")
        print("Please train models first: python smart_ml_orchestrator.py train")
        return

    # 2. Try to import validator
    print("\n2. Importing validator...")
    try:
        from smart_ml_validator import SmartValidator
        print("✓ Validator imported successfully")
    except ImportError as e:
        print(f"ERROR: Failed to import validator: {e}")
        return

    # 3. Create validator instance
    print("\n3. Creating validator instance...")
    try:
        validator = SmartValidator()
        print("✓ Validator created successfully")
    except Exception as e:
        print(f"ERROR: Failed to create validator: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Check loaded models
    print("\n4. Checking loaded models...")
    if validator.models:
        print(f"✓ Loaded {len(validator.models)} models:")
        for name in validator.models.keys():
            print(f"  - {name}")
    else:
        print("ERROR: No models loaded!")
        return

    # 5. Test validation on first model
    print("\n5. Testing validation on first model...")
    first_model = list(validator.models.keys())[0]
    print(f"Testing: {first_model}")

    try:
        # Test loading data
        print("\n  5a. Loading test data...")
        df = validator.load_test_data(first_model, start_days_ago=7, end_days_ago=2)

        if df.empty:
            print(f"  WARNING: No test data for {first_model}")
            print("  Need data from 48+ hours ago with known outcomes")
        else:
            print(f"  ✓ Loaded {len(df)} samples")

            # Test backtest
            print("\n  5b. Running backtest...")
            result = validator.backtest_model(first_model, days=7)

            if result:
                print(f"  ✓ Backtest complete:")
                print(f"    - Total signals: {result.get('total_signals', 0)}")
                print(f"    - Trades taken: {result.get('trades_taken', 0)}")
                print(f"    - Win rate: {result.get('win_rate', 0):.1%}")
            else:
                print("  WARNING: Backtest returned no results")

    except Exception as e:
        print(f"ERROR during validation: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_validation()