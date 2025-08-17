#!/usr/bin/env python3
"""Test installation of Smart ML Trading System."""

import sys
import os
from dotenv import load_dotenv

def test_imports():
    """Test all required imports."""
    print("Testing imports...")

    try:
        import pandas
        print("✓ pandas:", pandas.__version__)
    except ImportError as e:
        print("✗ pandas:", e)

    try:
        import numpy
        print("✓ numpy:", numpy.__version__)
    except ImportError as e:
        print("✗ numpy:", e)

    try:
        import sklearn
        print("✓ scikit-learn:", sklearn.__version__)
    except ImportError as e:
        print("✗ scikit-learn:", e)

    try:
        import xgboost
        print("✓ xgboost:", xgboost.__version__)
    except ImportError as e:
        print("✗ xgboost:", e)

    try:
        import lightgbm
        print("✓ lightgbm:", lightgbm.__version__)
    except ImportError as e:
        print("✗ lightgbm:", e)

    try:
        import psycopg2
        print("✓ psycopg2:", psycopg2.__version__)
    except ImportError as e:
        print("✗ psycopg2:", e)

def test_database():
    """Test database connection."""
    print("\nTesting database connection...")
    load_dotenv()

    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        conn.close()
        print("✓ Database connection successful")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def test_directories():
    """Test directory structure."""
    print("\nChecking directory structure...")

    dirs = ['logs', 'models/smart_ml', 'model_backups']
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path} exists")
        else:
            print(f"✗ {dir_path} missing")

if __name__ == "__main__":
    print("="*50)
    print("Smart ML Trading System - Installation Test")
    print("="*50)

    test_imports()
    test_directories()

    if test_database():
        print("\n✅ System is ready for use!")
    else:
        print("\n⚠️ System installed but database not configured")
        print("Please edit .env file with your database credentials")
