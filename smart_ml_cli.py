#!/usr/bin/env python3
"""
Smart ML Trading System - Command Line Interface
Cross-platform management interface for the trading system.
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
import argparse


# Add colors for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header():
    """Print system header."""
    print(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}    Smart ML Trading System - Control Panel{Colors.RESET}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print()


def print_status(message):
    """Print status message."""
    print(f"{Colors.GREEN}[✓]{Colors.RESET} {message}")


def print_error(message):
    """Print error message."""
    print(f"{Colors.RED}[✗]{Colors.RESET} {message}")


def print_warning(message):
    """Print warning message."""
    print(f"{Colors.YELLOW}[!]{Colors.RESET} {message}")


def print_info(message):
    """Print info message."""
    print(f"{Colors.CYAN}[i]{Colors.RESET} {message}")


def check_environment():
    """Check if environment is properly set up."""
    issues = []

    # Check for .env file
    if not os.path.exists('.env'):
        issues.append(".env file not found")

    # Check directories
    required_dirs = ['logs', 'models/smart_ml']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print_info(f"Created directory: {dir_path}")

    # Check Python imports
    try:
        import pandas
        import numpy
        import sklearn
        import xgboost
        import psycopg2
        print_status("All core packages available")
    except ImportError as e:
        issues.append(f"Missing package: {e}")

    # Check database connection
    try:
        from dotenv import load_dotenv
        load_dotenv()

        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        conn.close()
        print_status("Database connection successful")
    except Exception as e:
        issues.append(f"Database connection failed: {e}")

    if issues:
        print_warning("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    return True


def run_command(cmd, description=None):
    """Run a Python command."""
    if description:
        print_info(description)

    try:
        result = subprocess.run(
            [sys.executable] + cmd,
            capture_output=True,
            text=True,
            check=True
        )

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print_warning(result.stderr)

        return True

    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False
    except FileNotFoundError:
        print_error(f"File not found: {cmd[0]}")
        return False


def show_menu():
    """Show main menu."""
    print(f"\n{Colors.BOLD}Main Menu:{Colors.RESET}")
    print("1) System Status")
    print("2) Quick Start (Train + Predict)")
    print("3) Train Models")
    print("4) Run Predictions")
    print("5) Validate Models")
    print("6) Monitor Performance")
    print("7) Start Continuous Mode")
    print("8) View Recent Logs")
    print("9) Advanced Options")
    print("0) Exit")
    print()


def show_train_menu():
    """Show training menu."""
    print(f"\n{Colors.BOLD}Training Options:{Colors.RESET}")
    print("1) Train all models")
    print("2) Train BULL models (BUY + SELL)")
    print("3) Train NEUTRAL models (BUY + SELL)")
    print("4) Train BEAR models (BUY + SELL)")
    print("5) Train specific model")
    print("0) Back")
    print()


def show_advanced_menu():
    """Show advanced menu."""
    print(f"\n{Colors.BOLD}Advanced Options:{Colors.RESET}")
    print("1) Full Pipeline (train + validate + predict)")
    print("2) Generate Performance Report")
    print("3) Check Model Drift")
    print("4) Run Backtesting")
    print("5) A/B Test Models")
    print("6) Cleanup Old Data")
    print("7) Export Recent Predictions")
    print("0) Back")
    print()


def system_status():
    """Show system status."""
    print_header()
    run_command(['smart_ml_orchestrator.py', 'status'], "Checking system status...")


def quick_start():
    """Quick start - train essential models and run predictions."""
    print_header()
    print_info("Quick Start - Training essential models and running predictions")

    # Check current market regime
    try:
        from smart_ml_predictor import SmartPredictor
        predictor = SmartPredictor()
        regime = predictor.get_current_market_regime()
        print_status(f"Current market regime: {regime}")

        # Train models for current regime
        models_to_train = [f"{regime}_BUY", f"{regime}_SELL"]
        print_info(f"Training models: {models_to_train}")

        if run_command(['smart_ml_orchestrator.py', 'train', '--models'] + models_to_train):
            print_status("Models trained successfully")

            # Run predictions
            if run_command(['smart_ml_orchestrator.py', 'predict']):
                print_status("Predictions completed")
            else:
                print_error("Prediction failed")
        else:
            print_error("Training failed")

    except Exception as e:
        print_error(f"Quick start failed: {e}")


def train_models():
    """Train models menu."""
    while True:
        show_train_menu()
        choice = input("Select option: ").strip()

        if choice == '1':
            run_command(['smart_ml_orchestrator.py', 'train'], "Training all models...")
        elif choice == '2':
            run_command(['smart_ml_orchestrator.py', 'train', '--models', 'BULL_BUY', 'BULL_SELL'])
        elif choice == '3':
            run_command(['smart_ml_orchestrator.py', 'train', '--models', 'NEUTRAL_BUY', 'NEUTRAL_SELL'])
        elif choice == '4':
            run_command(['smart_ml_orchestrator.py', 'train', '--models', 'BEAR_BUY', 'BEAR_SELL'])
        elif choice == '5':
            print("\nAvailable models:")
            print("  BULL_BUY, BULL_SELL, NEUTRAL_BUY, NEUTRAL_SELL, BEAR_BUY, BEAR_SELL")
            model = input("Enter model name: ").strip().upper()
            if model in ['BULL_BUY', 'BULL_SELL', 'NEUTRAL_BUY', 'NEUTRAL_SELL', 'BEAR_BUY', 'BEAR_SELL']:
                run_command(['smart_ml_orchestrator.py', 'train', '--models', model])
            else:
                print_error("Invalid model name")
        elif choice == '0':
            break
        else:
            print_error("Invalid option")

        input("\nPress Enter to continue...")


def view_logs():
    """View recent log entries."""
    print_header()
    log_dir = Path('logs')

    if not log_dir.exists():
        print_error("Log directory not found")
        return

    # Find today's log file
    today = datetime.now().strftime('%Y%m%d')
    log_file = log_dir / f'smart_ml_{today}.log'

    if log_file.exists():
        print_info(f"Showing last 50 lines from {log_file.name}:")
        print("-" * 60)

        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-50:]:
                # Color code log levels
                if 'ERROR' in line or 'CRITICAL' in line:
                    print(f"{Colors.RED}{line.rstrip()}{Colors.RESET}")
                elif 'WARNING' in line:
                    print(f"{Colors.YELLOW}{line.rstrip()}{Colors.RESET}")
                elif 'SUCCESS' in line or '✅' in line:
                    print(f"{Colors.GREEN}{line.rstrip()}{Colors.RESET}")
                else:
                    print(line.rstrip())
    else:
        print_warning(f"No log file found for today ({today})")

        # Show available logs
        log_files = sorted(log_dir.glob('*.log'))
        if log_files:
            print_info("Available log files:")
            for log_file in log_files[-5:]:
                print(f"  - {log_file.name}")


def advanced_options():
    """Advanced options menu."""
    while True:
        show_advanced_menu()
        choice = input("Select option: ").strip()

        if choice == '1':
            run_command(['smart_ml_orchestrator.py', 'pipeline'], "Running full pipeline...")
        elif choice == '2':
            print_header()
            print_info("Generating performance report...")
            try:
                from smart_ml_monitor import SmartMonitor
                monitor = SmartMonitor()
                report = monitor.generate_performance_report()
                print(report)
            except Exception as e:
                print_error(f"Failed to generate report: {e}")
        elif choice == '3':
            print_header()
            print_info("Checking model drift...")
            try:
                from smart_ml_monitor import SmartMonitor
                monitor = SmartMonitor()
                for model in ['BULL_BUY', 'BULL_SELL', 'NEUTRAL_BUY', 'NEUTRAL_SELL', 'BEAR_BUY', 'BEAR_SELL']:
                    drift = monitor.calculate_model_drift(model, 24)
                    if drift:
                        severity = drift.get('severity', 'UNKNOWN')
                        color = Colors.RED if severity == 'HIGH' else Colors.YELLOW if severity == 'MEDIUM' else Colors.GREEN
                        print(f"{model}: {color}{severity}{Colors.RESET} (KL={drift.get('kl_divergence', 0):.4f})")
            except Exception as e:
                print_error(f"Failed to check drift: {e}")
        elif choice == '4':
            run_command(['smart_ml_validator.py'], "Running backtesting...")
        elif choice == '5':
            print("\nSelect models to compare:")
            print("1) BULL_BUY vs NEUTRAL_BUY")
            print("2) BEAR_SELL vs NEUTRAL_SELL")
            print("3) Custom")
            ab_choice = input("Select: ").strip()

            if ab_choice == '1':
                try:
                    from smart_ml_validator import SmartValidator
                    validator = SmartValidator()
                    validator.ab_test_models('BULL_BUY', 'NEUTRAL_BUY')
                except Exception as e:
                    print_error(f"A/B test failed: {e}")
            elif ab_choice == '2':
                try:
                    from smart_ml_validator import SmartValidator
                    validator = SmartValidator()
                    validator.ab_test_models('BEAR_SELL', 'NEUTRAL_SELL')
                except Exception as e:
                    print_error(f"A/B test failed: {e}")
        elif choice == '6':
            days = input("Days to keep (default 30): ").strip() or '30'
            run_command(['smart_ml_orchestrator.py', 'cleanup', '--days', days])
        elif choice == '7':
            print_info("Exporting recent predictions...")
            try:
                import pandas as pd
                from dotenv import load_dotenv
                import psycopg2

                load_dotenv()
                conn = psycopg2.connect(
                    host=os.getenv('DB_HOST'),
                    port=os.getenv('DB_PORT'),
                    database=os.getenv('DB_NAME'),
                    user=os.getenv('DB_USER'),
                    password=os.getenv('DB_PASSWORD')
                )

                query = """
                SELECT * FROM smart_ml.predictions 
                WHERE created_at >= NOW() - INTERVAL '24 hours'
                ORDER BY created_at DESC
                """

                df = pd.read_sql(query, conn)
                conn.close()

                filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(filename, index=False)
                print_status(f"Exported {len(df)} predictions to {filename}")
            except Exception as e:
                print_error(f"Export failed: {e}")
        elif choice == '0':
            break
        else:
            print_error("Invalid option")

        input("\nPress Enter to continue...")


def main():
    """Main entry point."""
    print_header()

    # Check environment first
    print_info("Checking environment...")
    if not check_environment():
        print_warning("Please fix the issues above before continuing")
        response = input("\nContinue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return

    # Main menu loop
    while True:
        show_menu()
        choice = input("Select option: ").strip()

        if choice == '1':
            system_status()
            input("\nPress Enter to continue...")
        elif choice == '2':
            quick_start()
            input("\nPress Enter to continue...")
        elif choice == '3':
            train_models()
        elif choice == '4':
            run_command(['smart_ml_orchestrator.py', 'predict'], "Running predictions...")
            input("\nPress Enter to continue...")
        elif choice == '5':
            run_command(['smart_ml_orchestrator.py', 'validate'], "Validating models...")
            input("\nPress Enter to continue...")
        elif choice == '6':
            run_command(['smart_ml_orchestrator.py', 'monitor'], "Monitoring performance...")
            input("\nPress Enter to continue...")
        elif choice == '7':
            interval = input("Check interval in minutes (default 15): ").strip() or '15'
            print_warning("Starting continuous mode. Press Ctrl+C to stop.")
            run_command(['smart_ml_orchestrator.py', 'continuous', '--interval', interval])
        elif choice == '8':
            view_logs()
            input("\nPress Enter to continue...")
        elif choice == '9':
            advanced_options()
        elif choice == '0':
            print_status("Exiting Smart ML Trading System")
            break
        else:
            print_error("Invalid option")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n")
        print_warning("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)