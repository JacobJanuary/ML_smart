#!/bin/bash

#################################################
# Smart ML Trading System Manager
# Управление Market-Adaptive ML Trading System
#################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_PATH="venv"
LOG_DIR="logs"
MODEL_DIR="models/smart_ml"
PYTHON_CMD="python"

# Create necessary directories
mkdir -p $LOG_DIR
mkdir -p $MODEL_DIR

# Functions
print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}    Smart ML Trading System Manager${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
}

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

check_venv() {
    if [ ! -d "$VENV_PATH" ]; then
        print_error "Virtual environment not found!"
        echo "Creating virtual environment..."
        $PYTHON_CMD -m venv $VENV_PATH
        source $VENV_PATH/bin/activate
        pip install -r requirements.txt
    else
        source $VENV_PATH/bin/activate
        print_status "Virtual environment activated"
    fi
}

check_database() {
    echo -e "${BLUE}Checking database connection...${NC}"
    $PYTHON_CMD -c "
import psycopg2
from dotenv import load_dotenv
import os
load_dotenv()
try:
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    conn.close()
    print('Database connection: OK')
    exit(0)
except Exception as e:
    print(f'Database connection failed: {e}')
    exit(1)
"
    if [ $? -eq 0 ]; then
        print_status "Database connected"
    else
        print_error "Database connection failed"
        exit 1
    fi
}

show_menu() {
    echo -e "${BLUE}Main Menu:${NC}"
    echo "1) System Status"
    echo "2) Train Models"
    echo "3) Validate Models"
    echo "4) Run Predictions"
    echo "5) Monitor Performance"
    echo "6) Run Full Pipeline"
    echo "7) Start Continuous Mode"
    echo "8) View Logs"
    echo "9) Cleanup Old Data"
    echo "10) Advanced Options"
    echo "0) Exit"
    echo ""
}

show_advanced_menu() {
    echo -e "${BLUE}Advanced Options:${NC}"
    echo "1) Train specific model"
    echo "2) Force retrain all models"
    echo "3) Run A/B testing"
    echo "4) Generate performance report"
    echo "5) Check model drift"
    echo "6) Export predictions to CSV"
    echo "7) Backup models"
    echo "8) Restore models from backup"
    echo "0) Back to main menu"
    echo ""
}

system_status() {
    print_header
    echo -e "${BLUE}System Status${NC}"
    echo "------------------------"
    $PYTHON_CMD smart_ml_orchestrator.py status
}

train_models() {
    print_header
    echo -e "${BLUE}Training Models${NC}"
    echo "------------------------"
    echo "1) Train all models"
    echo "2) Train BULL models"
    echo "3) Train NEUTRAL models"
    echo "4) Train BEAR models"
    echo "5) Custom selection"
    read -p "Select option: " train_option

    case $train_option in
        1)
            $PYTHON_CMD smart_ml_orchestrator.py train
            ;;
        2)
            $PYTHON_CMD smart_ml_orchestrator.py train --models BULL_BUY BULL_SELL
            ;;
        3)
            $PYTHON_CMD smart_ml_orchestrator.py train --models NEUTRAL_BUY NEUTRAL_SELL
            ;;
        4)
            $PYTHON_CMD smart_ml_orchestrator.py train --models BEAR_BUY BEAR_SELL
            ;;
        5)
            echo "Enter model names (space-separated):"
            echo "Options: BULL_BUY BULL_SELL NEUTRAL_BUY NEUTRAL_SELL BEAR_BUY BEAR_SELL"
            read -p "Models: " models
            $PYTHON_CMD smart_ml_orchestrator.py train --models $models
            ;;
        *)
            print_error "Invalid option"
            ;;
    esac
}

validate_models() {
    print_header
    echo -e "${BLUE}Validating Models${NC}"
    echo "------------------------"
    $PYTHON_CMD smart_ml_orchestrator.py validate

    read -p "Press Enter to continue..."
}

run_predictions() {
    print_header
    echo -e "${BLUE}Running Predictions${NC}"
    echo "------------------------"
    $PYTHON_CMD smart_ml_orchestrator.py predict

    read -p "Press Enter to continue..."
}

monitor_performance() {
    print_header
    echo -e "${BLUE}Monitoring Performance${NC}"
    echo "------------------------"
    $PYTHON_CMD smart_ml_orchestrator.py monitor

    read -p "Press Enter to continue..."
}

run_pipeline() {
    print_header
    echo -e "${BLUE}Running Full Pipeline${NC}"
    echo "------------------------"
    echo "Options:"
    echo "1) Full pipeline (train + validate + predict)"
    echo "2) Skip training"
    echo "3) Skip validation"
    echo "4) Predictions only"
    read -p "Select option: " pipeline_option

    case $pipeline_option in
        1)
            $PYTHON_CMD smart_ml_orchestrator.py pipeline
            ;;
        2)
            $PYTHON_CMD smart_ml_orchestrator.py pipeline --skip-training
            ;;
        3)
            $PYTHON_CMD smart_ml_orchestrator.py pipeline --skip-validation
            ;;
        4)
            $PYTHON_CMD smart_ml_orchestrator.py pipeline --skip-training --skip-validation
            ;;
        *)
            print_error "Invalid option"
            ;;
    esac

    read -p "Press Enter to continue..."
}

start_continuous() {
    print_header
    echo -e "${BLUE}Starting Continuous Mode${NC}"
    echo "------------------------"
    read -p "Enter check interval in minutes (default 15): " interval
    interval=${interval:-15}

    print_warning "Starting continuous mode (Ctrl+C to stop)..."
    $PYTHON_CMD smart_ml_orchestrator.py continuous --interval $interval
}

view_logs() {
    print_header
    echo -e "${BLUE}Recent Logs${NC}"
    echo "------------------------"

    # Find today's log file
    today=$(date +%Y%m%d)
    log_file="$LOG_DIR/smart_ml_$today.log"

    if [ -f "$log_file" ]; then
        echo "Showing last 50 lines of today's log:"
        tail -n 50 "$log_file"
    else
        print_warning "No log file found for today"
        echo "Available log files:"
        ls -la $LOG_DIR/*.log 2>/dev/null || echo "No log files found"
    fi

    read -p "Press Enter to continue..."
}

cleanup_data() {
    print_header
    echo -e "${BLUE}Cleanup Old Data${NC}"
    echo "------------------------"
    read -p "Days to keep (default 30): " days
    days=${days:-30}

    print_warning "This will delete data older than $days days"
    read -p "Continue? (y/n): " confirm

    if [ "$confirm" = "y" ]; then
        $PYTHON_CMD smart_ml_orchestrator.py cleanup --days $days
    else
        print_status "Cleanup cancelled"
    fi

    read -p "Press Enter to continue..."
}

train_specific_model() {
    echo "Select model to train:"
    echo "1) BULL_BUY"
    echo "2) BULL_SELL"
    echo "3) NEUTRAL_BUY"
    echo "4) NEUTRAL_SELL"
    echo "5) BEAR_BUY"
    echo "6) BEAR_SELL"
    read -p "Select: " model_choice

    case $model_choice in
        1) model="BULL_BUY";;
        2) model="BULL_SELL";;
        3) model="NEUTRAL_BUY";;
        4) model="NEUTRAL_SELL";;
        5) model="BEAR_BUY";;
        6) model="BEAR_SELL";;
        *) print_error "Invalid option"; return;;
    esac

    read -p "Force retrain? (y/n): " force
    if [ "$force" = "y" ]; then
        $PYTHON_CMD smart_ml_orchestrator.py train --models $model --force
    else
        $PYTHON_CMD smart_ml_orchestrator.py train --models $model
    fi
}

generate_report() {
    print_header
    echo -e "${BLUE}Generating Performance Report${NC}"
    echo "------------------------"
    $PYTHON_CMD -c "
from smart_ml_monitor import SmartMonitor
monitor = SmartMonitor()
report = monitor.generate_performance_report()
print(report)
"
    read -p "Press Enter to continue..."
}

backup_models() {
    print_header
    echo -e "${BLUE}Backing up models${NC}"
    echo "------------------------"

    backup_dir="model_backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p $backup_dir

    if cp -r $MODEL_DIR/* $backup_dir/ 2>/dev/null; then
        print_status "Models backed up to $backup_dir"
    else
        print_error "Backup failed"
    fi

    read -p "Press Enter to continue..."
}

restore_models() {
    print_header
    echo -e "${BLUE}Restore Models from Backup${NC}"
    echo "------------------------"

    echo "Available backups:"
    ls -la model_backups/ 2>/dev/null || echo "No backups found"

    read -p "Enter backup directory name: " backup_name

    if [ -d "model_backups/$backup_name" ]; then
        cp -r model_backups/$backup_name/* $MODEL_DIR/
        print_status "Models restored from $backup_name"
    else
        print_error "Backup not found"
    fi

    read -p "Press Enter to continue..."
}

# Main loop
print_header
print_status "Initializing Smart ML Trading System..."

# Check virtual environment
check_venv

# Check database
check_database

while true; do
    echo ""
    show_menu
    read -p "Select option: " choice

    case $choice in
        1)
            system_status
            read -p "Press Enter to continue..."
            ;;
        2)
            train_models
            ;;
        3)
            validate_models
            ;;
        4)
            run_predictions
            ;;
        5)
            monitor_performance
            ;;
        6)
            run_pipeline
            ;;
        7)
            start_continuous
            ;;
        8)
            view_logs
            ;;
        9)
            cleanup_data
            ;;
        10)
            while true; do
                echo ""
                show_advanced_menu
                read -p "Select option: " adv_choice

                case $adv_choice in
                    1)
                        train_specific_model
                        ;;
                    2)
                        $PYTHON_CMD smart_ml_orchestrator.py train --force
                        ;;
                    3)
                        $PYTHON_CMD smart_ml_validator.py
                        ;;
                    4)
                        generate_report
                        ;;
                    5)
                        $PYTHON_CMD -c "
from smart_ml_monitor import SmartMonitor
monitor = SmartMonitor()
for model in ['BULL_BUY', 'BULL_SELL', 'NEUTRAL_BUY', 'NEUTRAL_SELL', 'BEAR_BUY', 'BEAR_SELL']:
    drift = monitor.calculate_model_drift(model, 24)
    if drift:
        print(f'{model}: KL={drift.get(\"kl_divergence\", 0):.4f}, PSI={drift.get(\"psi_score\", 0):.4f}')
"
                        read -p "Press Enter to continue..."
                        ;;
                    6)
                        $PYTHON_CMD -c "
import pandas as pd
predictions = pd.read_sql('SELECT * FROM smart_ml.predictions WHERE created_at >= NOW() - INTERVAL \"24 hours\"',
                          con='postgresql://...')
predictions.to_csv('recent_predictions.csv', index=False)
print(f'Exported {len(predictions)} predictions to recent_predictions.csv')
"
                        ;;
                    7)
                        backup_models
                        ;;
                    8)
                        restore_models
                        ;;
                    0)
                        break
                        ;;
                    *)
                        print_error "Invalid option"
                        ;;
                esac
            done
            ;;
        0)
            print_status "Exiting Smart ML Trading System"
            deactivate 2>/dev/null
            exit 0
            ;;
        *)
            print_error "Invalid option"
            ;;
    esac
done