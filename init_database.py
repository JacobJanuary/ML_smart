#!/usr/bin/env python3
"""
Initialize Smart ML Database Schema
Creates all necessary tables and indexes for the Smart ML Trading System.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Color codes for output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_status(message):
    print(f"{Colors.GREEN}[✓]{Colors.RESET} {message}")


def print_error(message):
    print(f"{Colors.RED}[✗]{Colors.RESET} {message}")


def print_warning(message):
    print(f"{Colors.YELLOW}[!]{Colors.RESET} {message}")


def print_info(message):
    print(f"{Colors.BLUE}[i]{Colors.RESET} {message}")


def create_database_schema():
    """Create smart_ml schema and all necessary tables."""

    # Database connection parameters
    conn_params = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }

    # SQL for creating schema and tables
    schema_sql = """
    -- Create schema if not exists
    CREATE SCHEMA IF NOT EXISTS smart_ml;

    -- Set search path
    SET search_path TO smart_ml, public;

    -- Drop existing tables if needed (for clean install)
    DROP TABLE IF EXISTS smart_ml.prediction_outcomes CASCADE;
    DROP TABLE IF EXISTS smart_ml.model_drift CASCADE;
    DROP TABLE IF EXISTS smart_ml.model_performance CASCADE;
    DROP TABLE IF EXISTS smart_ml.predictions CASCADE;
    DROP TABLE IF EXISTS smart_ml.training_history CASCADE;

    -- 1. Training History Table
    CREATE TABLE smart_ml.training_history (
        id SERIAL PRIMARY KEY,
        model_name VARCHAR(50) NOT NULL,
        market_regime VARCHAR(20) NOT NULL,
        signal_type VARCHAR(10) NOT NULL,
        training_window_days INT NOT NULL,
        samples_count INT NOT NULL,
        train_win_rate DECIMAL(5,4),
        val_win_rate DECIMAL(5,4),
        threshold DECIMAL(5,4),
        signals_percentage DECIMAL(5,4),
        expected_profit DECIMAL(7,4),
        feature_importance JSONB,
        model_params JSONB,
        model_version VARCHAR(100),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );

    -- 2. Predictions Table
    CREATE TABLE smart_ml.predictions (
        id SERIAL PRIMARY KEY,
        signal_id BIGINT NOT NULL,
        model_name VARCHAR(50) NOT NULL,
        market_regime VARCHAR(20),
        signal_type VARCHAR(10),
        prediction_proba DECIMAL(5,4) NOT NULL,
        prediction BOOLEAN NOT NULL,
        confidence_level VARCHAR(20),
        features_hash VARCHAR(32),
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(signal_id)
    );

    -- 3. Prediction Outcomes Table (NEW - для отслеживания результатов)
    CREATE TABLE smart_ml.prediction_outcomes (
        id SERIAL PRIMARY KEY,
        prediction_id INT REFERENCES smart_ml.predictions(id),
        signal_id BIGINT NOT NULL,
        model_name VARCHAR(50) NOT NULL,
        predicted_at TIMESTAMP NOT NULL,
        outcome_timestamp TIMESTAMP,
        outcome_achieved BOOLEAN,
        time_to_outcome_hours DECIMAL(7,2),
        max_favorable_move_pct DECIMAL(7,4),
        max_adverse_move_pct DECIMAL(7,4),
        final_pnl_pct DECIMAL(7,4),
        outcome_type VARCHAR(50), -- 'TP_HIT', 'SL_HIT', 'TIMEOUT', 'PENDING'
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(signal_id, model_name)
    );

    -- 4. Model Performance Table
    CREATE TABLE smart_ml.model_performance (
        id SERIAL PRIMARY KEY,
        model_name VARCHAR(50) NOT NULL,
        evaluation_date DATE NOT NULL,
        total_predictions INT DEFAULT 0,
        true_positives INT DEFAULT 0,
        false_positives INT DEFAULT 0,
        true_negatives INT DEFAULT 0,
        false_negatives INT DEFAULT 0,
        win_rate DECIMAL(5,4),
        precision_score DECIMAL(5,4),
        recall_score DECIMAL(5,4),
        f1_score DECIMAL(5,4),
        profit_factor DECIMAL(7,4),
        max_drawdown DECIMAL(5,4),
        sharpe_ratio DECIMAL(7,4),
        regime_stability DECIMAL(5,4),
        created_at TIMESTAMP DEFAULT NOW()
    );

    -- 5. Model Drift Table
    CREATE TABLE smart_ml.model_drift (
        id SERIAL PRIMARY KEY,
        model_name VARCHAR(50) NOT NULL,
        check_timestamp TIMESTAMP NOT NULL,
        kl_divergence DECIMAL(7,6),
        psi_score DECIMAL(7,6),
        feature_drift JSONB,
        target_drift DECIMAL(5,4),
        needs_retrain BOOLEAN DEFAULT FALSE,
        drift_severity VARCHAR(20),
        alert_sent BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT NOW()
    );

    -- Create indexes for better performance
    CREATE INDEX idx_training_history_model_name 
        ON smart_ml.training_history(model_name);
    CREATE INDEX idx_training_history_created_at 
        ON smart_ml.training_history(created_at DESC);

    CREATE INDEX idx_predictions_signal_id 
        ON smart_ml.predictions(signal_id);
    CREATE INDEX idx_predictions_created_at 
        ON smart_ml.predictions(created_at DESC);
    CREATE INDEX idx_predictions_model_name 
        ON smart_ml.predictions(model_name);

    CREATE INDEX idx_prediction_outcomes_signal_id
        ON smart_ml.prediction_outcomes(signal_id);
    CREATE INDEX idx_prediction_outcomes_model_name
        ON smart_ml.prediction_outcomes(model_name);
    CREATE INDEX idx_prediction_outcomes_outcome_type
        ON smart_ml.prediction_outcomes(outcome_type);
    CREATE INDEX idx_prediction_outcomes_pending
        ON smart_ml.prediction_outcomes(outcome_type) 
        WHERE outcome_type = 'PENDING';

    CREATE INDEX idx_performance_model_date 
        ON smart_ml.model_performance(model_name, evaluation_date DESC);

    CREATE INDEX idx_drift_model_timestamp 
        ON smart_ml.model_drift(model_name, check_timestamp DESC);
    CREATE INDEX idx_drift_needs_retrain 
        ON smart_ml.model_drift(needs_retrain) WHERE needs_retrain = TRUE;

    -- Create views for monitoring
    CREATE OR REPLACE VIEW smart_ml.latest_model_status AS
    SELECT 
        th.model_name,
        th.market_regime,
        th.signal_type,
        th.val_win_rate as latest_win_rate,
        th.signals_percentage,
        th.threshold,
        th.created_at as last_trained,
        EXTRACT(DAY FROM NOW() - th.created_at) as days_since_training,
        mp.win_rate as recent_performance,
        md.needs_retrain,
        md.drift_severity
    FROM smart_ml.training_history th
    LEFT JOIN LATERAL (
        SELECT win_rate 
        FROM smart_ml.model_performance 
        WHERE model_name = th.model_name 
        ORDER BY evaluation_date DESC 
        LIMIT 1
    ) mp ON TRUE
    LEFT JOIN LATERAL (
        SELECT needs_retrain, drift_severity 
        FROM smart_ml.model_drift 
        WHERE model_name = th.model_name 
        ORDER BY check_timestamp DESC 
        LIMIT 1
    ) md ON TRUE
    WHERE (th.model_name, th.created_at) IN (
        SELECT model_name, MAX(created_at)
        FROM smart_ml.training_history
        GROUP BY model_name
    );

    -- Create materialized view for quick stats
    CREATE MATERIALIZED VIEW IF NOT EXISTS smart_ml.model_stats AS
    SELECT 
        p.model_name,
        COUNT(DISTINCT DATE(p.created_at)) as days_active,
        COUNT(*) as total_predictions,
        AVG(CASE WHEN p.prediction THEN 1.0 ELSE 0.0 END) as prediction_rate,
        AVG(p.prediction_proba) as avg_confidence,
        COUNT(po.outcome_achieved) as evaluated_predictions,
        AVG(CASE WHEN po.outcome_achieved THEN 1.0 ELSE 0.0 END) as actual_win_rate
    FROM smart_ml.predictions p
    LEFT JOIN smart_ml.prediction_outcomes po ON p.id = po.prediction_id
    WHERE p.created_at >= NOW() - INTERVAL '30 days'
    GROUP BY p.model_name;

    -- Create function to update outcome
    CREATE OR REPLACE FUNCTION smart_ml.update_prediction_outcome(
        p_signal_id BIGINT,
        p_outcome_achieved BOOLEAN,
        p_time_to_outcome DECIMAL,
        p_max_favorable DECIMAL,
        p_max_adverse DECIMAL,
        p_outcome_type VARCHAR
    ) RETURNS void AS $$
    BEGIN
        UPDATE smart_ml.prediction_outcomes
        SET 
            outcome_achieved = p_outcome_achieved,
            time_to_outcome_hours = p_time_to_outcome,
            max_favorable_move_pct = p_max_favorable,
            max_adverse_move_pct = p_max_adverse,
            outcome_type = p_outcome_type,
            outcome_timestamp = NOW(),
            updated_at = NOW()
        WHERE signal_id = p_signal_id 
            AND outcome_type = 'PENDING';
    END;
    $$ LANGUAGE plpgsql;

    -- Create function to refresh stats
    CREATE OR REPLACE FUNCTION smart_ml.refresh_model_stats()
    RETURNS void AS $$
    BEGIN
        REFRESH MATERIALIZED VIEW smart_ml.model_stats;
    END;
    $$ LANGUAGE plpgsql;

    -- Grant permissions (adjust as needed)
    GRANT USAGE ON SCHEMA smart_ml TO PUBLIC;
    GRANT SELECT ON ALL TABLES IN SCHEMA smart_ml TO PUBLIC;
    GRANT ALL ON ALL TABLES IN SCHEMA smart_ml TO CURRENT_USER;
    GRANT ALL ON ALL SEQUENCES IN SCHEMA smart_ml TO CURRENT_USER;
    """

    try:
        # Connect to database
        print_info("Connecting to database...")
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = False
        cur = conn.cursor()

        # Execute schema creation
        print_info("Creating schema smart_ml...")
        cur.execute(schema_sql)

        # Verify tables were created
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'smart_ml'
            ORDER BY table_name;
        """)

        tables = cur.fetchall()

        print_status(f"Created {len(tables)} tables in smart_ml schema:")
        for table in tables:
            print(f"  ✓ smart_ml.{table[0]}")

        # Check if required source tables/views exist
        print_info("\nChecking required source tables and views...")

        # Check for materialized view
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 
                FROM pg_matviews 
                WHERE schemaname = 'fas' 
                AND matviewname = 'ml_training_data_direct'
            );
        """)

        mv_exists = cur.fetchone()[0]
        if mv_exists:
            print_status("fas.ml_training_data_direct (materialized view) exists")
        else:
            print_warning("fas.ml_training_data_direct (materialized view) not found")

        # Check regular tables
        required_tables = [
            ('fas', 'scoring_history'),
            ('fas', 'market_regime'),
            ('fas', 'indicators'),
            ('fas', 'poc_levels')
        ]

        missing_items = []
        for schema, table in required_tables:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 
                    FROM information_schema.tables 
                    WHERE table_schema = %s 
                    AND table_name = %s
                );
            """, (schema, table))

            exists = cur.fetchone()[0]
            if exists:
                print_status(f"{schema}.{table} exists")
            else:
                print_warning(f"{schema}.{table} not found")
                missing_items.append(f"{schema}.{table}")

        if not mv_exists:
            missing_items.append("fas.ml_training_data_direct (materialized view)")

        if missing_items:
            print_warning("\nSome source tables/views are missing:")
            for item in missing_items:
                print(f"  - {item}")
            print_info("The system requires these tables/views to function properly")

        # Commit changes
        conn.commit()
        print_status("\nDatabase schema created successfully!")

        # Create initial statistics
        print_info("\nCreating initial statistics...")
        cur.execute("ANALYZE smart_ml.training_history;")
        cur.execute("ANALYZE smart_ml.predictions;")
        cur.execute("ANALYZE smart_ml.prediction_outcomes;")
        cur.execute("ANALYZE smart_ml.model_performance;")
        cur.execute("ANALYZE smart_ml.model_drift;")

        conn.commit()
        print_status("Statistics updated")

        # Insert initial outcome tracking job info
        print_info("\nSetting up outcome tracking...")
        cur.execute("""
            INSERT INTO smart_ml.prediction_outcomes (
                signal_id, model_name, predicted_at, outcome_type
            )
            SELECT 
                -1, 'SYSTEM', NOW(), 'INITIALIZED'
            WHERE NOT EXISTS (
                SELECT 1 FROM smart_ml.prediction_outcomes 
                WHERE model_name = 'SYSTEM'
            );
        """)
        conn.commit()
        print_status("Outcome tracking initialized")

        return True

    except psycopg2.Error as e:
        print_error(f"Database error: {e}")
        if conn:
            conn.rollback()
        return False

    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if conn:
            conn.rollback()
        return False

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def check_existing_schema():
    """Check if smart_ml schema already exists."""
    conn_params = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }

    try:
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()

        cur.execute("""
            SELECT EXISTS(
                SELECT 1 FROM information_schema.schemata 
                WHERE schema_name = 'smart_ml'
            );
        """)

        exists = cur.fetchone()[0]

        if exists:
            # Check tables
            cur.execute("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = 'smart_ml';
            """)

            table_count = cur.fetchone()[0]

            if table_count > 0:
                print_warning(f"Schema smart_ml already exists with {table_count} tables")
                return True

        return False

    except Exception as e:
        print_error(f"Error checking schema: {e}")
        return False

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def main():
    """Main execution."""
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}    Smart ML Database Initialization{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print()

    # Check if schema exists
    if check_existing_schema():
        response = input("\nSchema already exists. Recreate? (y/n): ").strip().lower()
        if response != 'y':
            print_info("Initialization cancelled")
            return

    # Create schema and tables
    if create_database_schema():
        print()
        print(f"{Colors.GREEN}{Colors.BOLD}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.GREEN}{Colors.BOLD}    Initialization Complete!{Colors.RESET}")
        print(f"{Colors.GREEN}{Colors.BOLD}{'=' * 60}{Colors.RESET}")
        print()
        print_info("Next steps:")
        print("  1. Train models: python smart_ml_orchestrator.py train")
        print("  2. Check status: python smart_ml_orchestrator.py status")
        print("  3. Run predictions: python smart_ml_orchestrator.py predict")
    else:
        print_error("\nInitialization failed. Please check the errors above.")


if __name__ == "__main__":
    main()