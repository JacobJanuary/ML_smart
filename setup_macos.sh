#!/bin/bash

#################################################
# Smart ML Trading System - macOS Setup Script
#################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Smart ML System - macOS Installation${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# Check Python version
print_info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
    print_status "Python $PYTHON_VERSION detected"
else
    print_error "Python 3.9+ required (found $PYTHON_VERSION)"
    exit 1
fi

# Check for Homebrew (recommended for macOS)
if ! command -v brew &> /dev/null; then
    print_warning "Homebrew not found. Installing Homebrew is recommended for macOS."
    read -p "Install Homebrew? (y/n): " install_brew
    if [ "$install_brew" = "y" ]; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
fi

# Install PostgreSQL client libraries if needed
if ! brew list postgresql@15 &> /dev/null 2>&1; then
    print_info "Installing PostgreSQL client libraries..."
    brew install postgresql@15
    export PATH="/opt/homebrew/opt/postgresql@15/bin:$PATH"
fi

# Create virtual environment
print_info "Creating virtual environment..."
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Removing old venv..."
    rm -rf venv
fi

python3 -m venv venv

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip, setuptools, and wheel first
print_info "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# For Python 3.12, install setuptools explicitly
if [ "$PYTHON_MINOR" -ge 12 ]; then
    print_warning "Python 3.12+ detected, installing additional compatibility packages..."
    pip install --upgrade setuptools wheel cython
fi

# Install requirements step by step for better error handling
print_info "Installing core dependencies..."

# Install database drivers first
print_info "Installing database drivers..."
pip install psycopg2-binary python-dotenv sqlalchemy

# Install numpy first (required by pandas)
print_info "Installing numpy..."
pip install "numpy>=1.26.0"

# Install pandas
print_info "Installing pandas..."
pip install "pandas>=2.1.0"

# Install scipy
print_info "Installing scipy..."
pip install "scipy>=1.11.3"

# Install scikit-learn
print_info "Installing scikit-learn..."
pip install "scikit-learn>=1.3.0"

# Install ML libraries
print_info "Installing ML libraries..."
pip install "xgboost>=2.0.0"

# LightGBM might need special handling on macOS
print_info "Installing LightGBM..."
if [[ $(uname -m) == 'arm64' ]]; then
    # For Apple Silicon (M1/M2)
    print_warning "Apple Silicon detected, installing LightGBM with special flags..."
    brew install libomp
    pip install lightgbm --config-settings=cmake.define.USE_OPENMP=OFF
else
    pip install "lightgbm>=4.1.0"
fi

# Install remaining packages
print_info "Installing remaining packages..."
pip install joblib schedule requests

# Optional packages
print_info "Installing optional packages..."
pip install matplotlib seaborn pytest black flake8 || print_warning "Some optional packages failed to install"

# Create necessary directories
print_info "Creating project directories..."
mkdir -p logs
mkdir -p models/smart_ml
mkdir -p model_backups

# Create .env file if not exists
if [ ! -f ".env" ]; then
    print_info "Creating .env file..."
    cat > .env << EOF
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database
DB_USER=your_user
DB_PASSWORD=your_password

# Telegram Alerts (optional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
EOF
    print_warning "Please edit .env file with your database credentials"
fi

# Verify installation
print_info "Verifying installation..."
python3 -c "
import pandas
import numpy
import sklearn
import xgboost
import lightgbm
import psycopg2
print('✓ All core packages imported successfully')
" && print_status "Installation verified" || print_error "Some packages failed to import"

# Create test script
print_info "Creating test script..."
cat > test_installation.py << 'EOF'
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
EOF

chmod +x test_installation.py

# Final message
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}        Installation Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
print_info "Next steps:"
echo "  1. Edit .env file with your database credentials"
echo "  2. Run: source venv/bin/activate"
echo "  3. Test: python test_installation.py"
echo "  4. Start: python smart_ml_orchestrator.py status"
echo ""
print_status "Virtual environment is activated in current shell"
print_warning "Remember to activate venv in new terminals: source venv/bin/activate"