#!/bin/bash
# Demo Preparation Script for CKPT2
# Run this script before your demo to ensure everything is ready

echo "=================================="
echo "CLV CKPT2 Demo Preparation"
echo "=================================="
echo ""

# Check Python version
echo "[1/5] Checking Python..."
python --version
echo ""

# Install dependencies
echo "[2/5] Installing dependencies..."
pip install -q pandas numpy scikit-learn matplotlib seaborn openpyxl jupyter
echo "✓ Core dependencies installed"
echo ""

# Install optional dependencies
echo "[3/5] Installing optional dependencies..."
pip install -q xgboost lifetimes 2>/dev/null || echo "⚠ Optional packages may need manual install"
echo ""

# Check data files
echo "[4/5] Checking data files..."
if [ -f "data/online_retail_II_Year_2009-2010.xlsx" ] && [ -f "data/online_retail_II_Year_2010-2011.xlsx" ]; then
    echo "✓ Data files found!"
else
    echo "⚠ Data files missing!"
    echo "Please download from: https://archive.ics.uci.edu/ml/datasets/online+retail+II"
    echo "Place files in data/ directory"
fi
echo ""

# Verify notebook
echo "[5/5] Verifying demo notebook..."
if [ -f "notebooks/CKPT2_Complete_Demo.ipynb" ]; then
    echo "✓ Demo notebook ready!"
else
    echo "⚠ Demo notebook missing!"
fi
echo ""

echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To start demo:"
echo "  cd clv-stacking"
echo "  jupyter notebook notebooks/CKPT2_Complete_Demo.ipynb"
echo ""
