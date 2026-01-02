#!/bin/bash
# =============================================================================
# AI Agent Framework - Demo Runner
# Team Falcons | Intel® Unnati Industrial Training Program 2025
# =============================================================================

set -e  # Exit on error

echo "============================================================"
echo "  AI Agent Framework - Running All Demos"
echo "  Team Falcons | Intel® Unnati Program 2025"
echo "============================================================"
echo ""

# Set PYTHONPATH to project root
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)"
echo "PYTHONPATH set to: $PYTHONPATH"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

echo "Python version: $(python --version)"
echo ""

# Install dependencies if needed
if [ "$1" == "--install" ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

# Run demos
echo "============================================================"
echo "  1. Reference Agents Demo (Research + Data Processing)"
echo "============================================================"
python examples/agents_demo.py
echo ""

echo "============================================================"
echo "  2. Tool System Demo (Schema Validation)"
echo "============================================================"
python examples/tools_demo.py
echo ""

echo "============================================================"
echo "  3. YAML Orchestrator Demo (State Persistence)"
echo "============================================================"
python examples/orchestrator_demo.py
echo ""

echo "============================================================"
echo "  4. Structured Logging Demo"
echo "============================================================"
python examples/logging_demo.py
echo ""

echo "============================================================"
echo "  5. OpenVINO Benchmark Demo"
echo "============================================================"
python examples/openvino_benchmark.py
echo ""

echo "============================================================"
echo "  ✅ All Demos Completed Successfully!"
echo "============================================================"
