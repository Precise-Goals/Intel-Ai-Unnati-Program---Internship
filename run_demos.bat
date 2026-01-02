@echo off
REM =============================================================================
REM AI Agent Framework - Demo Runner (Windows)
REM Team Falcons | Intel® Unnati Industrial Training Program 2025
REM =============================================================================

echo ============================================================
echo   AI Agent Framework - Running All Demos
echo   Team Falcons ^| Intel Unnati Program 2025
echo ============================================================
echo.

REM Set PYTHONPATH to project root
set PYTHONPATH=%~dp0
echo PYTHONPATH set to: %PYTHONPATH%
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do echo Python version: %%i
echo.

REM Install dependencies if --install flag is passed
if "%1"=="--install" (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo.
)

echo ============================================================
echo   1. Reference Agents Demo ^(Research + Data Processing^)
echo ============================================================
python examples/agents_demo.py
if errorlevel 1 goto :error
echo.

echo ============================================================
echo   2. Tool System Demo ^(Schema Validation^)
echo ============================================================
python examples/tools_demo.py
if errorlevel 1 goto :error
echo.

echo ============================================================
echo   3. YAML Orchestrator Demo ^(State Persistence^)
echo ============================================================
python examples/orchestrator_demo.py
if errorlevel 1 goto :error
echo.

echo ============================================================
echo   4. Structured Logging Demo
echo ============================================================
python examples/logging_demo.py
if errorlevel 1 goto :error
echo.

echo ============================================================
echo   5. OpenVINO Benchmark Demo
echo ============================================================
python examples/openvino_benchmark.py
if errorlevel 1 goto :error
echo.

echo ============================================================
echo   [SUCCESS] All Demos Completed Successfully!
echo ============================================================
exit /b 0

:error
echo ============================================================
echo   [ERROR] Demo failed! Check the output above.
echo ============================================================
exit /b 1
