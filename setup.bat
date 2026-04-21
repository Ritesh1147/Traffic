@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

echo ============================================================
echo  FEDERATED TRAFFIC SIGNAL CONTROL - SETUP SCRIPT
echo  Windows 11 / CPU-only
echo ============================================================
echo.

REM ── 1. Check Python version ─────────────────────────────────
echo [1/6] Checking Python version...
python --version 2>NUL
IF ERRORLEVEL 1 (
    echo ERROR: Python not found.
    echo Please install Python 3.10 from https://www.python.org/downloads/release/python-31011/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

FOR /F "tokens=2 delims= " %%V IN ('python --version 2^>^&1') DO SET PYVER=%%V
FOR /F "tokens=1,2 delims=." %%A IN ("!PYVER!") DO (
    SET PYMAJOR=%%A
    SET PYMINOR=%%B
)

IF NOT "!PYMAJOR!"=="3" (
    echo ERROR: Python 3.x required. Found !PYVER!
    pause
    exit /b 1
)
IF !PYMINOR! LSS 10 (
    echo ERROR: Python 3.10 or higher required. Found !PYVER!
    echo Download Python 3.10: https://www.python.org/downloads/release/python-31011/
    pause
    exit /b 1
)
IF !PYMINOR! GTR 11 (
    echo WARNING: Python !PYVER! detected. Recommended: 3.10 or 3.11
    echo SUMO traci works best with Python 3.10/3.11.
    echo Continuing anyway...
    echo.
)
echo Python !PYVER! OK.
echo.

REM ── 2. Create virtual environment ───────────────────────────
echo [2/6] Creating virtual environment...
IF EXIST venv (
    echo Virtual environment already exists, skipping.
) ELSE (
    python -m venv venv
    IF ERRORLEVEL 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created at .\venv
)
echo.

REM ── 3. Activate venv ────────────────────────────────────────
echo [3/6] Activating virtual environment...
CALL venv\Scripts\activate.bat
IF ERRORLEVEL 1 (
    echo ERROR: Could not activate virtual environment.
    pause
    exit /b 1
)
echo Activated.
echo.

REM ── 4. Upgrade pip ──────────────────────────────────────────
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo pip upgraded.
echo.

REM ── 5. Install dependencies ─────────────────────────────────
echo [5/6] Installing Python packages (this may take 3-5 minutes)...
echo.

echo   Installing PyTorch (CPU version)...
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu --quiet
IF ERRORLEVEL 1 (
    echo ERROR: PyTorch installation failed. Check your internet connection.
    pause
    exit /b 1
)
echo   PyTorch installed.

echo   Installing Flower (federated learning)...
pip install flwr==1.6.0 --quiet
echo   Flower installed.

echo   Installing Gymnasium (RL environment)...
pip install gymnasium==0.29.1 --quiet
echo   Gymnasium installed.

echo   Installing scientific stack...
pip install numpy==1.26.4 pandas==2.1.4 matplotlib==3.8.2 seaborn==0.13.1 --quiet
echo   Scientific stack installed.

echo   Installing utilities...
pip install tqdm==4.66.1 pyyaml==6.0.1 tensorboard==2.15.1 --quiet
echo   Utilities installed.

echo   Installing SUMO (traffic simulator) Python bindings...
pip install traci==1.19.0 sumolib==1.19.0 eclipse-sumo==1.19.0 --quiet
IF ERRORLEVEL 1 (
    echo WARNING: eclipse-sumo pip install failed. This is expected on some systems.
    echo You will need to install SUMO manually - see instructions below.
    echo For now, the simulation will use the built-in mock environment.
)
echo   SUMO bindings done.

echo.
echo [6/6] Verifying installation...
python -c "import torch; print('  PyTorch', torch.__version__, '- CPU only:', not torch.cuda.is_available())"
python -c "import flwr; print('  Flower', flwr.__version__)"
python -c "import gymnasium; print('  Gymnasium', gymnasium.__version__)"
python -c "import numpy; print('  NumPy', numpy.__version__)"
echo.

echo ============================================================
echo  SETUP COMPLETE
echo ============================================================
echo.
echo SUMO MANUAL INSTALL (required for real simulation):
echo   1. Download SUMO 1.19 from: https://sumo.dlr.de/docs/Downloads.php
echo   2. Run the installer (sumo-win64-1.19.0.msi)
echo   3. Installer sets SUMO_HOME environment variable automatically
echo   4. Restart this terminal after installing SUMO
echo.
echo TO START TRAINING:
echo   venv\Scripts\activate
echo   python train_federated.py
echo.
echo TO VIEW TENSORBOARD LOGS:
echo   venv\Scripts\activate
echo   tensorboard --logdir results/logs
echo   Open http://localhost:6006 in your browser
echo.
pause
