@echo off

REM Set environment variables for conda build
set SPARSEIR_USE_BLAS=1

REM Ensure we're in the right directory
cd /d %SRC_DIR%

REM Build the package
python -m pip install . --no-deps --ignore-installed --verbose
