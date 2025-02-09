@echo off

SET SCRIPT_DIR=%~dp0
SET VENV_PATH=%SCRIPT_DIR%venv
SET SCRIPT_PATH=%SCRIPT_DIR%main.py

python -m venv venv

CALL "%VENV_PATH%\Scripts\activate.bat"

pip install -r "%SCRIPT_DIR%\requirements.txt"

python "%SCRIPT_PATH%"

deactivate

pause
