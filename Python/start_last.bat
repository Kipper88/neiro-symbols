@echo off

SET SCRIPT_DIR=%~dp0
SET VENV_PATH=%SCRIPT_DIR%venv
SET SCRIPT_PATH=%SCRIPT_DIR%main.py

CALL "%VENV_PATH%\Scripts\activate.bat"

python "%SCRIPT_PATH%"

deactivate

pause
