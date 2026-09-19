@echo off
setlocal

title Resonastra Training
cd /d "%~dp0"

set "VOICELAB_PROJECT_ROOT=%CD%"
set "VOICELAB_USER_ROOT=%CD%"

if exist "%CD%\runtime\env\python.exe" (
    set "PYTHON_EXE=%CD%\runtime\env\python.exe"
    set "PATH=%CD%\runtime\env\bin;%CD%\runtime\env;%CD%\runtime\env\Scripts;%CD%\runtime\env\Library\bin;%PATH%"
) else (
    set "PYTHON_EXE=python"
)

set "PYTHONPATH=%CD%;%CD%\src;%PYTHONPATH%"

set "DEFAULT_PORT=7863"

if not defined HF_HUB_CACHE set "HF_HUB_CACHE=%CD%\hf_cache\hub"
if not defined HF_DATASETS_CACHE set "HF_DATASETS_CACHE=%CD%\hf_cache\datasets"
if not defined HF_ASSETS_CACHE set "HF_ASSETS_CACHE=%CD%\hf_cache\assets"

if not exist "%HF_HUB_CACHE%" mkdir "%HF_HUB_CACHE%"
if not exist "%HF_DATASETS_CACHE%" mkdir "%HF_DATASETS_CACHE%"
if not exist "%HF_ASSETS_CACHE%" mkdir "%HF_ASSETS_CACHE%"

echo ============================================================
echo Resonastra Training
echo ============================================================
echo Project root      : %VOICELAB_PROJECT_ROOT%
echo Python            : %PYTHON_EXE%
echo Default URL       : http://127.0.0.1:%DEFAULT_PORT%
echo ============================================================

"%PYTHON_EXE%" scripts\webui_training_user_release.py --host 127.0.0.1 --port %DEFAULT_PORT% --auto-port --inbrowser

pause
