@echo off
setlocal

title Resonastra DataFactory
REM Resonastra DataFactory user launcher.

cd /d "%~dp0"

set "VOICELAB_PROJECT_ROOT=%CD%"
set "VOICELAB_USER_ROOT=%CD%"
set "PYTHONPATH=%VOICELAB_PROJECT_ROOT%;%PYTHONPATH%"

if not defined HF_HUB_CACHE set "HF_HUB_CACHE=%VOICELAB_PROJECT_ROOT%\hf_cache\hub"
if not defined TRANSFORMERS_CACHE set "TRANSFORMERS_CACHE=%VOICELAB_PROJECT_ROOT%\hf_cache\transformers"
if not defined HF_DATASETS_CACHE set "HF_DATASETS_CACHE=%VOICELAB_PROJECT_ROOT%\hf_cache\datasets"
if not defined HF_ASSETS_CACHE set "HF_ASSETS_CACHE=%VOICELAB_PROJECT_ROOT%\hf_cache\assets"
if not defined FTLANG_CACHE set "FTLANG_CACHE=%VOICELAB_PROJECT_ROOT%\GPT_SoVITS\pretrained_models\fast_langdetect"

if not exist "%HF_HUB_CACHE%" mkdir "%HF_HUB_CACHE%"
if not exist "%TRANSFORMERS_CACHE%" mkdir "%TRANSFORMERS_CACHE%"
if not exist "%HF_DATASETS_CACHE%" mkdir "%HF_DATASETS_CACHE%"
if not exist "%HF_ASSETS_CACHE%" mkdir "%HF_ASSETS_CACHE%"
if not exist "%FTLANG_CACHE%" mkdir "%FTLANG_CACHE%"

set "PYTHON_EXE=%VOICELAB_PROJECT_ROOT%\runtime\env\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

if exist "%VOICELAB_PROJECT_ROOT%\runtime\env" (
    set "PATH=%VOICELAB_PROJECT_ROOT%\runtime\env\bin;%VOICELAB_PROJECT_ROOT%\runtime\env;%VOICELAB_PROJECT_ROOT%\runtime\env\Scripts;%VOICELAB_PROJECT_ROOT%\runtime\env\Library\bin;%PATH%"
)

set "UI_HOST=127.0.0.1"
set "UI_PORT=7861"

"%PYTHON_EXE%" scripts\webui_data_factory_user_release.py --host %UI_HOST% --port %UI_PORT% --auto-port --inbrowser

pause
