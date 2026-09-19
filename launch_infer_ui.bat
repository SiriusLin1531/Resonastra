@echo off
setlocal EnableExtensions

title Resonastra Inference
REM Resonastra user inference launcher.
REM This file is intended for Windows users to double-click.
REM It prefers the bundled runtime at runtime\env\python.exe.

cd /d "%~dp0"

set "VOICELAB_USER_ROOT=%~dp0"
set "PYTHONPATH=%VOICELAB_USER_ROOT%;%VOICELAB_USER_ROOT%src;%PYTHONPATH%"

REM Keep all Hugging Face cache activity package-local and force Hub offline mode.
set "HF_HOME=%VOICELAB_USER_ROOT%hf_cache"
set "HF_HUB_CACHE=%VOICELAB_USER_ROOT%hf_cache\hub"
set "HF_HUB_OFFLINE=1"
if not exist "%HF_HUB_CACHE%" mkdir "%HF_HUB_CACHE%"

REM REL-1-9 clean-machine closure:
REM fast-langdetect does NOT use the Hugging Face cache. Without FTLANG_CACHE
REM it falls back to the Windows system TEMP directory and may download the
REM 125 MB FastText lid.176.bin model on first inference. The package already
REM bundles the accepted model under GPT_SoVITS\pretrained_models\fast_langdetect,
REM so bind the library to that directory and fail closed if the asset is absent.
set "FTLANG_CACHE=%VOICELAB_USER_ROOT%GPT_SoVITS\pretrained_models\fast_langdetect"
if not exist "%FTLANG_CACHE%\lid.176.bin" (
    echo [FAIL] Bundled FastText language-identification model is missing:
    echo        %FTLANG_CACHE%\lid.176.bin
    echo [FAIL] Resonastra will not download this model at runtime.
    echo.
    pause
    exit /b 1
)

set "BUNDLED_PYTHON=%VOICELAB_USER_ROOT%runtime\env\python.exe"
set "UI_HOST=127.0.0.1"
set "UI_PORT=7860"
set "UI_URL=http://%UI_HOST%:%UI_PORT%"

if exist "%BUNDLED_PYTHON%" (
    set "PYTHON_EXE=%BUNDLED_PYTHON%"
    set "PATH=%VOICELAB_USER_ROOT%runtime\env\bin;%VOICELAB_USER_ROOT%runtime\env;%VOICELAB_USER_ROOT%runtime\env\Scripts;%VOICELAB_USER_ROOT%runtime\env\Library\bin;%PATH%"
) else (
    set "PYTHON_EXE=python"
    echo [WARN] Bundled runtime was not found:
    echo        %BUNDLED_PYTHON%
    echo [WARN] Falling back to system Python.
    echo.
)

if not exist "scripts\webui_user_infer_release.py" (
    echo [FAIL] Resonastra inference entry script is missing.
    echo Please check the application files and try again.
    echo.
    pause
    exit /b 1
)

echo ========================================
echo Resonastra Inference
echo ========================================
echo Project root:
echo   %VOICELAB_USER_ROOT%
echo Python executable:
echo   %PYTHON_EXE%
echo UI address:
echo   %UI_URL%
echo Hugging Face mode:
echo   OFFLINE (cache: %HF_HUB_CACHE%)
echo FastText language-ID:
echo   OFFLINE (model: %FTLANG_CACHE%\lid.176.bin)
echo.
echo If the browser does not open automatically, copy this URL manually:
echo   %UI_URL%
echo ========================================
echo.

"%PYTHON_EXE%" scripts\webui_user_infer_release.py --host %UI_HOST% --port %UI_PORT% --inbrowser
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
    echo [OK] Resonastra Inference exited normally.
) else (
    echo [FAIL] Resonastra Inference exited with code %EXIT_CODE%.
    echo Please review the output above.
)

echo.
pause
exit /b %EXIT_CODE%
