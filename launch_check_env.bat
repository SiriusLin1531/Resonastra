@echo off
setlocal EnableExtensions

REM Resonastra - environment check launcher
REM This file is intended for Windows users to double-click.

set "LAUNCH_DIR=%~dp0"
set "VOICELAB_USER_ROOT="

if exist "%LAUNCH_DIR%scripts\check_user_env.py" if exist "%LAUNCH_DIR%src\voicelab_user" (
    for %%I in ("%LAUNCH_DIR%.") do set "VOICELAB_USER_ROOT=%%~fI"
)

if not defined VOICELAB_USER_ROOT (
    if exist "%LAUNCH_DIR%..\..\scripts\check_user_env.py" if exist "%LAUNCH_DIR%..\..\src\voicelab_user" (
        for %%I in ("%LAUNCH_DIR%..\..") do set "VOICELAB_USER_ROOT=%%~fI"
    )
)

if not defined VOICELAB_USER_ROOT (
    if exist "%LAUNCH_DIR%..\scripts\check_user_env.py" if exist "%LAUNCH_DIR%..\src\voicelab_user" (
        for %%I in ("%LAUNCH_DIR%..") do set "VOICELAB_USER_ROOT=%%~fI"
    )
)

if not defined VOICELAB_USER_ROOT (
    echo [FAIL] Could not locate Resonastra project root.
    echo Please place launch_check_env.bat in the Resonastra package root.
    echo Current launcher directory:
    echo   %LAUNCH_DIR%
    echo.
    pause
    exit /b 1
)

cd /d "%VOICELAB_USER_ROOT%"

echo ========================================
echo Resonastra Prerequisite Check
echo ========================================
echo.

if not exist "%VOICELAB_USER_ROOT%\scripts\check_vc_redist.ps1" (
    echo [FAIL] Missing scripts\check_vc_redist.ps1
    echo.
    pause
    exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%VOICELAB_USER_ROOT%\scripts\check_vc_redist.ps1" -MinimumVersion "14.44.35211"
set "VC_EXIT=%ERRORLEVEL%"
if not "%VC_EXIT%"=="0" (
    echo.
    echo [FAIL] Microsoft Visual C++ prerequisite is not satisfied.
    echo Install or update the official x64 Visual C++ v14 Redistributable and rerun this check.
    echo Official Microsoft permalink:
    echo   https://aka.ms/vc14/vc_redist.x64.exe
    echo.
    pause
    exit /b %VC_EXIT%
)

set "PYTHONPATH=%VOICELAB_USER_ROOT%;%VOICELAB_USER_ROOT%\src;%PYTHONPATH%"
set "BUNDLED_PYTHON=%VOICELAB_USER_ROOT%\runtime\env\python.exe"

if exist "%BUNDLED_PYTHON%" (
    set "PYTHON_EXE=%BUNDLED_PYTHON%"
    set "PATH=%VOICELAB_USER_ROOT%\runtime\env\bin;%VOICELAB_USER_ROOT%\runtime\env;%VOICELAB_USER_ROOT%\runtime\env\Scripts;%VOICELAB_USER_ROOT%\runtime\env\Library\bin;%PATH%"
    set "STRICT_RUNTIME=--strict-runtime"
) else (
    set "PYTHON_EXE=python"
    set "STRICT_RUNTIME="
    echo [WARN] Bundled runtime was not found:
    echo        %BUNDLED_PYTHON%
    echo [WARN] Falling back to system Python for development only.
    echo.
)

echo.
echo ========================================
echo Resonastra Environment Check
echo ========================================
echo Project root:
 echo   %VOICELAB_USER_ROOT%
echo Python executable:
 echo   %PYTHON_EXE%
echo.

"%PYTHON_EXE%" scripts\check_user_env.py %STRICT_RUNTIME% --project-root "%VOICELAB_USER_ROOT%" --output logs\check_user_env_report.json
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
    echo [OK] Environment check completed successfully.
) else (
    echo [FAIL] Environment check reported problems. Please review the output above.
)

echo.
echo JSON report:
echo   logs\check_user_env_report.json
echo.
pause
exit /b %EXIT_CODE%
