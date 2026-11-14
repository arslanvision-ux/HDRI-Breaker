@echo off
setlocal
echo ============================================
echo   Solaris HDRI Maker - Run Verification
echo ============================================
echo.

set "HYTHON="
for /f "delims=" %%D in ('dir /b /ad /o-n "C:\Program Files\Side Effects Software\Houdini*" 2^>nul') do (
    if not defined HYTHON (
        if exist "C:\Program Files\Side Effects Software\%%D\bin\hython.exe" (
            set "HYTHON=C:\Program Files\Side Effects Software\%%D\bin\hython.exe"
        )
    )
)

if not defined HYTHON (
    echo [ERROR] Could not find hython.exe under "C:\Program Files\Side Effects Software\".
    pause
    exit /b 1
)

echo Using: "%HYTHON%"
echo.
"%HYTHON%" "%~dp0tests\test_hdri_maker.py"
echo.
pause
