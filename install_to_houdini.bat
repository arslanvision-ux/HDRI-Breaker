@echo off
setlocal
echo ============================================
echo   HDRI Breaker (Solaris) - Houdini Installer
echo ============================================
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_to_houdini.ps1"
if errorlevel 1 (
    echo.
    echo [ERROR] Installation failed.
    pause
    exit /b 1
)

echo.
echo Done! Restart Houdini and open the "HDRI Breaker" Shelf or "HDRI Breaker (Solaris)" Python Panel.
pause
