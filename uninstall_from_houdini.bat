@echo off
setlocal
echo ============================================
echo   Solaris HDRI Maker - Houdini Uninstaller
echo ============================================
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
    "$docs = [Environment]::GetFolderPath('MyDocuments');" ^
    "$dirs = Get-ChildItem -Path $docs -Directory -Filter 'houdini*' -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^houdini(20|21|22)\.\d+$' };" ^
    "$removed = $false;" ^
    "foreach ($d in $dirs) { $p = Join-Path $d.FullName 'packages\hdri_maker.json'; if (Test-Path $p) { Remove-Item $p -Force; Write-Host ('Removed: ' + $p) -ForegroundColor Yellow; $removed = $true } };" ^
    "if (-not $removed) { Write-Host 'No HDRI Maker package files found to remove.' -ForegroundColor Cyan }"

echo.
pause
