# Install HDRI Breaker package into Houdini 20+ / 21+ user preferences (dynamic path)
$repoRoot = ($PSScriptRoot -replace '\\', '/').TrimEnd('/')
$pkgPayload = [ordered]@{
    env = @(
        @{ HDRI_MAKER_ROOT = $repoRoot },
        @{ PYTHONPATH      = '${HDRI_MAKER_ROOT}/python' },
        @{ HOUDINI_PATH    = '${HDRI_MAKER_ROOT}/houdini' }
    )
} | ConvertTo-Json -Depth 4

$repoPkg = Join-Path $PSScriptRoot "houdini\packages\hdri_maker.json"
[System.IO.File]::WriteAllText($repoPkg, $pkgPayload + "`n", (New-Object System.Text.UTF8Encoding($false)))

$docs = [Environment]::GetFolderPath("MyDocuments")
$houdiniDirs = Get-ChildItem -Path $docs -Directory -Filter "houdini*" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "^houdini(20|21|22)\.\d+$" }

if (-not $houdiniDirs) {
    Write-Host "No Houdini 20+ user preference folders found in $docs." -ForegroundColor Yellow
    Write-Host "You can manually copy $repoPkg to <HOUDINI_USER_PREF_DIR>\packages\hdri_maker.json"
    exit 0
}

foreach ($dir in $houdiniDirs) {
    $pkgDir = Join-Path $dir.FullName "packages"
    if (-not (Test-Path $pkgDir)) {
        New-Item -ItemType Directory -Path $pkgDir -Force | Out-Null
    }
    $dest = Join-Path $pkgDir "hdri_maker.json"
    [System.IO.File]::WriteAllText($dest, $pkgPayload + "`n", (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "[OK] Installed HDRI Breaker package -> $dest" -ForegroundColor Green
}
