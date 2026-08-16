$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectPython = Join-Path $ScriptRoot ".venv\Scripts\python.exe"
$ServiceManager = Join-Path $ScriptRoot "panel_service.py"

if (!(Test-Path -LiteralPath $ProjectPython)) {
    Write-Error "Project Python environment not found: $ProjectPython"
    exit 1
}

if (!(Test-Path -LiteralPath $ServiceManager)) {
    Write-Error "Panel service manager not found: $ServiceManager"
    exit 1
}

& $ProjectPython $ServiceManager stop
exit $LASTEXITCODE
