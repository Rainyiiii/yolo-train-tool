param(
    [switch]$NoStart,
    [switch]$ForceReinstall
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($ForceReinstall) {
    Write-Warning "ForceReinstall 不会自动删除运行环境；如需重建，请先备份并移走 .venv。"
}
& (Join-Path $ScriptRoot "install_runtime.ps1") -InstallRoot $ScriptRoot -NoStart:$NoStart
exit $LASTEXITCODE
