param([string]$OutputDirectory = "")

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $ScriptRoot "dist"
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$packageName = "MyAutoTrain-Team-$stamp"
$temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) $packageName
$zipPath = Join-Path $OutputDirectory "$packageName.zip"

if (Test-Path -LiteralPath $temporaryRoot) {
    Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $temporaryRoot | Out-Null

$allowedExtensions = @(".py", ".ps1", ".cmd", ".bat", ".sh", ".json", ".md", ".txt")
Get-ChildItem -LiteralPath $ScriptRoot -File | Where-Object {
    $_.Extension -in $allowedExtensions -and
    $_.Name -notin @("train_panel_defaults.json") -and
    $_.Name -notmatch "^\.train"
} | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $temporaryRoot $_.Name)
}

if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
Compress-Archive -LiteralPath $temporaryRoot -DestinationPath $zipPath -CompressionLevel Optimal
Remove-Item -LiteralPath $temporaryRoot -Recurse -Force

Write-Host "团队部署包已生成：" -ForegroundColor Green
Write-Host $zipPath
Write-Host "队友解压后双击“一键安装并启动.cmd”即可。"
