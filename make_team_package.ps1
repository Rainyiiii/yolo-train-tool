param([string]$OutputDirectory = "")

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $ScriptRoot "dist"
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$packageName = "YOLO-Team-Training-Platform-Source-$stamp"
$tempBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$temporaryRoot = [IO.Path]::GetFullPath((Join-Path $tempBase $packageName))
$zipPath = Join-Path $OutputDirectory "$packageName.zip"

if (!$temporaryRoot.StartsWith($tempBase, [StringComparison]::OrdinalIgnoreCase)) {
    throw "临时目录不在系统临时文件夹内，已停止打包：$temporaryRoot"
}

if (Test-Path -LiteralPath $temporaryRoot) {
    Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $temporaryRoot | Out-Null

$allowedExtensions = @(".py", ".ps1", ".cmd", ".bat", ".vbs", ".sh", ".json", ".md", ".txt")
$trackedTopLevel = $null
if (Test-Path -LiteralPath (Join-Path $ScriptRoot ".git")) {
    $gitCommand = Get-Command git.exe -ErrorAction SilentlyContinue
    if ($gitCommand) {
        $trackedTopLevel = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
        & $gitCommand.Source -c core.quotepath=false -C $ScriptRoot ls-files | Where-Object { $_ -notmatch "[/\\]" } | ForEach-Object {
            [void]$trackedTopLevel.Add($_)
        }
    }
}
Get-ChildItem -LiteralPath $ScriptRoot -File | Where-Object {
    $_.Extension -in $allowedExtensions -and
    $_.Name -notin @("train_panel_defaults.json", "model_registry.json", ".annotation_server.pid.json") -and
    $_.Name -notmatch "^\.train" -and
    ($null -eq $trackedTopLevel -or $trackedTopLevel.Contains($_.Name))
} | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $temporaryRoot $_.Name)
}

$docsSource = Join-Path $ScriptRoot "docs"
if (Test-Path -LiteralPath $docsSource) {
    Copy-Item -LiteralPath $docsSource -Destination (Join-Path $temporaryRoot "docs") -Recurse
}

if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
Compress-Archive -LiteralPath $temporaryRoot -DestinationPath $zipPath -CompressionLevel Optimal
Remove-Item -LiteralPath $temporaryRoot -Recurse -Force

Write-Host "源码包已生成：" -ForegroundColor Green
Write-Host $zipPath
Write-Host "普通队友请优先使用 dist 目录中的 Windows 安装器。"
