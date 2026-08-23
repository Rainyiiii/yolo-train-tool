param([string]$Configuration = "Release")

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$RepoRoot = [IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\.."))
$WorkspaceTemp = Join-Path $RepoRoot "workspace\temp"
$DependencyCache = Join-Path $RepoRoot "workspace\cache\installer-dependencies"
$StagingRoot = [IO.Path]::GetFullPath((Join-Path $WorkspaceTemp "windows-installer-build"))
$OutputDirectory = Join-Path $RepoRoot "dist"
$Project = Join-Path $RepoRoot "desktop\YOLOTeamTrainingPlatform.Desktop\YOLOTeamTrainingPlatform.Desktop.csproj"
$InnoScript = Join-Path $RepoRoot "installer\windows\setup.iss"
$Version = (Get-Content -LiteralPath (Join-Path $RepoRoot "VERSION.txt") -Raw).Trim()
$SafeVersion = $Version -replace "[^0-9A-Za-z.-]", "-"
$versionNumbers = @([regex]::Matches($Version, "\d+") | ForEach-Object { $_.Value })
while ($versionNumbers.Count -lt 3) { $versionNumbers += "0" }
$NumericVersion = ($versionNumbers[0..2] -join ".") + ".0"

if (!$StagingRoot.StartsWith([IO.Path]::GetFullPath($WorkspaceTemp), [StringComparison]::OrdinalIgnoreCase)) {
    throw "安装器暂存目录不在工作区 temp 内：$StagingRoot"
}
if (Test-Path -LiteralPath $StagingRoot) { Remove-Item -LiteralPath $StagingRoot -Recurse -Force }
$AppStage = Join-Path $StagingRoot "App"
$DesktopStage = Join-Path $StagingRoot "Desktop"
$ResourceStage = Join-Path $StagingRoot "Resources"
foreach ($path in @($AppStage, $DesktopStage, $ResourceStage, $OutputDirectory, $DependencyCache)) {
    New-Item -ItemType Directory -Path $path -Force | Out-Null
}

$runtimeFiles = @(
    "annotation_exports.py", "annotation_server.py", "annotation_service.py", "annotation_store.py", "annotation_ui.py",
    "device_profiles.py", "export_model.py", "host_train_export.py", "install_runtime.ps1", "model_assets.py",
    "model_test.py", "panel_service.py", "platform_paths.py", "remote_train_env.py", "requirements.txt",
    "system_check.py", "train_panel.py", "train_panel_defaults.example.json", "ultralytics_train_runner.py",
    "video_track_label.py", "vm_convert_pack.sh", "VERSION.txt", "README.md", "SECURITY.md", "THIRD_PARTY_NOTICES.md"
)
foreach ($relative in $runtimeFiles) {
    $source = Join-Path $RepoRoot $relative
    if (!(Test-Path -LiteralPath $source)) { throw "安装器缺少运行文件：$relative" }
    Copy-Item -LiteralPath $source -Destination (Join-Path $AppStage $relative)
}
Copy-Item -LiteralPath (Join-Path $RepoRoot "docs") -Destination (Join-Path $AppStage "docs") -Recurse
$bundledModel = Join-Path $RepoRoot "yolo11n.pt"
if (Test-Path -LiteralPath $bundledModel) { Copy-Item -LiteralPath $bundledModel -Destination (Join-Path $AppStage "yolo11n.pt") }

$dotnet = Join-Path $env:ProgramFiles "dotnet\dotnet.exe"
if (!(Test-Path -LiteralPath $dotnet)) { throw "未找到 .NET 8 SDK。" }
& $dotnet restore $Project
if ($LASTEXITCODE -ne 0) { throw "WebView2 桌面程序依赖还原失败。" }
& $dotnet publish $Project -c $Configuration --no-restore --self-contained false -o $DesktopStage
if ($LASTEXITCODE -ne 0) { throw "WebView2 桌面程序发布失败。" }

$desktopRuntimeCache = Join-Path $DependencyCache "windowsdesktop-runtime-8-win-x64.exe"
if (!(Test-Path -LiteralPath $desktopRuntimeCache) -or (Get-Item -LiteralPath $desktopRuntimeCache).Length -lt 40MB) {
    Invoke-WebRequest -UseBasicParsing -Uri "https://aka.ms/dotnet/8.0/windowsdesktop-runtime-win-x64.exe" -OutFile $desktopRuntimeCache
}
if ((Get-Item -LiteralPath $desktopRuntimeCache).Length -lt 40MB) { throw ".NET 8 Desktop Runtime 安装程序下载不完整。" }
Copy-Item -LiteralPath $desktopRuntimeCache -Destination (Join-Path $ResourceStage "windowsdesktop-runtime-8-win-x64.exe")

$webViewCache = Join-Path $DependencyCache "MicrosoftEdgeWebview2Setup.exe"
if (!(Test-Path -LiteralPath $webViewCache) -or (Get-Item -LiteralPath $webViewCache).Length -lt 1MB) {
    Invoke-WebRequest -UseBasicParsing -Uri "https://go.microsoft.com/fwlink/p/?LinkId=2124703" -OutFile $webViewCache
}
if ((Get-Item -LiteralPath $webViewCache).Length -lt 1MB) { throw "WebView2 引导程序下载不完整。" }
Copy-Item -LiteralPath $webViewCache -Destination (Join-Path $ResourceStage "MicrosoftEdgeWebview2Setup.exe")

$isccCandidates = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
)
$iscc = $isccCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (!$iscc) { throw "未找到 Inno Setup 6 编译器。" }

& $iscc "/DSourceDirectory=$StagingRoot" "/DOutputDirectory=$OutputDirectory" "/DProductVersion=$SafeVersion" "/DProductVersionNumeric=$NumericVersion" $InnoScript
if ($LASTEXITCODE -ne 0) { throw "Windows 安装程序编译失败。" }

$setup = Join-Path $OutputDirectory "YOLO-Team-Training-Platform-Setup-v$SafeVersion.exe"
if (!(Test-Path -LiteralPath $setup)) { throw "安装程序未生成：$setup" }
Write-Host "WINDOWS_INSTALLER=$setup" -ForegroundColor Green
