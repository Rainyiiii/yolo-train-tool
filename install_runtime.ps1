param(
    [string]$InstallRoot = "",
    [switch]$NoStart,
    [switch]$RepairRuntime
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Utf8OutputEncoding = New-Object Text.UTF8Encoding($false)
[Console]::OutputEncoding = $Utf8OutputEncoding
$OutputEncoding = $Utf8OutputEncoding
$SourceRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($InstallRoot)) { $InstallRoot = $SourceRoot }
$InstallRoot = [IO.Path]::GetFullPath($InstallRoot)
$DeployedAppRoot = Join-Path $InstallRoot "App"
$AppRoot = if (Test-Path -LiteralPath (Join-Path $DeployedAppRoot "requirements.txt")) { $DeployedAppRoot } else { $SourceRoot }
$IsDeployed = !([IO.Path]::GetFullPath($InstallRoot).Equals([IO.Path]::GetFullPath($SourceRoot), [StringComparison]::OrdinalIgnoreCase))
$WorkspaceRoot = Join-Path $InstallRoot "Workspace"
if (!$IsDeployed) { $WorkspaceRoot = Join-Path $InstallRoot "workspace" }
$RuntimeRoot = if ($IsDeployed) { Join-Path $InstallRoot "Runtime\Python" } else { Join-Path $AppRoot ".venv" }
$VenvPython = Join-Path $RuntimeRoot "Scripts\python.exe"
$LogDir = Join-Path $WorkspaceRoot "logs"
$LogFile = Join-Path $LogDir "installation.log"
$ConfigDir = Join-Path $WorkspaceRoot "config"
$SettingsFile = Join-Path $ConfigDir "settings.json"
$Requirements = Join-Path $AppRoot "requirements.txt"
$SystemCheck = Join-Path $AppRoot "system_check.py"
$DefaultsExample = Join-Path $AppRoot "train_panel_defaults.example.json"
$DependencyStateFile = Join-Path $RuntimeRoot ".yolo-dependency-state.json"

foreach ($path in @(
    $LogDir, $ConfigDir, (Join-Path $WorkspaceRoot "state"), (Join-Path $WorkspaceRoot "datasets"),
    (Join-Path $WorkspaceRoot "annotation-hub"), (Join-Path $WorkspaceRoot "training-runs"),
    (Join-Path $WorkspaceRoot "model-assets\base-models"), (Join-Path $WorkspaceRoot "exports\datasets"),
    (Join-Path $WorkspaceRoot "exports\deployments"), (Join-Path $WorkspaceRoot "test-results"),
    (Join-Path $WorkspaceRoot "cache"), (Join-Path $WorkspaceRoot "temp"), (Join-Path $WorkspaceRoot "backups")
)) { New-Item -ItemType Directory -Path $path -Force | Out-Null }

$env:YOLO_TEAM_PLATFORM_HOME = $InstallRoot
$env:YOLO_TEAM_PLATFORM_DATA = $WorkspaceRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Start-Transcript -Path $LogFile -Append | Out-Null

function Write-Step([string]$Message) {
    Write-Output ""
    Write-Output "==> $Message"
}

function Find-CompatiblePython {
    $candidates = @(
        @{ File = "py.exe"; Args = @("-3.14") }, @{ File = "py.exe"; Args = @("-3.13") },
        @{ File = "py.exe"; Args = @("-3.12") }, @{ File = "py.exe"; Args = @("-3.11") },
        @{ File = "py.exe"; Args = @("-3.10") }, @{ File = "python.exe"; Args = @() }
    )
    foreach ($candidate in $candidates) {
        $command = Get-Command $candidate.File -ErrorAction SilentlyContinue
        if (!$command) { continue }
        try {
            $version = & $command.Source @($candidate.Args) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($LASTEXITCODE -eq 0 -and $version -in @("3.10", "3.11", "3.12", "3.13", "3.14")) {
                return @{ File = $command.Source; Args = $candidate.Args; Version = $version }
            }
        } catch {}
    }
    return $null
}

function Invoke-Probe([scriptblock]$Action) {
    $savedErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $null = & $Action
        $probeExitCode = $LASTEXITCODE
        return $probeExitCode -eq 0
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $savedErrorPreference
    }
}

function Test-RuntimeDependencies {
    if (!(Test-Path -LiteralPath $VenvPython)) { return $false }
    if (!(Test-Path -LiteralPath $SystemCheck)) { return $false }
    $systemReady = Invoke-Probe { & $VenvPython $SystemCheck }
    if (!$systemReady) { return $false }
    return Invoke-Probe { & $VenvPython -m pip check }
}

function Read-DependencyState {
    if (!(Test-Path -LiteralPath $DependencyStateFile)) { return $null }
    try {
        return Get-Content -LiteralPath $DependencyStateFile -Raw | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Write-DependencyState([string]$RequirementsHash, [string]$TorchProfile) {
    $platformVersionFile = Join-Path $AppRoot "VERSION.txt"
    $platformVersion = if (Test-Path -LiteralPath $platformVersionFile) {
        (Get-Content -LiteralPath $platformVersionFile -Raw).Trim()
    } else { "unknown" }
    $state = [ordered]@{
        schema_version = 1
        requirements_sha256 = $RequirementsHash
        torch_profile = $TorchProfile
        platform_version = $platformVersion
        verified_at = [DateTime]::UtcNow.ToString("o")
    }
    [IO.File]::WriteAllText(
        $DependencyStateFile,
        (($state | ConvertTo-Json -Depth 4) + [Environment]::NewLine),
        (New-Object Text.UTF8Encoding($false))
    )
}

function Remove-RuntimeForRepair {
    if (!(Test-Path -LiteralPath $RuntimeRoot)) { return }
    $resolvedRuntime = [IO.Path]::GetFullPath($RuntimeRoot).TrimEnd('\')
    $expectedRuntime = if ($IsDeployed) {
        [IO.Path]::GetFullPath((Join-Path $InstallRoot "Runtime\Python")).TrimEnd('\')
    } else {
        [IO.Path]::GetFullPath((Join-Path $AppRoot ".venv")).TrimEnd('\')
    }
    if (!$resolvedRuntime.Equals($expectedRuntime, [StringComparison]::OrdinalIgnoreCase)) {
        throw "拒绝删除异常的运行环境目录：$resolvedRuntime"
    }
    foreach ($serviceScriptName in @("annotation_service.py", "panel_service.py")) {
        $serviceScript = Join-Path $AppRoot $serviceScriptName
        if ((Test-Path -LiteralPath $VenvPython) -and (Test-Path -LiteralPath $serviceScript)) {
            $null = Invoke-Probe { & $VenvPython $serviceScript stop }
        }
    }
    Write-Step "完整修复：删除旧运行环境"
    Remove-Item -LiteralPath $resolvedRuntime -Recurse -Force
}

try {
    Set-Location $AppRoot
    Write-Output "YOLO团队训练平台安装器"
    Write-Output "程序目录：$AppRoot"
    Write-Output "工作区：$WorkspaceRoot"

    if ($RepairRuntime) { Remove-RuntimeForRepair }
    $runtimeAlreadyExists = Test-Path -LiteralPath $VenvPython
    $python = $null
    if (!$runtimeAlreadyExists) { $python = Find-CompatiblePython }
    if (!$runtimeAlreadyExists -and !$python) {
        Write-Step "安装 Python 3.14"
        $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
        if (!$winget) { throw "未找到 Python 3.10-3.14，也没有可用的 winget。" }
        $savedErrorPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $winget.Source install --id Python.Python.3.14 --exact --scope machine --silent --accept-package-agreements --accept-source-agreements --disable-interactivity
        $machineInstallExitCode = $LASTEXITCODE
        $ErrorActionPreference = $savedErrorPreference
        if ($machineInstallExitCode -ne 0) {
            $ErrorActionPreference = "Continue"
            & $winget.Source install --id Python.Python.3.14 --exact --scope user --silent --accept-package-agreements --accept-source-agreements --disable-interactivity
            $userInstallExitCode = $LASTEXITCODE
            $ErrorActionPreference = $savedErrorPreference
            if ($userInstallExitCode -ne 0) { throw "Python 3.14 自动安装失败。" }
        }
        $python = Find-CompatiblePython
        if (!$python) {
            $machinePython = "C:\Program Files\Python314\python.exe"
            $userPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python314\python.exe"
            if (Test-Path -LiteralPath $machinePython) { $python = @{ File = $machinePython; Args = @(); Version = "3.14" } }
            elseif (Test-Path -LiteralPath $userPython) { $python = @{ File = $userPython; Args = @(); Version = "3.14" } }
        }
        if (!$python) { throw "Python 安装完成后仍无法定位解释器，请查看安装日志。" }
    }

    if (!$runtimeAlreadyExists) {
        Write-Step "创建隔离的 Python 运行环境（Python $($python.Version)）"
        & $python.File @($python.Args) -m venv $RuntimeRoot
        if ($LASTEXITCODE -ne 0) { throw "创建 Python 运行环境失败。" }
    } else {
        $runtimeVersion = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($LASTEXITCODE -ne 0) { throw "已有 Python 运行环境无法启动，请选择完整修复。" }
        Write-Step "复用已有 Python $runtimeVersion 运行环境"
    }

    $hasNvidia = $null -ne (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)
    $torchProfile = if ($hasNvidia) { "cu128" } else { "cpu" }
    $requirementsHash = (Get-FileHash -LiteralPath $Requirements -Algorithm SHA256).Hash
    $dependencyStateFileExists = Test-Path -LiteralPath $DependencyStateFile
    $dependencyState = Read-DependencyState
    $stateMatches = $null -ne $dependencyState -and
        $dependencyState.requirements_sha256 -eq $requirementsHash -and
        $dependencyState.torch_profile -eq $torchProfile
    $legacyRuntimeReady = !$dependencyStateFileExists -and $null -eq $dependencyState -and (Test-RuntimeDependencies)
    $incrementalRuntimeReady = !$RepairRuntime -and ($stateMatches -or $legacyRuntimeReady) -and (Test-RuntimeDependencies)

    if ($incrementalRuntimeReady) {
        Write-Step "运行环境与当前版本一致，跳过依赖下载"
    } else {
        Write-Step $(if ($RepairRuntime) { "完整安装模型训练依赖" } else { "增量补齐模型训练依赖" })
        if (!$runtimeAlreadyExists -or $RepairRuntime) {
            & $VenvPython -m pip install --upgrade pip setuptools wheel
            if ($LASTEXITCODE -ne 0) { throw "pip 初始化失败。" }
        }

        $torchReady = Invoke-Probe {
            & $VenvPython -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"
        }
        $torchProfileChanged = $null -ne $dependencyState -and $dependencyState.torch_profile -ne $torchProfile
        if (!$torchReady -or $torchProfileChanged) {
        $torchIndex = if ($hasNvidia) { "https://download.pytorch.org/whl/cu128" } else { "https://download.pytorch.org/whl/cpu" }
            $torchArguments = @("-m", "pip", "install", "--upgrade")
            if ($torchProfileChanged) { $torchArguments += "--force-reinstall" }
            $torchArguments += @("torch", "torchvision", "torchaudio", "--index-url", $torchIndex)
            & $VenvPython @torchArguments
            if ($LASTEXITCODE -ne 0) { throw "PyTorch 安装失败。" }
        }
        # Deliberately omit --upgrade here.  pip then keeps every installed
        # package that already satisfies requirements.txt and downloads only
        # missing or incompatible dependencies.
        & $VenvPython -m pip install -r $Requirements
        if ($LASTEXITCODE -ne 0) { throw "平台依赖安装失败。" }
        if (!(Test-RuntimeDependencies)) { throw "依赖安装完成，但运行环境健康检查未通过。" }
    }
    Write-DependencyState $requirementsHash $torchProfile

    Write-Step "创建规范化工作区配置"
    $settings = if (Test-Path -LiteralPath $SettingsFile) {
        Get-Content -LiteralPath $SettingsFile -Raw | ConvertFrom-Json
    } elseif (Test-Path -LiteralPath $DefaultsExample) {
        Get-Content -LiteralPath $DefaultsExample -Raw | ConvertFrom-Json
    } else { [pscustomobject]@{} }
    $baseModel = Join-Path $WorkspaceRoot "model-assets\base-models\yolo11n.pt"
    $settings.base_model = $baseModel
    $settings.dataset_root = Join-Path $WorkspaceRoot "datasets"
    $settings.train_images_dir = Join-Path $WorkspaceRoot "datasets\default\images"
    $settings.train_annotations_dir = Join-Path $WorkspaceRoot "datasets\default\annotations"
    $settings.export_output_dir = Join-Path $WorkspaceRoot "exports\deployments"
    $settings.test_output_dir = Join-Path $WorkspaceRoot "test-results"
    $settings.label_images_dir = Join-Path $WorkspaceRoot "annotation-hub\quick-label\images"
    $settings.label_annotations_dir = Join-Path $WorkspaceRoot "annotation-hub\quick-label\annotations"
    $settings.train_device = if ($hasNvidia) { "cuda" } else { "cpu" }
    $settings.torch_cuda = if ($hasNvidia) { "cu128" } else { "cpu" }
    if (!$hasNvidia) { $settings.train_cache = "False"; $settings.batch = "4"; $settings.train_workers = "2" }
    [IO.File]::WriteAllText($SettingsFile, (($settings | ConvertTo-Json -Depth 8) + [Environment]::NewLine), (New-Object Text.UTF8Encoding($false)))

    Write-Step "准备默认 YOLO11n 基础模型"
    if (!(Test-Path -LiteralPath $baseModel)) {
        $bundledModel = Join-Path $AppRoot "yolo11n.pt"
        if (Test-Path -LiteralPath $bundledModel) {
            Copy-Item -LiteralPath $bundledModel -Destination $baseModel
        } else {
            $env:YOLO_BASE_MODEL_DESTINATION = $baseModel
            & $VenvPython -c "import os,shutil; from pathlib import Path; from ultralytics import YOLO; m=YOLO('yolo11n.pt'); src=Path(getattr(m,'ckpt_path','yolo11n.pt')).resolve(); dst=Path(os.environ['YOLO_BASE_MODEL_DESTINATION']); dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)"
            if ($LASTEXITCODE -ne 0) { throw "默认模型下载失败。" }
        }
    }

    Write-Step "执行系统自检"
    & $VenvPython (Join-Path $AppRoot "system_check.py") --write-report
    if ($LASTEXITCODE -ne 0) { throw "系统自检未通过。" }

    if (!$NoStart) {
        Write-Step "启动 YOLO团队训练平台"
        & $VenvPython (Join-Path $AppRoot "panel_service.py") start
        if ($LASTEXITCODE -ne 0) { throw "平台启动失败。" }
    }

    Write-Output ""
    Write-Output "安装完成。"
    Stop-Transcript | Out-Null
    exit 0
} catch {
    Write-Output ""
    Write-Output "安装未完成：$($_.Exception.Message)"
    Write-Output "安装日志：$LogFile"
    try { Stop-Transcript | Out-Null } catch {}
    exit 1
}
