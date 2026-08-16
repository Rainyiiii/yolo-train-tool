param(
    [switch]$NoStart,
    [switch]$ForceReinstall
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ScriptRoot "logs"
$LogFile = Join-Path $LogDir "install.log"
$VenvPython = Join-Path $ScriptRoot ".venv\Scripts\python.exe"
$Requirements = Join-Path $ScriptRoot "requirements.txt"
$DefaultsExample = Join-Path $ScriptRoot "train_panel_defaults.example.json"
$DefaultsFile = Join-Path $ScriptRoot "train_panel_defaults.json"

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
Start-Transcript -Path $LogFile -Append | Out-Null

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Red
}

function Find-CompatiblePython {
    $candidates = @(
        @{ File = "py.exe"; Args = @("-3.14") },
        @{ File = "py.exe"; Args = @("-3.13") },
        @{ File = "py.exe"; Args = @("-3.12") },
        @{ File = "py.exe"; Args = @("-3.11") },
        @{ File = "py.exe"; Args = @("-3.10") },
        @{ File = "python.exe"; Args = @() }
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

try {
    Set-Location $ScriptRoot
    Write-Host "MyAutoTrain 团队版安装器" -ForegroundColor Red
    Write-Host "安装目录：$ScriptRoot"

    $python = Find-CompatiblePython
    if (!$python) {
        Write-Step "未找到 Python 3.10-3.14，正在自动安装 Python 3.14"
        $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
        if (!$winget) {
            throw "电脑没有 Python，也没有 Windows winget。请先从 python.org 安装 Python 3.14，然后重新双击本安装程序。"
        }
        & $winget.Source install --id Python.Python.3.14 --exact --scope user --silent --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) { throw "Python 3.14 自动安装失败（退出码 $LASTEXITCODE）。" }
        $python314 = Join-Path $env:LOCALAPPDATA "Programs\Python\Python314\python.exe"
        if (Test-Path -LiteralPath $python314) {
            $python = @{ File = $python314; Args = @(); Version = "3.14" }
        } else {
            $python = Find-CompatiblePython
        }
        if (!$python) { throw "Python 已安装，但当前窗口尚未找到它。请关闭本窗口后重新双击安装程序。" }
    }

    Write-Step "使用 Python $($python.Version) 创建项目专用环境"
    if ($ForceReinstall -and (Test-Path -LiteralPath $VenvPython)) {
        throw "安全起见，ForceReinstall 不会自动删除旧环境。请手动移走 .venv 文件夹后重试。"
    }
    if (!(Test-Path -LiteralPath $VenvPython)) {
        & $python.File @($python.Args) -m venv (Join-Path $ScriptRoot ".venv")
        if ($LASTEXITCODE -ne 0) { throw "创建 Python 环境失败。" }
    }

    Write-Step "更新安装工具"
    & $VenvPython -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw "pip 更新失败。" }

    $hasNvidia = $null -ne (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)
    if ($hasNvidia) {
        $savedErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            & $VenvPython -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>$null
            $torchReady = $LASTEXITCODE -eq 0
        } finally {
            $ErrorActionPreference = $savedErrorActionPreference
        }
        if (!$torchReady) {
            Write-Step "检测到 NVIDIA 显卡，安装 CUDA 12.8 训练组件"
            & $VenvPython -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
            if ($LASTEXITCODE -ne 0) { throw "PyTorch CUDA 安装失败。请检查网络后重新双击；安装器会从断点继续。" }
        } else {
            Write-Step "NVIDIA 训练组件已经就绪，跳过重复下载"
        }
    } else {
        $savedErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            & $VenvPython -c "import torch" 2>$null
            $torchReady = $LASTEXITCODE -eq 0
        } finally {
            $ErrorActionPreference = $savedErrorActionPreference
        }
        if (!$torchReady) {
            Write-Step "未检测到 NVIDIA 显卡，安装 CPU 兼容组件"
            & $VenvPython -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            if ($LASTEXITCODE -ne 0) { throw "PyTorch CPU 安装失败。请检查网络后重新双击；安装器会从断点继续。" }
        } else {
            Write-Step "CPU 训练组件已经就绪，跳过重复下载"
        }
    }

    Write-Step "安装 MyAutoTrain 功能组件"
    & $VenvPython -m pip install --upgrade -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw "项目依赖安装失败。请检查 logs\install.log。" }

    if (!(Test-Path -LiteralPath $DefaultsFile) -and (Test-Path -LiteralPath $DefaultsExample)) {
        Copy-Item -LiteralPath $DefaultsExample -Destination $DefaultsFile
        $newDefaults = Get-Content -LiteralPath $DefaultsFile -Raw | ConvertFrom-Json
        if ($hasNvidia) {
            $newDefaults.train_device = "cuda"
            $newDefaults.torch_cuda = "cu128"
            $vramText = & nvidia-smi.exe --query-gpu=memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
            $vramMb = 0
            [void][int]::TryParse(($vramText -replace "[^0-9]", ""), [ref]$vramMb)
            if ($vramMb -ge 7500) {
                $newDefaults.batch = "16"
                $newDefaults.train_workers = "4"
            } elseif ($vramMb -ge 5500) {
                $newDefaults.batch = "8"
                $newDefaults.train_workers = "4"
            } else {
                $newDefaults.batch = "4"
                $newDefaults.train_workers = "2"
            }
        } else {
            $newDefaults.train_device = "cpu"
            $newDefaults.torch_cuda = "cpu"
            $newDefaults.train_cache = "False"
            $newDefaults.batch = "4"
            $newDefaults.train_workers = "2"
        }
        $defaultsJson = $newDefaults | ConvertTo-Json -Depth 8
        [IO.File]::WriteAllText($DefaultsFile, $defaultsJson + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))
    }

    if (!(Test-Path -LiteralPath (Join-Path $ScriptRoot "yolo11n.pt"))) {
        Write-Step "下载默认 YOLO11n 基础模型"
        & $VenvPython -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); print('YOLO11n ready')"
        if ($LASTEXITCODE -ne 0) { throw "基础模型下载失败。请检查网络后重试。" }
    } else {
        Write-Step "默认 YOLO11n 基础模型已经就绪"
    }

    Write-Step "执行系统自检"
    & $VenvPython (Join-Path $ScriptRoot "system_check.py") --write-report
    if ($LASTEXITCODE -ne 0) { throw "系统自检未通过。请查看上方说明或 logs\system_check.json。" }

    if (!$NoStart) {
        Write-Step "启动 MyAutoTrain"
        & $VenvPython (Join-Path $ScriptRoot "panel_service.py") start
        if ($LASTEXITCODE -ne 0) { throw "面板启动失败。请查看 logs\panel.log。" }
    }

    Write-Host ""
    Write-Host "安装完成。以后只需双击“启动训练面板.cmd”。" -ForegroundColor Green
    Write-Host "网页地址：http://127.0.0.1:8989/"
    Stop-Transcript | Out-Null
    exit 0
} catch {
    Write-Host ""
    Write-Host "安装未完成：$($_.Exception.Message)" -ForegroundColor Red
    Write-Host "详细日志：$LogFile"
    try { Stop-Transcript | Out-Null } catch {}
    exit 1
}
