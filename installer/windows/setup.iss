#define ProductName "YOLO团队训练平台"
#define ProductCode "YOLOTeamTrainingPlatform"

[Setup]
AppId={{D6F293C2-46AE-4A8E-8CF2-73DDF6E866B1}
AppName={#ProductName}
AppVersion={#ProductVersion}
AppPublisher=YOLO Team Training Platform Contributors
AppPublisherURL=https://github.com/Rainyiiii/yolo-train-tool
DefaultDirName={code:GetDefaultInstallDir}
DefaultGroupName={#ProductName}
DisableProgramGroupPage=yes
OutputDir={#OutputDirectory}
OutputBaseFilename=YOLO-Team-Training-Platform-Setup-v{#ProductVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#ProductName}
VersionInfoDescription={#ProductName} 安装程序
VersionInfoProductName={#ProductName}
VersionInfoVersion={#ProductVersionNumeric}
VersionInfoProductVersion={#ProductVersionNumeric}
SetupLogging=yes
CloseApplications=yes
RestartApplications=no

[Dirs]
Name: "{app}\Workspace"; Permissions: users-modify

[Files]
Source: "{#SourceDirectory}\App\*"; DestDir: "{app}\App"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceDirectory}\Desktop\*"; DestDir: "{app}\Desktop"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceDirectory}\Resources\windowsdesktop-runtime-8-win-x64.exe"; Flags: dontcopy
Source: "{#SourceDirectory}\Resources\MicrosoftEdgeWebview2Setup.exe"; Flags: dontcopy

[Icons]
Name: "{autoprograms}\{#ProductName}"; Filename: "{app}\Desktop\YOLOTeamTrainingPlatform.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\{#ProductName}"; Filename: "{app}\Desktop\YOLOTeamTrainingPlatform.exe"; WorkingDir: "{app}"

[Run]
Filename: "{app}\Desktop\YOLOTeamTrainingPlatform.exe"; Description: "启动 {#ProductName}"; Flags: postinstall nowait skipifsilent; Check: RuntimeInstallationSucceeded

[UninstallRun]
Filename: "{app}\Runtime\Python\Scripts\python.exe"; Parameters: """{app}\App\annotation_service.py"" stop"; Flags: runhidden waituntilterminated skipifdoesntexist; RunOnceId: "StopAnnotationService"
Filename: "{app}\Runtime\Python\Scripts\python.exe"; Parameters: """{app}\App\panel_service.py"" stop"; Flags: runhidden waituntilterminated skipifdoesntexist; RunOnceId: "StopPanelService"

[UninstallDelete]
; Only remove directories owned by this product under the actual {app} path.
; Never use an {app}\* wildcard: unknown files at the install root are preserved.
Type: filesandordirs; Name: "{app}\Runtime"
Type: filesandordirs; Name: "{app}\App"
Type: filesandordirs; Name: "{app}\Desktop"
Type: dirifempty; Name: "{app}"

[Code]
var
  RuntimeReady: Boolean;
  RuntimeProgressPage: TOutputMarqueeProgressWizardPage;
  RuntimeLogMemo: TNewMemo;
  RepairOptionsPage: TInputOptionWizardPage;
  DeleteWorkspaceOnUninstall: Boolean;
  WorkspaceChoiceMade: Boolean;

function HasCommandLineSwitch(const SwitchName: String): Boolean;
var
  I: Integer;
begin
  Result := False;
  for I := 1 to ParamCount do
    if CompareText(ParamStr(I), SwitchName) = 0 then begin
      Result := True;
      exit;
    end;
end;

procedure SelectWorkspaceUninstallPolicy;
begin
  if WorkspaceChoiceMade then
    exit;

  WorkspaceChoiceMade := True;
  if HasCommandLineSwitch('/PURGEDATA') then
    DeleteWorkspaceOnUninstall := True
  else if HasCommandLineSwitch('/KEEPDATA') or UninstallSilent then
    DeleteWorkspaceOnUninstall := False
  else
    DeleteWorkspaceOnUninstall := SuppressibleMsgBox(
      '是否保留 Workspace 中的用户数据？' + #13#10 + #13#10 +
      '选择“是”：保留数据集、标注、训练模型、导出结果和个人配置。' + #13#10 +
      '选择“否”：彻底删除本次安装目录内的程序和所有用户数据。',
      mbConfirmation, MB_YESNO, IDYES) = IDNO;

  if DeleteWorkspaceOnUninstall then
    Log('Workspace uninstall policy: delete')
  else
    Log('Workspace uninstall policy: preserve');
end;

procedure PurgeWorkspace;
var
  InstallRoot: String;
  WorkspaceRoot: String;
begin
  InstallRoot := RemoveBackslashUnlessRoot(ExpandConstant('{app}'));
  WorkspaceRoot := PathCombine(InstallRoot, 'Workspace');
  if not PathSame(ExtractFileDir(WorkspaceRoot), InstallRoot) then begin
    Log('Refusing to delete unexpected Workspace path: ' + WorkspaceRoot);
    exit;
  end;

  Log('Deleting Workspace: ' + WorkspaceRoot);
  if DirExists(WorkspaceRoot) and not DelTree(WorkspaceRoot, True, True, True) then
    Log('Workspace could not be deleted completely: ' + WorkspaceRoot);
  if DirExists(InstallRoot) and not RemoveDir(InstallRoot) then
    Log('Install root retained because it contains unknown files: ' + InstallRoot);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
    SelectWorkspaceUninstallPolicy
  else if (CurUninstallStep = usPostUninstall) and DeleteWorkspaceOnUninstall then
    PurgeWorkspace;
end;

procedure InitializeWizard;
begin
  RepairOptionsPage := CreateInputOptionPage(
    wpSelectDir,
    '升级与运行环境修复',
    '选择本次安装如何处理已有运行环境',
    '默认执行增量更新：保留可用的 Python、PyTorch、ONNX Runtime 和其他依赖，只补装缺少或不兼容的组件。',
    False, False);
  RepairOptionsPage.Add('完整修复运行环境（删除并重建 Runtime，需要重新下载训练依赖）');
  RepairOptionsPage.Values[0] := False;

  RuntimeProgressPage := CreateOutputMarqueeProgressPage(
    '正在安装运行环境',
    '升级默认复用已有组件；首次安装或完整修复需要保持网络连接。');
  RuntimeProgressPage.SetText('正在准备安装...', '');

  RuntimeLogMemo := TNewMemo.Create(RuntimeProgressPage);
  RuntimeLogMemo.Parent := RuntimeProgressPage.Surface;
  RuntimeLogMemo.Left := 0;
  RuntimeLogMemo.Top := RuntimeProgressPage.ProgressBar.Top +
    RuntimeProgressPage.ProgressBar.Height + ScaleY(12);
  RuntimeLogMemo.Width := RuntimeProgressPage.SurfaceWidth;
  RuntimeLogMemo.Height := RuntimeProgressPage.SurfaceHeight - RuntimeLogMemo.Top;
  RuntimeLogMemo.Anchors := [akLeft, akTop, akRight, akBottom];
  RuntimeLogMemo.ReadOnly := True;
  RuntimeLogMemo.ScrollBars := ssBoth;
  RuntimeLogMemo.WordWrap := False;
  RuntimeLogMemo.Font.Name := 'Consolas';
  RuntimeLogMemo.Font.Size := 8;
end;

function ExistingRuntimeAvailable(): Boolean;
begin
  Result := FileExists(ExpandConstant('{app}\Runtime\Python\Scripts\python.exe'));
end;

function FullRepairSelected(): Boolean;
begin
  Result := ExistingRuntimeAvailable() and RepairOptionsPage.Values[0];
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := (PageID = RepairOptionsPage.ID) and not ExistingRuntimeAvailable();
end;

function DirectoryContainsVersionedFile(const RootPath, VersionPrefix, RelativeFile: String): Boolean;
var
  FindRec: TFindRec;
  Candidate: String;
begin
  Result := False;
  if not DirExists(RootPath) then
    exit;
  if FindFirst(AddBackslash(RootPath) + '*', FindRec) then begin
    try
      repeat
        if (VersionPrefix = '') or (Pos(VersionPrefix, FindRec.Name) = 1) then begin
          Candidate := PathCombine(PathCombine(RootPath, FindRec.Name), RelativeFile);
          if FileExists(Candidate) then begin
            Result := True;
            exit;
          end;
        end;
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
end;

function DotNetDesktopRuntimeAvailable(): Boolean;
begin
  Result := DirectoryContainsVersionedFile(
    ExpandConstant('{autopf}\dotnet\shared\Microsoft.WindowsDesktop.App'),
    '8.', 'Microsoft.WindowsDesktop.App.deps.json');
end;

function WebView2RuntimeAvailable(): Boolean;
begin
  Result := DirectoryContainsVersionedFile(
    ExpandConstant('{pf32}\Microsoft\EdgeWebView\Application'),
    '', 'msedgewebview2.exe');
end;

procedure AppendRuntimeLog(const S: String);
begin
  if RuntimeLogMemo.Lines.Count >= 3000 then
    RuntimeLogMemo.Lines.Delete(0);
  RuntimeLogMemo.Lines.Add(S);
  RuntimeLogMemo.SelStart := Length(RuntimeLogMemo.Text);
  RuntimeLogMemo.SelLength := 0;
  Log(S);
end;

procedure RuntimeProcessLog(const S: String; const Error, FirstLine: Boolean);
begin
  if Error then
    AppendRuntimeLog('[日志读取错误] ' + S)
  else
    AppendRuntimeLog(S);
end;

function GetDefaultInstallDir(Param: String): String;
begin
  if DirExists('D:\') then
    Result := 'D:\YOLOTeamTrainingPlatform'
  else
    Result := ExpandConstant('{autopf}\YOLOTeamTrainingPlatform');
end;

function RuntimeInstallationSucceeded(): Boolean;
begin
  Result := RuntimeReady;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  PowerShellPath: String;
  RuntimeScript: String;
  RuntimeArguments: String;
begin
  if CurStep <> ssPostInstall then
    exit;

  RuntimeReady := False;
  RuntimeLogMemo.Clear;
  RuntimeProgressPage.Show;
  RuntimeProgressPage.Animate;
  try
    if DotNetDesktopRuntimeAvailable() then begin
      RuntimeProgressPage.SetText('已检测到 .NET 8 Desktop Runtime', '步骤 1 / 3 · 跳过重复安装');
      AppendRuntimeLog('==> .NET 8 Desktop Runtime 已安装，跳过');
    end else begin
      RuntimeProgressPage.SetText('正在安装 .NET 8 Desktop Runtime...', '步骤 1 / 3');
      AppendRuntimeLog('==> 安装 .NET 8 Desktop Runtime');
      ExtractTemporaryFile('windowsdesktop-runtime-8-win-x64.exe');
      if not Exec(ExpandConstant('{tmp}\windowsdesktop-runtime-8-win-x64.exe'), '/install /quiet /norestart', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
        RaiseException('无法启动 .NET 8 Desktop Runtime 安装程序。');
      if (ResultCode <> 0) and (ResultCode <> 3010) then
        RaiseException(Format('.NET 8 Desktop Runtime 安装失败，退出码 %d。', [ResultCode]));
      AppendRuntimeLog(Format('.NET 8 Desktop Runtime 完成（退出码 %d）', [ResultCode]));
    end;

    if WebView2RuntimeAvailable() then begin
      RuntimeProgressPage.SetText('已检测到 Microsoft Edge WebView2 Runtime', '步骤 2 / 3 · 跳过重复安装');
      AppendRuntimeLog('==> Microsoft Edge WebView2 Runtime 已安装，跳过');
    end else begin
      RuntimeProgressPage.SetText('正在安装 Microsoft Edge WebView2 Runtime...', '步骤 2 / 3');
      AppendRuntimeLog('==> 安装 Microsoft Edge WebView2 Runtime');
      ExtractTemporaryFile('MicrosoftEdgeWebview2Setup.exe');
      if not Exec(ExpandConstant('{tmp}\MicrosoftEdgeWebview2Setup.exe'), '/silent /install', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
        RaiseException('无法启动 WebView2 Runtime 安装程序。');
      if ResultCode <> 0 then
        RaiseException(Format('WebView2 Runtime 安装失败，退出码 %d。', [ResultCode]));
      AppendRuntimeLog('Microsoft Edge WebView2 Runtime 完成');
    end;

    if FullRepairSelected() then begin
      RuntimeProgressPage.SetText('正在完整修复 Python、PyTorch、ONNX Runtime 与平台依赖...', '步骤 3 / 3 · 将重新下载训练组件');
      AppendRuntimeLog('==> 完整修复运行环境（用户已选择）');
    end else if ExistingRuntimeAvailable() then begin
      RuntimeProgressPage.SetText('正在检查并增量更新运行环境...', '步骤 3 / 3 · 已有依赖会直接复用');
      AppendRuntimeLog('==> 增量更新 Python、PyTorch、ONNX Runtime 与平台依赖');
    end else begin
      RuntimeProgressPage.SetText('正在首次安装 Python、PyTorch、ONNX Runtime 与平台依赖...', '步骤 3 / 3 · 首次安装可能需要较长时间');
      AppendRuntimeLog('==> 首次安装 Python、PyTorch、ONNX Runtime 与平台依赖');
    end;
    PowerShellPath := ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe');
    RuntimeScript := ExpandConstant('{app}\App\install_runtime.ps1');
    RuntimeArguments := '-NoLogo -NoProfile -NonInteractive -OutputFormat Text -ExecutionPolicy Bypass -File "' + RuntimeScript + '" -InstallRoot "' + ExpandConstant('{app}') + '" -NoStart';
    if FullRepairSelected() then
      RuntimeArguments := RuntimeArguments + ' -RepairRuntime';
    if not ExecAndLogOutput(PowerShellPath,
      RuntimeArguments,
      ExpandConstant('{app}\App'), SW_SHOWNORMAL, ewWaitUntilTerminated, ResultCode, @RuntimeProcessLog) then
      RaiseException('无法启动平台运行环境安装程序。');
    if ResultCode <> 0 then
      RaiseException(Format('平台运行环境安装失败，退出码 %d。请查看 Workspace\logs\installation.log。', [ResultCode]));

    AppendRuntimeLog('安装与系统自检全部完成。');
    RuntimeReady := True;
  finally
    RuntimeProgressPage.Hide;
  end;
end;
