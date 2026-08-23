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
  RuntimeProgressPage := CreateOutputMarqueeProgressPage(
    '正在安装运行环境',
    '首次安装需要下载模型训练组件，请保持网络连接。');
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
begin
  if CurStep <> ssPostInstall then
    exit;

  RuntimeReady := False;
  RuntimeLogMemo.Clear;
  RuntimeProgressPage.Show;
  RuntimeProgressPage.Animate;
  try
    RuntimeProgressPage.SetText('正在安装 .NET 8 Desktop Runtime...', '步骤 1 / 3');
    AppendRuntimeLog('==> 安装 .NET 8 Desktop Runtime');
    ExtractTemporaryFile('windowsdesktop-runtime-8-win-x64.exe');
    if not Exec(ExpandConstant('{tmp}\windowsdesktop-runtime-8-win-x64.exe'), '/install /quiet /norestart', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
      RaiseException('无法启动 .NET 8 Desktop Runtime 安装程序。');
    if (ResultCode <> 0) and (ResultCode <> 3010) then
      RaiseException(Format('.NET 8 Desktop Runtime 安装失败，退出码 %d。', [ResultCode]));
    AppendRuntimeLog(Format('.NET 8 Desktop Runtime 完成（退出码 %d）', [ResultCode]));

    RuntimeProgressPage.SetText('正在安装 Microsoft Edge WebView2 Runtime...', '步骤 2 / 3');
    AppendRuntimeLog('==> 安装 Microsoft Edge WebView2 Runtime');
    ExtractTemporaryFile('MicrosoftEdgeWebview2Setup.exe');
    if not Exec(ExpandConstant('{tmp}\MicrosoftEdgeWebview2Setup.exe'), '/silent /install', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
      RaiseException('无法启动 WebView2 Runtime 安装程序。');
    if ResultCode <> 0 then
      RaiseException(Format('WebView2 Runtime 安装失败，退出码 %d。', [ResultCode]));
    AppendRuntimeLog('Microsoft Edge WebView2 Runtime 完成');

    RuntimeProgressPage.SetText('正在安装 Python、PyTorch、ONNX Runtime 与平台依赖...', '步骤 3 / 3 · 首次安装可能需要较长时间');
    AppendRuntimeLog('==> 安装 Python、PyTorch、ONNX Runtime 与平台依赖');
    PowerShellPath := ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe');
    RuntimeScript := ExpandConstant('{app}\App\install_runtime.ps1');
    if not ExecAndLogOutput(PowerShellPath,
      '-NoLogo -NoProfile -NonInteractive -OutputFormat Text -ExecutionPolicy Bypass -File "' + RuntimeScript + '" -InstallRoot "' + ExpandConstant('{app}') + '" -NoStart',
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
