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
Filename: "{app}\Desktop\YOLOTeamTrainingPlatform.exe"; Description: "启动 {#ProductName}"; Flags: postinstall nowait skipifsilent

[UninstallRun]
Filename: "{app}\Runtime\Python\Scripts\python.exe"; Parameters: """{app}\App\annotation_service.py"" stop"; Flags: runhidden waituntilterminated skipifdoesntexist; RunOnceId: "StopAnnotationService"
Filename: "{app}\Runtime\Python\Scripts\python.exe"; Parameters: """{app}\App\panel_service.py"" stop"; Flags: runhidden waituntilterminated skipifdoesntexist; RunOnceId: "StopPanelService"

[Code]
function GetDefaultInstallDir(Param: String): String;
begin
  if DirExists('D:\') then
    Result := 'D:\YOLOTeamTrainingPlatform'
  else
    Result := ExpandConstant('{autopf}\YOLOTeamTrainingPlatform');
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  PowerShellPath: String;
  RuntimeScript: String;
begin
  if CurStep <> ssPostInstall then
    exit;

  WizardForm.StatusLabel.Caption := '正在安装 .NET 8 Desktop Runtime...';
  ExtractTemporaryFile('windowsdesktop-runtime-8-win-x64.exe');
  if not Exec(ExpandConstant('{tmp}\windowsdesktop-runtime-8-win-x64.exe'), '/install /quiet /norestart', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    RaiseException('无法启动 .NET 8 Desktop Runtime 安装程序。');
  if (ResultCode <> 0) and (ResultCode <> 3010) then
    RaiseException(Format('.NET 8 Desktop Runtime 安装失败，退出码 %d。', [ResultCode]));

  WizardForm.StatusLabel.Caption := '正在安装 Microsoft Edge WebView2 Runtime...';
  ExtractTemporaryFile('MicrosoftEdgeWebview2Setup.exe');
  if not Exec(ExpandConstant('{tmp}\MicrosoftEdgeWebview2Setup.exe'), '/silent /install', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    RaiseException('无法启动 WebView2 Runtime 安装程序。');
  if ResultCode <> 0 then
    RaiseException(Format('WebView2 Runtime 安装失败，退出码 %d。', [ResultCode]));

  WizardForm.StatusLabel.Caption := '正在安装 Python、PyTorch、ONNX Runtime 与平台依赖，首次安装可能需要较长时间...';
  PowerShellPath := ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe');
  RuntimeScript := ExpandConstant('{app}\App\install_runtime.ps1');
  if not Exec(PowerShellPath,
    '-NoProfile -ExecutionPolicy Bypass -File "' + RuntimeScript + '" -InstallRoot "' + ExpandConstant('{app}') + '" -NoStart',
    ExpandConstant('{app}\App'), SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    RaiseException('无法启动平台运行环境安装程序。');
  if ResultCode <> 0 then
    RaiseException(Format('平台运行环境安装失败，退出码 %d。请查看 Workspace\logs\installation.log。', [ResultCode]));
end;
