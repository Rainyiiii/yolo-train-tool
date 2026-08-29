Option Explicit

Dim shell, files, root, desktopExe, pythonExe, serviceScript, installerScript, powerShellExe
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
root = files.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
desktopExe = files.BuildPath(root, "Desktop\YOLOTeamTrainingPlatform.exe")
pythonExe = files.BuildPath(root, ".venv\Scripts\python.exe")
serviceScript = files.BuildPath(root, "panel_service.py")
installerScript = files.BuildPath(root, "install_and_start.ps1")
powerShellExe = shell.ExpandEnvironmentStrings("%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe")

If files.FileExists(desktopExe) Then
    shell.Run Quote(desktopExe), 1, False
ElseIf files.FileExists(pythonExe) And files.FileExists(serviceScript) Then
    shell.Run Quote(pythonExe) & " " & Quote(serviceScript) & " start", 0, False
ElseIf files.FileExists(installerScript) Then
    shell.Run Quote(powerShellExe) & " -NoLogo -NoProfile -ExecutionPolicy Bypass -File " & Quote(installerScript), 1, False
Else
    MsgBox "YOLO Team Training Platform files are incomplete. Run the Windows installer again.", 16, "YOLO Team Training Platform"
End If

Function Quote(value)
    Quote = Chr(34) & value & Chr(34)
End Function
