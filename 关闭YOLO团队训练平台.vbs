Option Explicit

Dim shell, files, root, pythonExe
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
root = files.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
pythonExe = files.BuildPath(root, ".venv\Scripts\python.exe")

If files.FileExists(pythonExe) Then
    shell.Run Quote(pythonExe) & " " & Quote(files.BuildPath(root, "annotation_service.py")) & " stop", 0, True
    shell.Run Quote(pythonExe) & " " & Quote(files.BuildPath(root, "panel_service.py")) & " stop", 0, True
End If

Function Quote(value)
    Quote = Chr(34) & value & Chr(34)
End Function
