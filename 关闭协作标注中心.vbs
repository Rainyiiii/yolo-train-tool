Option Explicit

Dim shell, files, root, pythonExe, serviceScript
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
root = files.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
pythonExe = files.BuildPath(root, ".venv\Scripts\python.exe")
serviceScript = files.BuildPath(root, "annotation_service.py")

If files.FileExists(pythonExe) And files.FileExists(serviceScript) Then
    shell.Run Quote(pythonExe) & " " & Quote(serviceScript) & " stop", 0, True
End If

Function Quote(value)
    Quote = Chr(34) & value & Chr(34)
End Function
