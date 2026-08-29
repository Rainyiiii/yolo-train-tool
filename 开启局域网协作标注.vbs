Option Explicit

Dim shell, files, root, pythonExe, serviceScript
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
root = files.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
pythonExe = files.BuildPath(root, ".venv\Scripts\python.exe")
serviceScript = files.BuildPath(root, "annotation_service.py")

If Not files.FileExists(pythonExe) Or Not files.FileExists(serviceScript) Then
    MsgBox "Python runtime was not found. Run the one-click installer first.", 16, "YOLO Team Training Platform"
    WScript.Quit 1
End If

shell.Run Quote(pythonExe) & " " & Quote(serviceScript) & " start --share", 0, False

Function Quote(value)
    Quote = Chr(34) & value & Chr(34)
End Function
