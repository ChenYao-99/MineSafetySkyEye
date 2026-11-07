param(
    [Parameter(Mandatory=$true)][int]$ProcessId,
    [Parameter(Mandatory=$true)][ValidateSet('capture','close')][string]$Mode,
    [Parameter(Mandatory=$true)][string]$Destination
)
$ErrorActionPreference = 'Stop'
$taskTarget = [IO.Path]::GetFullPath($Destination)
$taskAllowed = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\cache')) + '\'
if (-not $taskTarget.StartsWith($taskAllowed,[StringComparison]::OrdinalIgnoreCase)) {
    throw 'Acceptance output must remain in the owned D-drive validation directory.'
}
$taskOwner = Get-CimInstance Win32_Process -Filter ('ProcessId = '+$ProcessId)
if ($taskOwner.CommandLine -notlike '*D:\GeoMMAgent\desktop\app.py*') { throw 'Not the owned desktop app.' }
$taskProcess = Get-Process -Id $ProcessId
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public class KatWindowCapture {
    [StructLayout(LayoutKind.Sequential)] public struct Rect { public int Left,Top,Right,Bottom; }
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr window,out Rect rect);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr window);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr window,int state);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr window,IntPtr dc,uint flags);
    private delegate bool WindowCallback(IntPtr window,IntPtr parameter);
    [DllImport("user32.dll")] private static extern bool EnumWindows(WindowCallback callback,IntPtr parameter);
    [DllImport("user32.dll")] private static extern uint GetWindowThreadProcessId(IntPtr window,out uint pid);
    [DllImport("user32.dll",CharSet=CharSet.Unicode)] private static extern int GetClassName(IntPtr window,StringBuilder name,int capacity);
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr window,uint message,IntPtr wParam,IntPtr lParam);
    public static IntPtr FindOwnedWindow(uint processId) {
        IntPtr found=IntPtr.Zero;
        EnumWindows(delegate(IntPtr window,IntPtr parameter) {
            uint pid; GetWindowThreadProcessId(window,out pid);
            if(pid==processId) {
                var name=new StringBuilder(256);GetClassName(window,name,256);
                if(name.ToString().StartsWith("WindowsForms10.Window")) {found=window;return false;}
            }
            return true;
        },IntPtr.Zero);
        return found;
    }
}
'@
[KatWindowCapture]::SetProcessDPIAware() | Out-Null
$taskHandle = $taskProcess.MainWindowHandle
if ($taskHandle -eq [IntPtr]::Zero) { $taskHandle = [KatWindowCapture]::FindOwnedWindow($ProcessId) }
if ($taskHandle -eq [IntPtr]::Zero) { throw 'Owned desktop main window has no handle.' }
if ($Mode -eq 'close') {
    if (-not [KatWindowCapture]::PostMessage($taskHandle,16,[IntPtr]::Zero,[IntPtr]::Zero)) { throw 'Cannot request desktop window close.' }
    Write-Output 'Owned desktop window close requested.'
    exit 0
}
[KatWindowCapture]::ShowWindow($taskHandle,5) | Out-Null
Start-Sleep -Milliseconds 400
$taskRect = New-Object KatWindowCapture+Rect
[KatWindowCapture]::GetWindowRect($taskHandle,[ref]$taskRect) | Out-Null
$taskWidth = $taskRect.Right-$taskRect.Left; $taskHeight = $taskRect.Bottom-$taskRect.Top
$taskBitmap = New-Object System.Drawing.Bitmap($taskWidth,$taskHeight)
$taskGraphics = [System.Drawing.Graphics]::FromImage($taskBitmap)
try {
    $taskDc = $taskGraphics.GetHdc()
    try { $taskPrinted = [KatWindowCapture]::PrintWindow($taskHandle,$taskDc,2) }
    finally { $taskGraphics.ReleaseHdc($taskDc) }
    if (-not $taskPrinted) { throw 'Owned window capture failed; screen capture is not used as fallback.' }
    $taskBitmap.Save($taskTarget,[System.Drawing.Imaging.ImageFormat]::Png)
} finally { $taskGraphics.Dispose(); $taskBitmap.Dispose() }
Write-Output 'Owned desktop window captured.'
