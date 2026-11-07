param(
    [switch]$CheckOnly,
    [switch]$LocalFactsOnly,
    [ValidateRange(1024,65535)][int]$BackendPort = 5100,
    [ValidateRange(1024,65535)][int]$DetectorPort = 5101
)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$taskDesktopPython = Join-Path $taskRoot 'cache\desktop_runtime\Scripts\pythonw.exe'
$taskDesktopApp = Join-Path $taskRoot 'desktop\app.py'
if (-not (Test-Path -LiteralPath $taskDesktopPython -PathType Leaf)) {
    throw 'Desktop runtime missing. Run desktop\install.ps1 first. The original web launcher is still available.'
}
$taskArguments = @('-B', ('"' + $taskDesktopApp + '"'), '--backend-port', [string]$BackendPort, '--detector-port', [string]$DetectorPort)
if ($LocalFactsOnly) { $taskArguments += '--local-facts-only' }
if ($CheckOnly) {
    $taskConsolePython = Join-Path $taskRoot 'cache\desktop_runtime\Scripts\python.exe'
    & $taskConsolePython -B $taskDesktopApp --check-only --backend-port $BackendPort --detector-port $DetectorPort
    exit $LASTEXITCODE
}
# pythonw has no console. This is the user-requested interactive app, so hiding
# its startup window would also hide the first WinForms/WebView2 window.
Start-Process -FilePath $taskDesktopPython -ArgumentList $taskArguments -WorkingDirectory $taskRoot -WindowStyle Normal | Out-Null
