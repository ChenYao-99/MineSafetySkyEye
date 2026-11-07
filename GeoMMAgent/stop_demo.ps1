param(
    [string]$DetectorDirectory = 'D:\LMHLD',
    [ValidateRange(1024,65535)][int]$BackendPort = 5100,
    [ValidateRange(1024,65535)][int]$DetectorPort = 5101
)
$ErrorActionPreference = 'Stop'
$taskSystemModules = [IO.Path]::Combine($PSHOME, 'Modules')
$env:PSModulePath = $taskSystemModules + [IO.Path]::PathSeparator + $env:PSModulePath
$taskRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$taskServices = @{
    $BackendPort = (Join-Path $taskRoot 'backend.py')
    $DetectorPort = (Join-Path ([IO.Path]::GetFullPath($DetectorDirectory)) 'detector_service.py')
}
foreach ($taskPort in @($BackendPort,$DetectorPort)) {
    $taskListeners = @(Get-NetTCPConnection -State Listen -LocalPort $taskPort -ErrorAction SilentlyContinue)
    foreach ($taskOwner in @($taskListeners.OwningProcess | Select-Object -Unique)) {
        if (-not $taskOwner) { continue }
        $taskProcess = Get-CimInstance Win32_Process -Filter ("ProcessId = " + $taskOwner)
        $taskExpected = [regex]::Escape($taskServices[$taskPort]) + '("|\s|$)'
        if ($taskProcess.Name -eq 'python.exe' -and $taskProcess.CommandLine -match $taskExpected) {
            Stop-Process -Id $taskOwner -Force
            Write-Host ('Stopped project service on port ' + $taskPort + ', PID ' + $taskOwner)
        } else {
            Write-Host ('Port ' + $taskPort + ' belongs to another process; left running.')
        }
    }
}
