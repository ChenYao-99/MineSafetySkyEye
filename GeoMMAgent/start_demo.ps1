param(
    [switch]$NoBrowser,
    [switch]$CheckOnly,
    [switch]$LocalFactsOnly,
    [string]$AgentPython = 'D:\Miniconda3\envs\GeoMMA\python.exe',
    [string]$DetectorPython = 'D:\Miniconda3\envs\lmhld\python.exe',
    [string]$DetectorDirectory = 'D:\LMHLD',
    [ValidateRange(1024,65535)][int]$BackendPort = 5100,
    [ValidateRange(1024,65535)][int]$DetectorPort = 5101,
    [ValidateRange(5,180)][int]$TimeoutSeconds = 90
)
$ErrorActionPreference = 'Stop'
$taskSystemModules = [IO.Path]::Combine($PSHOME, 'Modules')
$env:PSModulePath = $taskSystemModules + [IO.Path]::PathSeparator + $env:PSModulePath
$taskRoot = [IO.Path]::GetFullPath($PSScriptRoot)
if ([IO.Path]::GetPathRoot($taskRoot) -ne 'D:\') { throw 'Project runtime and caches must stay on D:.' }
$taskPython = [IO.Path]::GetFullPath($AgentPython)
$taskDetectorPython = [IO.Path]::GetFullPath($DetectorPython)
$taskDetectorRoot = [IO.Path]::GetFullPath($DetectorDirectory)
if ([IO.Path]::GetPathRoot($taskDetectorRoot) -ne 'D:\') { throw 'Detector source and caches must stay on D:.' }
if ($BackendPort -eq $DetectorPort) { throw 'Backend and detector ports must differ.' }
$taskBackend = Join-Path $taskRoot 'backend.py'
foreach ($required in @($taskPython,$taskDetectorPython,$taskBackend,(Join-Path $taskDetectorRoot 'detector_service.py'),(Join-Path $taskRoot '矿安天眼_前端演示(后端版).html'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw ('Required file missing: ' + $required) }
}
$taskURL = 'http://127.0.0.1:' + $BackendPort
$taskDetectorURL = 'http://127.0.0.1:' + $DetectorPort
$taskCodeHash = (Get-FileHash -LiteralPath $taskBackend -Algorithm SHA256).Hash.ToLowerInvariant()
$taskRuntimeHash = & $taskPython (Join-Path $taskRoot 'runtime_identity.py') backend
if ($LASTEXITCODE -ne 0) { throw 'Runtime source identity could not be calculated.' }
if ($CheckOnly) {
    @{status='local_files_present';frontend=($taskURL + '/');backend_code_sha256=$taskCodeHash;
      runtime_drive='D:';detector_directory=$taskDetectorRoot;backend_port=$BackendPort;detector_port=$DetectorPort;
      starts_services=$false;checks_model_accuracy=$false} | ConvertTo-Json
    exit 0
}
function Read-KatHealth {
    # Backend health includes a bounded 3-second detector probe when it is down.
    try { return Invoke-RestMethod -Uri ($taskURL + '/api/health') -TimeoutSec 5 }
    catch { return $null }
}
$taskDoctor = Join-Path $taskRoot 'deployment\doctor.py'
if (Test-Path -LiteralPath $taskDoctor -PathType Leaf) {
    & $taskPython $taskDoctor --role agent --source $taskRoot
    if ($LASTEXITCODE -ne 0) { throw 'Backend deployment preflight failed;no service was started.' }
    & $taskDetectorPython $taskDoctor --role detector --source $taskDetectorRoot
    if ($LASTEXITCODE -ne 0) { throw 'Detector/default-model deployment preflight failed;no service was started.' }
}
$taskHealth = Read-KatHealth
if ($null -ne $taskHealth) {
    if ($taskHealth.service -ne 'kat-backend' -or $taskHealth.runtime_root -ne $taskRoot -or $taskHealth.backend_code_sha256 -ne $taskCodeHash -or $taskHealth.backend_runtime_sha256 -ne $taskRuntimeHash) {
        throw 'Backend port is running another deployment or an older runtime. No duplicate service was started.'
    }
    if ($env:KAT_CASE_ARCHIVE_PATH -and $taskHealth.archive_database_path -ne [IO.Path]::GetFullPath($env:KAT_CASE_ARCHIVE_PATH)) {
        throw 'Existing backend is using a different archive. Stop project services before activating the requested archive path.'
    }
    if ($LocalFactsOnly -and $taskHealth.reporting_mode -ne 'local_facts_only') {
        throw 'Existing service is not in the requested local-facts mode.'
    }
    if ($taskHealth.detector_online -and -not $taskHealth.detector_runtime_compatible) {
        throw 'Existing detector belongs to another source or runtime version. Stop the old project services before restarting.'
    }
    if (-not $taskHealth.detector_online -and -not $taskHealth.detector_recovery_enabled) {
        $taskDetectorListener = Get-NetTCPConnection -State Listen -LocalPort $DetectorPort -ErrorAction SilentlyContinue
        if ($taskDetectorListener) {
            throw 'Detector port is occupied but its health is unavailable. No additional detector was started.'
        }
        $env:KAT_BIND_HOST='127.0.0.1';$env:PYTHONIOENCODING='utf-8';$env:YOLO_AUTOINSTALL='false'
        $taskRecoveryLogs=Join-Path $taskRoot 'cache\launcher'
        New-Item -ItemType Directory -Path $taskRecoveryLogs -Force | Out-Null
        $taskRecoveryStamp=Get-Date -Format 'yyyyMMdd_HHmmss_fff'
        $taskDetectorArguments=@('-u',('"'+(Join-Path $taskDetectorRoot 'detector_service.py')+'"'),[string]$DetectorPort)
        $taskRecoveredDetector=Start-Process -FilePath $taskDetectorPython -ArgumentList $taskDetectorArguments -WorkingDirectory $taskDetectorRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRecoveryLogs ($taskRecoveryStamp+'_detector_recovery.stdout.log')) -RedirectStandardError (Join-Path $taskRecoveryLogs ($taskRecoveryStamp+'_detector_recovery.stderr.log'))
        Write-Host ('Recovering stopped detector. Logs: '+$taskRecoveryLogs)
    }
} else {
    $taskListener = Get-NetTCPConnection -State Listen -LocalPort $BackendPort -ErrorAction SilentlyContinue
    if ($taskListener) { throw 'Backend port is occupied but the health endpoint is unavailable. No duplicate backend was started.' }
    $taskDetectorListener = Get-NetTCPConnection -State Listen -LocalPort $DetectorPort -ErrorAction SilentlyContinue
    if ($taskDetectorListener) {
        try { $taskDetectorHealth = Invoke-RestMethod -Uri ($taskDetectorURL+'/api/health') -TimeoutSec 2 }
        catch { throw 'Detector port is occupied without a valid health endpoint.' }
        $taskExpectedDetectorHash = & $taskDetectorPython (Join-Path $taskDetectorRoot 'runtime_identity.py')
        if ($LASTEXITCODE -ne 0 -or $taskDetectorHealth.runtime_root -ne $taskDetectorRoot -or $taskDetectorHealth.runtime_sha256 -ne $taskExpectedDetectorHash) {
            throw 'Detector port belongs to a different deployment. No backend was started.'
        }
    }
    $env:PORT = [string]$BackendPort; $env:DETECTOR_PORT = [string]$DetectorPort; $env:YOLO_AUTOINSTALL = 'false'
    $env:KAT_LMHLD_DIR = $taskDetectorRoot; $env:KAT_DETECTOR_PY = $taskDetectorPython
    $env:KAT_BIND_HOST = '127.0.0.1'
    $env:KAT_LOCAL_FACTS_ONLY = $(if($LocalFactsOnly){'1'}else{'0'})
    $env:PYTHONIOENCODING = 'utf-8'
    $taskLogDir = Join-Path $taskRoot 'cache\launcher'
    New-Item -ItemType Directory -Path $taskLogDir -Force | Out-Null
    $taskLogStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $taskStdout = Join-Path $taskLogDir ($taskLogStamp + '_backend.stdout.log')
    $taskStderr = Join-Path $taskLogDir ($taskLogStamp + '_backend.stderr.log')
    $taskArguments = @('-u', ('"' + $taskBackend + '"'))
    $taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $taskStdout -RedirectStandardError $taskStderr
    Write-Host ('Backend starting. Logs: ' + $taskLogDir)
}
$taskDeadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
do {
    $taskHealth = Read-KatHealth
    if ($taskHealth -and $taskHealth.service -eq 'kat-backend' -and $taskHealth.runtime_root -eq $taskRoot -and $taskHealth.backend_code_sha256 -eq $taskCodeHash -and $taskHealth.backend_runtime_sha256 -eq $taskRuntimeHash -and $taskHealth.detector_online -and $taskHealth.detector_runtime_compatible) {
        $taskDetectorHealth = Invoke-RestMethod -Uri ($taskDetectorURL+'/api/health') -TimeoutSec 2
        if ($taskDetectorHealth.runtime_root -ne $taskDetectorRoot) { throw 'Detector is serving a different source directory.' }
        if ($LocalFactsOnly -and $taskHealth.reporting_mode -ne 'local_facts_only') { throw 'Existing service is not in the requested local-facts mode.' }
        Write-Host ('Ready: ' + $taskURL + '/')
        Write-Host 'Detector weights are loaded on first use; this startup check is not an accuracy validation.'
        if (-not $NoBrowser) { Start-Process ($taskURL + '/') }
        exit 0
    }
    if ($taskProcess -and $taskProcess.HasExited) { throw ('Backend stopped during startup. Inspect: ' + $taskStderr) }
    if ($taskRecoveredDetector -and $taskRecoveredDetector.HasExited) { throw 'Recovered detector stopped during startup. Inspect cache\launcher recovery logs.' }
    Start-Sleep -Milliseconds 500
} while ([DateTime]::UtcNow -lt $taskDeadline)
throw ('Services did not become ready within ' + $TimeoutSeconds + ' seconds. Inspect '+$taskRoot+'\cache\launcher and '+$taskDetectorURL+'/api/health . No success was reported.')
