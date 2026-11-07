# Guard against false readiness or duplicate service creation on occupied ports.
$ErrorActionPreference='Stop'
$taskRoot='D:\GeoMMAgent'
$taskLauncher=Join-Path $taskRoot 'start_demo.ps1'
$taskHash=(Get-FileHash -LiteralPath (Join-Path $taskRoot 'backend.py') -Algorithm SHA256).Hash.ToLowerInvariant()
$global:katLauncherTestStarted=0
$global:katLauncherTestHash=$taskHash
$global:katLauncherTestRuntimeHash=& 'D:\Miniconda3\envs\GeoMMA\python.exe' (Join-Path $taskRoot 'runtime_identity.py') backend
$taskPreviousArchive=$env:KAT_CASE_ARCHIVE_PATH
$taskArchive=Join-Path $taskRoot 'cache\launcher_contract_unused.sqlite3'
$env:KAT_CASE_ARCHIVE_PATH=$taskArchive
function Start-Process { $global:katLauncherTestStarted++; throw 'Test forbids process creation' }
function Get-NetTCPConnection { return @{LocalPort=$LocalPort;OwningProcess=999999} }
function Invoke-RestMethod {
    $state=@{service='kat-backend';runtime_root='D:\GeoMMAgent';backend_code_sha256=$global:katLauncherTestHash;
        backend_runtime_sha256=$global:katLauncherTestRuntimeHash;archive_database_path='D:\GeoMMAgent\cache\launcher_contract_unused.sqlite3';
        reporting_mode='local_facts_only';detector_online=$true;detector_runtime_compatible=$true}
    switch ($global:katLauncherTestScenario) {
        'other' {$state.service='other-service'}
        'stale' {$state.backend_runtime_sha256='old'}
        'wrong_root' {$state.runtime_root='D:\another-source'}
        'wrong_archive' {$state.archive_database_path='D:\other_archive.sqlite3'}
        'wrong_mode' {$state.reporting_mode='cloud_optional'}
        'wrong_detector' {$state.detector_runtime_compatible=$false}
        'offline_occupied' {$state.detector_online=$false;$state.detector_runtime_compatible=$false}
    }
    return $state
}
$taskChecks=@()
try {
    foreach ($taskScenario in @('other','stale','wrong_root','wrong_archive','wrong_mode','wrong_detector','offline_occupied')) {
        $global:katLauncherTestScenario=$taskScenario
        $taskExpected=switch ($taskScenario) {
            'wrong_archive' {'Existing backend is using a different archive*'}
            'wrong_mode' {'Existing service is not in the requested local-facts mode*'}
            'wrong_detector' {'Existing detector belongs to another source or runtime version*'}
            'offline_occupied' {'Detector port is occupied but its health is unavailable*'}
            default {'Backend port is running another deployment or an older runtime*'}
        }
        try {
            & $taskLauncher -NoBrowser -LocalFactsOnly -TimeoutSeconds 5
            throw 'Launcher unexpectedly succeeded'
        } catch {
            if ($_.Exception.Message -notlike $taskExpected) { throw }
            $taskChecks+=@{scenario=$taskScenario;expected_failure=$true;no_false_success=$true}
        }
    }
    if ($global:katLauncherTestStarted -ne 0) { throw 'Launcher started duplicate or forbidden processes' }
    $taskCache=Join-Path $taskRoot 'cache\engineering_hardening_20261007'
    @{passed=$true;checks=$taskChecks;processes_started=0;health_responses_mocked=$true;scope='launcher rejection contracts;actual recovery tested separately'} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskCache 'launcher_contract_verification.json') -Encoding utf8
    Write-Output 'PASS 7 mismatched or occupied-service launcher refusals;no duplicate processes'
} finally { $env:KAT_CASE_ARCHIVE_PATH=$taskPreviousArchive }
