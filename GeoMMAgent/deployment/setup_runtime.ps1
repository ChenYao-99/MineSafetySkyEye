param(
    [string]$AgentPython = 'D:\Miniconda3\envs\GeoMMA\python.exe',
    [string]$DetectorPython = 'D:\Miniconda3\envs\lmhld\python.exe',
    [string]$Destination = 'D:\GeoMMAgent\runtime_deployment'
)
$ErrorActionPreference='Stop'
$taskDestination=[IO.Path]::GetFullPath($Destination)
if ([IO.Path]::GetPathRoot($taskDestination) -ne 'D:\') { throw 'Deployment and caches must stay on D:.' }
if ($taskDestination -in @('D:\GeoMMAgent','D:\LMHLD','D:\Miniconda3')) { throw 'Use a dedicated deployment directory.' }
foreach ($taskPython in @($AgentPython,$DetectorPython)) { if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) { throw ('Python interpreter missing: '+$taskPython) } }
New-Item -ItemType Directory -Path $taskDestination,(Join-Path $taskDestination 'pip_cache'),(Join-Path $taskDestination 'tmp') -Force | Out-Null
$env:TEMP=Join-Path $taskDestination 'tmp';$env:TMP=$env:TEMP;$env:PIP_CACHE_DIR=Join-Path $taskDestination 'pip_cache'
$env:PIP_CONFIG_FILE='NUL';$env:PIP_INDEX_URL='https://pypi.org/simple';$env:PYTHONIOENCODING='utf-8'
$taskAgent=Join-Path $taskDestination 'GeoMMA';$taskDetector=Join-Path $taskDestination 'lmhld'
foreach ($taskPair in @(@{Python=$AgentPython;Target=$taskAgent},@{Python=$DetectorPython;Target=$taskDetector})) {
    if (-not (Test-Path -LiteralPath (Join-Path $taskPair.Target 'Scripts\python.exe'))) {
        & $taskPair.Python -m venv --copies $taskPair.Target
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
    }
}
& (Join-Path $taskAgent 'Scripts\python.exe') -m pip install --no-deps -r (Join-Path $PSScriptRoot 'requirements.GeoMMA.lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Agent dependency installation failed.' }
& (Join-Path $taskDetector 'Scripts\python.exe') -m pip install --no-deps -r (Join-Path $PSScriptRoot 'requirements.lmhld.cuda124.lock.txt') --extra-index-url 'https://download.pytorch.org/whl/cu124'
if ($LASTEXITCODE -ne 0) { throw 'Detector CUDA dependency installation failed.' }
foreach ($taskEnvironment in @($taskAgent,$taskDetector)) {
    & (Join-Path $taskEnvironment 'Scripts\python.exe') -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
}
& (Join-Path $taskDetector 'Scripts\python.exe') -c "import struct,torch; assert struct.calcsize('P')==8; assert torch.cuda.is_available(); assert torch.version.cuda=='12.4'; print('CUDA12.4 capability verified')"
if ($LASTEXITCODE -ne 0) { throw 'GPU build/driver capability check failed. Do not report GPU deployment success.' }
Write-Host ('Independent dependency environments ready on D:. Agent: '+$taskAgent+'; detector: '+$taskDetector)
Write-Host 'This does not install NVIDIA drivers or prove deployment on another physical machine.'
