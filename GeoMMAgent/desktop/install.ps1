param([string]$BasePython = 'D:\Miniconda3\envs\GeoMMA\python.exe')
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
if ([IO.Path]::GetPathRoot($taskRoot) -ne 'D:\') { throw 'Desktop runtime must stay on D:.' }
$taskCache = Join-Path $taskRoot 'cache\runtime'
$env:TEMP = Join-Path $taskCache 'tmp'; $env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $taskCache 'pip'
New-Item -ItemType Directory -Path $env:TEMP,$env:PIP_CACHE_DIR -Force | Out-Null
$taskEnvironment = Join-Path $taskRoot 'cache\desktop_runtime'
$taskPython = Join-Path $taskEnvironment 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    & $BasePython -B -m venv $taskEnvironment
    if ($LASTEXITCODE -ne 0) { throw 'Desktop environment creation failed.' }
}
& $taskPython -B -m pip install --disable-pip-version-check -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Desktop dependency installation failed.' }
Write-Host 'Desktop runtime ready. Original agent/detector environments were not changed.'
