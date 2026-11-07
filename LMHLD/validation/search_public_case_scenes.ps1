param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('CL0002','KG0001','BR0001','Amyntaio','LasCruces','Hpakant')]
    [string]$CaseId,
    [ValidateSet('sentinel-2-l2a','sentinel-2-l1c')]
    [string]$Collection = 'sentinel-2-l2a',
    [ValidateSet('EarthSearch','PlanetaryComputer')]
    [string]$Provider = 'EarthSearch'
)
$ErrorActionPreference = 'Stop'
$caseRoot = Join-Path (Split-Path $PSScriptRoot -Parent) 'datasets\mine_benchmark_20261002'
if ($CaseId -in @('Amyntaio','LasCruces','Hpakant')) {
    $contextName = switch ($CaseId) { 'Amyntaio' { 'amyntaio_context.json' }; 'LasCruces' { 'las_cruces_context.json' }; 'Hpakant' { 'hpakant_context.json' } }
    $caseAudit = Get-Content -LiteralPath (Join-Path $caseRoot ('downloads\' + $contextName)) -Raw | ConvertFrom-Json
    $longitudeCenter = ($caseAudit.bbox_wgs84[0] + $caseAudit.bbox_wgs84[2]) / 2
    $latitudeCenter = ($caseAudit.bbox_wgs84[1] + $caseAudit.bbox_wgs84[3]) / 2
} else {
    $boundaryPath = Join-Path $caseRoot "quality_audit\$CaseId\published_boundary_wgs84.geojson"
    $caseAudit = Get-Content -LiteralPath (Join-Path $caseRoot "quality_audit\$CaseId\audit.json") -Raw | ConvertFrom-Json
    $boundary = Get-Content -LiteralPath $boundaryPath -Raw | ConvertFrom-Json
    $ring = $boundary.features[0].geometry.coordinates[0]
    $longitudeValues = @($ring | ForEach-Object { [double]$_[0] })
    $latitudeValues = @($ring | ForEach-Object { [double]$_[1] })
    $longitudeCenter = (($longitudeValues | Measure-Object -Minimum -Maximum).Minimum + ($longitudeValues | Measure-Object -Minimum -Maximum).Maximum) / 2
    $latitudeCenter = (($latitudeValues | Measure-Object -Minimum -Maximum).Minimum + ($latitudeValues | Measure-Object -Minimum -Maximum).Maximum) / 2
}
$caseDate = [datetime]::ParseExact($caseAudit.event_date, 'yyyy-MM-dd', [Globalization.CultureInfo]::InvariantCulture)
$daysBefore = switch ($CaseId) { 'LasCruces' { -25 }; 'Hpakant' { -60 }; default { -15 } }
$daysAfter = if ($CaseId -eq 'Hpakant') { 60 } else { 35 }
$requestBody = @{
    collections = @($Collection)
    limit = 100
    datetime = $caseDate.AddDays($daysBefore).ToString('yyyy-MM-dd') + 'T00:00:00Z/' + $caseDate.AddDays($daysAfter).ToString('yyyy-MM-dd') + 'T23:59:59Z'
    intersects = @{ type = 'Point'; coordinates = @($longitudeCenter, $latitudeCenter) }
} | ConvertTo-Json -Depth 6
$outputFolder = Join-Path $caseRoot 'scene_catalogues'
New-Item -ItemType Directory -Path $outputFolder -Force | Out-Null
$requestPath = Join-Path $outputFolder ($CaseId + '_' + $Collection + '_request.json')
[IO.File]::WriteAllText($requestPath, $requestBody, [Text.UTF8Encoding]::new($false))
$responseName = if ($Collection -eq 'sentinel-2-l2a') { $CaseId + '_earthsearch.json' } else { $CaseId + '_' + $Collection + '_earthsearch.json' }
if ($Provider -eq 'PlanetaryComputer') { $responseName = $CaseId + '_planetarycomputer.json' }
$responsePath = Join-Path $outputFolder $responseName
$searchUri = if ($Provider -eq 'PlanetaryComputer') { 'https://planetarycomputer.microsoft.com/api/stac/v1/search' } else { 'https://earth-search.aws.element84.com/v1/search' }
Invoke-WebRequest -Uri $searchUri -Method Post -ContentType 'application/json' -Body $requestBody -OutFile $responsePath -TimeoutSec 45
$catalogue = Get-Content -LiteralPath $responsePath -Raw | ConvertFrom-Json
$catalogue.features | ForEach-Object {
    [PSCustomObject]@{ Id=$_.id; Date=$_.properties.datetime; Cloud=$_.properties.'eo:cloud_cover'; HasSCL=(($null -ne $_.assets.scl) -or ($null -ne $_.assets.SCL)) }
} | Format-Table
