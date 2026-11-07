$ErrorActionPreference = 'Stop'
$caseRoot = [IO.Path]::GetFullPath('D:\LMHLD\datasets\mine_benchmark_20261002\reserved_candidates\TR_COP2024')
$nestedRoot = [IO.Path]::GetFullPath((Join-Path $caseRoot 'TR_COP2024'))
$registryPath = 'D:\LMHLD\datasets\mine_benchmark_20261002\mine_case_registry.json'
$evidencePath = Join-Path $caseRoot 'directory_normalisation.json'
if (-not $nestedRoot.StartsWith($caseRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected directory' }
foreach ($path in @($caseRoot, $nestedRoot)) {
    if ((Get-Item -LiteralPath $path).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Reparse directory rejected' }
}
if (Test-Path -LiteralPath $evidencePath) { throw 'Directory normalisation already recorded' }
$allowed = @('catalogue.json','acquisition_plan.json','acquisition.json','pre_BGRN_10m.tif','post_BGRN_10m.tif','pre_SCL_10m.tif','post_SCL_10m.tif','pre_product.xml','post_product.xml')
$files = @(Get-ChildItem -LiteralPath $nestedRoot -Force)
foreach ($file in $files) {
    if ($file.PSIsContainer -or ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $file.Name -notin $allowed) { throw 'Unexpected nested member' }
    if (Test-Path -LiteralPath (Join-Path $caseRoot $file.Name)) { throw 'Existing destination rejected' }
}
$registry = Get-Content -LiteralPath $registryPath -Raw | ConvertFrom-Json -AsHashtable
$mapping = @()
foreach ($file in $files) {
    $digest = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $oldKey = 'reserved_candidates/TR_COP2024/TR_COP2024/' + $file.Name
    $newKey = 'reserved_candidates/TR_COP2024/' + $file.Name
    if ($registry.additional_frozen_assets.ContainsKey($oldKey)) {
        if ($registry.additional_frozen_assets[$oldKey] -ne $digest) { throw 'Frozen hash mismatch' }
        $registry.additional_frozen_assets.Remove($oldKey)
        $registry.additional_frozen_assets[$newKey] = $digest
    }
    $mapping += @{from=$file.FullName; to=(Join-Path $caseRoot $file.Name); sha256=$digest}
}
foreach ($entry in $mapping) {
    Move-Item -LiteralPath $entry.from -Destination $entry.to
    if ((Get-FileHash -LiteralPath $entry.to -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw 'Bytes changed during move' }
}
[IO.File]::WriteAllText($registryPath, ($registry | ConvertTo-Json -Depth 100) + "`n", [Text.UTF8Encoding]::new($false))
[IO.File]::WriteAllText($evidencePath, (@{reason='Removed one accidentally duplicated output directory; source bytes and holdout role preserved';files=$mapping;model_predictions=0} | ConvertTo-Json -Depth 20) + "`n", [Text.UTF8Encoding]::new($false))
# Only an empty, exact verified child directory is removed, without recursion.
if (@(Get-ChildItem -LiteralPath $nestedRoot -Force).Count -ne 0) { throw 'Directory is not empty' }
Remove-Item -LiteralPath $nestedRoot
Write-Output 'Nine source files normalised within the same D: case directory; all bytes preserved'
