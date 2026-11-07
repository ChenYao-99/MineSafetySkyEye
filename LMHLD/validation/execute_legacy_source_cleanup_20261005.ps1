$ErrorActionPreference = 'Stop'
$taskDirectory = 'D:\LMHLD\cache\source_cleanup_20261005'
$planPath = Join-Path $taskDirectory 'plan.json'
$receiptPath = Join-Path $taskDirectory 'deleted_files.json'
if (Test-Path -LiteralPath $receiptPath) { throw 'Cleanup receipt already exists; verify or restore instead of deleting twice.' }
$plan = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
$expectedRoots = @('D:\GeoMMAgent','D:\LMHLD')
if ($plan.schema_version -ne 1 -or $plan.source_data_labels_protocols_study_results_and_current_weights_removed) { throw 'Unexpected cleanup plan schema or protected artifact policy.' }
$archive = (Resolve-Path -LiteralPath $plan.archive.path).Path
if (-not $archive.StartsWith($taskDirectory + '\',[System.StringComparison]::OrdinalIgnoreCase)) { throw 'Archive escaped the exact D: task directory.' }
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $plan.archive.sha256) { throw 'Verified recovery archive changed.' }
$rows = @($plan.archived_files_to_remove) + @($plan.regenerable_bytecode_to_remove)
foreach ($row in $rows) {
    $absolute = (Resolve-Path -LiteralPath $row.path).Path
    $root = [string]$row.root
    if ($root -notin $expectedRoots -or -not $absolute.StartsWith($root + '\',[System.StringComparison]::OrdinalIgnoreCase)) { throw 'Deletion target is outside authorised D: workspaces.' }
    $item = Get-Item -LiteralPath $absolute
    if ($item.PSIsContainer -or ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) { throw 'Only exact ordinary files may be removed.' }
    $ancestor = $item.Directory
    while ($null -ne $ancestor) {
        if ($ancestor.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw 'A deletion ancestor is a reparse point.' }
        if ($ancestor.FullName.Equals($root,[System.StringComparison]::OrdinalIgnoreCase)) { break }
        $ancestor = $ancestor.Parent
    }
    if ($null -eq $ancestor) { throw 'Could not verify the exact workspace ancestor.' }
    $relative = $absolute.Substring($root.Length + 1)
    if ($relative -match '^(datasets|models_prior|竞赛材料|\.git|\.agents|\.aws|\.codex)\\' -or $item.Name.StartsWith('.env')) { throw 'Protected data, credentials or repository metadata entered cleanup.' }
    if ($row.kind -eq 'regenerable_project_bytecode' -and ($item.Extension -ne '.pyc' -or $item.Directory.Name -ne '__pycache__')) { throw 'Unexpected bytecode target.' }
    if ((Get-FileHash -LiteralPath $absolute -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256 -or $item.Length -ne $row.bytes) { throw 'File changed since verified archive/plan; no deletion performed.' }
}
# Preserve current source bytes and deployment assets before any removals.
foreach ($property in $plan.protected_remaining_source_sha256.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath $property.Name -Algorithm SHA256).Hash.ToLowerInvariant() -ne $property.Value) { throw 'Remaining project source changed during audit.' }
}
foreach ($property in $plan.protected_runtime_artifact_sha256.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath $property.Name -Algorithm SHA256).Hash.ToLowerInvariant() -ne $property.Value) { throw 'Current model/deployment artifact changed during audit.' }
}
$receipt = [pscustomobject]@{
    created_at = (Get-Date).ToString('s')
    plan_sha256 = (Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash.ToLowerInvariant()
    archive = $plan.archive
    deleted = @()
    removed_bytes = 0
    raw_data_labels_current_weights_or_credentials_removed = $false
    policy = 'Exact ordinary files only; sources/debug outputs recovered from verified ZIP;bytecode regenerated from retained Python.'
}
$receipt | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
foreach ($row in $rows) {
    Remove-Item -LiteralPath $row.path
    if (Test-Path -LiteralPath $row.path) { throw 'Exact-file cleanup did not complete.' }
    $receipt.deleted += $row
    $receipt.removed_bytes += $row.bytes
    $receipt | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
}
[pscustomobject]@{
    archived_sources_and_debug_outputs = @($plan.archived_files_to_remove).Count
    removed_bytecode_files = @($plan.regenerable_bytecode_to_remove).Count
    removed_total_files = @($receipt.deleted).Count
    removed_MiB = [math]::Round($receipt.removed_bytes/1MB,2)
    net_saved_MiB_before_bytecode_regeneration = [math]::Round(($receipt.removed_bytes - $plan.archive.bytes)/1MB,2)
    receipt = $receiptPath
} | ConvertTo-Json
