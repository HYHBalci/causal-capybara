param([string]$ResourceRoot = '')

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'build_path_safety.ps1')

if ($env:OS -ne 'Windows_NT') {
    throw 'The bundled runtime can only be smoke-tested on Windows.'
}

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$stageRoot = if ($ResourceRoot) { [IO.Path]::GetFullPath($ResourceRoot) } else { Join-Path $repoRoot '.capy-build/release/windows-x64' }
$pythonExe = Join-Path $stageRoot 'runtimes/python/python.exe'
$buildRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot '.capy-build'))
$smokeHome = Join-Path $buildRoot 'release/smoke-home'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Run tools/stage_windows_runtime.ps1 first; missing $pythonExe"
}

Assert-BuildPath -BuildRoot $buildRoot -Path $smokeHome
if (Test-Path -LiteralPath $smokeHome) {
    Assert-BuildTreeSafe -BuildRoot $buildRoot -Path $smokeHome
    Remove-Item -LiteralPath $smokeHome -Recurse -Force
}
New-Item -ItemType Directory -Path $smokeHome -Force | Out-Null
# CAPY_HOME isolates settings, but the default study workspace is still under
# the user's home. Keep synthetic smoke studies inside the verified build tree.
$smokeSettings = @{ workspace = (Join-Path $smokeHome 'workspace'); allow_downloads = $false } | ConvertTo-Json
[IO.File]::WriteAllText((Join-Path $smokeHome 'settings.json'), $smokeSettings, [Text.UTF8Encoding]::new($false))

$oldCapyHome = $env:CAPY_HOME
$oldPythonPath = $env:PYTHONPATH
$oldBytecode = $env:PYTHONDONTWRITEBYTECODE
try {
    $env:CAPY_HOME = $smokeHome
    $env:PYTHONPATH = 'Z:\this-path-must-not-be-used'
    $env:PYTHONDONTWRITEBYTECODE = '1'

    & $pythonExe -B -c 'import sys; assert sys.version_info[:3] == (3, 13, 15), sys.version; import fastapi, uvicorn, numpy, scipy, pandas, pyarrow, sklearn, statsmodels, linearmodels, polars, duckdb, docx, capy_sidecar, capy_py; assert all("this-path-must-not-be-used" not in p for p in sys.path); print("Bundled interpreter and analysis imports OK")'
    if ($LASTEXITCODE -ne 0) { throw 'Bundled Python import smoke check failed.' }

    & $pythonExe -B -m capy_knowledge validate
    if ($LASTEXITCODE -ne 0) { throw 'Bundled public knowledge validation failed.' }
    & $pythonExe -B -c 'from capy_sidecar.assistant import knowledge; k = knowledge(); c = k.build_context("AIPW ATT overlap", max_bytes=65536); assert c["citations"] and c["groups"], c; print("Bundled assistant and cited knowledge context OK", c["snapshot_id"])'
    if ($LASTEXITCODE -ne 0) { throw 'Bundled assistant knowledge smoke check failed.' }

    $projectLines = @(& $pythonExe -B -m capy_sidecar example medicaid | Select-Object -Last 1)
    if ($LASTEXITCODE -ne 0 -or $projectLines.Count -eq 0) { throw 'Bundled Python could not create a worked example.' }
    $project = $projectLines[-1].Trim()
    $specLines = @(& $pythonExe -B -m capy_sidecar spec $project --design did --estimand ATT --treatment expanded --outcome uninsured_rate --unit state --time year --cluster state | Select-Object -Last 1)
    if ($LASTEXITCODE -ne 0 -or $specLines.Count -eq 0) { throw 'Bundled Python could not create a study specification.' }
    $spec = $specLines[-1].Trim()
    & $pythonExe -B -m capy_sidecar run $project $spec did.callaway_santanna did.twfe
    if ($LASTEXITCODE -ne 0) { throw 'Bundled Python could not estimate the worked example.' }
    & $pythonExe -B -m capy_sidecar validate $project
    if ($LASTEXITCODE -ne 0) { throw 'Bundled Python produced an invalid worked-example result.' }

    Write-Host 'Bundled Python 3.13.15 smoke check passed without using system Python packages.'
}
finally {
    $env:CAPY_HOME = $oldCapyHome
    $env:PYTHONPATH = $oldPythonPath
    $env:PYTHONDONTWRITEBYTECODE = $oldBytecode
}
