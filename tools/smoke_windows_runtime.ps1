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

$oldCapyHome = $env:CAPY_HOME
$oldPythonPath = $env:PYTHONPATH
$oldBytecode = $env:PYTHONDONTWRITEBYTECODE
try {
    $env:CAPY_HOME = $smokeHome
    $env:PYTHONPATH = 'Z:\this-path-must-not-be-used'
    $env:PYTHONDONTWRITEBYTECODE = '1'

    & $pythonExe -B -c 'import sys; assert sys.version_info[:3] == (3, 13, 15), sys.version; import fastapi, uvicorn, numpy, scipy, pandas, pyarrow, sklearn, statsmodels, linearmodels, polars, duckdb, docx, capy_sidecar, capy_py; assert all("this-path-must-not-be-used" not in p for p in sys.path); print("Bundled interpreter and analysis imports OK")'
    if ($LASTEXITCODE -ne 0) { throw 'Bundled Python import smoke check failed.' }

    $project = (& $pythonExe -B -m capy_sidecar example medicaid 2>$null | Select-Object -Last 1).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $project) { throw 'Bundled Python could not create a worked example.' }
    $spec = (& $pythonExe -B -m capy_sidecar spec $project --design did --estimand ATT --treatment expanded --outcome uninsured_rate --unit state --time year --cluster state | Select-Object -Last 1).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $spec) { throw 'Bundled Python could not create a study specification.' }
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
