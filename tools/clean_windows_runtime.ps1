param([string]$ResourceRoot = '')

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'build_path_safety.ps1')

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$buildRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot '.capy-build'))
$stageRoot = if ($ResourceRoot) { [IO.Path]::GetFullPath($ResourceRoot) } else {
    [IO.Path]::GetFullPath((Join-Path $buildRoot 'release/windows-x64'))
}
Assert-BuildPath -BuildRoot $buildRoot -Path $stageRoot
$pythonRoot = Join-Path $stageRoot 'runtimes/python'
if (-not (Test-Path -LiteralPath (Join-Path $pythonRoot 'python.exe'))) {
    throw 'Stage the embedded Python runtime before sanitizing release resources.'
}
Assert-BuildTreeSafe -BuildRoot $buildRoot -Path $stageRoot
# pip --target writes unused console launchers with the host interpreter's
# absolute path. The app calls bundled Python directly and does not use them.
$launcherRoot = Join-Path $pythonRoot 'Lib/site-packages/bin'
if (Test-Path -LiteralPath $launcherRoot) {
    Assert-BuildTreeSafe -BuildRoot $buildRoot -Path $launcherRoot
    Remove-Item -LiteralPath $launcherRoot -Recurse -Force
}
# Embedded Python ignores PYTHONDONTWRITEBYTECODE. Smoke commands use -B,
# and this final sweep also catches bytecode from subprocesses during tests.
$bytecode = @(Get-ChildItem -LiteralPath $stageRoot -Recurse -File -Force |
    Where-Object { $_.Extension -in @('.pyc', '.pyo') })
foreach ($file in $bytecode) {
    Assert-BuildPath -BuildRoot $buildRoot -Path $file.FullName
    Remove-Item -LiteralPath $file.FullName -Force
}
Write-Host "Sanitized staged resources: removed $($bytecode.Count) bytecode files and unused console launchers."
