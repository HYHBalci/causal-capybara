param([string]$ResourceRoot = '')

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$buildRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot '.capy-build'))
$buildPrefix = $buildRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
$stageRoot = if ($ResourceRoot) { [IO.Path]::GetFullPath($ResourceRoot) } else {
    [IO.Path]::GetFullPath((Join-Path $buildRoot 'release/windows-x64'))
}
if (-not $stageRoot.StartsWith($buildPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Refusing to sanitize resources outside the build root.'
}
$ancestor = $stageRoot
while ($ancestor.StartsWith($buildPrefix, [StringComparison]::OrdinalIgnoreCase) -or $ancestor -eq $buildRoot) {
    if ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw 'Refusing to sanitize a build directory reached through a reparse point.'
    }
    $ancestor = Split-Path -Parent $ancestor
}
$pythonRoot = Join-Path $stageRoot 'runtimes/python'
if (-not (Test-Path -LiteralPath (Join-Path $pythonRoot 'python.exe'))) {
    throw 'Stage the embedded Python runtime before sanitizing release resources.'
}
$stagePrefix = $stageRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
function Assert-SafeTarget([string]$Path) {
    $fullPath = [IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith($stagePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Refusing to remove a file outside the staged resource directory.'
    }
    $ancestor = $fullPath
    while ($ancestor.StartsWith($stagePrefix, [StringComparison]::OrdinalIgnoreCase) -or $ancestor -eq $stageRoot) {
        if ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw 'Refusing to remove resources reached through a reparse point.'
        }
        $ancestor = Split-Path -Parent $ancestor
    }
}
# pip --target writes unused console launchers with the host interpreter's
# absolute path. The app calls bundled Python directly and does not use them.
$launcherRoot = Join-Path $pythonRoot 'Lib/site-packages/bin'
if (Test-Path -LiteralPath $launcherRoot) {
    Assert-SafeTarget $launcherRoot
    foreach ($child in Get-ChildItem -LiteralPath $launcherRoot -Recurse -Force) {
        Assert-SafeTarget $child.FullName
    }
    Remove-Item -LiteralPath $launcherRoot -Recurse -Force
}
# Embedded Python ignores PYTHONDONTWRITEBYTECODE. Smoke commands use -B,
# and this final sweep also catches bytecode from subprocesses during tests.
$bytecode = @(Get-ChildItem -LiteralPath $stageRoot -Recurse -File -Force |
    Where-Object { $_.Extension -in @('.pyc', '.pyo') })
foreach ($file in $bytecode) {
    Assert-SafeTarget $file.FullName
    Remove-Item -LiteralPath $file.FullName -Force
}
Write-Host "Sanitized staged resources: removed $($bytecode.Count) bytecode files and unused console launchers."
