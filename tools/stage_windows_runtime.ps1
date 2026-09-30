param(
    [string]$HostPython = 'python'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($env:OS -ne 'Windows_NT') {
    throw 'The bundled runtime can only be staged on Windows.'
}

# The ZIP SHA-256 is published on python.org's Python 3.13.15 release page.
$pythonVersion = '3.13.15'
$pythonZipName = "python-$pythonVersion-embed-amd64.zip"
$pythonZipSha256 = 'd1f04d990aee1253d8569e8e5104e30fa9f5fa830899f14843448872d936a2cf'
$pythonUrl = "https://www.python.org/ftp/python/$pythonVersion/$pythonZipName"

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$buildRoot = Join-Path $repoRoot '.capy-build'
$stageRoot = Join-Path $buildRoot 'release/windows-x64'
$pythonRoot = Join-Path $stageRoot 'runtimes/python'
$cacheRoot = Join-Path $buildRoot 'cache'
$pythonZip = Join-Path $cacheRoot $pythonZipName

function Assert-WithinBuildRoot([string]$Path) {
    $fullBuild = [IO.Path]::GetFullPath($buildRoot).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $fullPath = [IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith($fullBuild, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to change a path outside the build root: $fullPath"
    }
}

Assert-WithinBuildRoot $stageRoot
Assert-WithinBuildRoot $cacheRoot

New-Item -ItemType Directory -Path $cacheRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath $pythonZip)) {
    Write-Host "Downloading official CPython $pythonVersion x64 embeddable runtime..."
    Invoke-WebRequest -Uri $pythonUrl -OutFile $pythonZip
}
$actualSha256 = (Get-FileHash -LiteralPath $pythonZip -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualSha256 -ne $pythonZipSha256) {
    throw "CPython archive SHA-256 mismatch: expected $pythonZipSha256, got $actualSha256"
}

if (Test-Path -LiteralPath $stageRoot) {
    $stagePrefix = [IO.Path]::GetFullPath($stageRoot).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $running = @(Get-CimInstance Win32_Process | Where-Object {
        $_.ExecutablePath -and [IO.Path]::GetFullPath($_.ExecutablePath).StartsWith(
            $stagePrefix, [StringComparison]::OrdinalIgnoreCase)
    })
    if ($running.Count -gt 0) {
        $ids = ($running | ForEach-Object { $_.ProcessId }) -join ', '
        throw "Close staged release processes before restaging. Running process IDs: $ids"
    }
    Remove-Item -LiteralPath $stageRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $pythonRoot -Force | Out-Null
Expand-Archive -LiteralPath $pythonZip -DestinationPath $pythonRoot

$pathFile = Join-Path $pythonRoot 'python313._pth'
if (-not (Test-Path -LiteralPath $pathFile)) {
    throw "The official runtime is missing $pathFile"
}
# The embedded distribution ignores PYTHONPATH. These relative paths reach
# Tauri's resources after installation while keeping the runtime isolated
# from packages and environment settings elsewhere on the user's machine.
@(
    'python313.zip'
    '.'
    'Lib\site-packages'
    '..\..\sidecar'
    '..\..\engines\python'
    'import site'
) | Set-Content -LiteralPath $pathFile -Encoding ascii

$sitePackages = Join-Path $pythonRoot 'Lib/site-packages'
New-Item -ItemType Directory -Path $sitePackages -Force | Out-Null

# Resolve only binary wheels for the target interpreter, regardless of the
# host Python version. requirements.txt pins each dependency's version.
$requirements = Join-Path $repoRoot 'requirements.txt'
$pipArgs = @(
    '-m', 'pip', 'install', '--disable-pip-version-check', '--no-compile',
    '--only-binary=:all:', '--platform', 'win_amd64', '--implementation',
    'cp', '--python-version', '3.13', '--abi', 'cp313',
    '--target', $sitePackages, '-r', $requirements
)
& $HostPython @pipArgs
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install Python 3.13 Windows wheels (exit $LASTEXITCODE)."
}

# Copy only tracked source/resources. A developer checkout may contain ignored
# build directories and bytecode that must never enter a public installer.
$tracked = & git -C $repoRoot ls-files -- sidecar engines schemas docs/explain LICENSE NOTICE docs/GUIDE.md
if ($LASTEXITCODE -ne 0 -or -not $tracked) {
    throw 'Could not enumerate tracked release resources.'
}
foreach ($relative in $tracked) {
    $source = Join-Path $repoRoot $relative
    $destination = Join-Path $stageRoot $relative
    $parent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
}
if (-not (Test-Path -LiteralPath (Join-Path $stageRoot 'sidecar/capy_sidecar/__main__.py'))) {
    throw 'The staged source is missing the sidecar entrypoint.'
}

& (Join-Path $PSScriptRoot 'clean_windows_runtime.ps1') -ResourceRoot $stageRoot

Write-Host "Staged verified CPython $pythonVersion and tracked resources at $stageRoot"
