param(
    [string]$Installer = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'build_path_safety.ps1')

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$buildRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot '.capy-build'))
$installRoot = [IO.Path]::GetFullPath((Join-Path $buildRoot 'release/installed-smoke'))
Assert-BuildPath -BuildRoot $buildRoot -Path $installRoot
if (-not $Installer) {
    $files = @(Get-ChildItem (Join-Path $repoRoot 'app/src-tauri/target/release/bundle/nsis') -Filter '*.exe' -File)
    if ($files.Count -ne 1) { throw "Expected one NSIS installer, found $($files.Count)." }
    $Installer = $files[0].FullName
}
$Installer = [IO.Path]::GetFullPath($Installer)
if (-not (Test-Path -LiteralPath $Installer)) { throw "Missing installer: $Installer" }

# NSIS may otherwise upgrade an existing current-user installation. Never use
# this smoke test on a host with a real Causal Capybara already installed.
$uninstallRoot = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall'
$existing = @(Get-ChildItem -LiteralPath $uninstallRoot -ErrorAction SilentlyContinue |
    ForEach-Object { Get-ItemProperty -LiteralPath $_.PSPath -ErrorAction SilentlyContinue } |
    Where-Object { $_.PSObject.Properties['DisplayName'] -and $_.DisplayName -like 'Causal Capybara*' })
if ($existing.Count -gt 0) {
    throw 'An existing Causal Capybara installation is registered for this user. Use a clean Windows user for installer smoke testing.'
}
if (Test-Path -LiteralPath $installRoot) {
    Assert-BuildTreeSafe -BuildRoot $buildRoot -Path $installRoot
    Remove-Item -LiteralPath $installRoot -Recurse -Force
}

# NSIS /D= must be last and unquoted, even when its absolute path has spaces.
Assert-BuildPath -BuildRoot $buildRoot -Path $installRoot
Write-Host "Silently installing release candidate into $installRoot"
$installed = Start-Process -FilePath $Installer -ArgumentList "/S /D=$installRoot" -PassThru -Wait -WindowStyle Hidden
if ($installed.ExitCode -ne 0) {
    throw "NSIS installer returned exit code $($installed.ExitCode)."
}
$resourceRoot = $installRoot
if (-not (Test-Path -LiteralPath (Join-Path $resourceRoot 'runtimes/python/python.exe'))) {
    throw "The installer did not place bundled Python in the isolated location $resourceRoot"
}
foreach ($required in @(
    'sidecar/capy_sidecar/__main__.py',
    'engines/python/capy_py/__init__.py',
    'schemas/capy.result.v1.json',
    'LICENSE',
    'NOTICE',
    'licenses/PYTHON_LICENSES.txt',
    'licenses/NPM_LICENSES.txt',
    'licenses/RUST_LICENSES.txt',
    'licenses/STANDARD_LICENSES.txt'
)) {
    if (-not (Test-Path -LiteralPath (Join-Path $resourceRoot $required))) {
        throw "The NSIS installation omitted required resource: $required"
    }
}
Write-Host "Installed resources found at $resourceRoot"
& (Join-Path $PSScriptRoot 'smoke_windows_runtime.ps1') -ResourceRoot $resourceRoot
Write-Host 'Installed NSIS resources and analysis runtime passed the smoke check.'

$desktopExe = Join-Path $installRoot 'causal-capybara.exe'
if (-not (Test-Path -LiteralPath $desktopExe)) {
    throw 'The installer omitted the desktop executable.'
}
$preexisting = Get-NetTCPConnection -LocalPort 8760 -State Listen -ErrorAction SilentlyContinue
if ($preexisting) {
    throw 'Port 8760 is already in use; cannot attribute engine startup to the installed app.'
}
$launchHome = Join-Path $buildRoot 'release/launch-smoke-home'
Assert-BuildPath -BuildRoot $buildRoot -Path $launchHome
if (Test-Path -LiteralPath $launchHome) {
    Assert-BuildTreeSafe -BuildRoot $buildRoot -Path $launchHome
}
New-Item -ItemType Directory -Path $launchHome -Force | Out-Null
$oldCapyPython = $env:CAPY_PYTHON
$oldCapyHome = $env:CAPY_HOME
$oldPath = $env:PATH
$appProcess = $null
$childIds = @()
try {
    $env:CAPY_PYTHON = ''
    $env:CAPY_HOME = $launchHome
    $env:PATH = (Join-Path $env:SystemRoot 'System32') + ';' + $env:SystemRoot
    $appProcess = Start-Process -FilePath $desktopExe -WorkingDirectory $installRoot -WindowStyle Hidden -PassThru
    $deadline = [DateTime]::UtcNow.AddSeconds(60)
    $healthy = $false
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($appProcess.HasExited) {
            throw "Installed desktop app exited early with $($appProcess.ExitCode)."
        }
        try {
            $reply = Invoke-RestMethod -Uri 'http://127.0.0.1:8760/health' -TimeoutSec 2
            if ($reply.ok -eq $true -and $reply.app -eq 'Causal Capybara') {
                $healthy = $true
                break
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $healthy) { throw 'Installed desktop app did not start its engine.' }
    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($appProcess.Id)")
    $embeddedPython = Join-Path $resourceRoot 'runtimes/python/python.exe'
    $childIds = @($children | Where-Object { $_.ExecutablePath -eq $embeddedPython } |
        Select-Object -ExpandProperty ProcessId)
    if ($childIds.Count -ne 1) {
        throw "The desktop app did not launch exactly one bundled Python process; found $($childIds.Count)."
    }
    $closed = $appProcess.CloseMainWindow()
    if (-not $closed) {
        # CloseMainWindow cannot find a window launched with WindowStyle Hidden.
        # WM_CLOSE still reaches Tauri's exit handler.
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class CapyWindowCloser {
    private delegate bool EnumWindowsProc(IntPtr window, IntPtr data);
    [DllImport("user32.dll")] private static extern bool EnumWindows(EnumWindowsProc callback, IntPtr data);
    [DllImport("user32.dll")] private static extern uint GetWindowThreadProcessId(IntPtr window, out uint processId);
    [DllImport("user32.dll")] private static extern bool PostMessage(IntPtr window, uint message, IntPtr wParam, IntPtr lParam);
    public static bool Close(uint processId) {
        bool sent = false;
        EnumWindows((window, data) => {
            uint owner;
            GetWindowThreadProcessId(window, out owner);
            if (owner == processId) sent |= PostMessage(window, 0x0010, IntPtr.Zero, IntPtr.Zero);
            return true;
        }, IntPtr.Zero);
        return sent;
    }
}
'@
        $closed = [CapyWindowCloser]::Close([uint32]$appProcess.Id)
    }
    if (-not $closed) {
        throw 'The desktop app did not accept a graceful window-close request.'
    }
    if (-not $appProcess.WaitForExit(15000)) {
        throw 'The desktop app did not exit after its window closed.'
    }
    Start-Sleep -Seconds 1
    $orphan = @($childIds | ForEach-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
    if ($orphan.Count -gt 0) {
        throw 'The desktop app left its bundled Python child running after exit.'
    }
    $generatedBytecode = @(Get-ChildItem -LiteralPath $resourceRoot -Recurse -File |
        Where-Object { $_.Extension -in @('.pyc', '.pyo') })
    if ($generatedBytecode.Count -gt 0) {
        throw 'Installed app generated bytecode in its resource tree; uninstall may leave files behind.'
    }
    Write-Host 'Installed desktop startup, /health, engine cleanup, and clean resource tree passed.'
}
finally {
    $env:CAPY_PYTHON = $oldCapyPython
    $env:CAPY_HOME = $oldCapyHome
    $env:PATH = $oldPath
    if ($appProcess -and -not $appProcess.HasExited) {
        Stop-Process -Id $appProcess.Id -Force -ErrorAction SilentlyContinue
    }
    foreach ($id in $childIds) {
        Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
    }
}
$uninstaller = @(Get-ChildItem -LiteralPath $installRoot -Filter '*uninstall*.exe' -File | Select-Object -First 1)
if ($uninstaller.Count -ne 1) {
    throw 'Smoke check passed, but the isolated installation has no uninstaller to clean it up.'
}
Assert-BuildTreeSafe -BuildRoot $buildRoot -Path $installRoot
$removed = Start-Process -FilePath $uninstaller[0].FullName -ArgumentList "/S _?=$installRoot" -PassThru -Wait -WindowStyle Hidden
if ($removed.ExitCode -ne 0) {
    throw "The isolated smoke installation uninstaller returned $($removed.ExitCode)."
}
Write-Host 'Isolated test installation was uninstalled.'
