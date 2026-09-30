# Release scripts may remove staged files and ask installers to write beneath
# .capy-build. A textual path prefix is insufficient when a directory is a
# junction or symbolic link, so check each existing component from that root.
function Assert-BuildPath([string]$BuildRoot, [string]$Path) {
    $root = [IO.Path]::GetFullPath($BuildRoot).TrimEnd('\', '/')
    $fullPath = [IO.Path]::GetFullPath($Path)
    $prefix = $root + [IO.Path]::DirectorySeparatorChar
    if (-not $fullPath.Equals($root, [StringComparison]::OrdinalIgnoreCase) -and
        -not $fullPath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Refusing to access a path outside .capy-build.'
    }

    $component = $fullPath
    while ($true) {
        $item = $null
        try {
            $item = Get-Item -LiteralPath $component -Force -ErrorAction Stop
        } catch {
            if ($_.CategoryInfo.Category -ne [Management.Automation.ErrorCategory]::ObjectNotFound) {
                throw
            }
        }
        if ($item -and ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw 'Refusing to access .capy-build through a junction or symbolic link.'
        }
        if ($component.Equals($root, [StringComparison]::OrdinalIgnoreCase)) { break }
        $component = [IO.Path]::GetDirectoryName($component)
        if (-not $component) { throw 'Could not verify the build path.' }
    }
}

# Check children one level at a time. Get-ChildItem -Recurse may traverse a
# junction on some PowerShell versions, so never recurse until the child has
# been checked. Call this immediately before any recursive removal or before
# asking an external process to delete an existing build tree.
function Assert-BuildTreeSafe([string]$BuildRoot, [string]$Path) {
    Assert-BuildPath -BuildRoot $BuildRoot -Path $Path
    $root = [IO.Path]::GetFullPath($Path)
    $item = Get-Item -LiteralPath $root -Force -ErrorAction Stop
    if (-not $item.PSIsContainer) { return }
    $prefix = $root.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push($root)
    while ($pending.Count -gt 0) {
        foreach ($child in Get-ChildItem -LiteralPath $pending.Pop() -Force -ErrorAction Stop) {
            $fullChild = [IO.Path]::GetFullPath($child.FullName)
            if (-not $fullChild.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw 'Refusing to traverse a build-tree child outside its intended directory.'
            }
            if ($child.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw 'Refusing to traverse a junction or symbolic link inside a build tree.'
            }
            if ($child.PSIsContainer) { $pending.Push($fullChild) }
        }
    }
}
