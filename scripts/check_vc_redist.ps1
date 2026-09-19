param(
    [string]$MinimumVersion = "14.44.35211"
)

$ErrorActionPreference = "Stop"
$registryCandidates = @(
    "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
)

function Convert-ToVersion([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) { return $null }
    $clean = $Value.Trim().TrimStart('v','V')
    try { return [Version]$clean } catch { return $null }
}

$minimum = Convert-ToVersion $MinimumVersion
if ($null -eq $minimum) {
    Write-Host "[FAIL] Invalid minimum VC++ Redistributable version: $MinimumVersion"
    exit 2
}

$best = $null
$bestPath = $null
foreach ($path in $registryCandidates) {
    if (-not (Test-Path $path)) { continue }
    try {
        $item = Get-ItemProperty -Path $path
        if ($item.Installed -ne 1) { continue }
        $versionText = [string]$item.Version
        if ([string]::IsNullOrWhiteSpace($versionText) -and $null -ne $item.Major) {
            $versionText = "$($item.Major).$($item.Minor).$($item.Bld).$($item.Rbld)"
        }
        $parsed = Convert-ToVersion $versionText
        if ($null -ne $parsed -and ($null -eq $best -or $parsed -gt $best)) {
            $best = $parsed
            $bestPath = $path
        }
    } catch {
        continue
    }
}

Write-Host "========================================"
Write-Host "Resonastra Microsoft VC++ Prerequisite"
Write-Host "========================================"
Write-Host "Required architecture : x64"
Write-Host "Minimum version       : $minimum"

if ($null -eq $best) {
    Write-Host "[FAIL] Microsoft Visual C++ v14 x64 Redistributable was not detected."
    Write-Host "Install the latest supported Microsoft x64 Redistributable, then rerun this check:"
    Write-Host "  https://aka.ms/vc14/vc_redist.x64.exe"
    exit 1
}

Write-Host "Detected version      : $best"
Write-Host "Registry source       : $bestPath"
if ($best -lt $minimum) {
    Write-Host "[FAIL] Installed Microsoft Visual C++ v14 x64 Redistributable is too old."
    Write-Host "Install/update to the latest supported Microsoft x64 Redistributable:"
    Write-Host "  https://aka.ms/vc14/vc_redist.x64.exe"
    exit 1
}

Write-Host "[OK] Microsoft Visual C++ v14 x64 Redistributable prerequisite satisfied."
exit 0
