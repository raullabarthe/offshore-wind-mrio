param(
    [string]$VenvPath = ".venv",
    [string]$PythonExe = "python",
    [string]$KernelName = "offshore-wind-mrio",
    [string]$DisplayName = "offshore-wind-mrio"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Python = Join-Path $VenvPath "Scripts\python.exe"

function Get-PythonVersionParts($Executable) {
    $versionText = & $Executable -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    $parts = $versionText.Trim().Split(".")
    return [int]$parts[0], [int]$parts[1]
}

function Assert-SupportedPython($Executable) {
    $major, $minor = Get-PythonVersionParts $Executable
    if ($major -ne 3 -or $minor -ne 13) {
        throw "Unsupported Python $major.$minor. Use Python 3.13 for this model."
    }
}

Assert-SupportedPython $PythonExe

if (Test-Path $Python) {
    try {
        $major, $minor = Get-PythonVersionParts $Python
        if ($major -ne 3 -or $minor -ne 13) {
            Write-Host "Existing virtual environment uses unsupported Python $major.$minor. Recreating $VenvPath..."
            $ResolvedVenv = (Resolve-Path $VenvPath).Path
            if (-not $ResolvedVenv.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Refusing to remove venv outside project root: $ResolvedVenv"
            }
            Remove-Item -LiteralPath $ResolvedVenv -Recurse -Force
        }
    }
    catch {
        throw
    }
}

if (-not (Test-Path $Python)) {
    & $PythonExe -m venv $VenvPath
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.txt
& $Python -m ipykernel install --user --name $KernelName --display-name $DisplayName

Write-Host ""
Write-Host "Environment ready."
Write-Host "Select the Jupyter kernel: $DisplayName"
