[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$SkipInstaller,
    [switch]$DebugConsole
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
$buildEnvironment = Join-Path $projectRoot ".venv-build"
$specPath = Join-Path $PSScriptRoot "SSPython.spec"
$distPath = Join-Path $projectRoot "dist"
$workPath = Join-Path $projectRoot "build\pyinstaller"
$applicationPath = Join-Path $distPath "SSPython"
$executablePath = Join-Path $applicationPath "SSPython.exe"
$versionFile = Join-Path $projectRoot "core\version.py"

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,
        [Parameter(Mandatory)]
        [string[]]$ArgumentList
    )

    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($ArgumentList -join ' ')"
    }
}

if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    throw "uv is required to create the isolated build environment."
}

$versionMatch = Select-String -LiteralPath $versionFile -Pattern '^__version__\s*=\s*"([^"]+)"$'
if (-not $versionMatch) {
    throw "Could not read __version__ from $versionFile."
}
$applicationVersion = $versionMatch.Matches[0].Groups[1].Value

$previousProjectEnvironment = $env:UV_PROJECT_ENVIRONMENT
$previousConsoleSetting = $env:SSPYTHON_BUILD_CONSOLE
$previousQtPlatform = $env:QT_QPA_PLATFORM
$previousMatplotlibBackend = $env:MPLBACKEND

try {
    $env:UV_PROJECT_ENVIRONMENT = $buildEnvironment
    $env:SSPYTHON_BUILD_CONSOLE = if ($DebugConsole) { "1" } else { "0" }

    Push-Location $projectRoot
    try {
        Invoke-NativeCommand "uv" @(
            "sync",
            "--frozen",
            "--no-dev",
            "--group", "build"
        )

        if (-not $SkipTests) {
            $env:QT_QPA_PLATFORM = "offscreen"
            $env:MPLBACKEND = "Agg"
            Invoke-NativeCommand "uv" @(
                "run",
                "--frozen",
                "--no-dev",
                "--group", "build",
                "python", "-m", "unittest", "discover", "tests", "-v"
            )
        }

        Invoke-NativeCommand "uv" @(
            "run",
            "--frozen",
            "--no-dev",
            "--group", "build",
            "python", "-m", "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath", $distPath,
            "--workpath", $workPath,
            $specPath
        )
    }
    finally {
        Pop-Location
    }

    $requiredBundleFiles = @(
        "SSPython.exe",
        "_internal\assets\icon.png",
        "_internal\style\dark_theme.qss",
        "_internal\style\light_theme.qss",
        "_internal\mne_lsl\lsl\lib\lsl.dll",
        "_internal\mne_icalabel\iclabel\network\assets\ICLabelNet.pt"
    )
    foreach ($relativePath in $requiredBundleFiles) {
        $absolutePath = Join-Path $applicationPath $relativePath
        if (-not (Test-Path -LiteralPath $absolutePath -PathType Leaf)) {
            throw "The frozen bundle is missing required file: $relativePath"
        }
    }

    $env:QT_QPA_PLATFORM = "offscreen"
    $env:MPLBACKEND = "Agg"
    $smokeLogPath = Join-Path $workPath "frozen-smoke-test.log"
    $env:SSPYTHON_SMOKE_LOG = $smokeLogPath
    Remove-Item -LiteralPath $smokeLogPath -Force -ErrorAction SilentlyContinue
    $smokeProcess = Start-Process `
        -FilePath $executablePath `
        -ArgumentList "--smoke-test" `
        -WorkingDirectory $applicationPath `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($smokeProcess.ExitCode -ne 0) {
        if (Test-Path -LiteralPath $smokeLogPath -PathType Leaf) {
            Write-Host (Get-Content -LiteralPath $smokeLogPath -Raw)
        }
        throw "Frozen application smoke test failed with exit code $($smokeProcess.ExitCode)."
    }
    Remove-Item Env:SSPYTHON_SMOKE_LOG -ErrorAction SilentlyContinue

    if (-not $SkipInstaller) {
        $isccCommand = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
        $isccPath = if ($isccCommand) {
            $isccCommand.Source
        }
        else {
            "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
        }

        if (-not (Test-Path -LiteralPath $isccPath -PathType Leaf)) {
            throw "Inno Setup 6 was not found. Install it or run with -SkipInstaller."
        }

        Invoke-NativeCommand $isccPath @(
            "/Qp",
            "/DMyAppVersion=$applicationVersion",
            (Join-Path $projectRoot "inno_setup.iss")
        )
    }

    Write-Host "SSPython $applicationVersion release artifacts are in $distPath"
}
finally {
    $env:UV_PROJECT_ENVIRONMENT = $previousProjectEnvironment
    $env:SSPYTHON_BUILD_CONSOLE = $previousConsoleSetting
    $env:QT_QPA_PLATFORM = $previousQtPlatform
    $env:MPLBACKEND = $previousMatplotlibBackend
}
