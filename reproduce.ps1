#requires -Version 5.1
<#
.SYNOPSIS
    One-command reproduction for Windows.

.DESCRIPTION
    Ensures Docker Desktop, WSL2, and Docker Compose are available, then builds
    the pinned reproduction image and runs the full pipeline. If Docker is
    already installed and running, the script is non-interactive.

.PARAMETER Mode
    full (default), analysis, test, or rl.
#>
[CmdletBinding()]
param(
    [ValidateSet("full", "analysis", "test", "rl")]
    [string]$Mode = "full"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-StrictMode -Version Latest
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Fail {
    param([string]$Message)
    Write-Host "error: $Message" -ForegroundColor Red
    exit 1
}

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command -Name $Name -ErrorAction SilentlyContinue)
}

function Update-ProcessPath {
    # Merge the machine and user PATH into this process without discarding
    # entries that only exist for the current session.
    $parts = @()
    foreach ($scope in @("Machine", "User")) {
        $value = [Environment]::GetEnvironmentVariable("Path", $scope)
        if ($value) {
            $parts += @($value -split ";" | Where-Object { $_ })
        }
    }
    $parts += @($env:Path -split ";" | Where-Object { $_ })

    $seen = @{}
    $merged = foreach ($part in $parts) {
        $key = $part.ToLowerInvariant()
        if (-not $seen.ContainsKey($key)) {
            $seen[$key] = $true
            $part
        }
    }
    $env:Path = ($merged -join ";")
}

function Test-DockerDaemon {
    if (-not (Test-Command "docker")) { return $false }
    & docker info *> $null
    return ($LASTEXITCODE -eq 0)
}

function Test-DockerCompose {
    if (-not (Test-Command "docker")) { return $false }
    & docker compose version *> $null
    return ($LASTEXITCODE -eq 0)
}

function Wait-Docker {
    param([int]$TimeoutSeconds = 300)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-DockerDaemon) { return $true }
        Start-Sleep -Seconds 3
    }
    return $false
}

function Ensure-Wsl {
    Write-Step "Checking WSL2..."
    if (-not (Test-Command "wsl.exe")) {
        Fail "wsl.exe was not found. Install WSL2, restart Windows, and rerun this script."
    }

    & wsl.exe --status *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Step "Enabling WSL2 (this may require a restart)..."
        try {
            $process = Start-Process -FilePath "wsl.exe" `
                -ArgumentList @("--install", "--no-distribution") `
                -Verb RunAs -Wait -PassThru
        } catch {
            Fail "Failed to enable WSL2: $($_.Exception.Message)"
        }
        if ($process.ExitCode -ne 0) {
            Fail "wsl --install failed with exit code $($process.ExitCode). Install WSL2 manually and rerun."
        }

        & wsl.exe --status *> $null
        if ($LASTEXITCODE -ne 0) {
            Fail "WSL2 was enabled, but Windows must be restarted. Restart and run this script again."
        }
    }

    & wsl.exe --set-default-version 2 *> $null

    try {
        $cpu = Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1
        if ($cpu) {
            $virt = $cpu.PSObject.Properties["VirtualizationFirmwareEnabled"]
            if ($virt -and $virt.Value -eq $false) {
                Fail "Hardware virtualization is disabled. Enable VT-x/AMD-V in BIOS/UEFI, then rerun."
            }
        }
    } catch {
        # The check is best-effort; Docker Desktop reports the same problem.
    }
}

function Install-DockerDesktop {
    Write-Step "Installing Docker Desktop..."
    $installer = Join-Path $env:TEMP "DockerDesktopInstaller.exe"
    $url = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"
    try {
        Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing
    } catch {
        Fail "Failed to download the Docker Desktop installer: $($_.Exception.Message)"
    }

    try {
        $process = Start-Process -FilePath $installer `
            -ArgumentList @("install", "--quiet", "--accept-license", "--backend=wsl-2") `
            -Verb RunAs -Wait -PassThru
    } catch {
        Fail "Docker Desktop installation was cancelled or failed: $($_.Exception.Message)"
    } finally {
        Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue
    }

    if ($process.ExitCode -eq 3010) {
        Fail "Docker Desktop was installed, but Windows must be restarted. Restart and rerun this script."
    }
    if ($process.ExitCode -ne 0) {
        Fail "Docker Desktop installer exited with code $($process.ExitCode)."
    }

    Update-ProcessPath
}

function Start-DockerDesktop {
    param([int]$TimeoutSeconds = 300)
    $desktop = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path -LiteralPath $desktop)) { return $false }

    Write-Step "Starting Docker Desktop..."
    Start-Process -FilePath $desktop | Out-Null

    $cli = Join-Path $env:ProgramFiles "Docker\Docker\DockerCli.exe"
    if (Test-Path -LiteralPath $cli) {
        & $cli -SwitchLinuxEngine *> $null
    }

    return (Wait-Docker -TimeoutSeconds $TimeoutSeconds)
}

function Ensure-Docker {
    Update-ProcessPath
    $desktop = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"

    # If Docker already works, leave the host untouched, even if Docker
    # Desktop was installed in a non-default location.
    if ((Test-Command "docker") -and (Test-DockerDaemon)) { return }

    # Only touch WSL2 when Docker Desktop still has to be installed. If Docker
    # is already working (including a Hyper-V backend), leave the host alone.
    if (-not (Test-Command "docker") -and -not (Test-Path -LiteralPath $desktop)) {
        Ensure-Wsl
        Install-DockerDesktop
        Update-ProcessPath
    }

    if (-not (Test-Command "docker")) {
        Update-ProcessPath
    }
    if (-not (Test-Command "docker")) {
        Fail "The docker CLI was not found. Open Docker Desktop once, then rerun."
    }

    if (Test-DockerDaemon) { return }

    if (Start-DockerDesktop -TimeoutSeconds 120) { return }

    # A previously installed Docker Desktop may have lost its WSL backend.
    Ensure-Wsl
    if (Start-DockerDesktop -TimeoutSeconds 300) { return }

    Fail "Docker Desktop did not become ready. Open it once, accept the license, switch to Linux containers, and rerun."
}

function Ensure-LinuxContainers {
    $osType = (& docker info --format '{{.OSType}}' 2>$null)
    if ($LASTEXITCODE -ne 0) {
        Fail "Could not query Docker."
    }
    if ($osType -and $osType.Trim() -eq "linux") { return }

    $cli = Join-Path $env:ProgramFiles "Docker\Docker\DockerCli.exe"
    if (Test-Path -LiteralPath $cli) {
        Write-Step "Switching Docker Desktop to Linux containers..."
        & $cli -SwitchLinuxEngine *> $null
        if (Wait-Docker -TimeoutSeconds 180) {
            $osType = (& docker info --format '{{.OSType}}' 2>$null)
        }
    }
    if (-not $osType -or $osType.Trim() -ne "linux") {
        Fail "Docker is running Windows containers. Switch Docker Desktop to Linux containers and rerun."
    }
}

$arch = if ($env:PROCESSOR_ARCHITEW6432) {
    $env:PROCESSOR_ARCHITEW6432
} else {
    $env:PROCESSOR_ARCHITECTURE
}
if ($arch -eq "ARM64") {
    Fail "This reproduction image is supported on x86_64 only. ARM64 Windows is not supported."
}
if ($arch -ne "AMD64") {
    Fail "Unsupported architecture: $arch."
}

if (-not (Test-Command "git")) {
    Fail "git is required to initialize the submodules. Install Git for Windows and rerun."
}

Ensure-Docker

if (-not (Test-DockerCompose)) {
    Fail "Docker Compose v2 is required. Update Docker Desktop and rerun."
}
Ensure-LinuxContainers

Write-Step "Initializing submodules..."
& git -C $Root submodule update --init --recursive
if ($LASTEXITCODE -ne 0) {
    Fail "Failed to initialize the git submodules."
}

$gitCommit = (& git -C $Root rev-parse HEAD 2>$null | Select-Object -First 1)
if (-not $gitCommit) { $gitCommit = "unknown" }
$gitStatus = ((& git -C $Root status --porcelain 2>$null) -join "`n")
$gitDirty = if ($gitStatus) { "dirty" } else { "clean" }
$submodules = ((& git -C $Root submodule status --recursive 2>$null) -join "`n")

$env:GIT_COMMIT = $gitCommit
$env:GIT_DIRTY = $gitDirty
if (-not $env:REPRO_RUN_ID) {
    # The RL scripts parse this as %Y%m%d_%H%M%S, so keep underscores.
    $env:REPRO_RUN_ID = [DateTime]::UtcNow.ToString("yyyyMMdd_HHmmss")
}

$resultsDir = Join-Path $Root "results"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null
$hostInfo = @(
    "# Host repository state captured by reproduce.ps1",
    "date_utc: $([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ss') + 'Z')",
    "",
    "## commit",
    $gitCommit,
    "",
    "## status",
    $gitStatus,
    "",
    "## submodules",
    $submodules,
    ""
) -join "`n"
[System.IO.File]::WriteAllText(
    (Join-Path $resultsDir "host-git-info.txt"),
    $hostInfo,
    (New-Object System.Text.UTF8Encoding($false))
)

$composeFile = Join-Path $Root "docker\compose.yaml"

Write-Step "Building the reproduction image..."
& docker compose -f $composeFile build `
    --build-arg "GIT_COMMIT=$($env:GIT_COMMIT)" `
    --build-arg "GIT_DIRTY=$($env:GIT_DIRTY)"
if ($LASTEXITCODE -ne 0) {
    Fail "Image build failed."
}

$runArgs = @("compose", "-f", $composeFile, "run", "--rm")
foreach ($name in @(
    "ANALYSIS_STEPS",
    "REPRO_RUN_ID",
    "RL_SEEDS",
    "RL_JOBS",
    "RL_MAIN_STEPS",
    "RL_TRANSFER_DEFAULT_STEPS",
    "RL_TRANSFER_LARGE_STEPS"
)) {
    $value = [Environment]::GetEnvironmentVariable($name, "Process")
    if ($value) {
        $runArgs += @("-e", "$name=$value")
    }
}

$runArgs += "reproduce"
if ($Mode -eq "rl") {
    $suite = if ($env:RL_SUITE) { $env:RL_SUITE } else { "main" }
    $runArgs += @("rl", $suite)
} else {
    $runArgs += $Mode
}

Write-Step "Running reproduction mode: $Mode (run id: $($env:REPRO_RUN_ID))"
& docker @runArgs
$status = $LASTEXITCODE

if ($status -eq 0) {
    Write-Step "Reproduction complete."
    Write-Host "Analysis results: $resultsDir"
    Write-Host "RL output:        $(Join-Path $Root 'output')"
} else {
    Write-Host "error: reproduction failed with exit code $status." -ForegroundColor Red
}
exit $status
