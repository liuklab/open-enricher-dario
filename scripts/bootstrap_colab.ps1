# Provision a Colab Pro GPU session, mount Drive (videos only), sync code via git, run inference.
# Configuration lives in the repository .env (copy from .env.example).
# Push your commits first. Install the CLI locally: uv tool install google-colab-cli
#
# Google Colab CLI currently documents Linux and macOS only. On native Windows, install
# the CLI inside WSL and either run scripts/bootstrap_colab.sh from WSL, or run this
# script if `colab` is already on PATH.
#
# Usage from the repository root (PowerShell):
#   .\scripts\bootstrap_colab.ps1
#   .\scripts\bootstrap_colab.ps1 embeddings
#
# Extra flags after the mode are forwarded to the inference colab exec:
#   .\scripts\bootstrap_colab.ps1 metadata --env OPEN_ENRICHER_VIDEO=clip.mp4

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Mode = "metadata",

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

if ($Mode -notin @("metadata", "embeddings")) {
    Write-Error "Usage: .\scripts\bootstrap_colab.ps1 [metadata|embeddings] [extra colab exec args...]"
    exit 1
}

if (-not (Get-Command colab -ErrorAction SilentlyContinue)) {
    Write-Error "colab CLI not found on PATH. On Windows use WSL: wsl -e bash scripts/bootstrap_colab.sh"
    exit 1
}

$Root = Split-Path -Parent $PSScriptRoot
$EnvFile = Join-Path $Root ".env"

function Import-DotEnv {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        Write-Host "No .env found at $Path. Copy .env.example to .env and fill in tokens."
        return
    }
    Get-Content -Path $Path | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#") -or $line -notmatch "=") {
            return
        }
        $key, $value = $line.Split("=", 2)
        $key = $key.Trim()
        if ($key.StartsWith("export ")) {
            $key = $key.Substring(7).Trim()
        }
        $value = $value.Trim()
        if (
            $value.Length -ge 2 -and
            (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'")))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        if (-not [string]::IsNullOrWhiteSpace($key)) {
            Set-Item -Path "Env:$key" -Value $value
        }
    }
}

function Add-ForwardedEnv {
    param(
        [System.Collections.Generic.List[string]]$Dest,
        [string[]]$Keys
    )
    foreach ($key in $Keys) {
        $value = [Environment]::GetEnvironmentVariable($key)
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            $Dest.Add("--env")
            $Dest.Add("$key=$value")
        }
    }
}

Import-DotEnv -Path $EnvFile

$ForwardKeys = @(
    "HF_TOKEN",
    "GITHUB_TOKEN",
    "GIT_TOKEN",
    "OPEN_ENRICHER_GIT_URL",
    "OPEN_ENRICHER_GIT_REF",
    "OPEN_ENRICHER_REPO",
    "OPEN_ENRICHER_DATA",
    "OPEN_ENRICHER_OUTPUT",
    "OPEN_ENRICHER_VIDEO",
    "OPEN_ENRICHER_MODEL",
    "OPEN_ENRICHER_NUM_FRAMES",
    "OPEN_ENRICHER_MAX_NEW_TOKENS"
)

$Session = if ($env:COLAB_SESSION) { $env:COLAB_SESSION } else { "enricher" }
$Gpu = if ($env:COLAB_GPU) { $env:COLAB_GPU } else { "A100" }
$Timeout = if ($env:COLAB_TIMEOUT) { $env:COLAB_TIMEOUT } else { "7200" }
$InstallTimeout = if ($env:COLAB_INSTALL_TIMEOUT) { $env:COLAB_INSTALL_TIMEOUT } else { "600" }

function Invoke-Colab {
    param([Parameter(Mandatory = $true)][string[]]$ColabArgs)
    & colab @ColabArgs
    if ($LASTEXITCODE -ne 0) {
        throw "colab failed ($LASTEXITCODE): colab $($ColabArgs -join ' ')"
    }
}

if (-not $env:OPEN_ENRICHER_GIT_URL) {
    $env:OPEN_ENRICHER_GIT_URL = (git -C $Root remote get-url origin 2>$null)
}
if (-not $env:OPEN_ENRICHER_GIT_REF) {
    $env:OPEN_ENRICHER_GIT_REF = (git -C $Root rev-parse --abbrev-ref HEAD 2>$null)
}
if ($env:OPEN_ENRICHER_GIT_REF -eq "HEAD") {
    $env:OPEN_ENRICHER_GIT_REF = (git -C $Root rev-parse HEAD)
}

Write-Host "Starting session $Session on GPU $Gpu"
Invoke-Colab @("new", "-s", $Session, "--gpu", $Gpu, "--high-mem")

Write-Host "Mounting Google Drive for videos (browser OAuth may open once)"
Invoke-Colab @("drivemount", "-s", $Session)

Write-Host "Installing Python dependencies on the VM"
$Requirements = Join-Path $Root "requirements.txt"
Invoke-Colab @("install", "-s", $Session, "-r", $Requirements)

$InstallArgs = [System.Collections.Generic.List[string]]::new()
$InstallArgs.AddRange([string[]]@("exec", "-s", $Session, "--timeout", "$InstallTimeout"))
Add-ForwardedEnv -Dest $InstallArgs -Keys $ForwardKeys
$InstallScript = Join-Path $Root "scripts\install_package.py"
$InstallArgs.Add("-f")
$InstallArgs.Add($InstallScript)

Write-Host "Syncing library from git $($env:OPEN_ENRICHER_GIT_URL) @ $($env:OPEN_ENRICHER_GIT_REF). Push local commits first."
Invoke-Colab -ColabArgs $InstallArgs.ToArray()

$RunnerName = if ($Mode -eq "embeddings") { "run_embeddings.py" } else { "run_metadata.py" }
$Runner = Join-Path $Root "scripts\$RunnerName"

$ExecArgs = [System.Collections.Generic.List[string]]::new()
$ExecArgs.AddRange([string[]]@("exec", "-s", $Session, "--timeout", "$Timeout"))
Add-ForwardedEnv -Dest $ExecArgs -Keys $ForwardKeys
if ($ExtraArgs) {
    $ExecArgs.AddRange([string[]]$ExtraArgs)
}
$ExecArgs.Add("-f")
$ExecArgs.Add($Runner)

Write-Host "Running $Runner"
Invoke-Colab -ColabArgs $ExecArgs.ToArray()

Write-Host "Done. Session $Session is still running. Stop it with: colab stop -s $Session"
