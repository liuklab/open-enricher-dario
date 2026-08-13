# Provision a Colab Pro GPU session, mount Drive, install the library, run inference.
# Copy this repo to Drive first (default: MyDrive/open_enricher/code) and put videos under
# MyDrive/open_enricher/data. Install the CLI locally: uv tool install google-colab-cli
#
# Google Colab CLI currently documents Linux and macOS only. On native Windows, install
# the CLI inside WSL and either run scripts/bootstrap_colab.sh from WSL, or run this
# script if `colab` is already on PATH.
#
# Usage from the repository root (PowerShell):
#   $env:HF_TOKEN = "hf_..."    # optional, for Hugging Face
#   .\scripts\bootstrap_colab.ps1
#   .\scripts\bootstrap_colab.ps1 embeddings
#
# Extra flags after the mode are forwarded to colab exec:
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

$Session = if ($env:COLAB_SESSION) { $env:COLAB_SESSION } else { "enricher" }
$Gpu = if ($env:COLAB_GPU) { $env:COLAB_GPU } else { "A100" }
$Timeout = if ($env:COLAB_TIMEOUT) { $env:COLAB_TIMEOUT } else { "7200" }
$InstallTimeout = if ($env:COLAB_INSTALL_TIMEOUT) { $env:COLAB_INSTALL_TIMEOUT } else { "600" }
$Root = Split-Path -Parent $PSScriptRoot

function Invoke-Colab {
    param([Parameter(Mandatory = $true)][string[]]$ColabArgs)
    & colab @ColabArgs
    if ($LASTEXITCODE -ne 0) {
        throw "colab failed ($LASTEXITCODE): colab $($ColabArgs -join ' ')"
    }
}

Write-Host "Starting session $Session on GPU $Gpu"
Invoke-Colab @("new", "-s", $Session, "--gpu", $Gpu, "--high-mem")

Write-Host "Mounting Google Drive (browser OAuth may open once)"
Invoke-Colab @("drivemount", "-s", $Session)

Write-Host "Installing Python dependencies on the VM"
$Requirements = Join-Path $Root "requirements.txt"
Invoke-Colab @("install", "-s", $Session, "-r", $Requirements)

Write-Host "Editable-installing open-enricher from Drive"
$InstallScript = Join-Path $Root "scripts\install_package.py"
Invoke-Colab @("exec", "-s", $Session, "--timeout", "$InstallTimeout", "-f", $InstallScript)

$RunnerName = if ($Mode -eq "embeddings") { "run_embeddings.py" } else { "run_metadata.py" }
$Runner = Join-Path $Root "scripts\$RunnerName"

$ExecArgs = [System.Collections.Generic.List[string]]::new()
$ExecArgs.AddRange([string[]]@("exec", "-s", $Session, "--timeout", "$Timeout"))
if ($env:HF_TOKEN) {
    $ExecArgs.Add("--env")
    $ExecArgs.Add("HF_TOKEN=$($env:HF_TOKEN)")
}
if ($ExtraArgs) {
    $ExecArgs.AddRange([string[]]$ExtraArgs)
}
$ExecArgs.Add("-f")
$ExecArgs.Add($Runner)

Write-Host "Running $Runner"
Invoke-Colab -ColabArgs $ExecArgs.ToArray()

Write-Host "Done. Session $Session is still running. Stop it with: colab stop -s $Session"
