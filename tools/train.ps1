[CmdletBinding()]
param(
    [string]$Config = "training/local-endgame.toml",
    [Nullable[UInt64]]$TargetUpdate = $null,
    [switch]$PrepareOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$exitCode = 1
Push-Location -LiteralPath $projectRoot
try {
    & cargo build --profile fast --offline --bin chineseai
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) {
        $trainingArgs = @("az-train", $Config)
        if ($null -ne $TargetUpdate) {
            if ($TargetUpdate -eq 0) { throw "TargetUpdate must be greater than 0." }
            $trainingArgs += @("--target-update", $TargetUpdate.ToString())
        }
        if ($PrepareOnly) { $trainingArgs += "--prepare-only" }
        & (Join-Path $projectRoot "target/fast/chineseai.exe") @trainingArgs
        $exitCode = $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
exit $exitCode
