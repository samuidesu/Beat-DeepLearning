# Train every model, then score each one on the held-out test split.
#
# Usage (from the IMDB project root, with the torch env active):
#     .\run_all.ps1
#     .\run_all.ps1 -Models rnn,lstm            # subset
#     .\run_all.ps1 -Models transformer
#     .\run_all.ps1 -TrainArgs '--seed',7       # extra flags passed to train.py
#
# If PowerShell refuses to run the file ("running scripts is disabled"):
#     powershell -ExecutionPolicy Bypass -File .\run_all.ps1
#
# Stops at the first failure -- a crashed run leaves a half-written log, and
# continuing would bury the error under thousands of lines of the next run.

param(
    [ValidateSet('rnn', 'gru', 'lstm', 'transformer')]
    [string[]]$Models = @('rnn', 'gru', 'lstm', 'transformer'),
    [string[]]$TrainArgs = @()
)

$ErrorActionPreference = 'Stop'
$start = Get-Date

# train.py takes --cell for the three recurrent models and --model for the
# Transformer; eval.py finds the checkpoint the same way in both cases.
function Get-TrainArgs($m) {
    if ($m -eq 'transformer') { return @('--model', 'transformer') }
    return @('--cell', $m)
}
function Get-EvalArgs($m) {
    if ($m -eq 'transformer') { return @('--weights', 'outputs_transformer/best.pt') }
    return @('--cell', $m)
}

foreach ($m in $Models) {
    Write-Host "`n===== train $m =====" -ForegroundColor Cyan
    python train.py @(Get-TrainArgs $m) @TrainArgs
    # $LASTEXITCODE is how a native exe reports failure; $? is not reliable here.
    if ($LASTEXITCODE -ne 0) { throw "train.py ($m) exited with $LASTEXITCODE" }
}

foreach ($m in $Models) {
    Write-Host "`n===== test $m =====" -ForegroundColor Cyan
    python eval.py @(Get-EvalArgs $m) --split test --save-cm
    if ($LASTEXITCODE -ne 0) { throw "eval.py ($m) exited with $LASTEXITCODE" }
}

$mins = ((Get-Date) - $start).TotalMinutes
Write-Host ("`nAll done in {0:N1} min. Results: outputs_*/training_log.json" -f $mins) -ForegroundColor Green
