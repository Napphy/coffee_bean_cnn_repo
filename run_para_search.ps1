# run_hyperparameter_search.ps1
# ============================================================
# Two-stage hyperparameter search for coffee defect classifiers
#
# Stage 1 - Find best optimizer + learning rate
#   Fixed: batch_size=32, dropout=0.3, max_epochs=100
#   Grid : 5 optimizers x 3 learning rates = 15 runs per model/variety
#
# Stage 2 - Find best batch size + dropout (uses best opt+LR from Stage 1)
#   Fixed: best optimizer + LR from Stage 1, max_epochs=100
#   Grid : 3 batch sizes x 3 dropouts = 9 runs per model/variety
#
# Phase 3 - Cross-variety fine-tuning using best overall config per model
#
# Total runs: (15 + 9) x 4 models x 2 varieties = 192
# Estimated time on GTX 1660 Ti: ~64-96 hours
#
# Usage:
#   .\run_hyperparameter_search.ps1              # run everything
#   .\run_hyperparameter_search.ps1 -DryRun      # preview commands only
#   .\run_hyperparameter_search.ps1 -Stage 1     # run Stage 1 only
#   .\run_hyperparameter_search.ps1 -Stage 2     # run Stage 2 only (needs Stage 1 done)
#   .\run_hyperparameter_search.ps1 -Stage 3     # run Phase 3 only (needs Stage 2 done)
# ============================================================

param(
    [switch]$DryRun,
    [ValidateSet("all", "1", "2", "3")]
    [string]$Stage = "all"
)

# ── Config ────────────────────────────────────────────────────────────────────
$Models     = @("squeezenet", "shufflenet", "mobilenetv2", "mobilenetv3", "efficientnet")
$Varieties  = @("arabica","robusta") 
$LRs        = @("1e-3", "1e-4", "1e-5")
$Optimizers = @("adam", "adamw", "sgd", "rmsprop", "adan")
$BatchSizes = @(8, 16, 32)
$Dropouts   = @(0.3, 0.5, 0.7)
$MaxEpochs  = 100

# Stage 1 fixed params
$S1_BatchSize = 32
$S1_Dropout   = 0.3

# ── Helpers ───────────────────────────────────────────────────────────────────
function Invoke-Train {
    param([string]$Cmd)
    if ($DryRun) {
        Write-Host "  [DRY] $Cmd" -ForegroundColor Cyan
    } else {
        Write-Host "" 
        Write-Host "  >>> $Cmd" -ForegroundColor Green
        Invoke-Expression $Cmd
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] Training failed. Check logs above." -ForegroundColor Red
            exit 1
        }
    }
}

function Get-BestConfig {
    param(
        [string]$Variety,
        [string]$Model,
        [string]$SearchPattern
    )
    $best   = $null
    $bestF1 = -1.0
    $found  = Get-ChildItem -Path "runs\$SearchPattern\results.json" -ErrorAction SilentlyContinue

    foreach ($f in $found) {
        $json = Get-Content $f.FullName | ConvertFrom-Json
        $f1   = $json.results[0].report.'weighted avg'.'f1-score'
        if ($f1 -gt $bestF1) {
            $bestF1 = $f1
            $best   = $json
        }
    }

    if ($null -eq $best) {
        Write-Host "  [WARN] No results found for pattern: $SearchPattern" -ForegroundColor Yellow
        return $null
    }

    $msg = "  Best config for {0}/{1}: opt={2} lr={3} bs={4} do={5} F1={6:F4}" -f `
        $Variety, $Model, $best.optimizer, $best.lr, $best.batch_size, $best.dropout, $bestF1
    Write-Host $msg -ForegroundColor Magenta
    return $best
}

function Get-BestModelPath {
    param([string]$RunName)
    $path = "runs\$RunName\best_model.pth"
    if (Test-Path $path) { return $path }
    $match = Get-ChildItem -Path "runs\${RunName}*\best_model.pth" -ErrorAction SilentlyContinue |
             Sort-Object LastWriteTime -Descending |
             Select-Object -First 1
    if ($match) { return $match.FullName }
    return $null
}

# ── Stage 1: Optimizer x Learning Rate ───────────────────────────────────────
if ($Stage -eq "all" -or $Stage -eq "1") {
    Write-Host ""
    Write-Host ("=" * 65) -ForegroundColor Yellow
    Write-Host "  STAGE 1 - Optimizer x Learning Rate Search" -ForegroundColor Yellow
    Write-Host "  Fixed: batch_size=$S1_BatchSize  dropout=$S1_Dropout  max_epochs=$MaxEpochs" -ForegroundColor Yellow
    Write-Host ("=" * 65) -ForegroundColor Yellow

    foreach ($variety in $Varieties) {
        foreach ($model in $Models) {
            Write-Host ""
            Write-Host "  -- $variety / $model --" -ForegroundColor White
            foreach ($opt in $Optimizers) {
                foreach ($lr in $LRs) {
                    $cmd = "python train.py" +
                           " --variety $variety" +
                           " --model $model" +
                           " --optimizer $opt" +
                           " --lr $lr" +
                           " --batch_size $S1_BatchSize" +
                           " --dropout $S1_Dropout" +
                           " --max_epochs $MaxEpochs"
                    Invoke-Train $cmd
                }
            }
        }
    }
}

# ── Stage 2: Batch Size x Dropout ────────────────────────────────────────────
if ($Stage -eq "all" -or $Stage -eq "2") {
    Write-Host ""
    Write-Host ("=" * 65) -ForegroundColor Yellow
    Write-Host "  STAGE 2 - Batch Size x Dropout Search" -ForegroundColor Yellow
    Write-Host "  Using best optimizer+LR from Stage 1 per model/variety" -ForegroundColor Yellow
    Write-Host ("=" * 65) -ForegroundColor Yellow

    foreach ($variety in $Varieties) {
        foreach ($model in $Models) {
            Write-Host ""
            Write-Host "  -- $variety / $model --" -ForegroundColor White

            $best = Get-BestConfig `
                -Variety $variety `
                -Model $model `
                -SearchPattern "phase12_${variety}_${model}_*_bs${S1_BatchSize}_do${S1_Dropout}_*"

            if ($null -eq $best) {
                Write-Host "  [SKIP] No Stage 1 results found. Run Stage 1 first." -ForegroundColor Yellow
                continue
            }

            $bestOpt = $best.optimizer
            $bestLR  = $best.lr

            foreach ($bs in $BatchSizes) {
                foreach ($do in $Dropouts) {
                    if ($bs -eq $S1_BatchSize -and $do -eq $S1_Dropout) { continue }

                    $cmd = "python train.py" +
                           " --variety $variety" +
                           " --model $model" +
                           " --optimizer $bestOpt" +
                           " --lr $bestLR" +
                           " --batch_size $bs" +
                           " --dropout $do" +
                           " --max_epochs $MaxEpochs"
                    Invoke-Train $cmd
                }
            }
        }
    }
}

# ── Phase 3: Cross-variety fine-tuning ───────────────────────────────────────
if ($Stage -eq "all" -or $Stage -eq "3") {
    Write-Host ""
    Write-Host ("=" * 65) -ForegroundColor Yellow
    Write-Host "  PHASE 3 - Cross-variety Fine-tuning" -ForegroundColor Yellow
    Write-Host "  Arabica->Robusta and Robusta->Arabica" -ForegroundColor Yellow
    Write-Host ("=" * 65) -ForegroundColor Yellow

    foreach ($model in $Models) {
        foreach ($srcVariety in $Varieties) {
            if ($srcVariety -eq "arabica") {
                $dstVariety = "robusta"
            } else {
                $dstVariety = "arabica"
            }

            Write-Host ""
            Write-Host "  -- $srcVariety -> $dstVariety / $model --" -ForegroundColor White

            $best = Get-BestConfig `
                -Variety $srcVariety `
                -Model $model `
                -SearchPattern "phase12_${srcVariety}_${model}_*"

            if ($null -eq $best) {
                Write-Host "  [SKIP] No Phase 1/2 results found." -ForegroundColor Yellow
                continue
            }

            $runPrefix = "phase12_${srcVariety}_${model}_$($best.optimizer)_lr$($best.lr)_bs$($best.batch_size)_do$($best.dropout)"
            $ckpt = Get-BestModelPath -RunName $runPrefix

            if ($null -eq $ckpt) {
                Write-Host "  [SKIP] Checkpoint not found for: $runPrefix" -ForegroundColor Yellow
                continue
            }

            $cmd = "python train.py" +
                   " --variety $dstVariety" +
                   " --model $model" +
                   " --optimizer $($best.optimizer)" +
                   " --lr $($best.lr)" +
                   " --batch_size $($best.batch_size)" +
                   " --dropout $($best.dropout)" +
                   " --max_epochs $MaxEpochs" +
                   " --pretrained_from `"$ckpt`""
            Invoke-Train $cmd
        }
    }
}

Write-Host ""
Write-Host ("=" * 65) -ForegroundColor Green
Write-Host "  ALL STAGES COMPLETE" -ForegroundColor Green
Write-Host "  Run: python aggregate_results.py" -ForegroundColor Green
Write-Host ("=" * 65) -ForegroundColor Green