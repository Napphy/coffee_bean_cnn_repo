# run_para_search.ps1
# ============================================================
# Two-stage para search for coffee defect classifiers
#
# Stage 1 - Find best learning rate PER OPTIMIZER
#   Fixed: batch_size=32, dropout=0.3, max_epochs=100
#   Grid : 5 optimizers x 3 learning rates = 15 runs per model/variety
#
# Stage 2 - Find best batch size + dropout PER OPTIMIZER
#   (uses each optimizer's best LR from Stage 1)
#   Fixed: max_epochs=100
#   Grid : 5 optimizers x 3 batch sizes x 2 dropouts = 30 runs per model/variety
#   
#
# Phase 3 - Cross-variety fine-tuning using best overall config PER OPTIMIZER
#   5 optimizers x 4 models x 2 directions = 40 runs
#
# Rationale:
#   We keep every optimizer's best configuration through every stage
#   (rather than collapsing to a single "best optimizer") so that Adan
#   can be directly compared against RMSProp/SGD/Adam/AdamW under its
#   own best-tuned settings, not just relative to whichever optimizer
#   happened to win Stage 1.
#
# Total runs: (15 + 25) x 4 models x 2 varieties + 40 = 320 + 40 = 360
# Estimated time on GTX 1660 Ti: significantly longer than the
# single-best-optimizer approach - plan accordingly.
#
# Usage:
#   .\run_para_search.ps1              # run everything
#   .\run_para_search.ps1 -DryRun      # preview commands only
#   .\run_para_search.ps1 -Stage 1     # run Stage 1 only
#   .\run_para_search.ps1 -Stage 2     # run Stage 2 only (needs Stage 1 done)
#   .\run_para_search.ps1 -Stage 3     # run Phase 3 only (needs Stage 2 done)
# ============================================================

param(
    [switch]$DryRun,
    [ValidateSet("all", "1", "2", "3")]
    [string]$Stage = "all"
)

# ── Config ────────────────────────────────────────────────────────────────────
$Models     = @("squeezenet", "shufflenet", "mobilenetv2", "mobilenetv3", "efficientnet")
$Varieties  = @("arabica", "robusta")
$LRs        = @("1e-3", "1e-4", "1e-5")
$Optimizers = @("adam", "adamw", "sgd", "rmsprop", "adan")
$BatchSizes = @(16, 32, 64)
$Dropouts   = @(0.3, 0.5)
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

function Test-RunExists {
    <#
    Checks whether a results.json already exists for the given run prefix.
    A run is considered "finished" only if results.json is present -
    a checkpoint alone (best_model.pth) without results.json means the
    run was interrupted mid-training and should be re-run.
    #>
    param([string]$RunPrefix)
    $match = Get-ChildItem -Path "runs\${RunPrefix}*\results.json" -ErrorAction SilentlyContinue |
             Select-Object -First 1
    return ($null -ne $match)
}

function Invoke-TrainIfNeeded {
    <#
    Runs training only if no completed run (results.json) matches RunPrefix.
    Otherwise skips with a message. RunPrefix should match the naming
    convention produced by train.py:
      phase12_<variety>_<model>_<opt>_lr<lr>_bs<bs>_do<do>
      phase3_<variety>_<model>_<opt>_lr<lr>_bs<bs>_do<do>
    #>
    param(
        [string]$Cmd,
        [string]$RunPrefix
    )
    if (Test-RunExists -RunPrefix $RunPrefix) {
        Write-Host "  [SKIP] Already completed: $RunPrefix" -ForegroundColor DarkGray
        return
    }
    Invoke-Train $Cmd
}

function Get-BestLRForOptimizer {
    <#
    Returns the best (highest weighted-avg F1 on same-variety test set)
    run for a SPECIFIC optimizer, searched within a glob pattern.
    #>
    param(
        [string]$Variety,
        [string]$Model,
        [string]$Optimizer,
        [string]$SearchPattern
    )
    $best   = $null
    $bestF1 = -1.0
    $found  = Get-ChildItem -Path "runs\$SearchPattern\results.json" -ErrorAction SilentlyContinue

    foreach ($f in $found) {
        $json = Get-Content $f.FullName | ConvertFrom-Json
        if ($json.optimizer -ne $Optimizer) { continue }
        $f1 = $json.results[0].report.'weighted avg'.'f1-score'
        if ($f1 -gt $bestF1) {
            $bestF1 = $f1
            $best   = $json
        }
    }

    if ($null -eq $best) {
        Write-Host "  [WARN] No results found for $Optimizer matching: $SearchPattern" -ForegroundColor Yellow
        return $null
    }

    $msg = "    Best {0} config for {1}/{2}: lr={3} bs={4} do={5} F1={6:F4}" -f `
        $Optimizer, $Variety, $Model, $best.lr, $best.batch_size, $best.dropout, $bestF1
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

# ── Stage 1: Learning Rate search, per optimizer ─────────────────────────────
if ($Stage -eq "all" -or $Stage -eq "1") {
    Write-Host ""
    Write-Host ("=" * 65) -ForegroundColor Yellow
    Write-Host "  STAGE 1 - Learning Rate Search (per optimizer)" -ForegroundColor Yellow
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
                    $prefix = "phase12_${variety}_${model}_${opt}_lr${lr}_bs${S1_BatchSize}_do${S1_Dropout}"
                    Invoke-TrainIfNeeded -Cmd $cmd -RunPrefix $prefix
                }
            }
        }
    }
}

# ── Stage 2: Batch Size x Dropout, per optimizer ─────────────────────────────
if ($Stage -eq "all" -or $Stage -eq "2") {
    Write-Host ""
    Write-Host ("=" * 65) -ForegroundColor Yellow
    Write-Host "  STAGE 2 - Batch Size x Dropout Search (per optimizer)" -ForegroundColor Yellow
    Write-Host "  Each optimizer uses its own best LR from Stage 1" -ForegroundColor Yellow
    Write-Host ("=" * 65) -ForegroundColor Yellow

    foreach ($variety in $Varieties) {
        foreach ($model in $Models) {
            Write-Host ""
            Write-Host "  -- $variety / $model --" -ForegroundColor White

            foreach ($opt in $Optimizers) {
                $best = Get-BestLRForOptimizer `
                    -Variety $variety `
                    -Model $model `
                    -Optimizer $opt `
                    -SearchPattern "phase12_${variety}_${model}_${opt}_*_bs${S1_BatchSize}_do${S1_Dropout}_*"

                if ($null -eq $best) {
                    Write-Host "  [SKIP] No Stage 1 results for $opt. Run Stage 1 first." -ForegroundColor Yellow
                    continue
                }

                $bestLR = $best.lr

                foreach ($bs in $BatchSizes) {
                    foreach ($do in $Dropouts) {
                        # Skip the combo already run in Stage 1 (bs=32, do=0.3)
                        if ($bs -eq $S1_BatchSize -and $do -eq $S1_Dropout) { continue }

                        $cmd = "python train.py" +
                               " --variety $variety" +
                               " --model $model" +
                               " --optimizer $opt" +
                               " --lr $bestLR" +
                               " --batch_size $bs" +
                               " --dropout $do" +
                               " --max_epochs $MaxEpochs"
                        $prefix = "phase12_${variety}_${model}_${opt}_lr${bestLR}_bs${bs}_do${do}"
                        Invoke-TrainIfNeeded -Cmd $cmd -RunPrefix $prefix
                    }
                }
            }
        }
    }
}

# ── Phase 3: Cross-variety fine-tuning, per optimizer ────────────────────────
if ($Stage -eq "all" -or $Stage -eq "3") {
    Write-Host ""
    Write-Host ("=" * 65) -ForegroundColor Yellow
    Write-Host "  PHASE 3 - Cross-variety Fine-tuning (per optimizer)" -ForegroundColor Yellow
    Write-Host "  Arabica->Robusta and Robusta->Arabica" -ForegroundColor Yellow
    Write-Host "  Each optimizer's best overall config is carried forward" -ForegroundColor Yellow
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

            foreach ($opt in $Optimizers) {
                # Search across ALL Stage 1 + Stage 2 results for this optimizer
                $best = Get-BestLRForOptimizer `
                    -Variety $srcVariety `
                    -Model $model `
                    -Optimizer $opt `
                    -SearchPattern "phase12_${srcVariety}_${model}_${opt}_*"

                if ($null -eq $best) {
                    Write-Host "  [SKIP] No Phase 1/2 results for $opt." -ForegroundColor Yellow
                    continue
                }

                $runPrefix = "phase12_${srcVariety}_${model}_${opt}_lr$($best.lr)_bs$($best.batch_size)_do$($best.dropout)"
                $ckpt = Get-BestModelPath -RunName $runPrefix

                if ($null -eq $ckpt) {
                    Write-Host "  [SKIP] Checkpoint not found for: $runPrefix" -ForegroundColor Yellow
                    continue
                }

                $cmd = "python train.py" +
                       " --variety $dstVariety" +
                       " --model $model" +
                       " --optimizer $opt" +
                       " --lr $($best.lr)" +
                       " --batch_size $($best.batch_size)" +
                       " --dropout $($best.dropout)" +
                       " --max_epochs $MaxEpochs" +
                       " --pretrained_from `"$ckpt`""
                $prefix = "phase3_${dstVariety}_${model}_${opt}_lr$($best.lr)_bs$($best.batch_size)_do$($best.dropout)"
                Invoke-TrainIfNeeded -Cmd $cmd -RunPrefix $prefix
            }
        }
    }
}

Write-Host ""
Write-Host ("=" * 65) -ForegroundColor Green
Write-Host "  ALL STAGES COMPLETE" -ForegroundColor Green
Write-Host "  Run: python aggregate_results.py" -ForegroundColor Green
Write-Host ("=" * 65) -ForegroundColor Green