# run_all_experiments.ps1
# Runs all 24 training + evaluation combinations in the correct order.
# Phase 1 & 2 must finish before Phase 3 (they produce the weights Phase 3 needs).
#
# Usage:
#   .\run_all_experiments.ps1
#   .\run_all_experiments.ps1 -DryRun      # print commands without running them

param(
    [switch]$DryRun
)

$Models   = @("mobilenetv1", "mobilenetv2", "mobilenetv3", "efficientnet")
$Varieties = @("arabica", "robusta")

function Invoke-Train {
    param([string]$Cmd)
    if ($DryRun) {
        Write-Host "[DRY RUN] $Cmd" -ForegroundColor Cyan
    } else {
        Write-Host "`n>>> $Cmd" -ForegroundColor Green
        Invoke-Expression $Cmd
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] Command failed. Stopping." -ForegroundColor Red
            exit 1
        }
    }
}

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 & 2 — Train each model on each variety from ImageNet weights
# Produces: runs/phase12_arabica_<model>/ and runs/phase12_robusta_<model>/
# ──────────────────────────────────────────────────────────────────────────────
Write-Host "`n========== PHASE 1 & 2: Train from ImageNet ==========" -ForegroundColor Yellow

foreach ($variety in $Varieties) {
    foreach ($model in $Models) {
        Invoke-Train "python train.py --variety $variety --model $model"
    }
}

# ──────────────────────────────────────────────────────────────────────────────
# Resolve Phase 3 weight paths (most recent run for each variety+model)
# ──────────────────────────────────────────────────────────────────────────────
function Get-BestModel {
    param([string]$Variety, [string]$Model)
    $pattern = "runs\phase12_${Variety}_${Model}_*\best_model.keras"
    $match = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue |
             Sort-Object LastWriteTime -Descending |
             Select-Object -First 1
    if ($null -eq $match) {
        Write-Host "[WARN] No Phase 1/2 checkpoint found for $Variety/$Model" -ForegroundColor Yellow
        return $null
    }
    return $match.FullName
}

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3A — Arabica-trained model fine-tuned on Robusta
# ──────────────────────────────────────────────────────────────────────────────
Write-Host "`n========== PHASE 3A: Arabica → Robusta ==========" -ForegroundColor Yellow

foreach ($model in $Models) {
    $weights = Get-BestModel -Variety "arabica" -Model $model
    if ($weights) {
        Invoke-Train "python train.py --variety robusta --model $model --pretrained_from `"$weights`""
    }
}

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3B — Robusta-trained model fine-tuned on Arabica
# ──────────────────────────────────────────────────────────────────────────────
Write-Host "`n========== PHASE 3B: Robusta → Arabica ==========" -ForegroundColor Yellow

foreach ($model in $Models) {
    $weights = Get-BestModel -Variety "robusta" -Model $model
    if ($weights) {
        Invoke-Train "python train.py --variety arabica --model $model --pretrained_from `"$weights`""
    }
}

Write-Host "`n========== ALL EXPERIMENTS COMPLETE ==========" -ForegroundColor Green
Write-Host "Results saved under: .\runs\" -ForegroundColor Green