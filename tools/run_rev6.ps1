# Exploratory runs requested by the sixth internal review (S4 row 20), in priority order.
# Resumable: re-running skips finished (architecture, recipe, fold, seed) runs.
# Every trained model is saved under results\exploratory_rev6\models\ (not committed; archive to OSF/Zenodo).
# Usage (repository root):  powershell -ExecutionPolicy Bypass -File tools\run_rev6.ps1
$ErrorActionPreference = "Continue"
$env:MOABB_DOWNLOAD_PROVIDER = "upstream"
$out = "results\exploratory_rev6"
New-Item -ItemType Directory -Force $out, "docs\exploratory_rev6" | Out-Null
$base = "..\csoanet2026_v2_deploy"
$all4 = "EEGNet", "ShallowConvNet", "MI-BMInet", "CSOANet-v2"
# stage: name, architectures, recipes   (about 4-5 min per run on the RTX 4060)
$stages = @(
  @("A_R2va",        $all4,                 @("R2va")),           # R2 checkpoint rule isolated   (120 runs)
  @("B_ShallowMN",   @("ShallowConvNet-MN"), @("R2", "R4")),      # max-norm ShallowConvNet        (60 runs)
  @("C_R1ct",        $all4,                 @("R1ct")),           # crop also at validation/test  (120 runs)
  @("D_R1ns",        $all4,                 @("R1ns")),           # crop without shift            (120 runs)
  @("E_ShallowMN2",  @("ShallowConvNet-MN"), @("R1", "R1nc", "R3")) # rest of the max-norm row     (90 runs)
)
foreach ($s in $stages) {
  foreach ($unit in "F1a", "F1b") {
    $log = "$out\run_$($s[0])_$unit.log"
    "=== $($s[0]) $unit start $((Get-Date).ToUniversalTime().ToString('s'))Z" | Tee-Object -Append $log
    python -u -m train.explore_rev6 --csoanet-root $base --unit $unit --out $out --archs $s[1] --recipes $s[2] 2>&1 |
      ForEach-Object { "$_" } | Tee-Object -Append $log
    if ($LASTEXITCODE -ne 0) { "=== $($s[0]) $unit FAILED (exit $LASTEXITCODE)" | Tee-Object -Append $log; exit 1 }
    "=== $($s[0]) $unit end $((Get-Date).ToUniversalTime().ToString('s'))Z" | Tee-Object -Append $log
  }
  Copy-Item "$out\F1a_long.csv" "docs\exploratory_rev6\F1a_rev6_long.csv" -ErrorAction SilentlyContinue
  Copy-Item "$out\F1b_long.csv" "docs\exploratory_rev6\F1b_rev6_long.csv" -ErrorAction SilentlyContinue
  git add train tools docs\exploratory_rev6
  git commit -m "Exploratory rev6 runs: stage $($s[0]) finished (S4 row 20)"
  git push
}
"=== all rev6 stages done $((Get-Date).ToUniversalTime().ToString('s'))Z"
