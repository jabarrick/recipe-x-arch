# Runs every confirmatory stage in the registered order (Amendment 1 §3.7).
# Every stage is resumable: rerunning this script skips finished runs.
# Usage (from the repository root):  powershell -ExecutionPolicy Bypass -File tools\run_confirmatory.ps1
# "Continue": Windows PowerShell turns any stderr line of a native program (e.g. a
# torch.onnx UserWarning) into an error record; with "Stop" that killed the run (S4 row 10).
# Failures are detected by the exit code below instead.
$ErrorActionPreference = "Continue"
$env:MOABB_DOWNLOAD_PROVIDER = "upstream"      # read the local GigaDB files; no NEMAR prefetch (S4 row 9)
$C = "..\csoanet2026_v2_deploy"
$stages = @(
  @("F1a","recipes"), @("F1b","recipes"),      # F1 R1-R4
  @("F2","recipes"),                            # F2 R1-R4
  @("F1a","T"), @("F1b","T"),                   # T (F1)
  @("F1a","Tprime"), @("F1b","Tprime"),         # T' (F1)
  @("F2","T"),                                  # T (F2)
  @("F2","Tprime")                              # T' (F2)
)
New-Item -ItemType Directory -Force results\confirmatory | Out-Null
foreach ($s in $stages) {
  $unit, $stage = $s
  $log = "results\confirmatory\run_${unit}_${stage}.log"
  "=== $unit $stage  start $((Get-Date).ToUniversalTime().ToString('s'))Z" | Tee-Object -Append $log
  python -u -m train.confirmatory --csoanet-root $C --unit $unit --stage $stage 2>&1 | ForEach-Object { "$_" } | Tee-Object -Append $log
  if ($LASTEXITCODE -ne 0) { "=== $unit $stage FAILED (exit $LASTEXITCODE)" | Tee-Object -Append $log; exit 1 }
  "=== $unit $stage  end $((Get-Date).ToUniversalTime().ToString('s'))Z" | Tee-Object -Append $log
}
"=== all stages finished"
