# Runs F1b under R1 without cropping (resumes from finished runs), then the exploratory
# analysis, then commits and pushes everything (manuscript V1.2 included).
# Usage (repository root):  powershell -ExecutionPolicy Bypass -File tools\after_r1nc.ps1
$ErrorActionPreference = "Continue"
$env:MOABB_DOWNLOAD_PROVIDER = "upstream"
$out = "results\exploratory_r1nocrop"
New-Item -ItemType Directory -Force $out, "docs\exploratory_r1nocrop" | Out-Null
"=== F1b R1nc start $((Get-Date).ToUniversalTime().ToString('s'))Z" | Tee-Object -Append "$out\run_F1b.log"
python -u -m train.explore_r1_nocrop --csoanet-root ..\csoanet2026_v2_deploy --unit F1b --out $out 2>&1 | ForEach-Object { "$_" } | Tee-Object -Append "$out\run_F1b.log"
if ($LASTEXITCODE -ne 0) { "=== F1b R1nc FAILED (exit $LASTEXITCODE)" | Tee-Object -Append "$out\run_F1b.log"; exit 1 }
"=== F1b R1nc end $((Get-Date).ToUniversalTime().ToString('s'))Z" | Tee-Object -Append "$out\run_F1b.log"
python -m analysis.explore_r1nc_analysis --res results\confirmatory --nc $out --out docs\exploratory_r1nocrop 2>&1 | ForEach-Object { "$_" } | Tee-Object "docs\exploratory_r1nocrop\analysis.log"
Copy-Item "$out\F1a_long.csv" "docs\exploratory_r1nocrop\F1a_R1nc_long.csv"
Copy-Item "$out\F1b_long.csv" "docs\exploratory_r1nocrop\F1b_R1nc_long.csv"
git add analysis train tools docs
git commit -m "Manuscript V1.2 (second review); exploratory F1 under R1 without cropping and its analysis (S4 row 15)"
git push
"=== all done $((Get-Date).ToUniversalTime().ToString('s'))Z"
