# Regenerate the submission file (default: candidate a17) from the committed models. Windows PowerShell 5.1 or PowerShell 7.
#   scripts\reproduce.ps1                   s: runs/candidates/s.csv, prints the md5 and compares it with the expected one
#   scripts\reproduce.ps1 -Name a17        the predecessor
#   scripts\reproduce.ps1 -Name a5c_fx2     the fallback candidate
# Needs (see README.md): the competition data bundle (MTAH_DATA_DIR), the main venv (MTAH_VENV, Python 3.12, requirements-main.txt),
# PathCRF checkout + torch venv (MTAH_PATHCRF_REPO, MTAH_PATHCRF_PY; third_party/pathcrf/SETUP.md).
# It does not retrain anything: every trained artefact the candidate needs is in models/<Name>/.
param([string]$Name = "s")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# candidate -> PathCRF input smoothing (s) and expected md5 of the CSV
$Known = @{
    "s"        = @{ smooth = 1.0; md5 = "4c2184b6807ae6f2b0c110c3dc164388" }
    "a17"      = @{ smooth = 1.0; md5 = "afb9e7b4ede118268fd0222fd5e11068" }
    "a5c_fx2"  = @{ smooth = 1.0; md5 = "107e7fc1157c05ad3ba8f893734606b7" }
}
if (-not $Known.ContainsKey($Name)) { Write-Host "unknown candidate $Name (known: $($Known.Keys -join ', '))"; exit 1 }
$Smooth = $Known[$Name].smooth; $Expected = $Known[$Name].md5

$Home1 = [Environment]::GetFolderPath("UserProfile")
$Venv = $env:MTAH_VENV;        if (-not $Venv)   { $Venv = Join-Path $Home1 "mtah-venv" }
$Py   = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "main venv not found at $Venv. Create it:"
    Write-Host "  py -3.12 -m venv $Venv"
    Write-Host "  $Py -m pip install -r requirements-main.txt"
    Write-Host "  $Py -m pip install -e . --no-deps"
    Write-Host "or set MTAH_VENV to an existing Python 3.12 venv."
    exit 1
}
if (-not $env:MTAH_DATA_DIR -or -not (Test-Path (Join-Path $env:MTAH_DATA_DIR "tactical"))) {
    Write-Host "MTAH_DATA_DIR must point at the unpacked competition bundle (the folder that contains tactical/)."; exit 1
}
if (-not $env:MTAH_PATHCRF_REPO) { $env:MTAH_PATHCRF_REPO = Join-Path $Home1 "mtah-ext\pathcrf" }
if (-not $env:MTAH_PATHCRF_PY)   { $env:MTAH_PATHCRF_PY   = Join-Path $Home1 "mtah-ext\pathcrf-venv\Scripts\python.exe" }
if (-not (Test-Path $env:MTAH_PATHCRF_REPO) -or -not (Test-Path $env:MTAH_PATHCRF_PY)) {
    Write-Host "PathCRF checkout or torch venv missing (repo: $env:MTAH_PATHCRF_REPO, python: $env:MTAH_PATHCRF_PY)."
    Write-Host "Follow third_party/pathcrf/SETUP.md, or set MTAH_PATHCRF_REPO / MTAH_PATHCRF_PY."; exit 1
}
$Sel = "models/$Name"
if (-not (Test-Path (Join-Path $Root "$Sel\config.json"))) { Write-Host "missing $Sel\config.json"; exit 1 }

# PathCRF on the real match is cached in runs/cache/pathcrf_fx (about 6.5 min with 4 threads when the cache is empty).
& $Py scripts\make_candidate.py --select $Sel --name $Name --pathcrf-smooth $Smooth
if ($LASTEXITCODE -ne 0) { exit 1 }

$Csv = Join-Path $Root "runs\candidates\$Name.csv"
$Md5 = (Get-FileHash -Algorithm MD5 $Csv).Hash.ToLower()
Write-Host "$Md5  runs/candidates/$Name.csv"
if ($Md5 -eq $Expected) { Write-Host "md5 matches the expected value ($Expected)" }
else { Write-Host "MD5 DIFFERS from the expected $Expected"; exit 2 }
