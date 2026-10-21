# Reproduce the MTAH possession submission. Windows PowerShell 5.1.
#   scripts\reproduce.ps1                  full chain: simulator -> PathCRF cache -> LOMO selection -> candidate
#   scripts\reproduce.ps1 -CandidateOnly   regenerate the CSV from the committed models/<Name>/ (config + dead model)
param(
    [switch]$CandidateOnly,
    [string]$Name = "a2"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# 1. mtah venv (Python 3.12). Pins: pyproject.toml ([project].dependencies); there is no requirements.txt.
$Venv = $env:MTAH_VENV
if (-not $Venv) { $Venv = "HOME/mtah-venv" }
$Py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "mtah venv not found at $Venv. Create it with:"
    Write-Host "  py -3.12 -m venv $Venv"
    Write-Host "  $Py -m pip install -e `".[dev]`"     (dependencies are pinned in pyproject.toml)"
    Write-Host "or set MTAH_VENV to an existing Python 3.12 venv."
    exit 1
}

# 2. competition data
if (-not $env:MTAH_DATA_DIR -or -not (Test-Path $env:MTAH_DATA_DIR)) {
    Write-Host "MTAH_DATA_DIR must point at the unpacked data bundle (tactical/, broadcast/)."
    exit 1
}

if ($CandidateOnly) {
    # 3a. committed artefacts -> runs/select/<Name>
    $Sel = Join-Path $Root "runs\select\$Name"
    $Mod = Join-Path $Root "models\$Name"
    New-Item -ItemType Directory -Force $Sel | Out-Null
    # config.json, the dead model and every other model artefact the candidate uses (team / merge / switch)
    $Files = @("config.json", "dead_model.joblib") + @(Get-ChildItem -Path $Mod -Filter "*.joblib" | Where-Object { $_.Name -ne "dead_model.joblib" } | ForEach-Object { $_.Name })
    foreach ($f in $Files) {
        if (-not (Test-Path (Join-Path $Mod $f))) { Write-Host "missing $Mod\$f"; exit 1 }
        Copy-Item -Force (Join-Path $Mod $f) (Join-Path $Sel $f)   # always overwrite with the committed artefact
        (Get-FileHash -Algorithm MD5 (Join-Path $Sel $f)).Hash.ToLower() + "  runs/select/$Name/$f"
    }
    # config.json may use PathCRF -> same PathCRF checks as the full mode
}

# 3. PathCRF (external, MPL-2.0, own venv) - see third_party/pathcrf/SETUP.md
$PcRepo = $env:MTAH_PATHCRF_REPO
if (-not $PcRepo) { $PcRepo = "HOME/mtah-ext\pathcrf" }
$PcPy = $env:MTAH_PATHCRF_PY
if (-not $PcPy) { $PcPy = "HOME/mtah-ext\pathcrf-venv\Scripts\python.exe" }
if (-not (Test-Path $PcRepo) -or -not (Test-Path $PcPy)) {
    Write-Host "PathCRF checkout or venv missing (repo: $PcRepo, python: $PcPy)."
    Write-Host "Follow third_party/pathcrf/SETUP.md, or set MTAH_PATHCRF_REPO / MTAH_PATHCRF_PY."
    exit 1
}

if ($CandidateOnly) {
    & $Py scripts\make_candidate.py --select "runs/select/$Name" --name $Name
    if ($LASTEXITCODE -ne 0) { exit 1 }
    $Csv = Join-Path $Root "runs\candidates\$Name.csv"
    (Get-FileHash -Algorithm MD5 $Csv).Hash.ToLower() + "  runs/candidates/$Name.csv"
    exit 0
}

# Full mode. Needs the DFL open-data parse (scripts/build_sim.py --dfl, default on E:) and runs/dfl_extra
# (match 9007, parsed from J03WR9). PathCRF threads are pinned to 4 inside the code.
# Raw DFL XML: MTAH_DFL_RAW, default E:\football-synthesizer\data\dfl_open. parse_dfl_match.py produces runs/dfl_extra.
$DflRaw = $env:MTAH_DFL_RAW
if (-not $DflRaw) { $DflRaw = "E:\football-synthesizer\data\dfl_open" }
if (-not (Test-Path $DflRaw)) { Write-Host "raw DFL XML not found at $DflRaw (set MTAH_DFL_RAW)."; exit 1 }
& $Py scripts\parse_dfl_match.py DFL-MAT-J03WR9 $DflRaw runs/dfl_extra 9007
if ($LASTEXITCODE -ne 0) { exit 1 }
& $Py scripts\build_sim.py
if ($LASTEXITCODE -ne 0) { exit 1 }
& $Py scripts\build_sim.py --extra-dir runs/dfl_extra --only 9007
if ($LASTEXITCODE -ne 0) { exit 1 }
& $Py scripts\precompute_pathcrf.py --sim runs/sim --seeds 100 101 --real
if ($LASTEXITCODE -ne 0) { exit 1 }
& $Py scripts\select_lomo.py --grid grid2 --sim runs/sim --seeds 100 101 --out "runs/select/$Name"
if ($LASTEXITCODE -ne 0) { exit 1 }
# The sealed read (select_lomo.py --sealed) is a deliberate, logged, one-off step and is not repeated here.
& $Py scripts\make_candidate.py --select "runs/select/$Name" --name $Name
if ($LASTEXITCODE -ne 0) { exit 1 }
$Csv = Join-Path $Root "runs\candidates\$Name.csv"
(Get-FileHash -Algorithm MD5 $Csv).Hash.ToLower() + "  runs/candidates/$Name.csv"
