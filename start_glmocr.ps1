[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$CONDA_DIR = Join-Path $SCRIPT_DIR "miniconda3"
$ENV_NAME = "glm-ocr"

function Write-Box($msg) {
    $line = "+" + ("-" * 50) + "+"
    Write-Host ""
    Write-Host $line -ForegroundColor Cyan
    Write-Host "| $msg" -ForegroundColor Cyan
    Write-Host $line -ForegroundColor Cyan
}

Write-Box "GLM-OCR Launcher"

$condaExe = $null
try {
    $sysConda = Get-Command conda -ErrorAction SilentlyContinue
    if ($sysConda) { $condaExe = $sysConda.Source }
} catch {}

if (-not $condaExe) {
    $localConda = Join-Path $CONDA_DIR "Scripts\conda.exe"
    if (Test-Path $localConda) {
        $condaExe = $localConda
        $env:PATH = "$CONDA_DIR;$CONDA_DIR\Scripts;$CONDA_DIR\Library\bin;$env:PATH"
    }
}

if (-not $condaExe) {
    Write-Host ""
    Write-Host "  [ERROR] Conda not found!" -ForegroundColor Red
    Write-Host "  Please run the setup script first (install_glmocr.bat)" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

$envExists = & $condaExe env list 2>&1 | Select-String "^$ENV_NAME\s"
if (-not $envExists) {
    Write-Host ""
    Write-Host "  [ERROR] Environment '$ENV_NAME' not found!" -ForegroundColor Red
    Write-Host "  Please run the setup script first (install_glmocr.bat)" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

$envPython = Join-Path $CONDA_DIR "envs\$ENV_NAME\python.exe"

Write-Host ""
Write-Host "  Starting GLM-OCR Web UI..." -ForegroundColor Green
Write-Host ""
Write-Host "  +----------------------------------------------+" -ForegroundColor Cyan
Write-Host "  | Browser will open automatically              |" -ForegroundColor Cyan
Write-Host "  | URL: http://localhost:7861                   |" -ForegroundColor Cyan
Write-Host "  |                                              |" -ForegroundColor Cyan
Write-Host "  | After launch, click 'Load Model' button      |" -ForegroundColor Cyan
Write-Host "  | Model loading takes ~10-30s, please wait...  |" -ForegroundColor Cyan
Write-Host "  +----------------------------------------------+" -ForegroundColor Cyan
Write-Host ""

$env:HF_ENDPOINT = "https://hf-mirror.com"

Push-Location $SCRIPT_DIR
& $envPython glmocr_webui.py --port 7861
Pop-Location

Write-Host ""
Write-Host "  GLM-OCR closed." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
