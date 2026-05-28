$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$CONDA_DIR = Join-Path $SCRIPT_DIR "miniconda3"
$ENV_NAME = "glm-ocr"
$PYTHON_VER = "3.12"
$MODEL_DIR = Join-Path $SCRIPT_DIR "models\GLM-OCR"

function Write-Box($msg) {
    $line = "+" + ("-" * 50) + "+"
    Write-Host ""
    Write-Host $line -ForegroundColor Cyan
    Write-Host "| $msg" -ForegroundColor Cyan
    Write-Host $line -ForegroundColor Cyan
}

function Write-Step($num, $total, $msg) {
    Write-Host ""
    Write-Host "[$num/$total] $msg" -ForegroundColor Yellow
}

function Write-OK($msg) {
    Write-Host "  [OK] $msg" -ForegroundColor Green
}

function Write-Warn($msg) {
    Write-Host "  [!!] $msg" -ForegroundColor Red
}

function Write-Info($msg) {
    Write-Host "  ... $msg" -ForegroundColor Gray
}

Write-Box "GLM-OCR One-Click Setup"

# ==================== Step 1: Conda ====================
Write-Step 1 5 "Checking Conda environment..."

$condaExe = $null

try {
    $sysConda = Get-Command conda -ErrorAction SilentlyContinue
    if ($sysConda) {
        $condaExe = $sysConda.Source
        Write-OK "System Conda found"
    }
} catch {}

if (-not $condaExe) {
    $localConda = Join-Path $CONDA_DIR "Scripts\conda.exe"
    if (Test-Path $localConda) {
        $condaExe = $localConda
        $env:PATH = "$CONDA_DIR;$CONDA_DIR\Scripts;$CONDA_DIR\Library\bin;$env:PATH"
        Write-OK "Local Miniconda found"
    }
}

if (-not $condaExe) {
    Write-Warn "Conda not found! Please install Miniconda first."
    Read-Host "Press Enter to exit"
    exit 1
}

# ==================== Step 2: Create Environment ====================
Write-Step 2 5 "Creating Python environment..."

Write-Info "Accepting conda channel Terms of Service..."
& $condaExe tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>&1 | Out-Null
& $condaExe tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>&1 | Out-Null
& $condaExe tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 2>&1 | Out-Null

$envExists = & $condaExe env list 2>&1 | Select-String "^$ENV_NAME\s"
if ($envExists) {
    Write-OK "Environment '$ENV_NAME' already exists"
} else {
    Write-Info "Creating environment with Python $PYTHON_VER..."
    & $condaExe create -n $ENV_NAME python=$PYTHON_VER -y
    if ($LASTEXITCODE -ne 0) {
        Write-Info "Default channels failed, trying conda-forge..."
        & $condaExe create -n $ENV_NAME python=$PYTHON_VER -y --override-channels -c conda-forge
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Failed to create environment!"
            Read-Host "Press Enter to exit"
            exit 1
        }
    }
    Write-OK "Environment created"
}

$envPython = Join-Path $CONDA_DIR "envs\$ENV_NAME\python.exe"
$envPip = Join-Path $CONDA_DIR "envs\$ENV_NAME\Scripts\pip.exe"

if (-not (Test-Path $envPython)) {
    Write-Warn "Python not found in environment!"
    Read-Host "Press Enter to exit"
    exit 1
}

# ==================== Step 3: Install Dependencies ====================
Write-Step 3 5 "Installing dependencies..."

Write-Info "Configuring pip mirror (Aliyun)..."
& $envPip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ 2>&1 | Out-Null
& $envPip config set install.trusted-host mirrors.aliyun.com 2>&1 | Out-Null

Write-Info "Installing PyTorch (CUDA 12.4)..."
& $envPip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) {
    Write-Info "CUDA 12.4 failed, trying CUDA 12.1..."
    & $envPip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
}
if ($LASTEXITCODE -ne 0) {
    Write-Info "CUDA 12.1 failed, trying pip mirror..."
    & $envPip install torch torchvision
}
Write-OK "PyTorch installed"

Write-Info "Installing core dependencies..."
$deps = @(
    "transformers>=4.50.0",
    "modelscope>=1.34.0",
    "accelerate",
    "safetensors",
    "sentencepiece",
    "protobuf"
)
foreach ($dep in $deps) {
    Write-Info "  Installing: $dep"
    & $envPip install $dep
}
Write-OK "Core dependencies installed"

Write-Info "Installing web UI dependencies..."
& $envPip install gradio pillow pymupdf
Write-OK "Web UI dependencies installed"

# ==================== Step 4: Download Model ====================
Write-Step 4 5 "Downloading GLM-OCR model..."

if (Test-Path (Join-Path $MODEL_DIR "config.json")) {
    Write-OK "Model already exists, skipping download"
} else {
    Write-Info "Downloading GLM-OCR model from ModelScope..."
    Write-Info "(Model is about 2GB, first download may take a while)"
    Write-Host ""

    New-Item -ItemType Directory -Force -Path $MODEL_DIR | Out-Null
    & $envPython -c @"
from modelscope import snapshot_download
snapshot_download('ZhipuAI/GLM-OCR', local_dir=r'$MODEL_DIR', resume_download=True)
"@

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Info "ModelScope download failed, trying HuggingFace mirror..."
        $env:HF_ENDPOINT = "https://hf-mirror.com"
        & $envPython -c @"
from huggingface_hub import snapshot_download
snapshot_download('zai-org/GLM-OCR', local_dir=r'$MODEL_DIR', resume_download=True)
"@
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Model download failed!"
            Write-Host "  Please download manually:"
            Write-Host "  1. Visit: https://modelscope.cn/models/ZhipuAI/GLM-OCR"
            Write-Host "  2. Download all files to: $MODEL_DIR"
            Read-Host "Press Enter to exit"
            exit 1
        }
    }
    Write-OK "Model downloaded"
}

# ==================== Step 5: Verify ====================
Write-Step 5 5 "Verifying installation..."

& $envPython -c @"
import torch, transformers, gradio
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  PyTorch: {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  Gradio: {gradio.__version__}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
else:
    print(f'  WARNING: No GPU detected, will use CPU (slower)')
"@

Write-Host ""
Write-Box "Setup Complete!"
Write-Host ""
Write-Host "  Double-click YiJianQiDongGLMOCR.bat to start" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to exit"
