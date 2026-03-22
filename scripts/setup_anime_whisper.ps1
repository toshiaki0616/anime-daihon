$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ModelOutput = Join-Path $ProjectRoot "models\anime-whisper-ct2"
$TempModelDir = Join-Path $env:TEMP "anime-whisper-src"

function Test-PythonCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [string[]]$PrefixArgs = @()
    )

    try {
        & $Executable @PrefixArgs -c "import sys; print(sys.version)" | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Resolve-PythonCommand {
    if ($env:ANIME_WHISPER_PYTHON) {
        $override = @{ Executable = $env:ANIME_WHISPER_PYTHON; PrefixArgs = @() }
        if (Test-PythonCommand -Executable $override.Executable -PrefixArgs $override.PrefixArgs) {
            return $override
        }
        throw "ANIME_WHISPER_PYTHON is set but could not be executed: $($env:ANIME_WHISPER_PYTHON)"
    }

    $candidates = @(
        @{ Executable = (Join-Path $ProjectRoot ".venv\Scripts\python.exe"); PrefixArgs = @() },
        @{ Executable = (Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\python.exe"); PrefixArgs = @() },
        @{ Executable = (Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\python3.exe"); PrefixArgs = @() },
        @{ Executable = "python"; PrefixArgs = @() },
        @{ Executable = "py"; PrefixArgs = @("-3") }
    )

    foreach ($candidate in $candidates) {
        if (Test-PythonCommand -Executable $candidate.Executable -PrefixArgs $candidate.PrefixArgs) {
            return $candidate
        }
    }

    throw "Python was not found. Rebuild .venv or install Python 3.10+ and run this script again."
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$PythonCommand,
        [Parameter(ValueFromRemainingArguments = $true)]
        [object[]]$Arguments
    )

    & $PythonCommand.Executable @($PythonCommand.PrefixArgs) @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed."
    }
}

function Invoke-PythonScript {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$PythonCommand,
        [Parameter(Mandatory = $true)]
        [string]$ScriptText
    )

    $tempScript = [System.IO.Path]::ChangeExtension([System.IO.Path]::GetTempFileName(), ".py")
    try {
        Set-Content -Path $tempScript -Value $ScriptText -Encoding UTF8
        Invoke-Python $PythonCommand $tempScript
    }
    finally {
        Remove-Item $tempScript -Force -ErrorAction SilentlyContinue
    }
}

function Test-ModelOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DirectoryPath
    )

    $requiredFiles = @("config.json", "model.bin", "tokenizer.json")
    foreach ($name in $requiredFiles) {
        if (-not (Test-Path (Join-Path $DirectoryPath $name))) {
            return $false
        }
    }
    return $true
}

$PythonCommand = Resolve-PythonCommand

Write-Host "Converting Anime Whisper to CTranslate2 format..." -ForegroundColor Cyan
Write-Host "Output: $ModelOutput" -ForegroundColor DarkGray
Write-Host "Temp dir: $TempModelDir" -ForegroundColor DarkGray
Write-Host "Python: $($PythonCommand.Executable) $($PythonCommand.PrefixArgs -join ' ')" -ForegroundColor DarkGray

Invoke-Python $PythonCommand -m pip install "transformers[torch]>=4.23" ctranslate2 huggingface_hub sentencepiece

if (Test-Path $TempModelDir) {
    Remove-Item $TempModelDir -Recurse -Force
}
if (Test-Path $ModelOutput) {
    Remove-Item $ModelOutput -Recurse -Force
}
New-Item -ItemType Directory -Path $TempModelDir | Out-Null
New-Item -ItemType Directory -Path $ModelOutput | Out-Null

Write-Host "Downloading Anime Whisper from Hugging Face..." -ForegroundColor Cyan
Invoke-PythonScript $PythonCommand @'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="litagin/anime-whisper",
    local_dir=r"TEMP_MODEL_DIR",
    local_dir_use_symlinks=False,
)
'@.Replace("TEMP_MODEL_DIR", $TempModelDir.Replace("\", "\\"))

Write-Host "Generating tokenizer.json..." -ForegroundColor Cyan
Invoke-PythonScript $PythonCommand @'
from transformers import AutoTokenizer
src = r"TEMP_MODEL_DIR"
out = r"TEMP_MODEL_DIR"
tokenizer = AutoTokenizer.from_pretrained(src)
tokenizer.save_pretrained(out, legacy_format=False)
'@.Replace("TEMP_MODEL_DIR", $TempModelDir.Replace("\", "\\"))

Write-Host "Running CTranslate2 conversion..." -ForegroundColor Cyan
Invoke-Python $PythonCommand -m ctranslate2.converters.transformers `
  --model $TempModelDir `
  --output_dir $ModelOutput `
  --copy_files tokenizer.json preprocessor_config.json tokenizer_config.json `
  --quantization int8 `
  --force

if (-not (Test-ModelOutput -DirectoryPath $ModelOutput)) {
    throw "Model conversion did not finish correctly. Check models\anime-whisper-ct2 for config.json, model.bin, and tokenizer.json."
}

Write-Host "Anime Whisper conversion finished." -ForegroundColor Green
Write-Host "Output: $ModelOutput" -ForegroundColor Green
