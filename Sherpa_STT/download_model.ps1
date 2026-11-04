$ErrorActionPreference = 'Stop'
$ModelName = 'sherpa-onnx-streaming-zipformer-korean-2024-06-16'
New-Item -ItemType Directory -Force models | Out-Null
$Archive = Join-Path 'models' "$ModelName.tar.bz2"
$Url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$ModelName.tar.bz2"
curl.exe -L --fail --output $Archive $Url
if ($LASTEXITCODE -ne 0) { throw 'Model download failed' }
tar -xjf $Archive -C models
if ($LASTEXITCODE -ne 0) { throw 'Model extraction failed' }
Remove-Item $Archive
Write-Host "Model ready: models/$ModelName"
