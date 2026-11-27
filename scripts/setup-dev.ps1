$ErrorActionPreference = 'Stop'

function Invoke-InstallCommand {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit ${LASTEXITCODE}): $Program $($Arguments -join ' ')"
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repoRoot
try {
    $venvPath = Join-Path $repoRoot 'video_recognition/.venv'
    $venvPython = Join-Path $venvPath 'Scripts/python.exe'
    $versionCheck = 'import sys; raise SystemExit(sys.version_info[:2] != (3, 14))'

    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            Invoke-InstallCommand 'py' @('-3.14', '-m', 'venv', $venvPath)
        } else {
            Invoke-InstallCommand 'python' @('-c', $versionCheck)
            Invoke-InstallCommand 'python' @('-m', 'venv', $venvPath)
        }
    }
    & $venvPython -c $versionCheck
    if ($LASTEXITCODE -ne 0) {
        throw "The existing environment at $venvPath must use Python 3.14. Move it aside and run this script again."
    }

    Invoke-InstallCommand $venvPython @('-m', 'pip', 'install', '--upgrade', 'pip')
    $requirements = @(
        'video_recognition/requirements-integrated.txt',
        'frontend/requirements.txt',
        'faster_whisper_stt/requirements.txt',
        'ai_agent/requirements.txt'
    )
    foreach ($file in $requirements) {
        Invoke-InstallCommand $venvPython @('-m', 'pip', 'install', '-r', $file)
    }
    Write-Host 'Setup complete. See README.md for the Streamlit launch command.'
} finally {
    Pop-Location
}
