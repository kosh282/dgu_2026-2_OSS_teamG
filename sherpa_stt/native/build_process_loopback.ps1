$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw 'Visual Studio 2022 C++ 빌드 도구가 필요합니다.'
}
$vsInstall = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsInstall) {
    throw 'Visual Studio Installer에서 MSVC x64/x86 빌드 도구를 찾지 못했습니다.'
}
$devCmd = Join-Path $vsInstall 'Common7\Tools\VsDevCmd.bat'
$source = Join-Path $scriptDir 'process_loopback.cpp'
$output = Join-Path $scriptDir 'process_loopback.exe'
$command = 'call "{0}" -arch=x64 -host_arch=x64 >nul && cd /d "{1}" && cl.exe /nologo /O2 /EHsc /std:c++17 /D_WIN32_WINNT=0x0A00 process_loopback.cpp /Fe:"{2}" /link ole32.lib mmdevapi.lib' -f $devCmd, $scriptDir, $output
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) {
    throw "프로세스 루프백 도우미 빌드 실패 (exit=$LASTEXITCODE)"
}
Write-Output "빌드 완료: $output"
