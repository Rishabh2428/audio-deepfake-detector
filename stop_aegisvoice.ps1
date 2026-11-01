$ErrorActionPreference = "Stop"
$projectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $MyInvocation.MyCommand.Path))
$pidFile = Join-Path $projectRoot "data\server.pid.json"
if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Host "No AegisVoice server started by this project was found."
    exit 0
}

$saved = Get-Content -Raw -LiteralPath $pidFile | ConvertFrom-Json
if ([IO.Path]::GetFullPath($saved.project_root) -ne $projectRoot) {
    throw "The saved process record belongs to a different project directory; no process was stopped."
}
$target = Get-CimInstance Win32_Process -Filter "ProcessId = $($saved.pid)" -ErrorAction SilentlyContinue
if (-not $target) {
    Remove-Item -LiteralPath $pidFile -Force
    Write-Host "Removed a stale AegisVoice process record."
    exit 0
}
$expectedExecutable = [IO.Path]::GetFullPath($saved.python)
$actualExecutable = [IO.Path]::GetFullPath($target.ExecutablePath)
$isAegisVoice = $target.CommandLine -match "app\.main:app|app\.local_server"
if ($actualExecutable -ne $expectedExecutable -or -not $isAegisVoice) {
    throw "The recorded process no longer matches this AegisVoice server. No process was stopped."
}
Stop-Process -Id ([int]$saved.pid)
Remove-Item -LiteralPath $pidFile -Force
Write-Host "Stopped the AegisVoice server started from $projectRoot."
