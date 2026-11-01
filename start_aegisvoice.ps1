$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$dataDir = Join-Path $projectRoot "data"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
$pidFile = Join-Path $dataDir "server.pid.json"
$stdoutPath = Join-Path $dataDir "server.log"
$stderrPath = Join-Path $dataDir "server-error.log"
$port = 8000
$url = "http://127.0.0.1:$port"

if (Test-Path -LiteralPath $pidFile) {
    try {
        $saved = Get-Content -Raw -LiteralPath $pidFile | ConvertFrom-Json
        if ([IO.Path]::GetFullPath($saved.project_root) -eq [IO.Path]::GetFullPath($projectRoot)) {
            $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $($saved.pid)" -ErrorAction SilentlyContinue
            if ($existing -and [IO.Path]::GetFullPath($existing.ExecutablePath) -eq [IO.Path]::GetFullPath($saved.python) -and $existing.CommandLine -match "app\.main:app|app\.local_server") {
                $runningPort = if ($saved.port) { [int]$saved.port } else { 8000 }
                Write-Host "AegisVoice is already running at http://127.0.0.1:$runningPort"
                exit 0
            }
        }
    } catch { }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pythonPrefix = @()
if (Test-Path -LiteralPath $venvPython) {
    $pythonPath = $venvPython
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $pythonPath = $pythonCommand.Source
    } else {
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if (-not $launcher) { throw "Python was not found. Install Python 3.12 or later, then retry." }
        $pythonPath = $launcher.Source
        $pythonPrefix = @("-3.12")
    }
}

$fullReady = $false
try {
    & $pythonPath @pythonPrefix -c "import fastapi, uvicorn, sqlalchemy, numpy, scipy, soundfile, multipart, sklearn, joblib, av"
    $fullReady = ($LASTEXITCODE -eq 0)
} catch { $fullReady = $false }

$portReady = $false
for ($candidate = 8000; $candidate -le 8010; $candidate++) {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $candidate)
    try {
        $listener.Start()
        $port = $candidate
        $portReady = $true
        break
    } catch {
        $socketException = $_.Exception
        while ($socketException -and -not ($socketException -is [System.Net.Sockets.SocketException])) { $socketException = $socketException.InnerException }
        if (-not $socketException -or $socketException.SocketErrorCode -notin @([System.Net.Sockets.SocketError]::AddressAlreadyInUse, [System.Net.Sockets.SocketError]::AccessDenied)) {
            throw "Could not bind to local port $candidate. $($_.Exception.Message)"
        }
    } finally {
        if ($listener) { $listener.Stop() }
    }
}
if (-not $portReady) { throw "Ports 8000 through 8010 are unavailable. Close another local service and retry." }
$url = "http://127.0.0.1:$port"

if ($fullReady) {
    $mode = "FastAPI full-format mode"
    $pythonArgs = @() + $pythonPrefix + @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$port")
} else {
    $mode = "offline PCM WAV mode"
    $pythonArgs = @() + $pythonPrefix + @("-m", "app.local_server", "--host", "127.0.0.1", "--port", "$port", "--data-dir", "data")
    Write-Host "Optional packages are unavailable; starting offline WAV mode (PCM WAV, 8 MB, 30 seconds)."
    Write-Host "For FLAC, OGG, AIFF, and the FastAPI API, install requirements.txt in .venv."
}

$process = Start-Process -FilePath $pythonPath -ArgumentList $pythonArgs -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
@{ pid = $process.Id; project_root = $projectRoot; python = $pythonPath; mode = $mode; port = $port } | ConvertTo-Json -Compress | Set-Content -LiteralPath $pidFile -Encoding UTF8

$healthy = $false
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    Start-Sleep -Milliseconds 500
    try {
        $health = Invoke-WebRequest -UseBasicParsing -Uri "$url/api/health" -TimeoutSec 2
        if ($health.StatusCode -eq 200) { $healthy = $true; break }
    } catch {
        $current = Get-Process -Id $process.Id -ErrorAction SilentlyContinue
        if (-not $current) { break }
    }
}

if (-not $healthy) {
    Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    $tail = if (Test-Path -LiteralPath $stderrPath) { Get-Content -Tail 20 -LiteralPath $stderrPath } else { @() }
    throw "AegisVoice did not become ready. Check data\server-error.log. $($tail -join ' ')"
}

Write-Host "Started $mode at $url"
Write-Host "Stop it with .\stop_aegisvoice.ps1. Server logs are in data\server.log and data\server-error.log."
if ($env:AEGISVOICE_OPEN_BROWSER -eq "1") { Start-Process $url }
