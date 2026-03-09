# JewelUX Unified Startup Script (PowerShell)

Write-Host "🚀 Starting JewelUX Project..." -ForegroundColor Cyan

# Define Paths
$BaseDir = $PSScriptRoot
$BackendDir = Join-Path $BaseDir "backend"
$FrontendDir = Join-Path $BaseDir "frontend"
$VenvPython = Join-Path $BaseDir ".venv\Scripts\python.exe"

# 1. Start Backend
Write-Host "📦 Starting Backend (FastAPI)..." -ForegroundColor Yellow
$BackendJob = Start-Process -FilePath $VenvPython -ArgumentList "run.py" -WorkingDirectory $BackendDir -PassThru -NoNewWindow

# 2. Start Frontend
Write-Host "🎨 Starting Frontend (Vite)..." -ForegroundColor Yellow
$FrontendJob = Start-Process -FilePath "npm.cmd" -ArgumentList "run", "dev" -WorkingDirectory $FrontendDir -PassThru -NoNewWindow

Write-Host "`n✅ Both services are starting!" -ForegroundColor Green
Write-Host "🔗 Frontend: http://localhost:5173" -ForegroundColor Blue
Write-Host "🔗 Backend:  http://localhost:8000" -ForegroundColor Blue
Write-Host "`nTo stop the servers, close this terminal or use: Stop-Process -Id $($BackendJob.Id), $($FrontendJob.Id)" -ForegroundColor Gray

# Optional: Keep window open if run by double-clicking
# Pause
