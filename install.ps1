# MKAngel PowerShell Installer
# Run: powershell -ExecutionPolicy Bypass -File install.ps1

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  MKAngel Installer for Windows" -ForegroundColor White
Write-Host "  Grammar Language Model - Pure Python, Zero Dependencies" -ForegroundColor Gray
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $pyver = python --version 2>&1
    Write-Host "  Python: $pyver" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Python not found." -ForegroundColor Red
    Write-Host "  Install Python 3.10+ from python.org" -ForegroundColor Yellow
    Write-Host "  Make sure 'Add Python to PATH' is checked." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

$InstallDir = "$env:USERPROFILE\MKAngel"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "[1/4] Installing to $InstallDir..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Copy-Item -Recurse -Force "$ScriptDir\glm" "$InstallDir\glm"
Copy-Item -Recurse -Force "$ScriptDir\app" "$InstallDir\app"
Copy-Item -Recurse -Force "$ScriptDir\assets" "$InstallDir\assets"
Copy-Item -Force "$ScriptDir\demo.py" "$InstallDir\"
Copy-Item -Force "$ScriptDir\pyproject.toml" "$InstallDir\"

Write-Host "[2/4] Creating launchers..." -ForegroundColor Yellow

# mkangel.bat - demo
@"
@echo off
cd /d "$InstallDir"
python demo.py
pause
"@ | Set-Content "$InstallDir\mkangel.bat"

# mkangel-chat.bat - interactive chat
@"
@echo off
cd /d "$InstallDir"
python -c "import sys; sys.path.insert(0,'.'); from app.conductor import AngelConductor; from app.chat import ChatSession; c = AngelConductor().awaken(); s = ChatSession(c); s.run()"
"@ | Set-Content "$InstallDir\mkangel-chat.bat"

Write-Host "[3/4] Adding to PATH..." -ForegroundColor Yellow
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$InstallDir", "User")
    Write-Host "  Added to user PATH. Open a NEW terminal for 'mkangel' to work." -ForegroundColor Gray
} else {
    Write-Host "  Already in PATH." -ForegroundColor Gray
}

Write-Host "[4/4] Creating desktop shortcut..." -ForegroundColor Yellow
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\MKAngel.lnk")
$Shortcut.TargetPath = "$InstallDir\mkangel.bat"
$Shortcut.WorkingDirectory = $InstallDir
$Shortcut.Description = "MKAngel - Grammar Language Model"
$Shortcut.Save()

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Close this terminal, open a new one, then:" -ForegroundColor White
Write-Host "    mkangel          (run the demo)" -ForegroundColor White
Write-Host "    mkangel-chat     (start chat)" -ForegroundColor White
Write-Host ""
Write-Host "  Or double-click MKAngel on your desktop." -ForegroundColor White
Write-Host ""
Write-Host "  23 domains | 47 grammars | 1954 words | 388 strange loops" -ForegroundColor Gray
Write-Host "  Pure Python | Zero dependencies | ~3 MB" -ForegroundColor Gray
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to close"
