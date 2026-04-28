@echo off
:: MKAngel Windows Installer
:: Creates a launcher command and desktop shortcut

echo.
echo ============================================================
echo   MKAngel Installer for Windows
echo   Grammar Language Model - Pure Python, Zero Dependencies
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org
    echo         Make sure "Add Python to PATH" is checked during install.
    pause
    exit /b 1
)

:: Set install directory
set "INSTALL_DIR=%USERPROFILE%\MKAngel"
set "SCRIPT_DIR=%~dp0"

echo [1/4] Installing to %INSTALL_DIR%...
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
xcopy /E /I /Y "%SCRIPT_DIR%glm" "%INSTALL_DIR%\glm" >nul
xcopy /E /I /Y "%SCRIPT_DIR%app" "%INSTALL_DIR%\app" >nul
xcopy /E /I /Y "%SCRIPT_DIR%assets" "%INSTALL_DIR%\assets" >nul
copy /Y "%SCRIPT_DIR%demo.py" "%INSTALL_DIR%\" >nul
copy /Y "%SCRIPT_DIR%pyproject.toml" "%INSTALL_DIR%\" >nul

:: Create mkangel.bat launcher
echo [2/4] Creating launcher...
(
echo @echo off
echo cd /d "%INSTALL_DIR%"
echo python demo.py
echo pause
) > "%INSTALL_DIR%\mkangel.bat"

:: Create mkangel-chat.bat
(
echo @echo off
echo cd /d "%INSTALL_DIR%"
echo python -c "import sys; sys.path.insert(0,'.'); from app.conductor import AngelConductor; from app.chat import ChatSession; c = AngelConductor().awaken(); s = ChatSession(c); s.run()"
) > "%INSTALL_DIR%\mkangel-chat.bat"

:: Add to PATH via user environment variable
echo [3/4] Adding to PATH...
for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "CURRENT_PATH=%%B"
echo %CURRENT_PATH% | findstr /I /C:"%INSTALL_DIR%" >nul
if errorlevel 1 (
    setx PATH "%CURRENT_PATH%;%INSTALL_DIR%" >nul 2>&1
    echo   Added %INSTALL_DIR% to user PATH
    echo   NOTE: Open a NEW terminal for 'mkangel' command to work
) else (
    echo   Already in PATH
)

:: Create desktop shortcut
echo [4/4] Creating desktop shortcut...
set "SHORTCUT=%USERPROFILE%\Desktop\MKAngel.lnk"
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath = '%INSTALL_DIR%\mkangel.bat'; $s.WorkingDirectory = '%INSTALL_DIR%'; $s.Description = 'MKAngel - Grammar Language Model'; $s.Save()"

echo.
echo ============================================================
echo   Installation complete!
echo.
echo   Close this terminal and open a new one, then type:
echo     mkangel          (run the demo)
echo     mkangel-chat     (start chat)
echo.
echo   Or double-click the MKAngel icon on your desktop.
echo.
echo   23 domains ^| 47 grammars ^| 1954 words ^| 388 strange loops
echo   Pure Python ^| Zero dependencies ^| ~3 MB
echo ============================================================
echo.
pause
