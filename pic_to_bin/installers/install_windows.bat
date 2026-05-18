@echo off
setlocal enabledelayedexpansion
title Pic-to-Bin Fusion Add-In Installer

echo ===========================================================
echo  Pic-to-Bin Fusion 360 Add-In Installer (Windows)
echo ===========================================================
echo.

REM Fusion's user-add-ins folder lives under %APPDATA% (Roaming). The
REM product directory has been "Autodesk Fusion 360" historically and
REM may appear as "Autodesk Fusion" on some newer installs; check both.
set "BASE=%APPDATA%\Autodesk"
set "TARGET="

for %%D in ("Autodesk Fusion 360" "Autodesk Fusion") do (
    if exist "%BASE%\%%~D\API\" (
        set "TARGET=%BASE%\%%~D\API\AddIns"
        goto :found
    )
)

echo Could not find a Fusion 360 user-API folder. Looked in:
echo   %BASE%\Autodesk Fusion 360\API
echo   %BASE%\Autodesk Fusion\API
echo.
echo If Fusion 360 isn't installed yet, install it from
echo https://www.autodesk.com/products/fusion/personal and open it
echo at least once, then re-run this installer.
echo.
echo (Note: the C:\...\AppData\Local\Autodesk\webdeploy\... folder
echo holds Fusion's built-in add-ins and is wiped on every Fusion
echo update — user add-ins do not go there.)
echo.
pause
exit /b 1

:found
echo Installing to:
echo   !TARGET!\pic_to_bin
echo.

if not exist "!TARGET!" mkdir "!TARGET!"

REM /E recurse, /I assume dir, /Y overwrite, /Q quiet, /EXCLUDE filters dirs
xcopy /E /I /Y /Q "%~dp0AddIns\pic_to_bin" "!TARGET!\pic_to_bin\"
if errorlevel 1 (
    echo.
    echo Install failed. See the error above.
    pause
    exit /b 1
)

echo.
echo ===========================================================
echo  Success! Finish setup inside Fusion 360:
echo ===========================================================
echo   1. Open Fusion 360
echo   2. Press Shift+S, switch to the Add-Ins tab
echo   3. Select 'pic_to_bin' and click Run
echo   4. Tick "Run on Startup" so it loads every session
echo.
echo The new button appears under:
echo   Solid - Create - Gridfinity Pic-to-Bin
echo.
pause
