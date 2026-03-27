@echo off
title Marine Radar Location Generator
cd /d "%~dp0marine-radar-location-generator"

pip install requests numpy >nul 2>&1

echo ============================================
echo   Marine Radar Location Generator
echo ============================================
echo.
echo Options:
echo   1. Generate location file (standard range)
echo   2. Look up water coordinates (for bays/ocean areas)
echo   3. Generate MARITIME location (auto-positions near coastline
echo      for DRS4DNXT short-range radar - best for Mode 12)
echo.

set /p CHOICE="Enter choice (1, 2, or 3): "

if "%CHOICE%"=="2" goto LOOKUP
if "%CHOICE%"=="3" goto MARITIME
goto GENERATE

:LOOKUP
echo.
echo ============================================
echo   Water Coordinate Lookup
echo ============================================
echo.
echo Enter the name of a water body to find its center coordinates.
echo Examples: Lake Murray, San Francisco Bay, Chesapeake Bay
echo.
set /p WATERNAME="Enter water body name: "
echo.
python lookup_water_coords.py "%WATERNAME%"
echo.
echo ============================================
pause
exit /b 0

:GENERATE
echo.
echo ============================================
echo   Generate Location File
echo ============================================
echo.
echo For LAKES: Enter the lake name (e.g., Lake Murray, South Carolina)
echo For BAYS/OCEAN: Use option 2 first to get water coordinates
echo.

set /p LOCATION="Enter location (name or lat,lon): "
set /p RANGE="Enter range in nautical miles (default 6): "
if "%RANGE%"=="" set RANGE=6

set /p TERRAIN="Include terrain/elevation data? (y/n, default n): "
set TFLAG=
if /i "%TERRAIN%"=="y" set TFLAG=--terrain

set /p OUTFILE="Output filename (without extension, .radarloc added automatically): "
set /p SAVEPATH="Save to folder path (leave blank for current directory): "
set OFLAG=
if not "%OUTFILE%"=="" (
    if not "%SAVEPATH%"=="" (
        set OFLAG=-o "%SAVEPATH%\%OUTFILE%"
    ) else (
        set OFLAG=-o "%OUTFILE%"
    )
) else (
    if not "%SAVEPATH%"=="" (
        echo ERROR: Must provide a filename when specifying a save path.
        pause
        exit /b 1
    )
)

echo.
echo Generating location file...
echo.
python generate_location.py "%LOCATION%" --range %RANGE% %TFLAG% %OFLAG%

echo.
echo ============================================
pause
exit /b 0

:MARITIME
echo.
echo ============================================
echo   Generate Maritime Location (Auto-Coastline)
echo ============================================
echo.
echo This mode auto-positions the radar near the nearest coastline
echo so features are within the DRS4DNXT's 0.125 NM range.
echo Great for Mode 12 in the Object Creation simulator.
echo.
echo For LAKES: Enter the lake name (e.g., Lake Murray, South Carolina)
echo For BAYS/OCEAN: Enter coordinates (e.g., 32.78,-79.93)
echo.

set /p MLOCATION="Enter location (name or lat,lon): "
set /p MRANGE="Search range in NM (default 6 - wider finds more coastlines): "
if "%MRANGE%"=="" set MRANGE=6

set /p MOUTFILE="Output filename (without extension): "
set /p MSAVEPATH="Save to folder path (leave blank for current directory): "
set MOFLAG=
if not "%MOUTFILE%"=="" (
    if not "%MSAVEPATH%"=="" (
        set MOFLAG=-o "%MSAVEPATH%\%MOUTFILE%"
    ) else (
        set MOFLAG=-o "%MOUTFILE%"
    )
)

echo.
echo Generating maritime location file...
echo.
python generate_location.py "%MLOCATION%" --range %MRANGE% --maritime %MOFLAG%

echo.
echo ============================================
pause
