@echo off
title Marine Radar Location Generator
cd /d "%~dp0marine-radar-location-generator"

pip install requests numpy >nul 2>&1

echo ============================================
echo   Marine Radar Location Generator
echo ============================================
echo.
echo Options:
echo   1. Generate location file (lakes work with names)
echo   2. Look up water coordinates (for bays/ocean areas)
echo.

set /p CHOICE="Enter choice (1 or 2): "

if "%CHOICE%"=="2" goto LOOKUP
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
