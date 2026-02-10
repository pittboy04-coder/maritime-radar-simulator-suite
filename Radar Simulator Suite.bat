@echo off
setlocal enabledelayedexpansion
title Marine Radar Simulator Suite
color 0B

:MAIN_MENU
cls
echo.
echo  ===============================================================================
echo                       MARINE RADAR SIMULATOR SUITE
echo                    Research ^& Training Data Generation
echo  ===============================================================================
echo.
echo   [1]  Launch Visual Simulator        - Interactive PPI with terrain ^& vessels
echo   [2]  Generate Training Data         - Batch create labeled datasets for ML
echo   [3]  Calibrate from Real Capture    - Extract clutter/RCS from your radar data
echo   [4]  Validate Simulation            - Compare simulation vs real capture
echo   [5]  Generate Location File         - Create .radarloc from GPS coordinates
echo   [6]  Quick Simulate (CLI)           - Fast command-line simulation
echo.
echo   [7]  Open CSV Data Folder           - View exported radar data
echo   [8]  Open Documentation             - User guides and reference
echo.
echo   [0]  Exit
echo.
echo  ===============================================================================
set /p choice="  Select option [0-8]: "

if "%choice%"=="1" goto VISUAL_SIMULATOR
if "%choice%"=="2" goto BATCH_GENERATE
if "%choice%"=="3" goto CALIBRATE
if "%choice%"=="4" goto VALIDATE
if "%choice%"=="5" goto GENERATE_LOCATION
if "%choice%"=="6" goto QUICK_SIMULATE
if "%choice%"=="7" goto OPEN_DATA
if "%choice%"=="8" goto OPEN_DOCS
if "%choice%"=="0" exit
goto MAIN_MENU

:VISUAL_SIMULATOR
cls
echo.
echo  ===============================================================================
echo                         VISUAL RADAR SIMULATOR
echo  ===============================================================================
echo.
echo   Launching interactive simulator...
echo.
echo   Controls:
echo     Space     Pause/Resume simulation
echo     T         Add terrain/island
echo     C         Toggle coastline display
echo     E         Export current sweep to CSV
echo     F5        Start/Stop recording
echo     +/-       Speed up/slow down
echo     Scroll    Change radar range
echo     Escape    Quit
echo.
echo   Features:
echo     - Load .radarloc files for real coastlines
echo     - Terrain occlusion with elevation data
echo     - Vessel tracking and annotation export
echo.
cd /d "%~dp0..\Validated-Terrain-Occluded-Radar-Simulation"
pip install numpy pygame-ce scipy >nul 2>&1
python main.py
echo.
pause
goto MAIN_MENU

:BATCH_GENERATE
cls
echo.
echo  ===============================================================================
echo                       BATCH TRAINING DATA GENERATOR
echo  ===============================================================================
echo.
echo   Generate labeled radar datasets for machine learning.
echo.
echo   Output includes:
echo     - CSV sweep data (720 sweeps x 1024 range bins)
echo     - JSON labels with vessel positions, RCS, and settings
echo     - Randomized clutter, gain, and vessel configurations
echo.
set /p num_scenarios="  Number of scenarios to generate [default=100]: "
if "%num_scenarios%"=="" set num_scenarios=100

set /p max_targets="  Maximum targets per scenario [default=15]: "
if "%max_targets%"=="" set max_targets=15

set /p range_nm="  Radar range in nautical miles [default=6]: "
if "%range_nm%"=="" set range_nm=6

set output_dir=%~dp0Training Data\batch_%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%
set output_dir=%output_dir: =0%

echo.
echo   Output folder: %output_dir%
echo.
echo   Generating %num_scenarios% scenarios...
echo.

cd /d "%~dp0..\Radar-Simulator-V3\cpp\build"
if exist "radar_sim" (
    .\radar_sim generate --scenarios %num_scenarios% --targets %max_targets% --range %range_nm% --output "%output_dir%" --verbose
) else (
    echo   ERROR: C++ simulator not built. Using Python fallback...
    cd /d "%~dp0..\Radar-Simulator-V3"
    python -c "from radar_v3.analysis.experiments import ParameterSweep; ParameterSweep().run()"
)

echo.
echo  ===============================================================================
echo   COMPLETE: Generated %num_scenarios% scenarios
echo   Location: %output_dir%
echo  ===============================================================================
pause
goto MAIN_MENU

:CALIBRATE
cls
echo.
echo  ===============================================================================
echo                    CALIBRATE FROM REAL CAPTURE
echo  ===============================================================================
echo.
echo   Analyze real radar data to extract:
echo     - Gain level
echo     - Sea clutter intensity
echo     - Rain clutter intensity
echo     - Detected objects with RCS estimates
echo.
echo   Drag-and-drop your CSV file path or enter it below.
echo.
set /p capture_file="  Real capture CSV file: "
if "%capture_file%"=="" (
    echo   No file specified.
    pause
    goto MAIN_MENU
)

set /p threshold="  Detection threshold [default=0.5]: "
if "%threshold%"=="" set threshold=0.5

echo.
echo   Analyzing capture...
echo.

cd /d "%~dp0..\Radar-Simulator-V3\cpp\build"
if exist "radar_sim" (
    .\radar_sim analyze --capture "%capture_file%" --threshold %threshold% --output "%~dp0Calibration\analysis_%date:~-4%%date:~4,2%%date:~7,2%.json"
) else (
    echo   ERROR: C++ simulator not built.
)

echo.
echo  ===============================================================================
echo   CALIBRATION COMPLETE
echo.
echo   Use these values when simulating:
echo     --gain [value]  --sea [value]  --rain [value]
echo.
echo   To match detected objects, use their RCS values when adding vessels.
echo  ===============================================================================
pause
goto MAIN_MENU

:VALIDATE
cls
echo.
echo  ===============================================================================
echo                      VALIDATE SIMULATION
echo  ===============================================================================
echo.
echo   Compare simulation output against real radar capture.
echo.
echo   Metrics produced:
echo     - Echo correlation (pattern matching)
echo     - Detection accuracy (Pd/Pfa)
echo     - Object matching (position errors)
echo     - Overall quality score
echo.
set /p real_file="  Real capture CSV: "
if "%real_file%"=="" (
    echo   No file specified.
    pause
    goto MAIN_MENU
)

set /p sim_file="  Simulated data CSV: "
if "%sim_file%"=="" (
    echo   No file specified.
    pause
    goto MAIN_MENU
)

echo.
echo   Running validation...
echo.

cd /d "%~dp0..\Radar-Simulator-V3\cpp\build"
if exist "radar_sim" (
    .\radar_sim validate --real "%real_file%" --simulated "%sim_file%" --report "%~dp0Validation Outputs\validation_%date:~-4%%date:~4,2%%date:~7,2%.json" --verbose
) else (
    echo   ERROR: C++ simulator not built.
    cd /d "%~dp0..\Validated-Terrain-Occluded-Radar-Simulation"
    python -m radar_sim.validation "%real_file%" "%sim_file%"
)

echo.
pause
goto MAIN_MENU

:GENERATE_LOCATION
cls
echo.
echo  ===============================================================================
echo                      GENERATE LOCATION FILE
echo  ===============================================================================
echo.
echo   Create a .radarloc file from real-world coordinates.
echo.
echo   Data sources (free):
echo     - OpenStreetMap coastlines
echo     - Open-Elevation terrain data
echo.
echo   Enter location name OR coordinates:
echo     Examples: "Lake Murray, SC" or "34.0818,-81.2169"
echo.
set /p location="  Location: "
if "%location%"=="" (
    echo   No location specified.
    pause
    goto MAIN_MENU
)

set /p range="  Range in nautical miles [default=6]: "
if "%range%"=="" set range=6

set /p use_terrain="  Include terrain elevation? (Y/N) [default=Y]: "
if "%use_terrain%"=="" set use_terrain=Y

echo.
echo   Generating location file...
echo.

cd /d "%~dp0marine-radar-location-generator"
if "%use_terrain%"=="Y" (
    python generate_location.py "%location%" --range %range% --terrain -o "%~dp0Locations\%location: =_%.radarloc"
) else (
    python generate_location.py "%location%" --range %range% -o "%~dp0Locations\%location: =_%.radarloc"
)

echo.
echo  ===============================================================================
echo   LOCATION FILE GENERATED
echo   Use "Load Location" in the visual simulator to import it.
echo  ===============================================================================
pause
goto MAIN_MENU

:QUICK_SIMULATE
cls
echo.
echo  ===============================================================================
echo                         QUICK SIMULATE (CLI)
echo  ===============================================================================
echo.
echo   Fast command-line simulation with custom parameters.
echo.
set /p targets="  Number of targets [default=5]: "
if "%targets%"=="" set targets=5

set /p range="  Range (nm) [default=6]: "
if "%range%"=="" set range=6

set /p rotations="  Antenna rotations [default=1]: "
if "%rotations%"=="" set rotations=1

set /p gain="  Gain (0-1) [default=0.5]: "
if "%gain%"=="" set gain=0.5

set /p sea="  Sea clutter (0-1) [default=0.3]: "
if "%sea%"=="" set sea=0.3

set /p rain="  Rain clutter (0-1) [default=0.0]: "
if "%rain%"=="" set rain=0.0

set output_file=%~dp0CSV data\simulation_%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%.csv
set output_file=%output_file: =0%

echo.
echo   Generating simulation...
echo.

cd /d "%~dp0..\Radar-Simulator-V3\cpp\build"
if exist "radar_sim" (
    .\radar_sim simulate --targets %targets% --range %range% --rotations %rotations% --gain %gain% --sea %sea% --rain %rain% --output "%output_file%" --verbose
) else (
    echo   ERROR: C++ simulator not built.
)

echo.
echo  ===============================================================================
echo   SIMULATION COMPLETE
echo   Output: %output_file%
echo  ===============================================================================
pause
goto MAIN_MENU

:OPEN_DATA
start "" "%~dp0CSV data"
goto MAIN_MENU

:OPEN_DOCS
if exist "%~dp0docs" (
    start "" "%~dp0docs"
) else (
    start "" "%~dp0"
)
goto MAIN_MENU
