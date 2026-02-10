@echo off
title Validated Terrain-Occluded Radar Simulation
cd /d "%~dp0..\Validated-Terrain-Occluded-Radar-Simulation"

pip install numpy pygame-ce scipy >nul 2>&1

echo ============================================================
echo   Validated Terrain-Occluded Radar Simulation
echo   Synthetic PPI Training Data Generator
echo ============================================================
echo.
echo Launching radar simulator...
echo.
echo   Controls:
echo     Space    = Pause/Resume
echo     T        = Add island terrain
echo     Delete   = Clear terrain
echo     C        = Toggle coastline
echo     E        = Export sweep to CSV
echo     F5       = Start/stop recording
echo     +/-      = Speed up/slow down
echo     Scroll   = Zoom PPI range
echo     Escape   = Quit
echo.
echo   Features:
echo     - LOAD LOCATION: Import .radarloc files with real coastlines
echo       * Lakes (Lake Murray, etc.): 98%+ accuracy vs real charts
echo       * Land creates 13m elevation for radar occlusion
echo       * Occluded vessels show [OCCLUDED] and gray out
echo     - Validation panel: Load real Furuno CSV, compare synthetic
echo     - Annotation export: JSON / COCO / YOLO bounding boxes
echo     - Batch generation: python generate_training_data.py
echo.
echo   Generate .radarloc files using: Generate Radar Location.bat
echo.
python main.py
echo.
pause
