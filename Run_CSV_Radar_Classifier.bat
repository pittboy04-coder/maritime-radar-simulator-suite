@echo off
title CSV Radar Classifier
echo ========================================
echo    CSV Radar Classifier
echo    Marine Radar Data Analysis Tool
echo ========================================
echo.

REM Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

REM Change to the csv-radar-classifier directory
cd /d "C:\Users\Noah\OneDrive - Strategy Communications\Desktop\csv-radar-classifier"

REM Check for pygame
python -c "import pygame" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required packages...
    pip install pygame numpy
    echo.
)

REM Run the application
echo Starting CSV Radar Classifier...
echo.
python main.py

if %errorlevel% neq 0 (
    echo.
    echo Application exited with an error.
    pause
)
