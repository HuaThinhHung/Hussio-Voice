@echo off
chcp 65001 >nul
title HUSSIO Voice - http://localhost:10601
cd /d "%~dp0"
echo.
echo   HUSSIO VOICE
echo   Dang khoi dong... trinh duyet se tu mo: http://localhost:10601
echo   Giu cua so nay mo trong luc dung. Dong cua so hoac bam Ctrl+C de tat.
echo.
uv run omnivoice-demo --mo-trinh-duyet
pause
