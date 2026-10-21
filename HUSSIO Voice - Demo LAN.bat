@echo off
chcp 65001 >nul
title HUSSIO Voice - DEMO cho sep (mang LAN)
cd /d "%~dp0"

rem Tat app dang chay o cong 10601 (neu co) de mo lai cho ca mang LAN
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":10601 .*LISTENING"') do taskkill /PID %%p /F >nul 2>&1

rem Lay IP LAN hien tai cua may (IP co the doi moi ngay)
set "IP="
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' -and $_.PrefixOrigin -ne 'WellKnown' } | Select-Object -First 1).IPAddress"`) do set "IP=%%i"
if not defined IP (
  echo.
  echo   Khong tim thay IP mang LAN. Kiem tra day mang / wifi roi chay lai.
  pause
  exit /b 1
)
echo http://%IP%:10601| clip

echo.
echo   ==========================================================
echo     HUSSIO VOICE - DEMO CHO SEP (cung mang LAN)
echo.
echo     Sep mo tren may sep:   http://%IP%:10601
echo     Link da chep san: dan vao Zalo gui sep.
echo.
echo     Dang khoi dong, doi khoang 15 giay...
echo     Xem xong: DONG CUA SO NAY de tat che do LAN.
echo     Dung lai tren may minh: bam "HUSSIO Voice.bat".
echo   ==========================================================
echo.
uv run omnivoice-demo --ip 0.0.0.0
pause
