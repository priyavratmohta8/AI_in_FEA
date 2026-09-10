@echo off
set LOCALHOST=%COMPUTERNAME%
if /i "%LOCALHOST%"=="Priyavrat" (taskkill /f /pid 20060)
if /i "%LOCALHOST%"=="Priyavrat" (taskkill /f /pid 28928)

del /F cleanup-ansys-Priyavrat-28928.bat
