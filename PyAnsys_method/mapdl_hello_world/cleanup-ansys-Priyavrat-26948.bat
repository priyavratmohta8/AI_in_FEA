@echo off
set LOCALHOST=%COMPUTERNAME%
if /i "%LOCALHOST%"=="Priyavrat" (taskkill /f /pid 17072)
if /i "%LOCALHOST%"=="Priyavrat" (taskkill /f /pid 26948)

del /F cleanup-ansys-Priyavrat-26948.bat
