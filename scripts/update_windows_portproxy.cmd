@echo off
setlocal

if "%~1"=="" (
  echo Usage: update_windows_portproxy.cmd ^<WSL-IP^>
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Start-Process powershell -Verb RunAs -Wait -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File ""%~dp0update_windows_portproxy.ps1"" -WslIp %~1'"

exit /b %ERRORLEVEL%
