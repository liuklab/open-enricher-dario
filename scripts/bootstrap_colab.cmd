@echo off
REM Windows cmd wrapper for scripts\bootstrap_colab.ps1
REM Usage: scripts\bootstrap_colab.cmd [metadata|embeddings] [extra colab exec args...]

setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap_colab.ps1" %*
exit /b %ERRORLEVEL%
