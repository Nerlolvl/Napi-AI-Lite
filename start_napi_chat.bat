@echo off
chcp 65001 > nul
cd /d "%~dp0"
python chat_terminal.py
pause
