@echo off
setlocal EnableDelayedExpansion

REM ====================================================
REM Step 2: Copy the "integration" directory into build\
xcopy integration build\integration /s /e /i /q

REM ====================================================
REM Step 3: Copy the "res\integration" directory into build\res\integration
xcopy res\integration build\res\integration /s /e /i /q

REM ====================================================
REM Step 4: Copy the installation script into build\
copy deploy\windows\scripts\install.bat build\

REM ====================================================
REM Step 5: Create a timestamp (format: DDMMYYhhmm)
for /f "skip=1 tokens=1" %%x in ('wmic os get LocalDateTime') do (
    if not defined ldt set ldt=%%x
)
set timestamp=%ldt:~6,2%%ldt:~4,2%%ldt:~2,2%%ldt:~8,2%%ldt:~10,2%

REM ====================================================
REM Step 6: Create the self-extracting archive with normal compression (-mx5)
"%ProgramFiles%\7-Zip\7z.exe" a -sfx lsa-win-%timestamp%-partial.7z build -mx5

echo Package created: lsa-win-%timestamp%.7z
pause
