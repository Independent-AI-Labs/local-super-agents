@echo off
REM ====================================================
REM Clean up any existing build folder
if exist build rd /s /q build

REM Create a fresh build folder
mkdir build

REM ====================================================
REM Step 1: Copy the 7z installer into the build folder
copy "C:\7z2409-x64.exe" build\

REM ====================================================
REM Step 2: Copy the "integration" directory into build\
xcopy integration build\integration /s /e /i /q

REM ====================================================
REM Step 3: Copy the "res\integration" directory into build\integration\res
xcopy res\integration build\integration\res /s /e /i /q

REM ====================================================
REM Step 4: Copy the installation script into build\
copy deploy\windows\scripts\install.bat build\

REM ====================================================
REM Step 5: Create a timestamp (format: DDMMYYhhmm)
REM Using WMIC to get the local datetime in a consistent format.
for /f "skip=1 tokens=1" %%x in ('wmic os get LocalDateTime') do (
    if not defined ldt set ldt=%%x
)
REM ldt is in the format: YYYYMMDDhhmmss....
set timestamp=%ldt:~6,2%%ldt:~4,2%%ldt:~2,2%%ldt:~8,2%%ldt:~10,2%

REM Create the self-extracting archive with normal compression level (-mx5)
"%ProgramFiles%\7-Zip\7z.exe" a -sfx lsa-win-%timestamp%.7z build -mx5

REM ====================================================
REM Step 6: Delete the build folder
rd /s /q build

REM End of script
echo Package created: lsa-win-%timestamp%.7z
pause
