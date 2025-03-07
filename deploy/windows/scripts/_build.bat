@echo off
setlocal EnableDelayedExpansion

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
REM Step 3: Copy the "res\integration" directory into build\res\integration
xcopy res\integration build\res\integration /s /e /i /q
xcopy res\modelfiles build\res\modelfiles /s /e /i /q

REM ====================================================
REM Step 3.5: Process *.7z.dir directories in ..\lsa-redist\res\integration
REM For every directory under the source that ends in ".7z.dir",
REM archive it (removing the trailing ".dir") and put the resulting
REM archive file directly in the corresponding build\res\integration folder.

set "sourceDir=..\lsa-redist\res\integration"
REM Convert sourceDir to an absolute path.
for %%A in ("%sourceDir%") do set "absSourceDir=%%~fA"
REM Ensure absSourceDir ends with a backslash.
if not "%absSourceDir:~-1%"=="\" set "absSourceDir=%absSourceDir%\"

set "destDir=build\res\integration"
REM Convert sourceDir to an absolute path.
for %%A in ("%destDir%") do set "absDestDir=%%~fA"
REM Ensure absSourceDir ends with a backslash.
if not "%absDestDir:~-1%"=="\" set "absDestDir=%absDestDir%\"

REM Change directory into the absolute source folder.
pushd "%absSourceDir%"
REM Process every subdirectory (recursively) matching the pattern.
for /D /R %%D in (*.7z.dir) do (
    REM %%D is the full path of a folder ending in .7z.dir.
    REM Extract only its name (e.g. "foo.7z.dir")
    set "folderName=%%~nxD"
    REM Remove the trailing ".dir" so that the archive becomes "foo.7z"
    set "zipFileName=!folderName:~0,-4!"

    REM Get the parent folder of %%D (where we want to put the archive).
    set "parentDir=%%~dpD"
    REM Compute the relative path (i.e. the portion after absSourceDir).
    set "relParent=!parentDir:%absSourceDir%=!"
    if defined relParent if "!relParent:~-1!"=="\" (
        set "relParent=!relParent:~0,-1!"
    )
    if "!relParent!"=="" (
        set "destDir=%absDestDir%"
    ) else (
        set "destDir=%absDestDir%!relParent!"
    )

    if not exist "!destDir!" mkdir "!destDir!"

    echo Archiving "%%D" to "!destDir!\!zipFileName!"
    "%ProgramFiles%\7-Zip\7z.exe" a -mx5 "!destDir!\!zipFileName!" "%%D"
)
popd

REM ====================================================
REM Step 4: Copy the installation script into build\
copy deploy\windows\scripts\*install* build\
copy deploy\windows\scripts\*update* build\

REM ====================================================
REM Step 5: Create a timestamp (format: DDMMYYhhmm)
for /f "skip=1 tokens=1" %%x in ('wmic os get LocalDateTime') do (
    if not defined ldt set ldt=%%x
)
set timestamp=%ldt:~6,2%%ldt:~4,2%%ldt:~2,2%%ldt:~8,2%%ldt:~10,2%

REM ====================================================
REM Step 6: Create the self-extracting archive with normal compression (-mx5)
"%ProgramFiles%\7-Zip\7z.exe" a -sfx lsa-win-%timestamp%.7z build -mx5

REM ====================================================
REM Step 7: Delete the build folder
REM rd /s /q build

echo Package created: lsa-win-%timestamp%.7z
pause
