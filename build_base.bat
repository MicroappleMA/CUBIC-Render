@echo off

set BUILD_TYPE=%~1
set BUILD_PATH=%~2

if "%BUILD_TYPE%"=="" (
    echo [CUBIC Builder] Error: BUILD_TYPE is not set
    exit /b 1
)

if "%BUILD_PATH%"=="" (
    echo [CUBIC Builder] Error: BUILD_PATH is not set
    exit /b 1
)

echo [CUBIC Builder] %BUILD_TYPE% Build: %BUILD_PATH%

if exist "%BUILD_PATH%/%BUILD_TYPE%" (
    echo [CUBIC Builder] Remove Outdated Build
    rd /S /Q "%BUILD_PATH%/%BUILD_TYPE%"
)

powershell -ExecutionPolicy ByPass -File decompress_lib.ps1

cmake -S . -B %BUILD_PATH%/%BUILD_TYPE% -DCMAKE_BUILD_TYPE=%BUILD_TYPE%
cmake --build %BUILD_PATH%/%BUILD_TYPE% --config %BUILD_TYPE%

if %ERRORLEVEL% EQU 0 (
    echo [CUBIC Builder] %BUILD_TYPE% Build Success
) else (
    echo [CUBIC Builder] %BUILD_TYPE% Build Failed
    exit /b 1
)