@echo off

set BUILD_TYPE=%~1
set BUILD_PATH=%~2

IF "%BUILD_TYPE%"=="" (
    echo [CUBIC Render] Error: BUILD_TYPE is not set
    exit /b 1
)

IF "%BUILD_PATH%"=="" (
    echo [CUBIC Render] Error: BUILD_PATH is not set
    exit /b 1
)

echo [CUBIC Render] %BUILD_TYPE% Build: %BUILD_PATH%

powershell -ExecutionPolicy ByPass -File decompress_lib.ps1

cmake -S . -B %BUILD_PATH%/%BUILD_TYPE% -DCMAKE_BUILD_TYPE=%BUILD_TYPE%
cmake --build %BUILD_PATH%/%BUILD_TYPE% --config %BUILD_TYPE%

IF %ERRORLEVEL% EQU 0 (
    ECHO [CUBIC Render] Build Success
) ELSE (
    ECHO [CUBIC Render] Build Failed
    exit /b 1
)