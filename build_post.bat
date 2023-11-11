@echo off

set BUILD_TYPE=%~1
set BUILD_PATH=%~2
set ARTIFACT_PATH=%~3

if "%BUILD_TYPE%"=="" (
    echo [CUBIC Builder] Error: BUILD_TYPE is not set
    exit /b 1
)

if "%BUILD_PATH%"=="" (
    echo [CUBIC Builder] Error: BUILD_PATH is not set
    exit /b 1
)

if "%ARTIFACT_PATH%"=="" (
    echo [CUBIC Builder] Error: ARTIFACT_PATH is not set
    exit /b 1
)

echo [CUBIC Builder] %BUILD_TYPE% Post Build: %ARTIFACT_PATH%

if not exist "%ARTIFACT_PATH%" (
    mkdir "%ARTIFACT_PATH%"
)

if exist "%ARTIFACT_PATH%/%BUILD_TYPE%" (
    echo [CUBIC Builder] Remove Outdated Artifact
    rd /S /Q "%ARTIFACT_PATH%/%BUILD_TYPE%"
)

echo [CUBIC Builder] Copy Build Artifacts
xcopy /E /I "%BUILD_PATH%\%BUILD_TYPE%\%BUILD_TYPE%\" "%ARTIFACT_PATH%\%BUILD_TYPE%\"

echo [CUBIC Builder] Copy defaultConfig.json to Build Folder
copy "config\defaultConfig.json" "%ARTIFACT_PATH%\%BUILD_TYPE%\defaultConfig.json"

echo [CUBIC Builder] Copy GLTF Model to Build Folder
xcopy /E /I "gltf\" "%ARTIFACT_PATH%\%BUILD_TYPE%\gltf\"

echo [CUBIC Builder] Copy README and LICENSE
copy "README.md" "%ARTIFACT_PATH%\%BUILD_TYPE%\README.md"
copy "LICENSE" "%ARTIFACT_PATH%\%BUILD_TYPE%\LICENSE"

if %ERRORLEVEL% EQU 0 (
    echo [CUBIC Builder] %BUILD_TYPE% Post Build Success
) else (
    echo [CUBIC Builder] %BUILD_TYPE% Post Build Failed
    exit /b 1
)