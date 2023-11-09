@echo off

echo =========================================
echo ============ CUBIC Render ===============
echo ============ Debug Build ================
echo =========================================

set BUILD_PATH="build"
set ARTIFACT_PATH="CUBIC-Render"

CALL build_base.bat "Debug" %BUILD_PATH%
CALL build_post.bat "Debug" %BUILD_PATH% %ARTIFACT_PATH%

pause