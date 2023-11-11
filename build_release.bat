@echo off

echo =========================================
echo ============ CUBIC Render ===============
echo =========== Release Build ===============
echo =========================================

set BUILD_PATH="build"
set ARTIFACT_PATH="CUBIC-Render"

CALL build_base.bat "Release" %BUILD_PATH%
CALL build_post.bat "Release" %BUILD_PATH% %ARTIFACT_PATH%

pause