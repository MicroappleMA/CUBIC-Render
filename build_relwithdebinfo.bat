@echo off

echo =========================================
echo ============ CUBIC Render ===============
echo ======== RelWithDebInfo Build ===========
echo =========================================

set BUILD_PATH="build"
set ARTIFACT_PATH="CUBIC-Render"

CALL build_base.bat "RelWithDebInfo" %BUILD_PATH%
CALL build_post.bat "RelWithDebInfo" %BUILD_PATH% %ARTIFACT_PATH%

pause