@echo off

echo =========================================
echo ============ CUBIC Render ===============
echo ============= Full Build ================
echo =========================================
echo.

set BUILD_PATH="build"
set ARTIFACT_PATH="CUBIC-Render"

echo ============ Debug Build ================
CALL build_base.bat "Debug" %BUILD_PATH%
CALL build_post.bat "Debug" %BUILD_PATH% %ARTIFACT_PATH%
echo.

echo =========== Release Build ===============
CALL build_base.bat "Release" %BUILD_PATH%
CALL build_post.bat "Release" %BUILD_PATH% %ARTIFACT_PATH%
echo.

echo ======== RelWithDebInfo Build ===========
CALL build_base.bat "RelWithDebInfo" %BUILD_PATH%
CALL build_post.bat "RelWithDebInfo" %BUILD_PATH% %ARTIFACT_PATH%
echo.

pause