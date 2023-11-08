@echo off

echo =========================================
echo ============ CUBIC Render ===============
echo ============= Full Build ================
echo =========================================
echo.

echo ============ Debug Build ================
CALL build_base.bat "Debug" "build"
echo.

echo =========== Release Build ===============
CALL build_base.bat "Release" "build"
echo.

echo ======== RelWithDebInfo Build ===========
CALL build_base.bat "RelWithDebInfo" "build"
echo.

pause