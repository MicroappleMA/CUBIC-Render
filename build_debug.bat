@echo off

echo =========================================
echo ============ CUBIC Render ===============
echo ============ Debug Build ================
echo =========================================

powershell -ExecutionPolicy ByPass -File decompress_lib.ps1

cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug --config Debug

pause