@echo off

echo =========================================
echo ============ CUBIC Render ===============
echo =========== Release Build ===============
echo =========================================

powershell -ExecutionPolicy ByPass -File decompress_lib.ps1

cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --config Release

pause