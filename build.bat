@echo off

echo =========================================
echo ============ CUBIC Render ===============
echo ============= Full Build ================
echo =========================================

powershell -ExecutionPolicy ByPass -File decompress_lib.ps1

cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug --config Debug

cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --config Release

cmake -S . -B build/relwithdebinfo -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/relwithdebinfo --config RelWithDebInfo

pause