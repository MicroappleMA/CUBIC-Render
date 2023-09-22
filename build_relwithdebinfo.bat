@echo off

echo =========================================
echo ============ CUBIC Render ===============
echo ======== RelWithDebInfo Build ===========
echo =========================================

powershell -ExecutionPolicy ByPass -File decompress_lib.ps1

cmake -S . -B build/relwithdebinfo -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/relwithdebinfo --config RelWithDebInfo

pause