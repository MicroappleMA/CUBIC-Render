@echo off

cmake -S . -B build/debug
cmake --build build/debug --config Debug

cmake -S . -B build/release
cmake --build build/release --config Release

cmake -S . -B build/relwithdebinfo
cmake --build build/relwithdebinfo --config RelWithDebInfo

pause