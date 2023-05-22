@echo off

cmake -S . -B build/debug
cmake --build build/debug --config Debug

pause