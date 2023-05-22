@echo off

cmake -S . -B build/release
cmake --build build/release --config Release

pause