@echo off

cmake -S . -B build/relwithdebinfo
cmake --build build/relwithdebinfo --config RelWithDebInfo

pause