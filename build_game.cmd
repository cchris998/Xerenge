@echo off
rem VR fork: builds the game (rexglue, branch vr) against the SDK (rexglue-sdk, branch vr) with the configured toolchain.
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=H:\tools\clang+llvm-20.1.8-x86_64-pc-windows-msvc\bin;C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%PATH%
cd /d H:\Xerenge\rexglue
cmake -S . -B build -DREXGLUE_USE_VULKAN=ON > H:\Xerenge\build_game_vr.log 2>&1
cmake --build build >> H:\Xerenge\build_game_vr.log 2>&1
echo exit %ERRORLEVEL%
