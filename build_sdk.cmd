@echo off
rem VR fork: builds the SDK (rexglue-sdk, branch vr) with the toolchain it was configured with.
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=H:\tools\clang+llvm-20.1.8-x86_64-pc-windows-msvc\bin;C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%PATH%
cd /d H:\Xerenge\rexglue-sdk
cmake -S . -B out\build\win-amd64 -DREXGLUE_USE_VULKAN=ON >H:\Xerenge\build_sdk.log 2>&1
cmake --build out\build\win-amd64 --config Release --target rexglue >> H:\Xerenge\build_sdk.log 2>&1
echo exit %ERRORLEVEL%
