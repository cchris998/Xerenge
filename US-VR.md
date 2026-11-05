# Building the US release with VR

The public 0.4.9-vr-0.2 installer contains only `burnout_eu.exe`. Its US disc
detection does not supply the missing `burnout_us.exe`. The US game must be
recompiled from its own executable, with the VR SDK and renderer dependencies.

This branch adds an explicit US build command. It verifies the supported retail
US `default.xex`, clears disc-derived shader caches from any earlier build, and
builds both the game and its stereo shaders. It does not bypass the installer’s
disc checks or rename a European executable to pretend it is the US build.

Use an extracted copy of your own US disc. Its `default.xex` must hash to
`5258c75d9ae649ed4fb71d4e2fe8808545db16092cfe764ea1a2504f5aae2e2d`.
The supported full US ISO hashes to
`7abf689f52b70666ba92052bd157cfcae9ed6290dae298fbf639baa1d169f92b`.

From a toolchain environment with Clang 18+, CMake, Ninja and (on Windows) the
Visual Studio C++ build tools and Windows SDK available:

```sh
./scripts/setup-sdk.sh
./scripts/setup-deps.sh
./scripts/build-us-vr.sh --game "/path/to/extracted USA game"
```

On Windows use Git for Windows’ Bash from an x64 Visual Studio developer
environment. The existing setup scripts fetch the `vr` branches of hookmanuk’s
SDK, plume and XenosRecomp forks. See the VR dependency table in `README.md`.

The output is `rexglue/build/burnout_us.exe` (`burnout_us` on Linux). A Windows
installer payload must include it as `payload/bin/burnout_us.exe` together with
the matching runtime DLLs, `openxr_loader.dll`, and the normal extraction tool
and launcher. The existing installer selects it for the supported US executable
and installs it as `bin/burnout.exe`.

Start the headset’s OpenXR runtime before launching **Burnout Revenge VR**.
Building successfully does not establish in-headset compatibility: menus,
video playback, racing, stereo views, tracking and cockpit positioning still
need to be tested with the US assets. Keep this build experimental until those
checks pass. Game files and generated recompiler/shader code stay local.

## Validation status (2026-11-04)

On Windows with Clang 20.1.8 and the VR dependencies:

- The US disc and executable matched the hashes above.
- The VR SDK and runtime compiled successfully.
- US code generation completed successfully (7.3 seconds).
- The executable scan translated 119 unique shaders, with none skipped.
- All 64 runtime shader containers rebuilt from the US disc, with none failing.
- Shell syntax, missing-input checks and wrong-executable rejection passed.

The build stopped before translating the runtime and stereo shader caches:
Microsoft Defender quarantined the locally compiled `XenosRecomp.exe` as
`Trojan:Win32/Bearfoos.B!ml`. This result does not establish whether the
detection is a false positive. No exclusion or security bypass was applied.
There is no completed US VR executable or successful headset test from this
attempt. Compatibility is unverified; this branch supplies a build path, not
a tested playable release.
