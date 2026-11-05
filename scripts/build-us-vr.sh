#!/bin/sh
# Build the VR branch from the supported US executable, rather than reusing
# the European executable shipped in the public VR installer.
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
. "$root/scripts/platform.sh"

if [ "$#" -ne 2 ] || [ "$1" != --game ]; then
    echo "usage: $0 --game <extracted USA game directory>" >&2
    exit 2
fi
game=$2
if [ ! -f "$game/default.xex" ]; then
    echo "$game holds no default.xex" >&2
    exit 1
fi
expected=5258c75d9ae649ed4fb71d4e2fe8808545db16092cfe764ea1a2504f5aae2e2d
actual=$(sha256sum "$game/default.xex" | cut -d' ' -f1)
if [ "$actual" != "$expected" ]; then
    echo "unsupported executable: expected the retail USA default.xex ($expected), got $actual" >&2
    exit 1
fi

# These files may have been made from another disc in this checkout. Rebuild
# both flat and stereo shaders from this disc; retain the expensive C++ objects.
project="$root/rexglue"
rm -f "$project/generated/"shader_cache*.cpp
rm -rf "$project/generated/xenos-scan" "$project/generated/runtime-containers" \
       "$project/generated/xenos-hlsl"
"$root/scripts/build.sh" --game "$game"
cp "$project/build/burnout$XR_EXE" "$project/build/burnout_us$XR_EXE"
echo "US VR build: $project/build/burnout_us$XR_EXE"
echo "Use the runtime libraries from this build with this executable."
