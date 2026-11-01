#!/bin/sh
# Fetches and builds the recompiler and system layer.
#
# The SDK is a separate checkout rather than vendored, and it is our fork: it
# carries fixes this title needs, including the vector pack defect that made the
# whole frontend draw blue. Upstream stays a remote, so the changes remain
# rebasable onto it.
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
. "$root/scripts/platform.sh"
sdk="$root/rexglue-sdk"
# HTTPS, so that no SSH key is needed to fetch it.
# The VR fork (hookmanuk's fork of Xerenge's SDK fork, its vr branch: the OpenXR layer).
fork=${XERENGE_SDK_REMOTE:-https://github.com/hookmanuk/rexglue-xerenge.git}
branch=${XERENGE_SDK_BRANCH:-vr}

if [ ! -d "$sdk/.git" ]; then
    echo "== fetching the SDK into $sdk"
    git clone -b "$branch" "$fork" "$sdk"
    git -C "$sdk" remote add upstream https://github.com/rexglue/rexglue-sdk 2>/dev/null || true
elif vr_working_copy "$sdk"; then
    echo "== the SDK is a VR working copy (branch vr, another remote): left as it is"
else
    echo "== the SDK is already there: $sdk; updating"
    git -C "$sdk" fetch origin "$branch"
    git -C "$sdk" checkout -q -f -B "$branch" FETCH_HEAD
fi

echo "== submodules"
retry git -C "$sdk" submodule update --init --recursive --depth 1

echo "== building the SDK"
# Vulkan on everywhere: the VR layer (src/ui/vulkan/vr_*.cpp) is part of it.
cmake --preset $XR_PRESET -S "$sdk" -DREXGLUE_USE_VULKAN=ON
cmake --build "$sdk/out/build/$XR_PRESET" --config Release --target rexglue -j"$XR_JOBS"

echo
echo "done. Code generator: $sdk/out/$XR_PRESET/rexglue$XR_EXE"
echo "Runtime libraries:  $sdk/out/$XR_PRESET"
