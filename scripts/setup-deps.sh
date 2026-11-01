#!/bin/sh
# Fetches the two other pieces the renderer is built from, and builds the
# shader translator.
#
#   plume         the Vulkan layer the plume renderer draws through (our fork:
#                 it adds reading a texture back, and mailbox presentation)
#   XenosRecomp   translates the title's Xenos shaders to SPIR-V (our fork:
#                 literal constants, the guest vertex index, 2D-array blending)
#
# Both are separate checkouts beside rexglue/, like the SDK.
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
. "$root/scripts/platform.sh"
plume="$root/plume"
xenos="$root/XenosRecomp"
# The VR forks (hookmanuk's forks of Xerenge's, their vr branches: multiview, the stereo shaders).
plume_remote=${XERENGE_PLUME_REMOTE:-https://github.com/hookmanuk/plume-xerenge.git}
xenos_remote=${XERENGE_XENOSRECOMP_REMOTE:-https://github.com/hookmanuk/XenosRecomp-xerenge.git}
plume_branch=${XERENGE_PLUME_BRANCH:-vr}
xenos_branch=${XERENGE_XENOSRECOMP_BRANCH:-vr}

if [ ! -d "$plume/.git" ]; then
    echo "== fetching plume into $plume"
    git clone --depth 1 -b "$plume_branch" "$plume_remote" "$plume"
elif vr_working_copy "$plume"; then
    echo "== plume is a VR working copy (branch vr, another remote): left as it is"
else
    echo "== plume is already there: $plume; updating"
    git -C "$plume" fetch --depth 1 origin "$plume_branch"
    git -C "$plume" checkout -q -f -B "$plume_branch" FETCH_HEAD
fi
echo "== plume submodules"
retry git -C "$plume" submodule update --init --recursive --depth 1

if [ ! -d "$xenos/.git" ]; then
    echo "== fetching XenosRecomp into $xenos"
    git clone -b "$xenos_branch" "$xenos_remote" "$xenos"
elif vr_working_copy "$xenos"; then
    echo "== XenosRecomp is a VR working copy (branch vr, another remote): left as it is"
else
    echo "== XenosRecomp is already there: $xenos; updating"
    git -C "$xenos" fetch origin "$xenos_branch"
    git -C "$xenos" checkout -q -f -B "$xenos_branch" FETCH_HEAD
fi
echo "== XenosRecomp submodules"
retry git -C "$xenos" submodule update --init --recursive --depth 1

echo "== building XenosRecomp"
# On a Mac XenosRecomp also compiles every shader to Metal AIR, which needs the
# metal compiler from a full Xcode. plume takes SPIR-V only, so leave it out.
xenos_options=
if [ "$XR_OS" = mac ]; then
    xenos_options=-DXENOS_RECOMP_AIR=OFF
fi
cmake -S "$xenos" -B "$xenos/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ $xenos_options
cmake --build "$xenos/build" --target XenosRecomp -j"$XR_JOBS"

echo
echo "done. Shader translator: $xenos/build/XenosRecomp/XenosRecomp$XR_EXE"
