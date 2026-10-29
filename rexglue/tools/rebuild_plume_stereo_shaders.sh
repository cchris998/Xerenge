#!/bin/sh
# VR (the vr branch): the multiview (stereo) variants of plume's shader caches, for VR on plume (plume_vr.cpp).
# Translates the same shaders as rebuild_plume_shaders.sh and translate_runtime_shaders.sh (their inputs, the scanned
# executable and the runtime containers, must have been made first) with shader_common_stereo.h: every draw renders
# both eyes (SV_ViewID), 2D textures are read as arrays at the eye's layer, and perspective geometry moves by the
# eye's clip shift. The results carry their own symbols (Stereo, StereoBootstrap) and are linked beside the others.
set -e
cd "$(dirname "$0")/.."

for candidate in ../XenosRecomp ../Xerenge/XenosRecomp; do
    [ -d "$candidate/XenosRecomp" ] && XENOS_ROOT=$candidate && break
done
XENOS_ROOT=${XENOS_ROOT:-../XenosRecomp}
RECOMP=${XENOSRECOMP:-$XENOS_ROOT/build/XenosRecomp/XenosRecomp}
HEADER=$XENOS_ROOT/XenosRecomp/shader_common_stereo.h
RENAME=${XENOSRECOMP_RENAME:-tools/prepare_aux_shader_cache}

[ -f "$HEADER" ] || { echo "no $HEADER (XenosRecomp's vr branch)" >&2; exit 1; }
[ -d generated/xenos-scan ] || { echo "run tools/rebuild_plume_shaders.sh first" >&2; exit 1; }
[ -d generated/runtime-containers ] || { echo "run tools/translate_runtime_shaders.sh --from-disc first" >&2; exit 1; }

"$RECOMP" generated/xenos-scan generated/shader_cache_stereo_raw.cpp "$HEADER"
"$RENAME" generated/shader_cache_stereo_raw.cpp generated/shader_cache_stereo.cpp --suffix Stereo
"$RECOMP" generated/runtime-containers generated/shader_cache_runtime_stereo_raw.cpp "$HEADER"
"$RENAME" generated/shader_cache_runtime_stereo_raw.cpp generated/shader_cache_runtime_stereo.cpp \
    --suffix StereoBootstrap
echo "stereo shader caches: generated/shader_cache_stereo.cpp, generated/shader_cache_runtime_stereo.cpp"
