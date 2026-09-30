#!/bin/bash
# Run the scene-building pipeline end to end, or any subset of its stages.
#
# Usage:
#   ./tools/run_pipeline.sh --scene scene05 --config iphone12_600
#   ./tools/run_pipeline.sh --scene scene05 --config iphone12_600 --stages scale,train
#   ./tools/run_pipeline.sh --scene scene05 --config iphone12_600 --stages export,verify
#   ./tools/run_pipeline.sh --scene scene05 --config iphone12_600 --stages scale --tag-rule present
#     (--tag-rule present: use every image containing the marker id, ignoring
#      spurious detections of other ids on grass; see build_gsplat.py)
#     (--level-ground 12: after scaling, rotate so the ground within 12 m is
#      horizontal; fixes a tag that was not lying flat)
#     (--min-tag-px 0: use every tag observation for the scale solve, as upstream
#      does; the default 100 keeps only close-range ones. scene04 was built with 0)
#
# Stages, in order:
#   frames  extract frames from the video by farthest point sampling
#   sfm     hloc SuperPoint + SuperGlue, exhaustive matching  (slowest stage)
#   check   verify SfM registration rate and tag-frame count  (changes nothing)
#   scale   solve Sim(3) from ArUco, write metric transforms.json and sparse_pc.ply
#   train   ns-train splatfacto
#   export  back-project a dense point cloud from the trained model
#   verify  automatic scale checks (ground plane, detected planes)
#
# The video must sit in repos/SousVide/gsplats/capture/ named <scene>.MOV or
# <scene>_<anything>.MOV, matching exactly one file.

set -o pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
# conda install directory: $CONDA_BASE if set, else derived from $CONDA_EXE (exported by
# `conda init`), else ~/anaconda3. Set CONDA_BASE yourself if conda lives elsewhere.
CONDA_BASE=${CONDA_BASE:-$(dirname "$(dirname "${CONDA_EXE:-$HOME/anaconda3/bin/conda}")")}

# PATH must satisfy two things at once. Getting this wrong is the single most
# common way to break a run, because the failures look unrelated:
#   1. droneenv/bin FIRST  -> ns-train, ns-viewer, and the right python live here.
#      Missing it gives:  FileNotFoundError: 'ns-train'
#   2. sfmtools/bin LAST   -> only to expose the colmap and ffmpeg executables.
#      Missing it gives:  Could not find ffmpeg. Please install ffmpeg.
#      (nerfstudio's ColmapConverterToNerfstudioDataset.__post_init__ calls
#       check_ffmpeg_installed() and check_colmap_installed() unconditionally
#       and sys.exit(1)s, even when sfm_tool="hloc" does not need them.)
# It must go last, not first, so sfmtools' python cannot shadow droneenv's.
export PATH="$CONDA_BASE/envs/droneenv/bin:$PATH:$CONDA_BASE/envs/sfmtools/bin"
# Drop a leaked ROS 2 PYTHONPATH; it points at a python3.12 site-packages that
# would be inserted ahead of droneenv's python3.10 packages.
unset PYTHONPATH
export PYTHONUNBUFFERED=1

PY="$CONDA_BASE/envs/droneenv/bin/python -u"
W="$ROOT/repos/SousVide/gsplats/workspace"
SCENE=""; CFG=""; SELECT="uniform"; TAGRULE="exact"; LEVEL=0; MINPX=""
STAGES="frames,sfm,check,scale,train,export,verify"

while [ $# -gt 0 ]; do
  case "$1" in
    --scene)  SCENE="$2";  shift 2 ;;
    --config) CFG="$2";    shift 2 ;;
    --select) SELECT="$2"; shift 2 ;;
    --tag-rule) TAGRULE="$2"; shift 2 ;;
    --level-ground) LEVEL="$2"; shift 2 ;;
    --min-tag-px) MINPX="$2"; shift 2 ;;
    --stages) STAGES="$2"; shift 2 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done
[ -n "$SCENE" ] && [ -n "$CFG" ] || { echo "--scene and --config are required"; exit 2; }

command -v ns-train >/dev/null || { echo "ns-train not on PATH; check CONDA_BASE"; exit 1; }
command -v colmap   >/dev/null || { echo "colmap not on PATH; check CONDA_BASE";   exit 1; }

log () { echo ""; echo "════════════════════════════════════════════════"; \
         echo "[$(date '+%F %T')] $*"; echo "════════════════════════════════════════════════"; }

run_build_stage () {
  log "stage: $1"
  cd "$ROOT"
  $PY tools/build_gsplat.py --scene "$SCENE" --config "$CFG" --select "$SELECT" --tag-rule "$TAGRULE" --level-ground "$LEVEL" ${MINPX:+--min-tag-px "$MINPX"} --stage "$1" \
    || { echo "[$(date '+%F %T')] FAILED: $1"; exit 1; }
  echo "[$(date '+%F %T')] done: $1"
}

latest_config () { echo "$(ls -td "$W/outputs/$SCENE/splatfacto/"*/ | head -1)config.yml"; }

IFS=',' read -ra WANTED <<< "$STAGES"
for s in "${WANTED[@]}"; do
  case "$s" in
    frames|sfm|check|scale|train) run_build_stage "$s" ;;

    export)
      log "stage: export"
      CONF=$(latest_config)
      echo "config: $CONF"
      mkdir -p "$W/exports"
      # config.yml stores relative paths (data: <scene>, output_dir: outputs),
      # so eval_setup must run with the workspace as its working directory.
      cd "$W"
      $PY "$ROOT/tools/export_pointcloud.py" --config "$CONF" \
          --out "$W/exports/${SCENE}_dense.ply" --num-points 1000000 \
        || { echo "[$(date '+%F %T')] FAILED: export"; exit 1; }
      echo "[$(date '+%F %T')] done: export"
      ;;

    verify)
      log "stage: verify"
      cd "$ROOT"
      $PY tools/verify_scale.py --scene "$SCENE"
      [ -f "$W/exports/${SCENE}_dense.ply" ] && \
        $PY tools/verify_scale.py --scene "$SCENE" --pcd "$W/exports/${SCENE}_dense.ply"
      echo ""
      echo "Automatic checks only. Measure a known object yourself to confirm scale:"
      echo "  $ROOT/tools/open_pointcloud.sh $SCENE"
      ;;

    *) echo "unknown stage: $s"; exit 2 ;;
  esac
done

echo ""
echo "[$(date '+%F %T')] pipeline finished"
