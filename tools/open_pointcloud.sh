#!/bin/bash
# Open a scene's dense point cloud in an Open3D window.
# Shift + left-click picks points; distances are printed when the window closes.
#
# Usage: ./tools/open_pointcloud.sh [scene]        (default: scene04)
set -e
SCENE=${1:-scene04}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# conda install directory: $CONDA_BASE if set, else derived from $CONDA_EXE (exported by
# `conda init`), else ~/anaconda3. Set CONDA_BASE yourself if conda lives elsewhere.
CONDA_BASE=${CONDA_BASE:-$(dirname "$(dirname "${CONDA_EXE:-$HOME/anaconda3/bin/conda}")")}

export PATH="$CONDA_BASE/envs/droneenv/bin:$PATH"
unset PYTHONPATH
export DISPLAY=${DISPLAY:-:1}

PLY="$ROOT/repos/SousVide/gsplats/workspace/exports/${SCENE}_dense.ply"
[ -f "$PLY" ] || PLY="$ROOT/scenes/$SCENE/${SCENE}_dense.ply"
[ -f "$PLY" ] || { echo "dense point cloud for $SCENE not found"; exit 1; }

echo "Opening $PLY"
echo "  orbit = left drag | pan = middle drag | zoom = scroll"
echo "  measure = Shift + left-click two points, then press Q"
cd "$ROOT"
python tools/verify_scale.py --scene "$SCENE" --pcd "$PLY" --pick
