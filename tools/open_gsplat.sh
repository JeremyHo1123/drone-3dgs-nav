#!/bin/bash
# Open a scene's trained 3DGS in the nerfstudio web viewer.
# Uses the most recent training run for that scene.
#
# Usage: ./tools/open_gsplat.sh [scene]           (default: scene04)
# Stop with Ctrl+C -- closing the browser tab leaves it holding GPU memory.
set -e
SCENE=${1:-scene04}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONDA_BASE=${CONDA_BASE:-/home/jeremy/anaconda3}

export PATH="$CONDA_BASE/envs/droneenv/bin:$PATH:$CONDA_BASE/envs/sfmtools/bin"
unset PYTHONPATH

W="$ROOT/repos/SousVide/gsplats/workspace"
cd "$W" || { echo "workspace not found: $W"; exit 1; }
CONF=$(ls -td "outputs/$SCENE/splatfacto/"*/ 2>/dev/null | head -1)config.yml
[ -f "$CONF" ] || { echo "no trained model found for $SCENE"; exit 1; }

echo "Opening $CONF"
echo "  then browse to http://localhost:7007"
echo "  stop with Ctrl+C"
ns-viewer --load-config "$CONF"
