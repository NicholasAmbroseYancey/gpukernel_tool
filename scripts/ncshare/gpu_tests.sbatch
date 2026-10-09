#!/bin/bash
# Run the full test suite (unit + kernel + gpu benchmarks) on an NCShare H200.
#
# GitHub CI only runs on CPUs (mocked tests + the Triton interpreter); this
# covers what it can't: kernels compiled for a real GPU, and the `gpu` tests.
#
# Usage (from an NCShare login node):
#   sbatch gpu_tests.sbatch                  # origin/main
#   sbatch gpu_tests.sbatch my-branch        # a pushed branch
#   sbatch gpu_tests.sbatch 7                # GitHub PR #7
#   sbatch gpu_tests.sbatch /work/$USER/gpukernel_tool   # a copied working tree
#
# Output goes to gk-gpu-tests-<jobid>.out in the directory you ran sbatch from.
# The job's exit code is pytest's, so `sacct -j <jobid>` shows pass/fail.

#SBATCH --job-name=gk-gpu-tests
#SBATCH --partition=interactive-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=gk-gpu-tests-%j.out

set -euo pipefail

REF="${1:-main}"
REPO_URL="${REPO_URL:-https://github.com/NicholasAmbroseYancey/gpukernel_tool.git}"
# -devel, not -runtime: Triton compiles a small C launcher at run time and
# needs gcc, which the runtime image doesn't have.
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel}"

# Home is only 50GB, so keep the image and pip packages on /work.
CACHE="${CACHE:-/work/$USER/gk-ci}"
SIF="$CACHE/$(basename "${IMAGE#docker://}" | tr ':' '_').sif"
PYDEPS="$CACHE/pydeps"
export APPTAINER_CACHEDIR="$CACHE/apptainer-cache"
mkdir -p "$CACHE" "$APPTAINER_CACHEDIR"

if [[ ! -f "$SIF" ]]; then
    echo "Pulling $IMAGE (first run only)..."
    apptainer pull "$SIF" "$IMAGE"
fi

# Check out the code to test into job-local storage.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/gk-ci.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
if [[ -d "$REF" ]]; then
    echo "Testing working tree $REF"
    cp -r "$REF" "$WORK/repo"
    rm -rf "$WORK/repo/kernels" "$WORK/repo/reports" "$WORK/repo/__pycache__"
else
    git clone --quiet "$REPO_URL" "$WORK/repo"
    if [[ "$REF" =~ ^[0-9]+$ ]]; then
        git -C "$WORK/repo" fetch --quiet origin "pull/$REF/head:pr-$REF"
        git -C "$WORK/repo" checkout --quiet "pr-$REF"
    else
        git -C "$WORK/repo" checkout --quiet "$REF"
    fi
    echo "Testing $REF at $(git -C "$WORK/repo" log -1 --format='%h %s')"
fi

echo "Node: $(uname -n)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# The image has torch, triton and numpy; add the rest once and reuse them.
apptainer exec --nv "$SIF" bash -c "
    set -euo pipefail
    if [[ ! -d '$PYDEPS/pytest' ]]; then
        pip install --quiet --target '$PYDEPS' pytest requests matplotlib
    fi
"

apptainer exec --nv --cleanenv \
    --env PYTHONPATH="$PYDEPS" \
    --pwd "$WORK/repo" \
    "$SIF" bash -c '
        set -euo pipefail
        unset TRITON_INTERPRET
        # Without a GPU every kernel/gpu test is skipped, which would look
        # like a pass, so fail loudly instead.
        python -c "import torch, triton; assert torch.cuda.is_available(), \"no CUDA in container\"; print(\"torch\", torch.__version__, \"triton\", triton.__version__)"
        python -m pytest -rxXs
    '
