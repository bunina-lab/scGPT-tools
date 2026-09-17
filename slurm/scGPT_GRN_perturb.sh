#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# scGPT in-silico perturbation of the attention GRN.
#
# Usage:
#   sbatch scGPT_GRN_perturb.sh <config.yaml> [PERTURB_LEVEL]
#
#   The optional 2nd argument overrides run.perturb_level, so one config file
#   covers the whole verification sequence:
#
#     sbatch scGPT_GRN_perturb.sh config_scGPT_GRN_perturb.yaml sham
#     sbatch scGPT_GRN_perturb.sh config_scGPT_GRN_perturb.yaml bin:0
#     sbatch scGPT_GRN_perturb.sh config_scGPT_GRN_perturb.yaml q:0.9
#
#   ##SBATCH directives below are DEFAULTS. Override at submission time, e.g.:
#     sbatch --time=12:00:00 --mem=256G scGPT_GRN_perturb.sh my_config.yaml
#
# GPU REQUIREMENT: this PyTorch build (1.13.0+cu117) ships kernels for
#   sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 only.
#   A40 = sm_86 OK   A100 = sm_80 OK   H200 = sm_90 NOT SUPPORTED
# Running on an unsupported arch previously produced a silently all-NaN GRN.
# The preflight check below aborts rather than letting that happen again.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J scGPT_GRN_perturb
#SBATCH --time=06:00:00
#SBATCH --mem=256G
#SBATCH --export=ALL
#SBATCH --cpus-per-task=4
#SBATCH --output=%j_scGPT_GRN_perturb.out
#SBATCH --error=%j_scGPT_GRN_perturb.err
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:a40:1

set -o pipefail

# ── Environment setup ─────────────────────────────────────────────────────────
GUIX_PROFILE="$HOME/.guix-profile"
source "$GUIX_PROFILE/etc/profile"
source /etc/profile.d/slurm.sh
eval "$(${HOME}/bin/micromamba shell hook --shell bash)"
micromamba activate scGPTpy310

# ── Validate inputs ───────────────────────────────────────────────────────────
YAML_CONFIG="${1}"
LEVEL_OVERRIDE="${2}"

if [[ -z "$YAML_CONFIG" ]]; then
    echo "ERROR: No config file provided."
    echo "Usage: sbatch scGPT_GRN_perturb.sh <config.yaml> [PERTURB_LEVEL]"
    exit 1
fi

if [[ ! -f "$YAML_CONFIG" ]]; then
    echo "ERROR: Config file '$YAML_CONFIG' not found."
    exit 1
fi

if ! command -v yq &>/dev/null; then
    echo "ERROR: 'yq' is not installed or not on PATH."
    exit 1
fi

echo "Using config: $YAML_CONFIG"
echo ""

# ── Preflight: refuse to run on a GPU this PyTorch cannot target ─────────────
PYTHONNOUSERSITE=1 python - <<'PYEOF' || exit 1
import sys
import torch

if not torch.cuda.is_available():
    sys.exit("ERROR: no CUDA device visible to PyTorch.")

name = torch.cuda.get_device_name(0)
cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
arches = torch.cuda.get_arch_list()
print(f"GPU:            {name} ({cap})")
print(f"torch {torch.__version__} supports: {' '.join(arches)}")

if cap not in arches:
    sys.exit(
        f"\nERROR: {name} is {cap}, which this PyTorch build does not support.\n"
        f"Kernels may run without raising and produce garbage -- this is what\n"
        f"caused an entirely NaN GRN on an H200 (sm_90).\n"
        f"Resubmit against a supported GPU, e.g.:\n"
        f"    sbatch --gres=gpu:a40:1  scGPT_GRN_perturb.sh <config.yaml>\n"
        f"    sbatch --gres=gpu:a100:1 scGPT_GRN_perturb.sh <config.yaml>\n"
    )
print("Preflight OK: GPU architecture is supported.\n")
PYEOF

# ── Parse config with yq ──────────────────────────────────────────────────────
SCRIPT_PATH=$(yq -r '.paths.grn_script'   "$YAML_CONFIG")
QUERY_H5AD=$(yq -r  '.paths.query_h5ad'   "$YAML_CONFIG")
TF_PATH=$(yq -r     '.paths.tf_path'      "$YAML_CONFIG")
MODEL_PT=$(yq -r    '.paths.model_pt'     "$YAML_CONFIG")
MODEL_ARGS=$(yq -r  '.paths.model_args'   "$YAML_CONFIG")
MODEL_VOCAB=$(yq -r '.paths.model_vocab'  "$YAML_CONFIG")
OUTPUT_ROOT=$(yq -r '.paths.output_dir'   "$YAML_CONFIG")
GENE_COLUMN=$(yq -r '.run.gene_column'    "$YAML_CONFIG")
CELL_COLUMN=$(yq -r '.run.cell_column'    "$YAML_CONFIG")
LAYER_KEY=$(yq -r   '.run.layer_key'      "$YAML_CONFIG")
BATCH_SIZE=$(yq -r  '.run.batch_size'     "$YAML_CONFIG")
ATTN_THRESH=$(yq -r '.run.attention_threshold' "$YAML_CONFIG")
COMPRESS=$(yq -r    '.run.compress_output'     "$YAML_CONFIG")
PERTURB_LEVEL=$(yq -r '.run.perturb_level'     "$YAML_CONFIG")
EXTRA_ARGS=$(yq -r  '.run.extra_args'     "$YAML_CONFIG")

# Command-line override wins over the config value
if [[ -n "$LEVEL_OVERRIDE" ]]; then
    PERTURB_LEVEL="$LEVEL_OVERRIDE"
fi

# Gene list: YAML sequence -> bash array
mapfile -t PERTURB_GENES < <(yq -r '.run.perturb_genes[]' "$YAML_CONFIG")
if [[ ${#PERTURB_GENES[@]} -eq 0 ]]; then
    echo "ERROR: run.perturb_genes is empty. Name at least one gene to perturb."
    exit 1
fi

# One subdirectory per level so parallel submissions cannot clobber each other
LEVEL_SLUG="${PERTURB_LEVEL//:/}"
OUTPUT_DIR="${OUTPUT_ROOT}/${LEVEL_SLUG}"

echo "Perturbing:   ${PERTURB_GENES[*]}"
echo "Level:        $PERTURB_LEVEL"
echo "Output:       $OUTPUT_DIR"
echo ""

# ── Build command ─────────────────────────────────────────────────────────────
CMD=(python "$SCRIPT_PATH"
    --query-h5ad   "$QUERY_H5AD"
    --model-path   "$MODEL_PT"
    --model-args   "$MODEL_ARGS"
    --model-vocab  "$MODEL_VOCAB"
    --output-dir   "$OUTPUT_DIR"
    --cell-column  "$CELL_COLUMN"
    --gene-column  "$GENE_COLUMN"
    --layer-key    "$LAYER_KEY"
    --tf-path      "$TF_PATH"
    --batch-size   "$BATCH_SIZE"
    --use-attention
    --attention-threshold "$ATTN_THRESH"
    --perturb-level "$PERTURB_LEVEL"
    --perturb-genes "${PERTURB_GENES[@]}"
)

[[ "$COMPRESS" == "true" ]] && CMD+=(--compress-output)
[[ "$EXTRA_ARGS" != "null" && -n "$EXTRA_ARGS" ]] && CMD+=($EXTRA_ARGS)

# ── Execute ───────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

echo "Command: ${CMD[*]}"
echo ""

PYTHONNOUSERSITE=1 "${CMD[@]}"
STATUS=$?

echo ""
echo "Exit status: $STATUS"
echo "Done: $(date)"
exit $STATUS
