#!/usr/bin/env bash
# Reproduce every result in the report, from raw downloads to figures, in order.
# Downloads resume and skip finished files, so the script can be re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."

# Stages 3 and 4 start their graph side from the best stage 2 graph model: GAT with CNN node features
GRAPH_ARGS=(--arch gat --cnn-nodes --pretrained-gnn stage2_gat_segment_cnn_nodes)

run() {
  echo "+ $*"
  uv run "$@"
}

# ---- Data
run python scripts/download_data.py musiccaps-csv fma-small
run python scripts/download_musiccaps_audio.py
run python scripts/download_musiccaps_audio.py --timeout 90   # second pass for clips that timed out
run python -m src.datasets
run python -m src.preprocess musiccaps
run python -m src.preprocess fma_small
run python -m src.export_samples

# ---- Stage 1: tags from text
run python -m src.train random_tags
run python -m src.train bert_tags --text raw
run python -m src.train bert_tags --text masked

# ---- Stage 2: genre from structure
run python -m src.train majority_genre
run python -m src.train cnn_genre
for arch in graphsage gat; do
  for graph in segment chord; do
    run python -m src.train gnn_genre --arch "$arch" --graph "$graph"
  done
  run python -m src.train gnn_genre --arch "$arch" --graph segment --crop 0.5
done
run python -m src.train gnn_genre --arch gat --graph segment --cnn-nodes

# ---- Stage 3: fusion ablation on masked captions
for variant in gnn_only bert_only concat cross_attention; do
  run python -m src.train fusion --variant "$variant" --text masked "${GRAPH_ARGS[@]}"
done

# ---- Stage 4: contrastive retrieval, then the automatic relevance check
run python -m src.train contrastive "${GRAPH_ARGS[@]}"
run python -m src.relevance

# ---- Figures (PDF copies go to report/figures) and notebooks
run python -m src.plots
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb notebooks/demo_context.ipynb
