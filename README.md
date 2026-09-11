# Leitmotif

**Understanding musical context by reading a song two ways: as a graph of its own structure, and through the words people use to describe it.**

Songs repeat themselves. Choruses come back, riffs loop, and a bridge breaks the pattern. Leitmotif turns each clip into a graph: short segments are nodes, and edges link segments that follow each other or sound alike, so a returning chorus becomes a loop in the graph. A graph neural network reads that structure, BERT reads the caption or tags, and a cross-attention layer lets each side look at the other.

The goal is a model that can:

- **Tag** a clip with genre, mood and instrumentation
- **Search** music by description, e.g. *"lo-fi hip hop with a mellow piano loop"*
- **Point to where it happens**: show which parts of a clip each word of a caption attends to

> Solo project for CSE425 (Neural Networks), BRAC University. Work in progress.

## How it works

```mermaid
flowchart LR
    audio["Audio clip"] --> seg["Segments<br/>log-mel, chroma, MFCC"]
    seg --> graph["Segment graph<br/>time + similarity edges"]
    graph --> gnn["GNN<br/>GraphSAGE / GAT"]
    text["Caption or tags"] --> bert["BERT"]
    gnn -->|"graph embedding"| fuse["Cross-attention fusion"]
    bert -->|"token embeddings"| fuse
    fuse --> tags["Genre, mood,<br/>instrument tags"]
    gnn -.-> search["Text-to-music search<br/>(contrastive)"]
    bert -.-> search
```

## Roadmap

| Stage | What | Model | Data | Status |
|---|---|---|---|---|
| 1 | Tags from text | BERT multi-label classifier | MusicCaps | Planned |
| 2 | Genre from structure | GraphSAGE / GAT on segment graphs, compared with a CNN on mel-spectrograms | FMA-small | Planned |
| 3 | Fusion | GNN + BERT cross-attention, with ablations | MusicCaps | Planned |
| 4 | Text-to-music search | Contrastive dual encoder (InfoNCE) | MusicCaps | Planned |
| Demo | Try it in the browser | Gradio app | | Planned |

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12. PyTorch comes from the CUDA 12.8 index, which also covers RTX 50-series GPUs.

```bash
uv sync                                               # creates .venv with all dependencies
uv run python scripts/download_data.py musiccaps-csv
uv run python scripts/download_data.py fma-small      # ~7.6 GB, resumes if interrupted
```

Without uv: `pip install -r requirements.txt`.

uv's package cache and Hugging Face model downloads are kept in `.cache/` inside the repo, so deleting the folder removes everything.

## Data

| Dataset | What it provides | License |
|---|---|---|
| [MusicCaps](https://huggingface.co/datasets/google/MusicCaps) | 5,521 ten-second clips with musician-written captions and aspect tags | CC BY-SA 4.0 (audio comes from YouTube) |
| [FMA](https://github.com/mdeff/fma) (small) | 8,000 thirty-second tracks across 8 genres, with artist-aware splits | Metadata CC BY 4.0; audio under each artist's Creative Commons license |

Downloads go to `data/raw/` and are not committed.

## Repository layout

```
leitmotif/
├── config.yaml             # preprocessing, graph and model settings
├── pyproject.toml          # dependencies (uv); uv.lock pins exact versions
├── requirements.txt        # pip-compatible export of uv.lock
├── data/
│   ├── raw/                # downloads (not committed)
│   ├── processed/          # features, graphs, BERT caches; samples/ is committed
│   └── splits/             # train / val / test JSON
├── notebooks/              # EDA and end-to-end demo
├── scripts/
│   └── download_data.py
├── src/
│   ├── audio_features.py   # log-mel, chroma, MFCC, segmentation
│   ├── graph_builder.py    # segment and chord graphs
│   ├── datasets.py         # loaders, tag vocabulary, splits
│   ├── bert_encoder.py     # stage 1: BERT tag classifier
│   ├── gnn_model.py        # stage 2: GraphSAGE / GAT
│   ├── baselines.py        # random and CNN baselines
│   ├── fusion_model.py     # stage 3: cross-attention fusion
│   ├── contrastive.py      # stage 4: InfoNCE dual encoder
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── results/                # metrics, plots, retrieval examples
└── report/                 # final report
```

## References

- Defferrard et al. *FMA: A Dataset for Music Analysis.* ISMIR 2017.
- Agostinelli et al. *MusicLM: Generating Music From Text.* arXiv:2301.11325, 2023. (Introduces MusicCaps.)
- Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019.
- Hamilton, Ying and Leskovec. *Inductive Representation Learning on Large Graphs.* NeurIPS 2017. (GraphSAGE)
- Veličković et al. *Graph Attention Networks.* ICLR 2018.

## License

Code is MIT licensed. Datasets keep their own licenses.
