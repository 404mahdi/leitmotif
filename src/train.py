"""Training entry point for all four stages.

    uv run python -m src.train random_tags                   # B1 for tags: random guesses
    uv run python -m src.train bert_tags --text raw          # stage 1 on captions as written
    uv run python -m src.train bert_tags --text masked       # stage 1 with tag words hidden
    uv run python -m src.train majority_genre                # B1 for genre: always the most common genre
    uv run python -m src.train cnn_genre                     # B2: CNN on log-mel spectrograms
    uv run python -m src.train gnn_genre --arch graphsage --graph segment    # stage 2
    uv run python -m src.train fusion --variant cross_attention --text masked \
        --pretrained-gnn stage2_graphsage_segment            # stage 3 (also bert_only, gnn_only, concat)
    uv run python -m src.train contrastive --pretrained-gnn stage2_graphsage_segment    # stage 4

Each run writes checkpoints/<name>.pt plus <name>_args.json (the arguments and config it used),
and graph models also write <name>_node_stats.pt so new audio can be normalized the same way.
"""

import argparse
import json
import math
import time

import numpy as np
import torch
from scipy.special import expit, softmax
from sklearn.metrics import average_precision_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from src import ROOT
from src.evaluate import (
    RESULTS,
    best_threshold,
    multiclass_metrics,
    multilabel_metrics,
    retrieval_metrics,
    save_history,
    save_metrics,
)
from src.utils import load_config, resolve, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINTS = ROOT / "checkpoints"
SPLITS = ("train", "val", "test")


def warmup_then_linear_decay(optimizer: torch.optim.Optimizer, total_steps: int, warmup_fraction: float = 0.1):
    warmup = max(1, int(total_steps * warmup_fraction))

    def scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def adamw_with_bert_groups(model: nn.Module, bert_lr: float, lr: float, weight_decay: float = 0.01):
    """A small learning rate for pretrained BERT layers and a larger one for everything trained from scratch."""
    bert = [p for n, p in model.named_parameters() if n.startswith("text.") and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith("text.") and p.requires_grad]
    groups = [group for group in ({"params": bert, "lr": bert_lr}, {"params": rest, "lr": lr}) if group["params"]]
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


@torch.no_grad()
def collect(model: nn.Module, step, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader; return (outputs, targets) as numpy arrays."""
    model.eval()
    outputs, targets = [], []
    for batch in loader:
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16):
            output, target = step(model, batch)
        outputs.append(output.float().cpu())
        targets.append(target.cpu())
    return torch.cat(outputs).numpy(), torch.cat(targets).numpy()


def run_epochs(name, model, step, loss_fn, loaders, optimizer, scheduler, epochs, evaluate, select="macro_f1"):
    """Train, keep the weights with the best validation score, and return (history, best epoch).

    `step(model, batch)` returns (outputs, targets); `evaluate(model, split)` returns a metrics dict.
    """
    CHECKPOINTS.mkdir(exist_ok=True)
    checkpoint = CHECKPOINTS / f"{name}.pt"
    history, best, best_epoch = [], -math.inf, 0
    for epoch in range(1, epochs + 1):
        model.train()
        started, total, count = time.time(), 0.0, 0
        for batch in loaders["train"]:
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16):
                output, target = step(model, batch)
            loss = loss_fn(output.float(), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += loss.item() * len(target)
            count += len(target)

        record = {"epoch": epoch, "train_loss": total / count, "val": evaluate(model, "val"), "test": evaluate(model, "test")}
        history.append(record)
        improved = record["val"][select] > best
        if improved:
            best, best_epoch = record["val"][select], epoch
            torch.save(model.state_dict(), checkpoint)
        print(
            f"[{name}] epoch {epoch}/{epochs} | loss {record['train_loss']:.4f} | val {select} {record['val'][select]:.4f}"
            f" | test {select} {record['test'][select]:.4f} | {time.time() - started:.0f}s{' *' if improved else ''}",
            flush=True,
        )
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    return history, best_epoch


def finish_multilabel(name: str, model: nn.Module, step, loaders: dict, history: list, best_epoch: int) -> None:
    """Choose one decision threshold on validation, report test metrics with it, and save everything."""
    val_logits, val_targets = collect(model, step, loaders["val"])
    test_logits, test_targets = collect(model, step, loaders["test"])
    threshold = best_threshold(expit(val_logits), val_targets)
    metrics = {
        "best_epoch": best_epoch,
        "val": multilabel_metrics(expit(val_logits), val_targets, threshold),
        "test": multilabel_metrics(expit(test_logits), test_targets, threshold),
    }
    save_metrics(name, metrics)
    save_history(name, history)
    np.save(CHECKPOINTS / f"{name}_test_probs.npy", expit(test_logits))
    print(json.dumps(metrics["test"], indent=1))


def finish_multiclass(name: str, model: nn.Module, step, loaders: dict, history: list, best_epoch: int, classes: list) -> None:
    metrics = {"best_epoch": best_epoch, "classes": classes}
    for split in ("val", "test"):
        logits, labels = collect(model, step, loaders[split])
        metrics[split] = multiclass_metrics(softmax(logits, axis=1), labels)
    metrics["test_confusion"] = confusion_matrix(labels, logits.argmax(axis=1)).tolist()
    save_metrics(name, metrics)
    save_history(name, history)
    print(json.dumps(metrics["test"], indent=1))


# ---------------------------------------------------------------- MusicCaps tags from text


def musiccaps_with_targets(cfg: dict):
    from src.datasets import load_musiccaps, multi_hot

    vocab = list(json.loads((resolve(cfg["paths"]["splits"]) / "musiccaps_tags.json").read_text(encoding="utf-8")))
    df = load_musiccaps(cfg)
    df["y"] = df["tags"].map(lambda tags: multi_hot(tags, vocab))
    return df, vocab


def random_tags(cfg: dict, args: argparse.Namespace) -> str:
    from src.baselines import random_tag_baseline

    df, _ = musiccaps_with_targets(cfg)
    train, test = (np.stack(df.loc[df["split"] == split, "y"]) for split in ("train", "test"))
    metrics = {"test": random_tag_baseline(train, test, cfg["seed"])}
    save_metrics("baseline_random_tags", metrics)
    print(json.dumps(metrics["test"], indent=1))
    return "baseline_random_tags"


def bert_tags(cfg: dict, args: argparse.Namespace) -> str:
    from transformers import AutoTokenizer

    from src.bert_encoder import BertTagger, CaptionDataset
    from src.datasets import mask_tags, tag_mask_pattern

    tcfg = cfg["train"]["bert_tags"]
    df, vocab = musiccaps_with_targets(cfg)
    if args.text == "masked":
        pattern = tag_mask_pattern(vocab)
        df["caption"] = df["caption"].map(lambda caption: mask_tags(caption, pattern))

    tokenizer = AutoTokenizer.from_pretrained(cfg["text"]["model"])
    loaders = {}
    for split in SPLITS:
        part = df[df["split"] == split]
        dataset = CaptionDataset(part["caption"].tolist(), np.stack(part["y"]), tokenizer, cfg["text"]["max_length"])
        loaders[split] = DataLoader(dataset, batch_size=tcfg["batch_size"], shuffle=split == "train")

    model = BertTagger(cfg["text"]["model"], len(vocab)).to(DEVICE)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": tcfg["lr"]},
            {"params": model.head.parameters(), "lr": tcfg["head_lr"]},
        ],
        weight_decay=0.01,
    )
    scheduler = warmup_then_linear_decay(optimizer, tcfg["epochs"] * len(loaders["train"]))

    def step(model, batch):
        logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
        return logits, batch["labels"].to(DEVICE)

    def evaluate(model, split):
        logits, targets = collect(model, step, loaders[split])
        return multilabel_metrics(expit(logits), targets)

    name = f"stage1_bert_{args.text}"
    history, best_epoch = run_epochs(
        name, model, step, nn.BCEWithLogitsLoss(), loaders, optimizer, scheduler, tcfg["epochs"], evaluate
    )
    finish_multilabel(name, model, step, loaders, history, best_epoch)
    return name


# ---------------------------------------------------------------- FMA genres from audio


def fma_splits(cfg: dict, available) -> tuple[dict[str, tuple[list[int], list[int]]], list[str]]:
    """(track ids, genre indices) per split, limited to tracks that preprocessed successfully."""
    from src.datasets import load_fma_small

    df = load_fma_small(cfg)
    genres = sorted(df["genre"].unique())
    label = {genre: i for i, genre in enumerate(genres)}
    df = df[df["track_id"].isin(list(available))]
    splits = {}
    for split in SPLITS:
        part = df[df["split"] == split]
        splits[split] = (part["track_id"].tolist(), [label[g] for g in part["genre"]])
    return splits, genres


def standardize_nodes(graphs_by_split: dict[str, list]) -> dict[str, torch.Tensor]:
    """Scale node features with the mean and std of the training graphs only, and return those statistics."""
    train_x = torch.cat([g.x for g in graphs_by_split["train"]])
    stats = {"mean": train_x.mean(dim=0), "std": train_x.std(dim=0) + 1e-6}
    for graphs in graphs_by_split.values():
        for g in graphs:
            g.x = (g.x - stats["mean"]) / stats["std"]
    return stats


def build_graph_encoder(cfg: dict, sample, arch: str, pretrained: str | None = None):
    """A GraphEncoder sized from config.yaml, optionally initialized from a stage 2 checkpoint."""
    from src.gnn_model import GraphEncoder

    gcfg = cfg["train"]["gnn_genre"]
    encoder = GraphEncoder(
        sample.num_node_features, gcfg["hidden"], gcfg["layers"], arch, gcfg["dropout"], edge_dim=sample.edge_attr.shape[1]
    )
    if pretrained:
        state = torch.load(CHECKPOINTS / f"{pretrained}.pt", map_location="cpu")
        encoder.load_state_dict({k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")})
        print(f"Graph encoder initialized from {pretrained}", flush=True)
    return encoder


def majority_genre(cfg: dict, args: argparse.Namespace) -> str:
    from src.baselines import majority_class_baseline

    ids = json.loads((resolve(cfg["paths"]["processed"]) / "fma_small" / "mel_ids.json").read_text(encoding="utf-8"))
    splits, genres = fma_splits(cfg, ids)
    metrics = {"test": majority_class_baseline(np.array(splits["train"][1]), np.array(splits["test"][1]), len(genres))}
    save_metrics("baseline_majority_genre", metrics)
    print(json.dumps(metrics["test"], indent=1))
    return "baseline_majority_genre"


def cnn_genre(cfg: dict, args: argparse.Namespace) -> str:
    from src.baselines import MelCNN, MelDataset

    tcfg = cfg["train"]["cnn_genre"]
    root = resolve(cfg["paths"]["processed"]) / "fma_small"
    ids = json.loads((root / "mel_ids.json").read_text(encoding="utf-8"))
    mel = np.load(root / "mel.npy", mmap_mode="r")
    row_of = {track: row for row, track in enumerate(ids)}
    splits, genres = fma_splits(cfg, ids)
    loaders = {
        split: DataLoader(
            MelDataset(mel, [row_of[t] for t in tracks], labels), batch_size=tcfg["batch_size"], shuffle=split == "train"
        )
        for split, (tracks, labels) in splits.items()
    }

    model = MelCNN(len(genres), dropout=tcfg["dropout"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = warmup_then_linear_decay(optimizer, tcfg["epochs"] * len(loaders["train"]), 0.05)

    def step(model, batch):
        mel_batch, labels = batch
        return model(mel_batch.to(DEVICE)), labels.to(DEVICE)

    def evaluate(model, split):
        logits, labels = collect(model, step, loaders[split])
        return multiclass_metrics(softmax(logits, axis=1), labels)

    name = "baseline_cnn_genre"
    history, best_epoch = run_epochs(
        name, model, step, nn.CrossEntropyLoss(), loaders, optimizer, scheduler, tcfg["epochs"], evaluate
    )
    finish_multiclass(name, model, step, loaders, history, best_epoch, genres)
    return name


def gnn_genre(cfg: dict, args: argparse.Namespace) -> str:
    from torch_geometric.loader import DataLoader as GraphLoader

    from src.gnn_model import GraphClassifier

    tcfg = cfg["train"]["gnn_genre"]
    name = f"stage2_{args.arch}_{args.graph}"
    graphs = torch.load(
        resolve(cfg["paths"]["processed"]) / "fma_small" / f"{args.graph}_graphs.pt", weights_only=False
    )
    splits, genres = fma_splits(cfg, graphs.keys())
    graphs_by_split = {}
    for split, (tracks, labels) in splits.items():
        graphs_by_split[split] = []
        for track, label in zip(tracks, labels):
            data = graphs[track]
            data.y = torch.tensor([label])
            graphs_by_split[split].append(data)
    stats = standardize_nodes(graphs_by_split)
    loaders = {
        split: GraphLoader(items, batch_size=tcfg["batch_size"], shuffle=split == "train")
        for split, items in graphs_by_split.items()
    }

    encoder = build_graph_encoder(cfg, graphs_by_split["train"][0], args.arch)
    model = GraphClassifier(encoder, len(genres), tcfg["dropout"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = warmup_then_linear_decay(optimizer, tcfg["epochs"] * len(loaders["train"]), 0.05)

    def step(model, batch):
        batch = batch.to(DEVICE)
        return model(batch), batch.y

    def evaluate(model, split):
        logits, labels = collect(model, step, loaders[split])
        return multiclass_metrics(softmax(logits, axis=1), labels)

    history, best_epoch = run_epochs(
        name, model, step, nn.CrossEntropyLoss(), loaders, optimizer, scheduler, tcfg["epochs"], evaluate
    )
    torch.save(stats, CHECKPOINTS / f"{name}_node_stats.pt")
    finish_multiclass(name, model, step, loaders, history, best_epoch, genres)
    return name


# ---------------------------------------------------------------- MusicCaps graphs + captions


def musiccaps_graph_text(cfg: dict, text_mode: str):
    """One PyG Data per MusicCaps clip carrying its segment graph, tokenized caption and tag targets."""
    from transformers import AutoTokenizer

    from src.datasets import mask_tags, tag_mask_pattern

    df, vocab = musiccaps_with_targets(cfg)
    graphs = torch.load(resolve(cfg["paths"]["processed"]) / "musiccaps" / "segment_graphs.pt", weights_only=False)
    df = df[df["ytid"].isin(list(graphs))].reset_index(drop=True)
    captions = df["caption"]
    if text_mode == "masked":
        pattern = tag_mask_pattern(vocab)
        captions = captions.map(lambda caption: mask_tags(caption, pattern))

    tokenizer = AutoTokenizer.from_pretrained(cfg["text"]["model"])
    encoded = tokenizer(
        captions.tolist(), padding="max_length", truncation=True, max_length=cfg["text"]["max_length"], return_tensors="pt"
    )
    by_split = {split: [] for split in SPLITS}
    for i, (ytid, split, y) in enumerate(zip(df["ytid"], df["split"], df["y"])):
        data = graphs[ytid]
        data.input_ids = encoded["input_ids"][i : i + 1]  # (1, L): batches stack into (B, L)
        data.attention_mask = encoded["attention_mask"][i : i + 1]
        data.y = torch.from_numpy(y).unsqueeze(0)
        data.ytid = ytid
        by_split[split].append(data)
    stats = standardize_nodes(by_split)
    print({split: len(items) for split, items in by_split.items()}, "clips with audio", flush=True)
    return by_split, vocab, df, tokenizer, stats


def fusion(cfg: dict, args: argparse.Namespace) -> str:
    from torch_geometric.loader import DataLoader as GraphLoader

    from src.fusion_model import FusionTagger

    tcfg = cfg["train"]["fusion"]
    name = f"stage3_{args.variant}_{args.text}" + ("_pretrained" if args.pretrained_gnn else "")
    by_split, vocab, _, _, stats = musiccaps_graph_text(cfg, args.text)
    loaders = {
        split: GraphLoader(items, batch_size=tcfg["batch_size"], shuffle=split == "train")
        for split, items in by_split.items()
    }
    encoder = None
    if args.variant != "bert_only":
        encoder = build_graph_encoder(cfg, by_split["train"][0], args.arch, args.pretrained_gnn)
    model = FusionTagger(
        args.variant, cfg["text"]["model"], encoder, len(vocab),
        tcfg["dim"], tcfg["heads"], tcfg["dropout"], tcfg["trainable_bert_layers"],
    ).to(DEVICE)
    optimizer = adamw_with_bert_groups(model, tcfg["bert_lr"], tcfg["lr"])
    scheduler = warmup_then_linear_decay(optimizer, tcfg["epochs"] * len(loaders["train"]))

    def step(model, batch):
        batch = batch.to(DEVICE)
        return model(batch), batch.y

    def evaluate(model, split):
        logits, targets = collect(model, step, loaders[split])
        return multilabel_metrics(expit(logits), targets)

    history, best_epoch = run_epochs(
        name, model, step, nn.BCEWithLogitsLoss(), loaders, optimizer, scheduler, tcfg["epochs"], evaluate
    )
    torch.save(stats, CHECKPOINTS / f"{name}_node_stats.pt")
    finish_multilabel(name, model, step, loaders, history, best_epoch)

    # Fused representations of the test clips, for the t-SNE plots
    model.eval()
    zs, ids = [], []
    with torch.no_grad():
        for batch in loaders["test"]:
            batch = batch.to(DEVICE)
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16):
                zs.append(model.fuse(batch).float().cpu())
            ids += batch.ytid
    np.save(CHECKPOINTS / f"{name}_test_z.npy", torch.cat(zs).numpy())
    (CHECKPOINTS / f"{name}_test_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    return name


# ---------------------------------------------------------------- Text-to-music retrieval

TAG_PROMPTS = ("{}", "this music is {}", "a recording featuring {}")


@torch.no_grad()
def embed_pairs(model: nn.Module, loader) -> tuple[np.ndarray, np.ndarray]:
    """Graph and caption embeddings for every clip in a loader, in loader order."""
    model.eval()
    graphs, texts = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16):
            graphs.append(model.embed_graph(batch).float().cpu())
            texts.append(model.embed_text(batch.input_ids, batch.attention_mask).float().cpu())
    return torch.cat(graphs).numpy(), torch.cat(texts).numpy()


@torch.no_grad()
def tag_embeddings(model: nn.Module, tokenizer, vocab: list[str], max_length: int) -> np.ndarray:
    """Average text embedding of a few prompt templates per tag, for zero-shot tagging."""
    model.eval()
    per_template = []
    for template in TAG_PROMPTS:
        encoded = tokenizer(
            [template.format(tag) for tag in vocab], padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(DEVICE)
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16):
            per_template.append(model.embed_text(encoded["input_ids"], encoded["attention_mask"]).float())
    return nn.functional.normalize(torch.stack(per_template).mean(dim=0), dim=-1).cpu().numpy()


def write_retrieval_examples(items: list, df, graphs: np.ndarray, texts: np.ndarray, seed: int, n: int = 10, k: int = 3) -> None:
    """Save n random test captions with their top-k retrieved clips, for the report and the demo."""
    info = df.set_index("ytid")
    rng = np.random.default_rng(seed)
    examples = []
    for q in sorted(rng.choice(len(items), size=min(n, len(items)), replace=False)):
        scores = texts[q] @ graphs.T
        query = items[q].ytid
        examples.append({
            "query_ytid": query,
            "query_caption": info.at[query, "caption"],
            "rank_of_true_clip": int((scores > scores[q]).sum()) + 1,
            "top": [
                {
                    "ytid": items[j].ytid,
                    "score": round(float(scores[j]), 4),
                    "caption": info.at[items[j].ytid, "caption"],
                    "url": f"https://www.youtube.com/watch?v={items[j].ytid}&t={int(info.at[items[j].ytid, 'start_s'])}s",
                }
                for j in np.argsort(-scores)[:k]
            ],
        })
    path = RESULTS / "retrieval_examples" / "examples.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(examples, indent=2), encoding="utf-8")


def contrastive(cfg: dict, args: argparse.Namespace) -> str:
    from torch_geometric.loader import DataLoader as GraphLoader

    from src.contrastive import DualEncoder, info_nce

    tcfg = cfg["train"]["contrastive"]
    name = "stage4_contrastive" + ("_pretrained" if args.pretrained_gnn else "")
    by_split, vocab, df, tokenizer, stats = musiccaps_graph_text(cfg, "raw")
    loaders = {
        split: GraphLoader(items, batch_size=tcfg["batch_size"], shuffle=split == "train", drop_last=split == "train")
        for split, items in by_split.items()
    }
    encoder = build_graph_encoder(cfg, by_split["train"][0], args.arch, args.pretrained_gnn)
    model = DualEncoder(
        cfg["text"]["model"], encoder, tcfg["dim"], tcfg["temperature"], tcfg["trainable_bert_layers"]
    ).to(DEVICE)
    optimizer = adamw_with_bert_groups(model, tcfg["bert_lr"], tcfg["lr"])
    scheduler = warmup_then_linear_decay(optimizer, tcfg["epochs"] * len(loaders["train"]))

    def step(model, batch):
        similarity = model(batch.to(DEVICE))
        return similarity, torch.arange(len(similarity), device=DEVICE)

    def evaluate(model, split):
        graphs, texts = embed_pairs(model, loaders[split])
        return retrieval_metrics(texts @ graphs.T)

    history, best_epoch = run_epochs(
        name, model, step, lambda similarity, _: info_nce(similarity), loaders, optimizer, scheduler,
        tcfg["epochs"], evaluate, select="mean_recall",
    )
    torch.save(stats, CHECKPOINTS / f"{name}_node_stats.pt")

    graphs, texts = embed_pairs(model, loaders["test"])
    targets = np.concatenate([data.y.numpy() for data in by_split["test"]])
    zero_shot = graphs @ tag_embeddings(model, tokenizer, vocab, cfg["text"]["max_length"]).T
    has_positive = targets.sum(axis=0) > 0
    metrics = {
        "best_epoch": best_epoch,
        "temperature": model.temperature,
        "test": retrieval_metrics(texts @ graphs.T),
        "zero_shot_tags_test": {
            "auc_pr": float(average_precision_score(targets[:, has_positive], zero_shot[:, has_positive], average="macro"))
        },
    }
    save_metrics(name, metrics)
    save_history(name, history)
    write_retrieval_examples(by_split["test"], df, graphs, texts, cfg["seed"])
    np.save(CHECKPOINTS / f"{name}_test_graph_emb.npy", graphs)
    np.save(CHECKPOINTS / f"{name}_test_text_emb.npy", texts)
    (CHECKPOINTS / f"{name}_test_ids.json").write_text(json.dumps([d.ytid for d in by_split["test"]]), encoding="utf-8")

    # A search index over every clip with audio, for the demo notebook and app
    everything = [data for split in SPLITS for data in by_split[split]]
    all_graphs, all_texts = embed_pairs(model, GraphLoader(everything, batch_size=tcfg["batch_size"]))
    torch.save(
        {
            "ytids": [data.ytid for data in everything],
            "graph_embeddings": torch.from_numpy(all_graphs),
            "text_embeddings": torch.from_numpy(all_texts),
        },
        CHECKPOINTS / f"{name}_index.pt",
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "best_epoch"}, indent=1))
    return name


TASKS = {
    "random_tags": random_tags,
    "bert_tags": bert_tags,
    "majority_genre": majority_genre,
    "cnn_genre": cnn_genre,
    "gnn_genre": gnn_genre,
    "fusion": fusion,
    "contrastive": contrastive,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--text", choices=["raw", "masked"], default="raw", help="captions as written or with tag words masked")
    parser.add_argument("--arch", choices=["graphsage", "gat"], default="graphsage", help="GNN layer type")
    parser.add_argument("--graph", choices=["segment", "chord"], default="segment", help="stage 2 graph type")
    parser.add_argument("--variant", choices=["bert_only", "gnn_only", "concat", "cross_attention"], default="cross_attention")
    parser.add_argument("--pretrained-gnn", metavar="CHECKPOINT", help="start the graph encoder from a stage 2 checkpoint")
    parser.add_argument("--epochs", type=int, help="override the epoch count in config.yaml")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    if args.epochs is not None and args.task in cfg["train"]:
        cfg["train"][args.task]["epochs"] = args.epochs
    name = TASKS[args.task](cfg, args)

    CHECKPOINTS.mkdir(exist_ok=True)
    (CHECKPOINTS / f"{name}_args.json").write_text(json.dumps({"args": vars(args), "config": cfg}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
