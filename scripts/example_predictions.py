"""Five example stage-1 predictions, as the brief asks for.

    uv run python scripts/example_predictions.py

For five test captions it prints, and saves to results/stage1_examples.json, the clip's true tags and the top
predictions of both stage-1 models: the one trained on captions as written and the one trained with tag words masked.
The two side by side show how much of the raw model's score comes from reading the label off the caption.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets import load_musiccaps, mask_tags, tag_mask_pattern  # noqa: E402
from src.utils import load_config, resolve  # noqa: E402

TOP = 5


def main() -> None:
    cfg = load_config()
    vocab = list(json.loads((resolve(cfg["paths"]["splits"]) / "musiccaps_tags.json").read_text(encoding="utf-8")))
    df = load_musiccaps(cfg)
    test = df[df["split"] == "test"].reset_index(drop=True)  # the order the stage-1 test loader used
    probs = {
        text: np.load(ROOT / "checkpoints" / f"stage1_bert_{text}_test_probs.npy") for text in ("raw", "masked")
    }
    pattern = tag_mask_pattern(vocab)

    rng = np.random.default_rng(cfg["seed"])
    chosen = sorted(rng.choice(len(test), size=TOP, replace=False))
    examples = []
    for row in chosen:
        clip = test.iloc[row]
        example = {
            "ytid": clip["ytid"],
            "caption": clip["caption"],
            "masked_caption": mask_tags(clip["caption"], pattern),
            "true_tags": [tag for tag in clip["tags"] if tag in vocab],
        }
        for text, matrix in probs.items():
            order = np.argsort(-matrix[row])[:TOP]
            example[f"predicted_{text}"] = [[vocab[k], round(float(matrix[row][k]), 3)] for k in order]
        examples.append(example)

    path = ROOT / "results" / "stage1_examples.json"
    path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    for example in examples:
        print(f"\n{example['ytid']}: {example['caption'][:110]}")
        print(f"  true:   {', '.join(example['true_tags'])}")
        for text in ("raw", "masked"):
            print(f"  {text:6s}: " + ", ".join(f"{tag} ({score:.2f})" for tag, score in example[f"predicted_{text}"]))
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
