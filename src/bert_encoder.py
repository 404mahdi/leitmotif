"""BERT text encoder and the stage 1 multi-label tag classifier."""

import numpy as np
import torch
from torch import nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """Pretrained BERT returning token states H (batch, length, d) and the [CLS] vector t (batch, d)."""

    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dim = self.bert.config.hidden_size

    def freeze_below(self, trainable_layers: int) -> None:
        """Freeze the embeddings and every transformer layer except the top `trainable_layers`."""
        blocks = (self.bert.encoder if hasattr(self.bert, "encoder") else self.bert.transformer).layer
        frozen = [self.bert.embeddings, *blocks[: len(blocks) - trainable_layers]]
        for module in frozen:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return hidden, hidden[:, 0]


class BertTagger(nn.Module):
    """t = BERT_CLS(x), y_hat_k = sigmoid(w_k^T t + b_k). Returns logits; the loss applies the sigmoid."""

    def __init__(self, model_name: str, n_tags: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = TextEncoder(model_name)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.dim, n_tags)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        _, cls = self.encoder(input_ids, attention_mask)
        return self.head(self.dropout(cls))


class CaptionDataset(torch.utils.data.Dataset):
    """Captions tokenized once up front, paired with multi-hot tag targets."""

    def __init__(self, texts: list[str], targets: np.ndarray, tokenizer, max_length: int):
        self.encoded = tokenizer(
            texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
        )
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encoded["input_ids"][i],
            "attention_mask": self.encoded["attention_mask"][i],
            "labels": self.targets[i],
        }
