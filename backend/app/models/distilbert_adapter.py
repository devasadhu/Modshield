"""
DistilBERT multi-label adapter.

Primary classifier for TriageLoop. Validate the orchestrator/strategies/
calibration pipeline against LogRegAdapter first — this one is slower and
GPU-bound, so don't debug plumbing against it.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

from app.models.base import ModelAdapter, N_LABELS

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128


class _TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


class DistilBertAdapter(ModelAdapter):
    def __init__(self, device: str | None = None, lr: float = 2e-5, epochs: int = 2, batch_size: int = 16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=N_LABELS, problem_type="multi_label_classification"
        ).to(self.device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, texts: list[str], labels: np.ndarray) -> None:
        ds = _TextDataset(texts, labels, self.tokenizer)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.epochs):
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self.model(**batch)
                loss = out.loss
                loss.backward()
                optim.step()
                optim.zero_grad()

    @torch.no_grad()
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        self.model.eval()
        ds = _TextDataset(texts, None, self.tokenizer)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        all_probs = []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits = self.model(**batch).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)

    @torch.no_grad()
    def mc_dropout_predict(self, texts: list[str], n_passes: int = 10) -> np.ndarray:
        """
        Run N stochastic forward passes with dropout active, for uncertainty
        decomposition (Layer 2). Returns shape (n_passes, n_samples, N_LABELS).
        """
        self.model.train()  # keep dropout active
        ds = _TextDataset(texts, None, self.tokenizer)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        passes = []
        for _ in range(n_passes):
            batch_probs = []
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch).logits
                batch_probs.append(torch.sigmoid(logits).cpu().numpy())
            passes.append(np.concatenate(batch_probs, axis=0))
        self.model.eval()
        return np.stack(passes, axis=0)

    @torch.no_grad()
    def embed(self, texts: list[str]) -> np.ndarray:
        self.model.eval()
        ds = _TextDataset(texts, None, self.tokenizer)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        all_embs = []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model.distilbert(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
            all_embs.append(cls_emb)
        return np.concatenate(all_embs, axis=0)

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str) -> None:
        self.model = DistilBertForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(path)