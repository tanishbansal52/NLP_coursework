import json
import numpy as np
import torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

from data_analysis.preprocessing import mask_locations
import os
from huggingface_hub import login

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
login(token=HUGGINGFACE_TOKEN)

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN    = 256
BATCH_SIZE = 16
GRAD_ACCUM = 2
LR         = 2e-5
EPOCHS     = 5
SEED       = 42


class PCLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class PCLClassifier:
    def __init__(
        self,
        model_name=MODEL_NAME,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS,
        grad_accum=GRAD_ACCUM,
        mask_locs=True,
        output_dir="checkpoints/deberta_pcl",
        device=None,
    ):
        self.model_name  = model_name
        self.max_len     = max_len
        self.batch_size  = batch_size
        self.lr          = lr
        self.epochs      = epochs
        self.grad_accum  = grad_accum
        self.mask_locs   = mask_locs
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model       = None

        if device is None:
            self.device = torch.device(
                "mps"  if torch.backends.mps.is_available()  else
                "cuda" if torch.cuda.is_available()          else
                "cpu"
            )
        else:
            self.device = torch.device(device)

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if mask_locs:
            self.tokenizer.add_tokens(["[LOCATION]"])

        print(f"device={self.device}  model={model_name}  mask_locs={mask_locs}")

    def _preprocess(self, texts):
        texts = list(texts)
        if self.mask_locs:
            print("  applying location masking...")
            texts = mask_locations(texts)
        return texts

    def _class_weights(self, labels):
        counts  = np.bincount(labels)
        weights = len(labels) / (len(counts) * counts.astype(float))
        print(f"  class counts: {counts.tolist()}  weights: {np.round(weights, 3).tolist()}")
        return torch.tensor(weights, dtype=torch.float32).to(self.device)

    def _loader(self, texts, labels, shuffle):
        ds = PCLDataset(texts, labels, self.tokenizer, self.max_len)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=0)

    def _forward(self, batch):
        return self.model(
            input_ids      = batch["input_ids"].to(self.device),
            attention_mask = batch["attention_mask"].to(self.device),
            token_type_ids = batch["token_type_ids"].to(self.device) if "token_type_ids" in batch else None,
        )

    def fit(self, train_texts, train_labels, val_texts=None, val_labels=None):
        train_texts  = self._preprocess(train_texts)
        train_labels = list(train_labels)
        loss_fn      = nn.CrossEntropyLoss(weight=self._class_weights(train_labels))

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)

        loader      = self._loader(train_texts, train_labels, shuffle=True)
        total_steps = (len(loader) // self.grad_accum) * self.epochs
        optimizer   = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler   = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

        best_f1  = 0.0
        history  = []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(loader, 1):
                out   = self._forward(batch)
                loss  = loss_fn(out.logits.float(), batch["labels"].to(self.device))
                (loss / self.grad_accum).backward()
                total_loss += loss.item()

                if step % self.grad_accum == 0 or step == len(loader):
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            avg_loss   = total_loss / len(loader)
            epoch_info = {"epoch": epoch, "train_loss": round(avg_loss, 4)}

            if val_texts is not None:
                metrics = self.evaluate(val_texts, val_labels)
                epoch_info.update(metrics)
                f1 = metrics["f1_pcl"]
                print(f"epoch {epoch}/{self.epochs}  loss={avg_loss:.4f}  f1_pcl={f1:.4f}  f1_macro={metrics['f1_macro']:.4f}")
                if f1 > best_f1:
                    best_f1 = f1
                    self._save("best")
                    print(f"  new best: {best_f1:.4f}")
            else:
                print(f"epoch {epoch}/{self.epochs}  loss={avg_loss:.4f}")

            history.append(epoch_info)

        self._save("final")
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        return history

    @torch.no_grad()
    def predict(self, texts):
        assert self.model is not None, "call fit() or load() first"
        texts = self._preprocess(texts)
        self.model.eval()
        preds = []
        for batch in self._loader(texts, None, shuffle=False):
            preds.extend(self._forward(batch).logits.argmax(dim=-1).cpu().tolist())
        return preds

    @torch.no_grad()
    def predict_proba(self, texts):
        assert self.model is not None, "call fit() or load() first"
        texts = self._preprocess(texts)
        self.model.eval()
        probs = []
        for batch in self._loader(texts, None, shuffle=False):
            probs.append(torch.softmax(self._forward(batch).logits, dim=-1).cpu().numpy())
        return np.vstack(probs)

    def evaluate(self, texts, labels):
        preds  = self.predict(texts)
        labels = list(labels)
        p, r, f1_pcl, _ = precision_recall_fscore_support(labels, preds, pos_label=1, average="binary")
        f1_macro = f1_score(labels, preds, average="macro")
        print(classification_report(labels, preds, target_names=["Non-PCL", "PCL"], digits=4))
        return {
            "f1_pcl":       round(float(f1_pcl), 4),
            "precision_pcl": round(float(p), 4),
            "recall_pcl":    round(float(r), 4),
            "f1_macro":      round(float(f1_macro), 4),
        }

    def _save(self, tag):
        path = self.output_dir / tag
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"  saved → {path}")

    def load(self, tag="best"):
        path = self.output_dir / tag
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model     = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        self.model.eval()
        print(f"  loaded from {path}")
