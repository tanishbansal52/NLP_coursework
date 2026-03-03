import json
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

from model.classifier import PCLClassifier
from data_analysis.augmentation import augment_minority, oversample_minority

SEED = 42


class RobertaPCLClassifier(PCLClassifier):
    def __init__(
        self,
        model_name="roberta-base",
        max_len=256,
        batch_size=16,
        lr=1e-5,
        epochs=5,
        grad_accum=2,
        mask_locs=True,
        output_dir="checkpoints/roberta_pcl",
        device=None,
        target_ratio=0.18,
        use_aug=True,
        patience=2,
        threshold=0.35,
    ):
        super().__init__(
            model_name=model_name,
            max_len=max_len,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            grad_accum=grad_accum,
            mask_locs=mask_locs,
            output_dir=output_dir,
            device=device,
        )
        self.target_ratio = target_ratio
        self.use_aug      = use_aug
        self.patience     = patience
        self.threshold    = threshold

    def _checkpoint_path(self, epoch):
        return self.output_dir / f"checkpoint_epoch{epoch}.pt"

    # In _save_checkpoint, only save if it's the best epoch
    def _save_checkpoint(self, epoch, optimizer, scheduler, best_f1, epochs_no_impr):
        # Only keep latest checkpoint to save disk space
        for old in self.output_dir.glob("checkpoint_epoch*.pt"):
            old.unlink()   # delete previous checkpoint before saving new one
        
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_f1": best_f1,
            "epochs_no_impr": epochs_no_impr,
        }, self.output_dir / f"checkpoint_epoch{epoch}.pt")

    def _load_checkpoint(self, last_epoch, optimizer, scheduler):
        path = self._checkpoint_path(last_epoch)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        print(f"resumed from {path}  (epoch {ckpt['epoch']}, best_f1={ckpt['best_f1']:.4f})")
        return ckpt["epoch"], ckpt["best_f1"], ckpt["epochs_no_impr"]

    def fit(self, train_texts, train_labels, val_texts=None, val_labels=None,
            use_checkpoint=False, last_epoch=0):

        train_texts  = self._preprocess(list(train_texts))
        train_labels = list(train_labels)

        if self.use_aug:
            train_texts, train_labels = augment_minority(
                train_texts, train_labels, target_ratio=self.target_ratio
            )
        else:
            train_texts, train_labels = oversample_minority(
                train_texts, train_labels, target_ratio=self.target_ratio
            )

        # always build model fresh first — then optionally overwrite with checkpoint
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        if not use_checkpoint:
            # fresh init for new head
            nn.init.normal_(self.model.classifier.dense.weight, std=0.02)
            nn.init.zeros_(self.model.classifier.dense.bias)
            nn.init.normal_(self.model.classifier.out_proj.weight, std=0.02)
            nn.init.zeros_(self.model.classifier.out_proj.bias)

        self.model.to(self.device)

        loss_fn = nn.CrossEntropyLoss()   # no class weights — augmentation handles imbalance

        loader      = self._loader(train_texts, train_labels, shuffle=True)
        total_steps = (len(loader) // self.grad_accum) * self.epochs
        optimizer   = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=0.05
        )
        scheduler   = get_linear_schedule_with_warmup(
            optimizer, int(total_steps * 0.06), total_steps
        )

        # --- resume from checkpoint if requested ---
        if use_checkpoint and last_epoch > 0:
            start_epoch, best_f1, epochs_no_impr = self._load_checkpoint(
                last_epoch, optimizer, scheduler
            )
            start_epoch += 1   # continue from NEXT epoch
        else:
            start_epoch    = 1
            best_f1        = -1.0
            epochs_no_impr = 0

        history = []

        for epoch in range(start_epoch, self.epochs + 1):
            self.model.train()
            total_loss, nan_steps = 0.0, 0
            optimizer.zero_grad()

            for step, batch in enumerate(loader, 1):
                out  = self._forward(batch)
                loss = loss_fn(out.logits.float(), batch["labels"].to(self.device))

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_steps += 1
                    optimizer.zero_grad()
                    continue

                (loss / self.grad_accum).backward()
                total_loss += loss.item()

                if step % self.grad_accum == 0 or step == len(loader):
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            if nan_steps:
                print(f"  WARNING: skipped {nan_steps}/{len(loader)} NaN steps")

            avg_loss   = total_loss / max(len(loader) - nan_steps, 1)
            epoch_info = {"epoch": epoch, "train_loss": round(avg_loss, 4)}

            if val_texts is not None:
                metrics = self.evaluate(list(val_texts), list(val_labels))
                epoch_info.update(metrics)
                f1 = metrics["f1_pcl"]
                print(f"epoch {epoch}/{self.epochs}  loss={avg_loss:.4f}  "
                      f"f1_pcl={f1:.4f}  f1_macro={metrics['f1_macro']:.4f}")

                if f1 > best_f1:
                    best_f1        = f1
                    epochs_no_impr = 0
                    self._save("best")
                    print(f"  new best: {best_f1:.4f}")
                else:
                    epochs_no_impr += 1
                    print(f"  no improvement ({epochs_no_impr}/{self.patience})")
                    if epochs_no_impr >= self.patience:
                        print(f"early stopping at epoch {epoch}")
                        self._save_checkpoint(epoch, optimizer, scheduler, best_f1, epochs_no_impr)
                        history.append(epoch_info)
                        break
            else:
                print(f"epoch {epoch}/{self.epochs}  loss={avg_loss:.4f}")

            # save checkpoint after every epoch
            self._save_checkpoint(epoch, optimizer, scheduler, best_f1, epochs_no_impr)
            history.append(epoch_info)

        self._save("final")
        return history

    @torch.no_grad()
    def predict(self, texts, preprocess=True):
        assert self.model is not None, "call fit() or load() first"
        if preprocess:
            texts = self._preprocess(list(texts))
        self.model.eval()
        probs = []
        for batch in self._loader(texts, None, shuffle=False):
            p = torch.softmax(self._forward(batch).logits.float(), dim=-1)
            probs.extend(p[:, 1].cpu().tolist())
        return [1 if p >= self.threshold else 0 for p in probs]

    def evaluate(self, texts, labels, preprocess=True):
        preds  = self.predict(texts, preprocess=preprocess)
        labels = list(labels)
        p, r, f1_pcl, _ = precision_recall_fscore_support(
            labels, preds, pos_label=1, average="binary"
        )
        f1_macro = f1_score(labels, preds, average="macro")
        print(classification_report(labels, preds, target_names=["Non-PCL", "PCL"], digits=4))
        return {
            "f1_pcl":        round(float(f1_pcl),  4),
            "precision_pcl": round(float(p),        4),
            "recall_pcl":    round(float(r),        4),
            "f1_macro":      round(float(f1_macro), 4),
        }