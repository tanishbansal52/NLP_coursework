import os
import argparse
from dotenv import load_dotenv

load_dotenv()

from data_analysis.data_loader import load_training_set, load_validation_set
from model.classifier import PCLClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--batch-size", type=int,   default=16)
    p.add_argument("--no-mask",    action="store_true")
    p.add_argument("--eval-only",  action="store_true")
    p.add_argument("--output-dir", default="checkpoints/deberta_pcl")
    return p.parse_args()


def main():
    args = parse_args()

    print("loading train split...")
    train_df = load_training_set()
    print(f"  {len(train_df)} examples  (PCL={train_df['label'].sum()}, Non-PCL={(train_df['label']==0).sum()})")

    print("loading dev split...")
    dev_df = load_validation_set()
    print(f"  {len(dev_df)} examples  (PCL={dev_df['label'].sum()}, Non-PCL={(dev_df['label']==0).sum()})")

    clf = PCLClassifier(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        mask_locs=not args.no_mask,
        output_dir=args.output_dir,
    )

    if args.eval_only:
        clf.load("best")
    else:
        clf.fit(train_df["text"], train_df["label"], dev_df["text"], dev_df["label"])
        clf.load("best")

    print("\nfinal eval on dev set:")
    clf.evaluate(dev_df["text"], dev_df["label"])


if __name__ == "__main__":
    main()