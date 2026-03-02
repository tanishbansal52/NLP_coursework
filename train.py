import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

from data_analysis.data_loader import load_training_set, load_validation_set
from model.classifier import PCLClassifier

# --- config ---
EPOCHS     = 5
LR         = 2e-5
BATCH_SIZE = 16
MASK_LOCS  = True
EVAL_ONLY  = False
OUTPUT_DIR = "checkpoints/deberta_pcl"
# --------------

print("loading train split...")
train_df = load_training_set()
print(f"  {len(train_df)} examples  (PCL={train_df['label'].sum()}, Non-PCL={(train_df['label']==0).sum()})")

print("loading dev split...")
dev_df = load_validation_set()
print(f"  {len(dev_df)} examples  (PCL={dev_df['label'].sum()}, Non-PCL={(dev_df['label']==0).sum()})")

clf = PCLClassifier(
    epochs=EPOCHS,
    lr=LR,
    batch_size=BATCH_SIZE,
    mask_locs=MASK_LOCS,
    output_dir=OUTPUT_DIR,
)

if EVAL_ONLY:
    clf.load("best")
else:
    clf.fit(train_df["text"], train_df["label"], dev_df["text"], dev_df["label"])
    clf.load("best")

print("\nfinal eval on dev set:")
clf.evaluate(dev_df["text"], dev_df["label"])