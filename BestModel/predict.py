import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

from data_analysis.data_loader import load_validation_set, load_test
from BestModel.roberta_classifier import RobertaPCLClassifier

# ── config ──────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints/roberta_pcl"
MODEL_TAG      = "best"          # which saved subfolder to load
THRESHOLD      = 0.35            # threshold from run_config
MODEL_NAME     = "roberta-base"
OUT_DEV        = "dev.txt"
OUT_TEST       = "test.txt"
# ────────────────────────────────────────────────────────────────────────────

print("loading model …")
clf = RobertaPCLClassifier(
    model_name  = MODEL_NAME,
    output_dir  = CHECKPOINT_DIR,
    threshold   = THRESHOLD,
    mask_locs   = True,
)
clf.load(MODEL_TAG)

# ── Dev set predictions ──────────────────────────────────────────────────────
print("\ngenerating dev-set predictions …")
dev_df   = load_validation_set()
dev_preds = clf.predict(dev_df["text"].tolist())

with open(OUT_DEV, "w") as f:
    for p in dev_preds:
        f.write(f"{p}\n")
print(f"  {len(dev_preds)} predictions written → {OUT_DEV}")

# quick sanity-check with known labels
from sklearn.metrics import f1_score, classification_report
dev_labels = dev_df["label"].tolist()
f1_pcl   = f1_score(dev_labels, dev_preds, pos_label=1, average="binary")
f1_macro = f1_score(dev_labels, dev_preds, average="macro")
print(f"  dev F1-PCL  = {f1_pcl:.4f}")
print(f"  dev F1-macro= {f1_macro:.4f}")
print(classification_report(dev_labels, dev_preds,
                             target_names=["Non-PCL", "PCL"], digits=4))

# ── Test set predictions ─────────────────────────────────────────────────────
print("generating test-set predictions …")
test_df   = load_test()
test_preds = clf.predict(test_df["text"].tolist())

with open(OUT_TEST, "w") as f:
    for p in test_preds:
        f.write(f"{p}\n")
print(f"  {len(test_preds)} predictions written → {OUT_TEST}")
print("done.")
