import os
import sys
import json
from dotenv import load_dotenv
from huggingface_hub import login

# ensure project root is on the path regardless of where this script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

from data_analysis.data_loader import load_training_set, load_validation_set
from BestModel.roberta_classifier import RobertaPCLClassifier

# --- config ---
EPOCHS          = 3
LR              = 1.5e-5
BATCH_SIZE      = 16
MASK_LOCS       = True
TARGET_RATIO    = 0.30
USE_AUG         = False
EVAL_ONLY       = False
OUTPUT_DIR      = "checkpoints/roberta_pcl"
MODEL_NAME      = "roberta-base"
PATIENCE        = 2
THRESHOLD       = 0.30
USE_CHECKPOINT  = False
LAST_EPOCH      = 3
# --------------

print("loading train split...")
train_df = load_training_set()
print(f"  {len(train_df)} examples  (PCL={train_df['label'].sum()}, "
      f"Non-PCL={(train_df['label']==0).sum()})")

print("loading dev split...")
dev_df = load_validation_set()
print(f"  {len(dev_df)} examples  (PCL={dev_df['label'].sum()}, "
      f"Non-PCL={(dev_df['label']==0).sum()})")

clf = RobertaPCLClassifier(
    model_name=MODEL_NAME,
    epochs=EPOCHS,
    lr=LR,
    batch_size=BATCH_SIZE,
    mask_locs=MASK_LOCS,
    target_ratio=TARGET_RATIO,
    use_aug=USE_AUG,
    output_dir=OUTPUT_DIR,
    patience=PATIENCE,
    threshold=THRESHOLD,
)

if EVAL_ONLY:
    clf.load("best")
else:
    history = clf.fit(
        train_df["text"], train_df["label"],
        dev_df["text"],   dev_df["label"],
        use_checkpoint=USE_CHECKPOINT,
        last_epoch=LAST_EPOCH,
    )

    # save history
    history_path = os.path.join(OUTPUT_DIR, "history.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"history saved -> {history_path}")

    # save run config
    config = {
        "model_name":   MODEL_NAME,
        "epochs":       EPOCHS,
        "lr":           LR,
        "batch_size":   BATCH_SIZE,
        "mask_locs":    MASK_LOCS,
        "target_ratio": TARGET_RATIO,
        "use_aug":      USE_AUG,
        "patience":     PATIENCE,
        "threshold":    THRESHOLD,
    }
    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    clf.load("best")

print("\nfinal eval on dev set:")
metrics = clf.evaluate(dev_df["text"], dev_df["label"])
with open(os.path.join(OUTPUT_DIR, "final_eval.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"eval results saved -> {os.path.join(OUTPUT_DIR, 'final_eval.json')}")

# --- threshold sweep on dev set ---
print("\nthreshold sweep on dev set...")
from sklearn.metrics import precision_recall_fscore_support, f1_score as _f1

probs = clf.predict_proba(dev_df["text"], preprocess=True)
true_labels = list(dev_df["label"])

sweep_results = []
for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    preds = [1 if p >= t else 0 for p in probs]
    _, _, f1_pcl, _ = precision_recall_fscore_support(
        true_labels, preds, pos_label=1, average="binary", zero_division=0
    )
    f1_mac = _f1(true_labels, preds, average="macro", zero_division=0)
    sweep_results.append({"threshold": t, "f1_pcl": round(f1_pcl, 4), "f1_macro": round(f1_mac, 4)})
    print(f"  threshold={t:.2f}  f1_pcl={f1_pcl:.4f}  f1_macro={f1_mac:.4f}")

best = max(sweep_results, key=lambda x: x["f1_pcl"])
print(f"  best threshold: {best['threshold']}  f1_pcl={best['f1_pcl']}")

with open(os.path.join(OUTPUT_DIR, "threshold_sweep.json"), "w") as f:
    json.dump({"sweep": sweep_results, "best": best}, f, indent=2)
print(f"sweep saved -> {os.path.join(OUTPUT_DIR, 'threshold_sweep.json')}")