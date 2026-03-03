import os
import json
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

from data_analysis.data_loader import load_training_set, load_validation_set
from model.roberta_classifier import RobertaPCLClassifier

# --- config ---
EPOCHS          = 3            # We know it peaks early, don't waste time on 4
LR              = 1.5e-5       # Slightly higher learning rate to explore better minima
BATCH_SIZE      = 16
MASK_LOCS       = True
TARGET_RATIO    = 0.30         # Higher ratio! (Make the dataset ~30% PCL)
USE_AUG         = False        # IMPORTANT: Turn OFF synonym replacement!
EVAL_ONLY       = True
OUTPUT_DIR      = "checkpoints/roberta_pcl_oversample" # Save to a new folder!
MODEL_NAME      = "roberta-base"
PATIENCE        = 2
THRESHOLD       = 0.30         # Start evaluating at 0.30
USE_CHECKPOINT  = False        # START FRESH
LAST_EPOCH      = 1            # No checkpoint to load, start at epoch 0
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