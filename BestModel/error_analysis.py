import os
import sys
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
)

from data_analysis.data_loader import load_validation_set, load_pcl, load_categories
from BestModel.roberta_classifier import RobertaPCLClassifier

# ── config ───────────────────────────────────────────────────────────────────
CHECKPOINT_DIR  = "checkpoints/roberta_pcl"
MODEL_NAME      = "roberta-base"
THRESHOLD       = 0.35
OUT_DIR         = "error_analysis"
HISTORY_PATH    = "checkpoints/roberta_pcl/history.json"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

# ── load model ────────────────────────────────────────────────────────────────
print("loading model …")
clf = RobertaPCLClassifier(
    model_name = MODEL_NAME,
    output_dir = CHECKPOINT_DIR,
    threshold  = THRESHOLD,
    mask_locs  = True,
)
clf.load("best")

# ── load dev split and auxiliary metadata ────────────────────────────────────
print("loading dev split …")
dev_df = load_validation_set()
texts  = dev_df["text"].tolist()
labels = dev_df["label"].tolist()

# also grab keyword / country from the PCL master file for the dev par_ids
pcl_df = load_pcl()
meta   = pcl_df[["par_id", "keyword", "country_code"]].drop_duplicates("par_id")
dev_df = dev_df.merge(meta, on="par_id", how="left")

# ── run inference ─────────────────────────────────────────────────────────────
print("running inference …")
probs = clf.predict_proba(texts, preprocess=True)
preds = [1 if p >= THRESHOLD else 0 for p in probs]

dev_df["prob"]  = probs
dev_df["pred"]  = preds
dev_df["label"] = labels

# ── helper: error category ────────────────────────────────────────────────────
def error_type(row):
    if row["label"] == 1 and row["pred"] == 0:
        return "FN"
    if row["label"] == 0 and row["pred"] == 1:
        return "FP"
    if row["label"] == 1 and row["pred"] == 1:
        return "TP"
    return "TN"

dev_df["error_type"] = dev_df.apply(error_type, axis=1)

# ════════════════════════════════════════════════════════════════════════════
# 1. Confusion Matrix
# ════════════════════════════════════════════════════════════════════════════
print("plotting confusion matrix …")
cm  = confusion_matrix(labels, preds)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Non-PCL (0)", "PCL (1)"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix – Dev Set", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

tn, fp, fn, tp = cm.ravel()
print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

# ════════════════════════════════════════════════════════════════════════════
# 2. Precision-Recall Curve
# ════════════════════════════════════════════════════════════════════════════
print("plotting PR curve …")
precision_vals, recall_vals, thresholds_pr = precision_recall_curve(labels, probs)
ap = average_precision_score(labels, probs)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(recall_vals, precision_vals, lw=2, color="steelblue",
        label=f"RoBERTa base (AP={ap:.3f})")
ax.axhline(sum(labels) / len(labels), ls="--", color="grey",
           label=f"Baseline (random) precision={sum(labels)/len(labels):.2f}")
# mark operating point
op_idx = np.argmin(np.abs(np.array(list(thresholds_pr)) - THRESHOLD))
ax.scatter([recall_vals[op_idx]], [precision_vals[op_idx]],
           s=80, zorder=5, color="red", label=f"threshold={THRESHOLD}")
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve – Dev Set", fontsize=13)
ax.legend(fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pr_curve.png"), dpi=150)
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# 3. Training History
# ════════════════════════════════════════════════════════════════════════════
if os.path.exists(HISTORY_PATH):
    print("plotting training history …")
    with open(HISTORY_PATH) as fh:
        history = json.load(fh)

    epochs      = [h["epoch"]      for h in history]
    train_loss  = [h["train_loss"] for h in history]
    f1_pcl_hist = [h.get("f1_pcl", None) for h in history]
    f1_mac_hist = [h.get("f1_macro", None) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, train_loss, marker="o", color="tomato", lw=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss per Epoch"); ax1.set_xticks(epochs)

    if any(v is not None for v in f1_pcl_hist):
        ax2.plot(epochs, f1_pcl_hist,  marker="o", color="steelblue", lw=2, label="F1-PCL")
        ax2.plot(epochs, f1_mac_hist, marker="s", color="seagreen",  lw=2, label="F1-macro")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1 Score")
        ax2.set_title("Validation F1 per Epoch"); ax2.legend()
        ax2.set_xticks(epochs); ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_history.png"), dpi=150)
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# 4. Error Examples (FP / FN)
# ════════════════════════════════════════════════════════════════════════════
print("collecting error examples …")

def _sample(df_sub, n=20):
    rows = df_sub.sort_values("prob", ascending=False) if "FP" in df_sub["error_type"].iloc[0] \
           else df_sub.sort_values("prob", ascending=True)
    return [
        {
            "par_id":   int(r["par_id"]),
            "text":     str(r["text"])[:400],
            "label":    int(r["label"]),
            "pred":     int(r["pred"]),
            "prob_pcl": round(float(r["prob"]), 4),
            "keyword":  str(r.get("keyword", "")),
            "country":  str(r.get("country_code", "")),
        }
        for _, r in rows.head(n).iterrows()
    ]

fp_df = dev_df[dev_df["error_type"] == "FP"]
fn_df = dev_df[dev_df["error_type"] == "FN"]
tp_df = dev_df[dev_df["error_type"] == "TP"]
tn_df = dev_df[dev_df["error_type"] == "TN"]

error_examples = {
    "false_positives":  _sample(fp_df) if len(fp_df) else [],
    "false_negatives":  _sample(fn_df) if len(fn_df) else [],
    "true_positives":   _sample(tp_df.sample(min(20, len(tp_df)), random_state=42)) if len(tp_df) else [],
    "true_negatives":   _sample(tn_df.sample(min(20, len(tn_df)), random_state=42)) if len(tn_df) else [],
}
with open(os.path.join(OUT_DIR, "error_examples.json"), "w") as fh:
    json.dump(error_examples, fh, indent=2, ensure_ascii=False)
print(f"  FP={len(fp_df)}  FN={len(fn_df)}  TP={len(tp_df)}  TN={len(tn_df)}")

# ════════════════════════════════════════════════════════════════════════════
# 5. Keyword Analysis
# ════════════════════════════════════════════════════════════════════════════
print("keyword analysis …")
kw_stats = []
for kw, grp in dev_df.groupby("keyword"):
    if len(grp) < 5:
        continue
    y_true = grp["label"].tolist()
    y_pred = grp["pred"].tolist()
    f1 = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    prevalence = sum(y_true) / len(y_true)
    fp_rate = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1) / max(y_true.count(0), 1)
    fn_rate = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0) / max(y_true.count(1), 1)
    kw_stats.append({
        "keyword":    kw,
        "n":          len(grp),
        "prevalence": round(prevalence, 3),
        "f1_pcl":     round(f1, 4),
        "fp_rate":    round(fp_rate, 3),
        "fn_rate":    round(fn_rate, 3),
    })
kw_stats.sort(key=lambda x: x["f1_pcl"])
with open(os.path.join(OUT_DIR, "keyword_analysis.json"), "w") as fh:
    json.dump(kw_stats, fh, indent=2)
print(f"  analysed {len(kw_stats)} keywords")

# keyword bar chart (worst 10 / best 10)
if len(kw_stats) >= 4:
    worst = kw_stats[:min(10, len(kw_stats)//2)]
    best  = kw_stats[-min(10, len(kw_stats)//2):]
    shown = worst + best
    names = [d["keyword"] for d in shown]
    f1s   = [d["f1_pcl"]  for d in shown]
    colors = ["tomato"] * len(worst) + ["steelblue"] * len(best)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, f1s, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("F1-PCL")
    ax.set_title("Per-Keyword F1-PCL (red=worst, blue=best)")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, f1s):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "keyword_f1.png"), dpi=150)
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# 6. Text Length Analysis
# ════════════════════════════════════════════════════════════════════════════
print("length analysis …")
dev_df["text_len"] = dev_df["text"].str.split().str.len()
bins  = [0, 30, 60, 100, 150, 300, 99999]
labels_bins = ["<30", "30-60", "60-100", "100-150", "150-300", ">300"]
dev_df["len_bin"] = pd.cut(dev_df["text_len"], bins=bins, labels=labels_bins)

len_stats = []
for bin_label, grp in dev_df.groupby("len_bin", observed=True):
    y_true = grp["label"].tolist()
    y_pred = grp["pred"].tolist()
    f1 = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    len_stats.append({
        "length_bin": str(bin_label),
        "n":          len(grp),
        "f1_pcl":     round(f1, 4),
        "pcl_rate":   round(sum(y_true) / len(y_true), 3),
        "fp_count":   sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1),
        "fn_count":   sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0),
    })
with open(os.path.join(OUT_DIR, "length_analysis.json"), "w") as fh:
    json.dump(len_stats, fh, indent=2)

# ════════════════════════════════════════════════════════════════════════════
# 7. Country Analysis
# ════════════════════════════════════════════════════════════════════════════
print("country analysis …")
country_stats = []
for cc, grp in dev_df.groupby("country_code"):
    if len(grp) < 5:
        continue
    y_true = grp["label"].tolist()
    y_pred = grp["pred"].tolist()
    f1 = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    country_stats.append({
        "country": str(cc),
        "n":       len(grp),
        "f1_pcl":  round(f1, 4),
        "pcl_rate": round(sum(y_true) / len(y_true), 3),
    })
country_stats.sort(key=lambda x: x["f1_pcl"])
with open(os.path.join(OUT_DIR, "country_analysis.json"), "w") as fh:
    json.dump(country_stats, fh, indent=2)

# ════════════════════════════════════════════════════════════════════════════
# 8. Probability distribution by class
# ════════════════════════════════════════════════════════════════════════════
print("plotting probability distributions …")
pcl_probs    = [p for p, l in zip(probs, labels) if l == 1]
nonpcl_probs = [p for p, l in zip(probs, labels) if l == 0]
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(nonpcl_probs, bins=40, alpha=0.6, color="steelblue", label="Non-PCL (true)")
ax.hist(pcl_probs,    bins=40, alpha=0.6, color="tomato",    label="PCL (true)")
ax.axvline(THRESHOLD, color="black", ls="--", lw=1.5, label=f"threshold={THRESHOLD}")
ax.set_xlabel("P(PCL)"); ax.set_ylabel("Count")
ax.set_title("Model Output Probability Distribution by True Class")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "prob_distribution.png"), dpi=150)
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# 9. Confusion matrix by keyword (heatmap for top keywords)
# ════════════════════════════════════════════════════════════════════════════
print("keyword confusion heatmap …")
top_kws = sorted(kw_stats, key=lambda x: x["n"], reverse=True)[:15]
top_kw_names = [d["keyword"] for d in top_kws]
top_kw_f1    = [d["f1_pcl"]  for d in top_kws]
top_kw_fp    = [d["fp_rate"] for d in top_kws]
top_kw_fn    = [d["fn_rate"] for d in top_kws]

x = np.arange(len(top_kw_names))
width = 0.28
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width, top_kw_f1,  width, label="F1-PCL",   color="steelblue")
ax.bar(x,         top_kw_fp,  width, label="FP rate",   color="orange",    alpha=0.8)
ax.bar(x + width, top_kw_fn,  width, label="FN rate",   color="tomato",    alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(top_kw_names, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Rate"); ax.set_ylim(0, 1)
ax.set_title("Top-15 Keywords: F1-PCL, FP rate, FN rate")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "keyword_error_breakdown.png"), dpi=150)
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# 10. Summary text report
# ════════════════════════════════════════════════════════════════════════════
f1_pcl_final   = f1_score(labels, preds, pos_label=1, average="binary", zero_division=0)
f1_macro_final = f1_score(labels, preds, average="macro", zero_division=0)
report_str     = classification_report(labels, preds,
                                        target_names=["Non-PCL", "PCL"], digits=4)

worst_kws = [d for d in kw_stats[:5]]
best_kws  = [d for d in kw_stats[-5:]]

# find high-confidence errors
high_conf_fp = dev_df[(dev_df["error_type"]=="FP") & (dev_df["prob"] >= 0.70)].sort_values("prob", ascending=False)
high_conf_fn = dev_df[(dev_df["error_type"]=="FN") & (dev_df["prob"] <= 0.20)].sort_values("prob")

summary = f"""
Model : roberta-base  (threshold={THRESHOLD})
Dataset: official dev split  ({len(dev_df)} examples,
         PCL={sum(labels)}, Non-PCL={len(labels)-sum(labels)})

════════════════════════════════════════
 Overall Performance
════════════════════════════════════════
{report_str}
  F1-PCL  = {f1_pcl_final:.4f}
  F1-macro= {f1_macro_final:.4f}
  AP      = {ap:.4f}

  |         | Pred Non-PCL | Pred PCL |
  |---------|-------------|---------|
  | True Non-PCL |    {tn:5d}   |  {fp:5d}  |
  | True PCL     |    {fn:5d}   |  {tp:5d}  |

════════════════════════════════════════
 Error Breakdown
════════════════════════════════════════
  True Positives  (TP) : {tp}
  False Positives (FP) : {fp}   ← Non-PCL misclassified as PCL
  False Negatives (FN) : {fn}   ← PCL missed by the model
  True Negatives  (TN) : {tn}

  Error rate (all errors): {(fp+fn)/len(labels):.3f}
  Precision: {tp/(tp+fp) if (tp+fp)>0 else 0:.4f}
  Recall   : {tp/(tp+fn) if (tp+fn)>0 else 0:.4f}

════════════════════════════════════════
 High-Confidence Errors
════════════════════════════════════════
  High-confidence FP (prob ≥ 0.70): {len(high_conf_fp)}
  High-confidence FN (prob ≤ 0.20): {len(high_conf_fn)}

- Top 3 High-confidence False Positives -
""" + "\n".join(
    f"  [{i+1}] prob={r['prob']:.3f}  kw={r.get('keyword','')}  country={r.get('country_code','')}\n"
    f"      \"{str(r['text'])[:200]}...\""
    for i, (_, r) in enumerate(high_conf_fp.head(3).iterrows())
) + f"""

- Top 3 High-confidence False Negatives -
""" + "\n".join(
    f"  [{i+1}] prob={r['prob']:.3f}  kw={r.get('keyword','')}  country={r.get('country_code','')}\n"
    f"      \"{str(r['text'])[:200]}...\""
    for i, (_, r) in enumerate(high_conf_fn.head(3).iterrows())
) + f"""

════════════════════════════════════════
 Keyword-Level Analysis
════════════════════════════════════════
 ↓ Worst 5 keywords by F1-PCL:
""" + "\n".join(
    f"   {d['keyword']:20s}  n={d['n']:4d}  prev={d['prevalence']:.2f}  "
    f"F1={d['f1_pcl']:.4f}  FP={d['fp_rate']:.3f}  FN={d['fn_rate']:.3f}"
    for d in worst_kws
) + """

 ↑ Best 5 keywords by F1-PCL:
""" + "\n".join(
    f"   {d['keyword']:20s}  n={d['n']:4d}  prev={d['prevalence']:.2f}  "
    f"F1={d['f1_pcl']:.4f}  FP={d['fp_rate']:.3f}  FN={d['fn_rate']:.3f}"
    for d in best_kws
) + f"""

════════════════════════════════════════
 Text-Length Analysis
════════════════════════════════════════
 Bin           |  n   | F1-PCL | PCL-rate | FP  | FN
""" + "\n".join(
    f" {d['length_bin']:14s}| {d['n']:5d}| {d['f1_pcl']:.4f} | {d['pcl_rate']:.3f}    | "
    f"{d['fp_count']:3d} | {d['fn_count']:3d}"
    for d in len_stats
) + f"""

════════════════════════════════════════
 Generated Artefacts
════════════════════════════════════════
  {OUT_DIR}/confusion_matrix.png
  {OUT_DIR}/pr_curve.png
  {OUT_DIR}/training_history.png
  {OUT_DIR}/prob_distribution.png
  {OUT_DIR}/keyword_f1.png
  {OUT_DIR}/keyword_error_breakdown.png
  {OUT_DIR}/error_examples.json
  {OUT_DIR}/keyword_analysis.json
  {OUT_DIR}/length_analysis.json
  {OUT_DIR}/country_analysis.json
"""

summary_path = os.path.join(OUT_DIR, "error_analysis_summary.txt")
with open(summary_path, "w") as fh:
    fh.write(summary)
print(summary)
print(f"\nSummary saved → {summary_path}")
print("Error analysis complete.")
