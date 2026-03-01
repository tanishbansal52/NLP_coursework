import ast
import pandas as pd

PCL_THERSHOLD = 2

# Convert annotator scoring to binary labels (PCL vs non-PCL) based on a threshold.
def _parse_label(val):
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            return int(sum(parsed) >= PCL_THERSHOLD)
        return int(parsed)
    except Exception:
        return int(val)

# Merge PCL scores with paragraph texts
def _load_split(pcl_path, labels_path):
    pcl_df = load_pcl(pcl_path)
    labels = pd.read_csv(labels_path)
    labels.columns = [c.strip().lower() for c in labels.columns]
    if "par_id" not in labels.columns:
        labels = labels.rename(columns={labels.columns[0]: "par_id", labels.columns[1]: "label"})
    labels["label"] = labels["label"].apply(_parse_label)
    df = labels.merge(pcl_df[["par_id", "text"]], on="par_id", how="left")
    df = df.dropna(subset=["text"]).copy()
    return df[["par_id", "text", "label"]]

# Load training dataset
def load_training_set(pcl_path="datasets/labels/dontpatronizeme_pcl.tsv",
                       labels_path="datasets/train_semeval_parids-labels.csv"):
    return _load_split(pcl_path, labels_path)

# Load validation dataset
def load_validation_set(pcl_path="datasets/labels/dontpatronizeme_pcl.tsv",
                     labels_path="datasets/dev_semeval_parids-labels.csv"):
    return _load_split(pcl_path, labels_path)

# Load test dataset (without labels)
def load_test(path="datasets/task4_test.tsv"):
    return pd.read_csv(path, sep="\t", header=None,
                       names=["par_id", "art_id", "keyword", "country_code", "text"])

# Load the full PCL dataset
def load_pcl(path="datasets/labels/dontpatronizeme_pcl.tsv"):
    df = pd.read_csv(path, sep="\t", header=None, skiprows=4,
                     names=["par_id", "art_id", "keyword", "country_code", "text", "label"])
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["pcl"] = (df["label"] >= PCL_THERSHOLD).astype(int)
    return df

# Load categories for PCL text spans
def load_categories(path="datasets/labels/dontpatronizeme_categories.tsv"):
    return pd.read_csv(path, sep="\t", header=None, skiprows=4,
                       names=["par_id", "art_id", "text", "keyword", "country_code",
                              "span_start", "span_finish", "span_text", "pcl_category", "num_annotators"])