import pandas as pd

PCL_THERSHOLD = 2

def load_pcl(path="datasets/labels/dontpatronizeme_pcl.tsv"):
    df = pd.read_csv(path, sep="\t", header=None, skiprows=4,
                        names=["par_id", "art_id", "keyword", "country_code", "text", "label"])
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["pcl"] = (df["label"] >= PCL_THERSHOLD).astype(int)
    return df

def load_categories(path="datasets/labels/dontpatronizeme_categories.tsv"):
    return pd.read_csv(path, sep="\t", header=None, skiprows=4,
                        names=["par_id", "art_id", "text", "keyword", "country_code",
                                "span_start", "span_finish", "span_text", "pcl_category", "num_annotators"])
def load_train_labels(path="datasets/train_semeval_parids-labels.csv"):
    return pd.read_csv(path)

def load_dev_labels(path="datasets/dev_semeval_parids-labels.csv"):
    return pd.read_csv(path)

def load_test(path="datasets/task4_test.tsv"):
    return pd.read_csv(path, sep="\t", header=None,
                        names=["par_id", "art_id", "keyword", "country_code", "text"])