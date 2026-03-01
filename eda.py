import gc
from data_analysis.data_loader import load_pcl, load_test
from data_analysis.ngram import plot_ngrams
from data_analysis.ner import plot_ner_comparison

# Loading data
pcl_df = load_pcl()

# Separate raw text for each class
pcl_texts = pcl_df[pcl_df["pcl"] == 1]["text"].dropna().reset_index(drop=True)
non_pcl_texts = pcl_df[pcl_df["pcl"] == 0]["text"].dropna().reset_index(drop=True)

# N-gram analysis
plot_ngrams(pcl_texts, non_pcl_texts)
gc.collect()

# Named entity recognition
plot_ner_comparison(pcl_texts, non_pcl_texts)