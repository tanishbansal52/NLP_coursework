import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def _top_ngrams(texts, n, top_k=20):
    vec = CountVectorizer(ngram_range=(n, n), stop_words="english", max_features=10000)
    matrix = vec.fit_transform(texts)
    counts = matrix.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    top_idx = counts.argsort()[::-1][:top_k]
    return [(vocab[i], counts[i]) for i in top_idx]

def plot_ngrams(pcl_texts, non_pcl_texts, top_k=20, save=True):
    for n, label in [(1, "Unigram"), (2, "Bigram"), (3, "Trigram")]:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        for ax, texts, title, color in [
            (axes[0], pcl_texts, f"PCL {label}s", "salmon"),
            (axes[1], non_pcl_texts, f"Non-PCL {label}s", "steelblue"),
        ]:
            terms, counts = zip(*_top_ngrams(texts, n=n, top_k=top_k))
            ax.barh(list(reversed(terms)), list(reversed(counts)), color=color)
            ax.set_title(f"Top {top_k} {title}")
            ax.set_xlabel("Count")

        plt.tight_layout()
        if save:
            fig.savefig(f"{label.lower()}_analysis.png", dpi=150)
        plt.show()

def plot_discriminative_ngrams(pcl_texts, non_pcl_texts, n=1, top_k=20, alpha=1.0, save=True):
    # Build a shared vocabulary across both classes
    vec = CountVectorizer(ngram_range=(n, n), stop_words="english", max_features=20000)
    vec.fit(list(pcl_texts) + list(non_pcl_texts))
    vocab = vec.get_feature_names_out()

    pcl_counts = vec.transform(pcl_texts).sum(axis=0).A1
    non_pcl_counts = vec.transform(non_pcl_texts).sum(axis=0).A1

    # Normalise to frequencies with Laplace smoothing to avoid log(0)
    pcl_freq = (pcl_counts + alpha) / (pcl_counts.sum() + alpha * len(vocab))
    non_pcl_freq = (non_pcl_counts + alpha) / (non_pcl_counts.sum() + alpha * len(vocab))

    log_odds = np.log(pcl_freq / non_pcl_freq)

    pcl_idx = np.argsort(log_odds)[::-1][:top_k]
    non_pcl_idx = np.argsort(log_odds)[:top_k]

    label_name = {1: "Unigram", 2: "Bigram", 3: "Trigram"}.get(n, f"{n}-gram")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    pcl_terms = vocab[pcl_idx]
    pcl_scores = log_odds[pcl_idx]
    axes[0].barh(pcl_terms[::-1], np.abs(pcl_scores[::-1]), color="salmon", alpha=0.9)
    axes[0].set_title(f"PCL-distinctive {label_name}s")
    axes[0].set_xlabel("|Log-Odds Ratio|")

    non_pcl_terms = vocab[non_pcl_idx[::-1]]
    non_pcl_scores = log_odds[non_pcl_idx[::-1]]
    axes[1].barh(non_pcl_terms[::-1], np.abs(non_pcl_scores[::-1]), color="steelblue", alpha=0.9)
    axes[1].set_title(f"Non-PCL-distinctive {label_name}s")
    axes[1].set_xlabel("|Log-Odds Ratio|")

    plt.suptitle(f"Discriminative {label_name}s: PCL vs Non-PCL", fontsize=12)
    plt.tight_layout()
    if save:
        fig.savefig(f"discriminative_{label_name.lower()}s.png", dpi=150)
    plt.show()