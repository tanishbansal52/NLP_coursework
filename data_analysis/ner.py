import gc
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

BATCH_SIZE = 10

def _extract_entities(texts, sample_size=100):
    if len(texts) > sample_size:
        texts = texts.sample(sample_size, random_state=42)

    type_counts = Counter()
    text_by_type = defaultdict(Counter)

    for i, text in enumerate(texts):
        tokens = None
        tagged = None
        chunked = None
        try:
            tokens = word_tokenize(str(text))
            tagged = pos_tag(tokens)
            chunked = ne_chunk(tagged, binary=False)
            for node in chunked:
                if isinstance(node, Tree):
                    label = node.label()
                    type_counts[label] += 1
                    text_by_type[label][" ".join(w for w, _ in node.leaves()).lower()] += 1
        except Exception:
            pass
        finally:
            del tokens, tagged, chunked

        # free NLTK's internal tree allocations every BATCH_SIZE texts
        if (i + 1) % BATCH_SIZE == 0:
            gc.collect()

    gc.collect()
    return type_counts, text_by_type

def plot_ner_comparison(pcl_texts, non_pcl_texts, sample_size=500, top_entities=15, save=True):
    print(f"Running NER on PCL texts (sample={min(sample_size, len(pcl_texts))})...")
    pcl_counts, pcl_by_type = _extract_entities(pcl_texts, sample_size)
    gc.collect()
    print(f"Running NER on Non-PCL texts (sample={min(sample_size, len(non_pcl_texts))})...")
    non_pcl_counts, non_pcl_by_type = _extract_entities(non_pcl_texts, sample_size)
    gc.collect()

    pcl_n = min(sample_size, len(pcl_texts))
    non_pcl_n = min(sample_size, len(non_pcl_texts))
    all_labels = sorted(set(pcl_counts) | set(non_pcl_counts))
    pcl_rates = [pcl_counts.get(l, 0) / pcl_n for l in all_labels]
    non_pcl_rates = [non_pcl_counts.get(l, 0) / non_pcl_n for l in all_labels]

    # sort bars by difference so the most discriminative types appear first
    order = np.argsort([abs(p - n) for p, n in zip(pcl_rates, non_pcl_rates)])[::-1]
    all_labels = [all_labels[i] for i in order]
    pcl_rates = [pcl_rates[i] for i in order]
    non_pcl_rates = [non_pcl_rates[i] for i in order]

    x, width = np.arange(len(all_labels)), 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, pcl_rates, width, label="PCL", color="salmon", alpha=0.9)
    ax.bar(x + width / 2, non_pcl_rates, width, label="Non-PCL", color="steelblue", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha="right")
    ax.set_title("Named Entity Type Rate: PCL vs Non-PCL")
    ax.set_ylabel("Entities per Text")
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig("ner_type_rates.png", dpi=150)
    plt.show()

    # second plot: actual entity strings for the most discriminative type
    focus = all_labels[0]
    pcl_top = pcl_by_type[focus].most_common(top_entities)
    non_pcl_top = non_pcl_by_type[focus].most_common(top_entities)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, title, color in [
        (axes[0], pcl_top, f"PCL — top {focus} entities", "salmon"),
        (axes[1], non_pcl_top, f"Non-PCL — top {focus} entities", "steelblue"),
    ]:
        if data:
            labels, vals = zip(*data)
            ax.barh(list(reversed(labels)), list(reversed(vals)), color=color, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Count")
    plt.tight_layout()
    if save:
        fig.savefig(f"ner_{focus.lower()}_strings.png", dpi=150)
    plt.show()