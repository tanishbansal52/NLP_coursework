import random
import nltk
from nltk.corpus import wordnet

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

SEED = 42
random.seed(SEED)


def _get_synonyms(word: str) -> list[str]:
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != word.lower():
                synonyms.add(candidate)
    return list(synonyms)


def synonym_replacement(text: str, n: int = 3) -> str:
    words     = text.split()
    new_words = words.copy()
    candidates = [i for i, w in enumerate(words) if _get_synonyms(w)]
    random.shuffle(candidates)
    for idx in candidates[:n]:
        syns = _get_synonyms(words[idx])
        if syns:
            new_words[idx] = random.choice(syns)
    return " ".join(new_words)


def _resample(texts, labels, target_ratio, augment_fn=None):
    """Shared logic for both augmentation and oversampling."""
    pos_texts = [t for t, l in zip(texts, labels) if l == 1]
    neg_texts = [t for t, l in zip(texts, labels) if l == 0]

    n_neg      = len(neg_texts)
    n_pos_target = int((target_ratio * n_neg) / (1 - target_ratio))
    n_to_add     = max(0, n_pos_target - len(pos_texts))

    print(f"  resampling: {len(pos_texts)} → {len(pos_texts) + n_to_add} PCL examples "
          f"(target ratio {target_ratio})")

    aug_texts, aug_labels = [], []
    for i in range(n_to_add):
        src = pos_texts[i % len(pos_texts)]
        aug_texts.append(augment_fn(src) if augment_fn else src)
        aug_labels.append(1)

    all_texts  = texts + aug_texts
    all_labels = labels + aug_labels
    combined   = list(zip(all_texts, all_labels))
    random.shuffle(combined)
    all_texts, all_labels = zip(*combined)
    return list(all_texts), list(all_labels)


def augment_minority(texts, labels, target_ratio=0.18):
    """Upsample minority with synonym replacement."""
    return _resample(texts, labels, target_ratio, augment_fn=synonym_replacement)


def oversample_minority(texts, labels, target_ratio=0.18):
    """Pure random oversampling — no text modification."""
    return _resample(texts, labels, target_ratio, augment_fn=None)