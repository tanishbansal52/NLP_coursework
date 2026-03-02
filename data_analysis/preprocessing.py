import re
import json
import hashlib
from pathlib import Path
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

CACHE_DIR = Path("cache")

def _mask_single(text):
    try:
        tokens = word_tokenize(str(text))
        tagged = pos_tag(tokens)
        chunked = ne_chunk(tagged, binary=False)
    except Exception:
        return text

    out = []
    for node in chunked:
        if isinstance(node, Tree) and node.label() in ("GPE", "LOC"):
            out.append("[LOCATION]")
        elif isinstance(node, Tree):
            out.extend(w for w, _ in node.leaves())
        else:
            out.append(node[0])

    masked = " ".join(out)
    masked = re.sub(r"(\[LOCATION\]\s*)+", "[LOCATION] ", masked).strip()
    return masked


def mask_locations(texts):
    texts = list(texts)

    key = hashlib.md5("".join(texts).encode()).hexdigest()
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"masked_{key}.json"

    if cache_path.exists():
        print(f"loading masked texts from cache ({cache_path})")
        return json.loads(cache_path.read_text())

    results = []
    for i, text in enumerate(texts):
        results.append(_mask_single(text))
        if (i + 1) % 500 == 0:
            print(f"masked {i + 1}/{len(texts)}")

    cache_path.write_text(json.dumps(results))
    print(f"saved to cache ({cache_path})")
    return results