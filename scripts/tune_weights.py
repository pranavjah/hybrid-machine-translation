"""
Tunes the translation-score / fluency-score weighting used by the hybrid
reranker, instead of leaving it at an arbitrary 0.6/0.4 split.

For each sentence in a held-out validation slice (NOT used in train_phrase_smt.py's
training subsample — same shuffle + seed, so it's guaranteed non-overlapping):
  1. Generate NMT beam candidates from the actual pretrained model
     (Helsinki-NLP/opus-mt-en-fr — same model the browser demo uses).
  2. Generate the SMT decoder's own candidate (monotone phrase-based decode).
  3. For each candidate weighting in a small grid, rerank the pooled
     candidates and score the winner against the real French reference
     using a lightweight BLEU-style n-gram precision metric.
  4. Average that metric across the validation set, per weighting.

Prints the best-performing weights — hardcode these into lib/hybridMT.ts's
default weights.

Usage:
    python tune_weights.py
"""

import json
import math
import random
from collections import Counter

from decode_smt import decode as smt_decode

DATA_EN = "../data/cleaned_dataset.en"
DATA_FR = "../data/cleaned_dataset.fr"
SMT_DIR = "../models/smt"

SUBSAMPLE = 20000  # must match train_phrase_smt.py so validation doesn't overlap training
VALIDATION_SIZE = 150  # kept small since this calls the real NMT model per sentence — CPU inference isn't free
RANDOM_SEED = 42

WEIGHT_GRID = [
    (0.3, 0.7), (0.4, 0.6), (0.5, 0.5),
    (0.6, 0.4), (0.7, 0.3), (0.8, 0.2),
]

NULL_TOKEN = "<null>"
OOV_PROB = 1e-4
ALPHA = 0.4


def load_validation_set():
    with open(DATA_EN, encoding="utf-8") as f:
        en_lines = [l.strip() for l in f if l.strip()]
    with open(DATA_FR, encoding="utf-8") as f:
        fr_lines = [l.strip() for l in f if l.strip()]
    pairs = list(zip(en_lines, fr_lines))
    random.Random(RANDOM_SEED).shuffle(pairs)
    # Same slice train_phrase_smt.py trained on is [:SUBSAMPLE]; validation
    # comes from just after it, so it's unseen during training.
    return pairs[SUBSAMPLE : SUBSAMPLE + VALIDATION_SIZE]


def translation_model_score(en_tokens, fr_tokens, t):
    e_tokens = en_tokens + [NULL_TOKEN]
    logp = 0.0
    for f in fr_tokens:
        s = sum(t.get(e, {}).get(f, 0.0) for e in e_tokens)
        p = s / len(e_tokens) if s > 0 else OOV_PROB
        logp += math.log(p)
    return logp / len(fr_tokens) if fr_tokens else 0.0


def unigram_prob(w, lm, total_unigrams):
    return (lm["unigrams"].get(w, 0) + 1) / (total_unigrams + lm["vocab_size"])


def bigram_prob(a, b, lm, total_unigrams):
    c = lm["bigrams"].get(f"{a}\t{b}")
    ctx = lm["unigrams"].get(a, 0)
    if c and ctx > 0:
        return c / ctx
    return ALPHA * unigram_prob(b, lm, total_unigrams)


def trigram_prob(a, b, c_, lm, total_unigrams):
    c = lm["trigrams"].get(f"{a}\t{b}\t{c_}")
    ctx = lm["bigrams"].get(f"{a}\t{b}", 0)
    if c and ctx > 0:
        return c / ctx
    return ALPHA * bigram_prob(b, c_, lm, total_unigrams)


def language_model_score(fr_tokens, lm, total_unigrams):
    vocab = set(lm["unigrams"].keys())
    unked = [w if w in vocab else "<unk>" for w in fr_tokens]
    tokens = ["<s>", "<s>"] + unked + ["</s>"]
    logp = 0.0
    for i in range(2, len(tokens)):
        logp += math.log(trigram_prob(tokens[i - 2], tokens[i - 1], tokens[i], lm, total_unigrams))
    return logp / (len(tokens) - 2)


def ngram_precision_metric(candidate, reference):
    """Lightweight BLEU-style metric: unigram+bigram precision with a
    brevity penalty. Not full BLEU (no smoothing/4-gram), but enough to
    rank weight configs against each other consistently."""
    cand = candidate.split()
    ref = reference.split()
    if not cand:
        return 0.0

    def precision(n):
        cand_ngrams = Counter(tuple(cand[i : i + n]) for i in range(len(cand) - n + 1))
        ref_ngrams = Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
        if not cand_ngrams:
            return 0.0
        overlap = sum(min(c, ref_ngrams.get(g, 0)) for g, c in cand_ngrams.items())
        total = sum(cand_ngrams.values())
        return overlap / total

    p1, p2 = precision(1), precision(2)
    score = math.sqrt(p1 * max(p2, 1e-9)) if p2 > 0 else p1 * 0.5
    bp = min(1.0, math.exp(1 - len(ref) / len(cand))) if len(cand) < len(ref) else 1.0
    return score * bp


def main():
    print("Loading SMT tables...")
    t = json.load(open(f"{SMT_DIR}/translation_probs.json", encoding="utf-8"))
    phrase_table = json.load(open(f"{SMT_DIR}/phrase_table.json", encoding="utf-8"))
    lm = json.load(open(f"{SMT_DIR}/lm.json", encoding="utf-8"))
    total_unigrams = sum(lm["unigrams"].values())

    print("Loading pretrained NMT model (Helsinki-NLP/opus-mt-en-fr)...")
    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

    print(f"Loading {VALIDATION_SIZE} held-out validation pairs...")
    val_pairs = load_validation_set()

    print("Generating NMT + SMT candidates for validation set (this is the slow part)...")
    all_candidates = []  # list of (en_text, [candidates], reference)
    for i, (en_text, fr_ref) in enumerate(val_pairs):
        inputs = tokenizer(en_text, return_tensors="pt")
        out = model.generate(
            **inputs, num_beams=5, num_return_sequences=5, max_length=64
        )
        nmt_candidates = [tokenizer.decode(o, skip_special_tokens=True) for o in out]

        smt_result = smt_decode(en_text.lower().split(), phrase_table, lm)
        smt_candidate = smt_result[0] if smt_result else None

        candidates = list(dict.fromkeys(nmt_candidates + ([smt_candidate] if smt_candidate else [])))
        all_candidates.append((en_text, candidates, fr_ref))

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(val_pairs)}")

    print("\nScoring weight grid...\n")
    results = []
    for w_trans, w_flu in WEIGHT_GRID:
        total_metric = 0.0
        for en_text, candidates, fr_ref in all_candidates:
            en_tokens = en_text.lower().split()
            best_score, best_cand = -1e18, candidates[0]
            for c in candidates:
                fr_tokens = c.lower().split()
                ts = translation_model_score(en_tokens, fr_tokens, t)
                ls = language_model_score(fr_tokens, lm, total_unigrams)
                total = w_trans * ts + w_flu * ls
                if total > best_score:
                    best_score, best_cand = total, c
            total_metric += ngram_precision_metric(best_cand, fr_ref)

        avg_metric = total_metric / len(all_candidates)
        results.append((w_trans, w_flu, avg_metric))
        print(f"  translation={w_trans:.1f}  fluency={w_flu:.1f}  ->  avg metric = {avg_metric:.4f}")

    best = max(results, key=lambda r: r[2])
    print(f"\nBest weights: translation={best[0]}, fluency={best[1]} (metric={best[2]:.4f})")
    print("Hardcode these into the default weights in lib/hybridMT.ts")


if __name__ == "__main__":
    main()
