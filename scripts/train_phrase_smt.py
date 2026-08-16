"""
Extended statistical MT trainer — upgrades the word-level IBM Model 1 system
to something closer to a real hybrid MT pipeline:

  1. IBM Model 2 (simplified): adds a learned position-bias on top of Model 1's
     translation probabilities. Real Model 2 conditions alignment probability
     on exact (i, j, len_e, len_f); to keep the exported table small enough
     to ship to a browser, this trains a *relative-position bucket* version
     instead — d(bucket) where bucket = round(10 * (i/len_e - j/len_f)),
     clipped to [-10, 10]. This is a documented simplification, not full
     Model 2, and the code says so.

  2. Word alignments are trained in BOTH directions (en->fr and fr->en),
     then symmetrized with the standard grow-diag heuristic (Och & Ney).

  3. Phrase pairs (up to length MAX_PHRASE_LEN) are extracted from the
     symmetrized alignment using the standard Moses-style consistent-phrase
     extraction algorithm, then counted into a phrase translation table
     phi(f_phrase | e_phrase).

  4. The trigram LM is trained the same way as before.

Outputs (all small JSON, no ONNX/conversion needed):
    ../models/smt/translation_probs.json   (Model 1/2 word probabilities — used for reranking)
    ../models/smt/phrase_table.json        (new — used by the independent SMT decoder)
    ../models/smt/lm.json                  (same trigram LM as before)

Usage:
    python train_phrase_smt.py
"""

import json
import os
import random
from collections import defaultdict, Counter

DATA_EN = "../data/cleaned_dataset.en"
DATA_FR = "../data/cleaned_dataset.fr"
OUTPUT_DIR = "../models/smt"

SUBSAMPLE = 20000
EM_ITERATIONS = 8
RANDOM_SEED = 42
MAX_SENT_LEN = 25  # sentences longer than this are skipped for alignment/phrase training (keeps EM and phrase extraction fast; Tatoeba sentences are almost all short)

MAX_VOCAB_FR = 8000
MAX_TRANSLATIONS_PER_WORD = 8
MIN_BIGRAM_COUNT = 2
MIN_TRIGRAM_COUNT = 2

MAX_PHRASE_LEN = 3
MAX_PHRASES_PER_SOURCE = 5
MIN_PHRASE_COUNT = 2

NULL_TOKEN = "<null>"
DIST_BUCKET_RANGE = 10  # buckets from -10..10


def load_corpus():
    with open(DATA_EN, encoding="utf-8") as f:
        en_lines = [line.strip() for line in f if line.strip()]
    with open(DATA_FR, encoding="utf-8") as f:
        fr_lines = [line.strip() for line in f if line.strip()]
    assert len(en_lines) == len(fr_lines), "EN/FR line counts don't match"

    pairs = list(zip(en_lines, fr_lines))
    random.Random(RANDOM_SEED).shuffle(pairs)
    pairs = pairs[:SUBSAMPLE]

    tokenized = [(en.split(), fr.split()) for en, fr in pairs]
    tokenized = [
        (e, f)
        for e, f in tokenized
        if e and f and len(e) <= MAX_SENT_LEN and len(f) <= MAX_SENT_LEN
    ]
    return tokenized


def dist_bucket(i, j, len_e, len_f):
    """Relative-position bucket for the simplified Model 2 distortion term.
    i, j are 0-indexed positions; len_e/len_f exclude NULL."""
    if len_e == 0 or len_f == 0:
        return 0
    raw = (i / len_e) - (j / len_f)
    b = round(raw * DIST_BUCKET_RANGE)
    return max(-DIST_BUCKET_RANGE, min(DIST_BUCKET_RANGE, b))


def train_ibm_model2(sentence_pairs, iterations=EM_ITERATIONS):
    """EM training for translation probs t(f|e) plus a position-bias term
    d(bucket), restricted to word pairs that co-occur (same practical
    sparsity approximation as Model 1)."""
    candidates = defaultdict(set)
    for en_words, fr_words in sentence_pairs:
        e_tokens = en_words + [NULL_TOKEN]
        for e in e_tokens:
            candidates[e].update(fr_words)

    t = {e: {f: 1.0 / len(fs) for f in fs} for e, fs in candidates.items()}
    d = {b: 1.0 for b in range(-DIST_BUCKET_RANGE, DIST_BUCKET_RANGE + 1)}  # uniform init

    for it in range(iterations):
        count_t = defaultdict(lambda: defaultdict(float))
        total_t = defaultdict(float)
        count_d = defaultdict(float)
        total_d = 0.0

        for en_words, fr_words in sentence_pairs:
            len_e, len_f = len(en_words), len(fr_words)
            e_tokens = en_words + [NULL_TOKEN]

            for j, f in enumerate(fr_words):
                weights = []
                z = 0.0
                for i, e in enumerate(e_tokens):
                    tp = t[e].get(f, 0.0)
                    if tp == 0:
                        weights.append(0.0)
                        continue
                    is_null = e == NULL_TOKEN
                    b = 0 if is_null else dist_bucket(i, j, len_e, len_f)
                    w = tp * d.get(b, 1.0)
                    weights.append(w)
                    z += w

                if z == 0:
                    continue

                for i, e in enumerate(e_tokens):
                    w = weights[i]
                    if w == 0:
                        continue
                    delta = w / z
                    count_t[e][f] += delta
                    total_t[e] += delta
                    if e != NULL_TOKEN:
                        b = dist_bucket(i, j, len_e, len_f)
                        count_d[b] += delta
                        total_d += delta

        for e in t:
            tot = total_t.get(e, 0.0)
            if tot == 0:
                continue
            for f in t[e]:
                t[e][f] = count_t[e].get(f, 0.0) / tot

        if total_d > 0:
            for b in d:
                d[b] = count_d.get(b, 0.0) / total_d

        print(f"  Model 2 EM iteration {it + 1}/{iterations} done")

    return t, d


def viterbi_alignment(en_words, fr_words, t, d):
    """For each French word, pick the English word (or NULL) with the
    highest posterior probability under the trained t/d tables."""
    len_e, len_f = len(en_words), len(fr_words)
    e_tokens = en_words + [NULL_TOKEN]
    alignment = []  # list of (i, j) with i = English index or -1 for NULL
    for j, f in enumerate(fr_words):
        best_i, best_score = -1, -1.0
        for i, e in enumerate(e_tokens):
            tp = t.get(e, {}).get(f, 0.0)
            if tp == 0:
                continue
            is_null = e == NULL_TOKEN
            b = 0 if is_null else dist_bucket(i, j, len_e, len_f)
            score = tp * d.get(b, 1.0)
            if score > best_score:
                best_score = score
                best_i = i if not is_null else -1
        if best_i != -1:
            alignment.append((best_i, j))
    return alignment  # set of (e_index, f_index) points, NULL-aligned words omitted


def grow_diag(fwd_align, rev_align, len_e, len_f):
    """Standard Och & Ney grow-diag symmetrization: start from the
    intersection of both directions' alignments, then grow by adding
    adjacent points from the union while at least one side of the new
    point is still unaligned."""
    fwd = set(fwd_align)
    rev = set(rev_align)
    inter = fwd & rev
    union = fwd | rev

    aligned = set(inter)
    aligned_e = {i for i, j in aligned}
    aligned_f = {j for i, j in aligned}

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    changed = True
    while changed:
        changed = False
        for i, j in list(aligned):
            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                if not (0 <= ni < len_e and 0 <= nj < len_f):
                    continue
                if (ni, nj) not in union:
                    continue
                if (ni not in aligned_e) or (nj not in aligned_f):
                    aligned.add((ni, nj))
                    aligned_e.add(ni)
                    aligned_f.add(nj)
                    changed = True

    return aligned


def extract_phrases(en_words, fr_words, alignment):
    """Standard consistent-phrase extraction (Koehn et al.): a phrase pair
    (e-span, f-span) is extracted if every alignment point touching the
    e-span lands inside the f-span, and vice versa, and there's at least
    one alignment point inside the box."""
    len_e, len_f = len(en_words), len(fr_words)
    phrases = []

    for i1 in range(len_e):
        for i2 in range(i1, min(len_e, i1 + MAX_PHRASE_LEN)):
            # French span aligned to this English span
            f_points = [j for (i, j) in alignment if i1 <= i <= i2]
            if not f_points:
                continue
            j1, j2 = min(f_points), max(f_points)
            if j2 - j1 + 1 > MAX_PHRASE_LEN:
                continue

            # Consistency check: no alignment point inside the f-span maps
            # outside the e-span, and vice versa.
            consistent = True
            for (i, j) in alignment:
                if j1 <= j <= j2 and not (i1 <= i <= i2):
                    consistent = False
                    break
                if i1 <= i <= i2 and not (j1 <= j <= j2):
                    consistent = False
                    break
            if not consistent:
                continue

            e_phrase = " ".join(en_words[i1 : i2 + 1])
            f_phrase = " ".join(fr_words[j1 : j2 + 1])
            phrases.append((e_phrase, f_phrase))

    return phrases


def build_phrase_table(sentence_pairs, t_fwd, d_fwd, t_rev, d_rev):
    counts = defaultdict(Counter)

    for idx, (en_words, fr_words) in enumerate(sentence_pairs):
        fwd = viterbi_alignment(en_words, fr_words, t_fwd, d_fwd)
        # reverse direction: aligns fr->en, so swap the returned pairs back to (i,j)=(en_idx,fr_idx)
        rev_raw = viterbi_alignment(fr_words, en_words, t_rev, d_rev)
        rev = [(i, j) for (j, i) in rev_raw]  # rev_raw is (fr_idx-as-"i", en_idx-as-"j"); flip

        sym = grow_diag(fwd, rev, len(en_words), len(fr_words))
        phrases = extract_phrases(en_words, fr_words, sym)
        for e_phrase, f_phrase in phrases:
            counts[e_phrase][f_phrase] += 1

        if (idx + 1) % 5000 == 0:
            print(f"  phrase extraction: {idx + 1}/{len(sentence_pairs)} sentences")

    phrase_table = {}
    for e_phrase, f_counter in counts.items():
        total = sum(f_counter.values())
        top = [(f, c) for f, c in f_counter.most_common(MAX_PHRASES_PER_SOURCE) if c >= MIN_PHRASE_COUNT]
        if not top:
            continue
        phrase_table[e_phrase] = {f: round(c / total, 5) for f, c in top}

    return phrase_table


def prune_translation_table(t):
    pruned = {}
    for e, fs in t.items():
        if e == NULL_TOKEN:
            continue
        top = sorted(fs.items(), key=lambda kv: kv[1], reverse=True)[:MAX_TRANSLATIONS_PER_WORD]
        if top:
            pruned[e] = {f: round(p, 5) for f, p in top}
    return pruned


def train_ngram_lm(sentence_pairs):
    fr_sentences = [fr for _, fr in sentence_pairs]
    word_freq = Counter(w for sent in fr_sentences for w in sent)
    vocab = {w for w, _ in word_freq.most_common(MAX_VOCAB_FR)}

    def unk(w):
        return w if w in vocab else "<unk>"

    unigrams, bigrams, trigrams = Counter(), Counter(), Counter()
    for sent in fr_sentences:
        tokens = ["<s>", "<s>"] + [unk(w) for w in sent] + ["</s>"]
        for i, w in enumerate(tokens):
            unigrams[w] += 1
            if i >= 1:
                bigrams[(tokens[i - 1], w)] += 1
            if i >= 2:
                trigrams[(tokens[i - 2], tokens[i - 1], w)] += 1

    bigrams_pruned = {f"{a}\t{b}": c for (a, b), c in bigrams.items() if c >= MIN_BIGRAM_COUNT}
    trigrams_pruned = {
        f"{a}\t{b}\t{c_}": c for (a, b, c_), c in trigrams.items() if c >= MIN_TRIGRAM_COUNT
    }
    return {
        "vocab_size": len(vocab) + 1,
        "unigrams": dict(unigrams),
        "bigrams": bigrams_pruned,
        "trigrams": trigrams_pruned,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading corpus...")
    sentence_pairs = load_corpus()
    print(f"Training on {len(sentence_pairs)} sentence pairs\n")

    print("Training forward Model 2 (en -> fr)...")
    t_fwd, d_fwd = train_ibm_model2(sentence_pairs)

    print("\nTraining reverse Model 2 (fr -> en)...")
    reversed_pairs = [(fr, en) for en, fr in sentence_pairs]
    t_rev, d_rev = train_ibm_model2(reversed_pairs)

    print("\nExtracting phrases (grow-diag symmetrization + phrase extraction)...")
    phrase_table = build_phrase_table(sentence_pairs, t_fwd, d_fwd, t_rev, d_rev)

    pruned_t = prune_translation_table(t_fwd)

    print("\nTraining trigram language model...")
    lm = train_ngram_lm(sentence_pairs)

    with open(os.path.join(OUTPUT_DIR, "translation_probs.json"), "w", encoding="utf-8") as f:
        json.dump(pruned_t, f, ensure_ascii=False)
    with open(os.path.join(OUTPUT_DIR, "distortion.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in d_fwd.items()}, f, ensure_ascii=False)
    with open(os.path.join(OUTPUT_DIR, "phrase_table.json"), "w", encoding="utf-8") as f:
        json.dump(phrase_table, f, ensure_ascii=False)
    with open(os.path.join(OUTPUT_DIR, "lm.json"), "w", encoding="utf-8") as f:
        json.dump(lm, f, ensure_ascii=False)

    print("\nDone. File sizes:")
    for fname in ["translation_probs.json", "distortion.json", "phrase_table.json", "lm.json"]:
        path = os.path.join(OUTPUT_DIR, fname)
        print(f"  {fname}: {os.path.getsize(path) / 1024:.1f} KB")
    print(f"\nPhrase table: {len(phrase_table)} source phrases")


if __name__ == "__main__":
    main()
