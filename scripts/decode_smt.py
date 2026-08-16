"""
A simplified monotone phrase-based decoder — the piece that makes the SMT
half an independent translation engine rather than only a reranker of NMT
output.

Simplification, stated honestly: real phrase-based decoders (Moses) allow
reordering within a distortion window, tracked via a coverage bitmap over
source positions ("stack decoding"). This decoder is monotone — it only
ever consumes source phrases left-to-right — which avoids the combinatorial
blowup of arbitrary reordering. For English-French, a language pair with
fairly similar word order, monotone decoding is a reasonable simplification
and still produces genuinely decoded (not templated) output. It is not
full Moses-style decoding.

Algorithm: beam search over prefixes of the source sentence. At each step,
each hypothesis extends by consuming a phrase of length 1..MAX_PHRASE_LEN
starting at its current source position, appending that phrase's best (or
several) translations, and scoring with phrase-table probability + LM
increment. Top-K hypotheses are kept at each prefix length (a "stack" per
position, standard stack-decoding terminology minus the reordering).
"""

import math
from collections import namedtuple

MAX_PHRASE_LEN = 3
BEAM_WIDTH = 8
OOV_PHRASE_PENALTY = math.log(1e-3)  # score for source words with no phrase-table entry (copied through)

Hypothesis = namedtuple("Hypothesis", ["pos", "tokens", "score"])


def lm_increment_score(prev2, prev1, new_tokens, lm):
    """Log-prob of appending new_tokens to a sequence, given LM trigram
    tables, using stupid backoff (same as the JS reranker)."""
    from math import log

    def unigram_prob(w):
        total = sum(lm["unigrams"].values())
        return (lm["unigrams"].get(w, 0) + 1) / (total + lm["vocab_size"])

    def bigram_prob(a, b):
        c = lm["bigrams"].get(f"{a}\t{b}")
        ctx = lm["unigrams"].get(a, 0)
        if c and ctx > 0:
            return c / ctx
        return 0.4 * unigram_prob(b)

    def trigram_prob(a, b, c_):
        c = lm["trigrams"].get(f"{a}\t{b}\t{c_}")
        ctx = lm["bigrams"].get(f"{a}\t{b}", 0)
        if c and ctx > 0:
            return c / ctx
        return 0.4 * bigram_prob(b, c_)

    vocab = set(lm["unigrams"].keys())
    score = 0.0
    a, b = prev2, prev1
    for w in new_tokens:
        w_unk = w if w in vocab else "<unk>"
        score += log(trigram_prob(a, b, w_unk))
        a, b = b, w_unk
    return score, a, b


def decode(source_tokens, phrase_table, lm, beam_width=BEAM_WIDTH, max_phrase_len=MAX_PHRASE_LEN):
    n = len(source_tokens)
    # hypotheses[k] = beam of Hypothesis objects that have consumed the
    # first k source tokens
    hypotheses = {0: [Hypothesis(pos=0, tokens=(), score=0.0)]}
    lm_state = {0: [("<s>", "<s>")]}  # parallel LM context per hypothesis in hypotheses[0]

    # We track (hypothesis, lm_context) pairs together to keep this simple.
    beams = {0: [(Hypothesis(pos=0, tokens=(), score=0.0), ("<s>", "<s>"))]}

    for k in range(1, n + 1):
        candidates = []
        for start in range(max(0, k - max_phrase_len), k):
            length = k - start
            if length < 1 or length > max_phrase_len:
                continue
            if start not in beams:
                continue
            src_phrase = " ".join(source_tokens[start:k])
            translations = phrase_table.get(src_phrase)

            if translations:
                options = list(translations.items())  # (fr_phrase, prob)
            else:
                if length == 1:
                    # OOV single word: copy through as a fallback so the
                    # decoder always produces *something*, with a heavy
                    # penalty so real phrase matches are always preferred.
                    options = [(source_tokens[start], None)]
                else:
                    options = []

            for fr_phrase, prob in options:
                phrase_score = math.log(prob) if prob else OOV_PHRASE_PENALTY
                fr_tokens = fr_phrase.split()
                for hyp, (a, b) in beams[start]:
                    lm_score, new_a, new_b = lm_increment_score(a, b, fr_tokens, lm)
                    new_score = hyp.score + phrase_score + lm_score
                    new_hyp = Hypothesis(pos=k, tokens=hyp.tokens + tuple(fr_tokens), score=new_score)
                    candidates.append((new_hyp, (new_a, new_b)))

        if not candidates:
            continue
        candidates.sort(key=lambda hc: hc[0].score, reverse=True)
        beams[k] = candidates[:beam_width]

    if n not in beams or not beams[n]:
        return None
    best_hyp, _ = max(beams[n], key=lambda hc: hc[0].score)
    return " ".join(best_hyp.tokens), best_hyp.score


if __name__ == "__main__":
    import json

    phrase_table = json.load(open("../models/smt/phrase_table.json", encoding="utf-8"))
    lm = json.load(open("../models/smt/lm.json", encoding="utf-8"))

    tests = [
        "the cat is black",
        "i am happy",
        "you like the dog",
        "the black cat likes water",
    ]
    for sent in tests:
        result = decode(sent.split(), phrase_table, lm)
        print(f"{sent!r:35s} -> {result}")
