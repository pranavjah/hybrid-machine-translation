# Hybrid Machine Translation (EN → FR)

Two independent translation engines translate the same English sentence, and a
tuned reranker picks the winner:

1. **NMT** — a pretrained transformer (`Helsinki-NLP/opus-mt-en-fr`) generates
   5 candidate translations via beam search.
2. **SMT** — a hand-rolled statistical decoder proposes its own candidate,
   built from scratch on this repo's training pipeline:
   - **IBM Model 2** word-alignment probabilities (word-translation
     probabilities + a learned relative-position bias), trained via EM in
     both directions (EN→FR and FR→EN).
   - Alignments are **symmetrized** with the standard grow-diag heuristic
     (Och & Ney), then used to **extract phrase pairs** (up to 3 words) via
     the standard Moses-style consistent-phrase-extraction algorithm.
   - A **trigram language model** (with stupid backoff smoothing) scores
     fluency.
   - A **monotone phrase-based decoder** does beam search over the phrase
     table + LM to generate a translation, left-to-right, with no
     reordering.

Both engines' candidates are pooled and reranked using IBM Model
1/2 translation probabilities + LM fluency, with the relative weighting
**tuned by grid search against 150 held-out reference translations**
(see `scripts/tune_weights.py`), not guessed.

Everything — NMT inference (via [transformers.js](https://github.com/xenova/transformers.js)),
the SMT decoder, and the reranker — runs **entirely client-side in the
browser**. No backend, no Flask API.

## Known limitation

The SMT decoder is **monotone** — it can only consume source words
left-to-right, with no reordering. This is a real, documented simplification
(full phrase-based decoders like Moses track a coverage bitmap and allow
reordering within a distortion window; that's a meaningfully larger
undertaking). The practical effect: word-order differences between English
and French — most visibly adjective placement — don't get fixed. For
example, "the black cat" decodes to "le noir chat" instead of the correct
"le chat noir". This shows up consistently and is expected, not a bug.

## Architecture

```
scripts/
  preprocess.py        # cleans + tokenizes the raw EN/FR corpus (preserves French diacritics)
  train_phrase_smt.py  # trains IBM Model 2 (both directions), extracts phrases, trains the LM
  decode_smt.py         # standalone monotone phrase-based decoder (Python reference implementation)
  tune_weights.py       # grid search for reranker weights against held-out references
  train_model.py         # (optional) fine-tunes opus-mt-en-fr on this corpus — NOT used by the live demo (see below)

models/smt/
  translation_probs.json  # IBM Model 1/2 word-translation probabilities
  distortion.json          # learned relative-position bias
  phrase_table.json        # extracted phrase pairs + probabilities
  lm.json                  # trigram language model counts
```

The live demo uses the **pretrained** `Helsinki-NLP/opus-mt-en-fr` model
(via its [Xenova ONNX port](https://huggingface.co/Xenova/opus-mt-en-fr)),
not a custom fine-tuned checkpoint. `train_model.py` is included and fully
functional (it fine-tunes and actually saves a checkpoint, which the
original version of this script did not do), but fine-tuning on generic
Tatoeba sentences doesn't meaningfully beat the base model's quality, and
shipping a custom fine-tuned checkpoint to the browser would require its
own ONNX conversion pipeline. It's here for anyone who wants to extend the
project in that direction.

## Training data

English-French sentence pairs from
[Tatoeba](https://tatoeba.org/), distributed via
[ManyThings.org](https://www.manythings.org/anki/) (`fra-eng.zip`), CC-BY.
~20,000 sentence pairs were used for training (subsampled from ~240k
available pairs for reasonable CPU training time).

The raw corpus isn't committed to this repo (`data/` is gitignored) to keep
the repo small. To regenerate it:

```powershell
Invoke-WebRequest -Uri "https://www.manythings.org/anki/fra-eng.zip" -OutFile "fra-eng.zip"
Expand-Archive -Path fra-eng.zip -DestinationPath .\tmp_fra_eng -Force
# split tmp_fra_eng/fra.txt (tab-separated EN<TAB>FR<TAB>attribution) into
# data/dataset.en and data/dataset.fr, then:
cd scripts
python preprocess.py
python train_phrase_smt.py
```

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r scripts/requirements.txt
# torch>=2.5 required; if you hit a version mismatch:
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
```

## Reproducing the tuned weights

```powershell
cd scripts
python tune_weights.py
```

This runs the real NMT model against 150 held-out validation sentences (not
seen during SMT training) and grid-searches the translation/fluency
weighting, scoring each combination with an n-gram precision metric against
the real French references. Current tuned weights: **translation = 0.4,
fluency = 0.6** (fluency scoring turned out more informative than raw
translation-adequacy for this data/model combination).