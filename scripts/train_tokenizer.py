import sentencepiece as spm

# Train English tokenizer
spm.SentencePieceTrainer.train(
    input='cleaned_dataset.en',
    model_prefix='english_tokenizer',
    vocab_size=8000,
    character_coverage=1.0,
    model_type='bpe'
)

# Train French tokenizer
spm.SentencePieceTrainer.train(
    input='cleaned_dataset.fr',
    model_prefix='french_tokenizer',
    vocab_size=8000,
    character_coverage=1.0,
    model_type='bpe'
)

print("Tokenizers trained successfully.")