import nltk
import re

# Download NLTK punkt tokenizer
nltk.download("punkt")

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation/symbols but KEEP letters with diacritics (é, è, ç, à, ô, etc.)
    # and basic apostrophes, since these are meaningful in French.
    text = re.sub(r"[^a-zA-Zà-öø-ÿÀ-ÖØ-ß'\s]", "", text)
    # Tokenize sentences
    sentences = nltk.sent_tokenize(text)
    return sentences

# Load datasets
with open("../data/dataset.en", "r", encoding="utf-8") as f:
    english_data = f.readlines()

with open("../data/dataset.fr", "r", encoding="utf-8") as f:
    french_data = f.readlines()

# Preprocess datasets
english_data = [" ".join(preprocess_text(line)) for line in english_data]
french_data = [" ".join(preprocess_text(line)) for line in french_data]

# Save preprocessed data
with open("../data/cleaned_dataset.en", "w", encoding="utf-8") as f:
    f.writelines("\n".join(english_data))

with open("../data/cleaned_dataset.fr", "w", encoding="utf-8") as f:
    f.writelines("\n".join(french_data))

print("Preprocessing completed!")