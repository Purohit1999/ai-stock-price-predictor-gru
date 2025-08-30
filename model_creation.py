"""
Train a next-word predictor and save:
- next_word_model.h5        (legacy format to keep current predict script working)
- next_word_model.keras     (new recommended Keras format)
- tokenizer.pkl             (contains tokenizer + max_sequence_len)

Enhancements:
- Clean text + lowercase
- NLTK sentence split (with auto-download)
- OOV handling in Tokenizer
- n-gram sequence creation (predict next token)
- LSTM + Embedding
- EarlyStopping + ModelCheckpoint (uses val_loss if we have a validation split)
- Reproducibility seeds
- Auto-detect external corpus.txt if present
"""

import os
import re
import random
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import nltk
from nltk.tokenize import sent_tokenize

# -----------------------
# Reproducibility
# -----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# -----------------------
# Config
# -----------------------
EPOCHS = 200
BATCH_SIZE = 32
EMBED_DIM = 64
LSTM_UNITS = 128
DROPOUT = 0.2
MIN_TOKENS_PER_NGRAM = 2

# Validation split (only applied if we have enough sequences)
# Set to 0.1 by default; will fall back to training on 'loss' if not enough data.
VAL_SPLIT = 0.1
MIN_SAMPLES_FOR_VAL = 50  # require at least this many sequences to use validation split

# -----------------------
# Paths
# -----------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH_H5 = ROOT / "next_word_model.h5"
MODEL_PATH_KERAS = ROOT / "next_word_model.keras"
TOKENIZER_PATH = ROOT / "tokenizer.pkl"
CORPUS_FILE = ROOT / "corpus.txt"  # If present, we read from this

# -----------------------
# Fallback demo corpus
# -----------------------
CORPUS_TEXT_FALLBACK = """
I love machine learning. I love deep learning.
Machine learning is fun. Deep learning is powerful.
I enjoy learning new things. Learning new skills takes practice.
Natural language processing enables machines to understand text.
Neural networks can generate text and predict the next word.
"""

# -----------------------
# Helpers
# -----------------------
def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


def clean_text(text: str) -> str:
    # basic normalization: lowercase, remove non-letters except apostrophes/spaces
    text = text.lower()
    text = re.sub(r"[^a-z'\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_ngrams(tokenizer, sentences, min_tokens=2):
    """
    Turn sentences into n-gram sequences:
    e.g. "i love ml" -> [i,love,ml]
    produce:
      [i] -> predict 'love'
      [i,love] -> predict 'ml'
    """
    sequences = []
    for sent in sentences:
        if not sent:
            continue
        token_list = tokenizer.texts_to_sequences([sent])[0]
        for i in range(2, len(token_list) + 1):
            n_gram = token_list[:i]
            if len(n_gram) >= min_tokens:
                sequences.append(n_gram)
    return sequences


def prepare_xy(sequences):
    max_len = max(len(s) for s in sequences)
    sequences = pad_sequences(sequences, maxlen=max_len, padding="pre")
    X = sequences[:, :-1]
    y = sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=None)  # auto-infer classes
    num_classes = y.shape[1]
    return X, y, max_len, num_classes


def read_corpus() -> str:
    # Prefer external file if present
    if CORPUS_FILE.exists():
        print(f"📄 Using external corpus file: {CORPUS_FILE.name}")
        return CORPUS_FILE.read_text(encoding="utf-8", errors="ignore")
    print("ℹ️ Using built-in demo corpus (create 'corpus.txt' to use your own).")
    return CORPUS_TEXT_FALLBACK


# -----------------------
# Main training
# -----------------------
def main():
    ensure_nltk()

    raw = read_corpus()

    # Split into sentences then clean
    sentences = [clean_text(s) for s in sent_tokenize(raw)]
    sentences = [s for s in sentences if s]  # drop empties
    if not sentences:
        raise RuntimeError("No sentences after cleaning. Check your corpus.")

    # Tokenizer with OOV to be safer at inference
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        oov_token="<OOV>", filters=""  # we already cleaned
    )
    tokenizer.fit_on_texts(sentences)
    total_words = len(tokenizer.word_index) + 1  # +1 for padding

    # Create n-gram sequences
    sequences = build_ngrams(tokenizer, sentences, min_tokens=MIN_TOKENS_PER_NGRAM)
    if not sequences:
        raise RuntimeError("No training sequences created. Check your corpus.")

    X, y, max_sequence_len, num_classes = prepare_xy(sequences)
    print(f"📊 Sequences: {len(sequences)} | Vocab: {total_words} | Max seq len: {max_sequence_len}")

    # Model
    model = Sequential([
        Embedding(input_dim=total_words,
                  output_dim=EMBED_DIM,
                  input_length=max_sequence_len - 1),
        LSTM(LSTM_UNITS, return_sequences=False),
        Dropout(DROPOUT),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    # Decide whether to use validation split
    use_val = len(sequences) >= MIN_SAMPLES_FOR_VAL and VAL_SPLIT > 0
    monitor = "val_loss" if use_val else "loss"
    if use_val:
        print(f"✅ Using validation split: {VAL_SPLIT:.2f} (monitor: {monitor})")
    else:
        print(f"ℹ️ Not enough data for validation split; monitoring '{monitor}'")

    # Checkpoints (save best in new Keras format)
    ckpt = ModelCheckpoint(
        filepath=str(MODEL_PATH_KERAS),
        monitor=monitor,
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    es = EarlyStopping(
        monitor=monitor,
        patience=10,
        restore_best_weights=True
    )

    # Train
    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[ckpt, es],
        validation_split=VAL_SPLIT if use_val else 0.0,
        shuffle=True,
    )

    # Ensure final saves (both formats)
    # New recommended format
    model.save(MODEL_PATH_KERAS)
    # Legacy H5 to keep backward compatibility with your current predict script
    model.save(MODEL_PATH_H5)

    # Save tokenizer + maxlen together
    with open(TOKENIZER_PATH, "wb") as f:
        payload = {
            "tokenizer": tokenizer,
            "max_sequence_len": max_sequence_len
        }
        pickle.dump(payload, f)

    print(f"✅ Saved model (Keras) -> {MODEL_PATH_KERAS.name}")
    print(f"✅ Saved model (H5)    -> {MODEL_PATH_H5.name}")
    print(f"✅ Saved tokenizer     -> {TOKENIZER_PATH.name}")

if __name__ == "__main__":
    main()
