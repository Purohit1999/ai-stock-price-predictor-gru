"""
Load model + tokenizer and predict next word(s).

Features:
- Loads next_word_model.keras (preferred) or falls back to next_word_model.h5
- Greedy / temperature / top-k sampling
- Masks special tokens (PAD and <OOV>) to avoid junk outputs
- CLI + interactive mode
- Input cleaning to match training

Examples:
  python predict_next_word.py --seed "I love"                 # greedy
  python predict_next_word.py --seed "deep" --temp 0.8        # temperature
  python predict_next_word.py --seed "machine learning" --top_k 5
  python predict_next_word.py --seed "machine learning" --num_words 10 --top_k 5
  python predict_next_word.py --interactive
"""

import argparse
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------
# Paths
# -----------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH_KERAS = ROOT / "next_word_model.keras"
MODEL_PATH_H5 = ROOT / "next_word_model.h5"
TOKENIZER_PATH = ROOT / "tokenizer.pkl"

# -----------------------
# Load artifacts
# -----------------------
def load_model_any():
    if MODEL_PATH_KERAS.exists():
        print(f"📦 Loading model: {MODEL_PATH_KERAS.name}")
        return tf.keras.models.load_model(MODEL_PATH_KERAS)
    if MODEL_PATH_H5.exists():
        print(f"📦 Loading model: {MODEL_PATH_H5.name} (legacy H5)")
        return tf.keras.models.load_model(MODEL_PATH_H5)
    raise FileNotFoundError(
        "Model not found. Expected 'next_word_model.keras' or 'next_word_model.h5' "
        f"in {ROOT}"
    )

def load_tokenizer_payload():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
    with open(TOKENIZER_PATH, "rb") as f:
        payload = pickle.load(f)
    if "tokenizer" not in payload or "max_sequence_len" not in payload:
        raise ValueError("Tokenizer payload missing expected keys.")
    return payload

loaded_model = load_model_any()
payload = load_tokenizer_payload()
tokenizer = payload["tokenizer"]
max_sequence_len = payload["max_sequence_len"]

# -----------------------
# Text cleaning (match training)
# -----------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z'\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------
# Sampling helpers
# -----------------------
def softmax_temperature(logits, temperature: float):
    logits = np.asarray(logits, dtype="float64") / max(temperature, 1e-8)
    exp = np.exp(logits - np.max(logits))
    return exp / np.sum(exp)

def sample_from_probs(probs):
    probs = np.asarray(probs, dtype="float64")
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))

def top_k_filter(probs, k=5):
    probs = np.asarray(probs, dtype="float64")
    if k <= 0 or k >= probs.size:
        s = probs.sum()
        return probs / s if s > 0 else probs
    idx = np.argpartition(probs, -k)[-k:]
    masked = np.full_like(probs, -np.inf, dtype="float64")
    masked[idx] = np.log(probs[idx] + 1e-12)
    x = np.exp(masked - np.max(masked))
    return x / x.sum()

def mask_special_tokens(probs, tokenizer):
    """
    Zero-out PAD (index 0) and the tokenizer's OOV token,
    then renormalize so they can't be sampled.
    """
    probs = np.asarray(probs, dtype="float64")
    if probs.size == 0:
        return probs

    # PAD index is always 0 in Keras Tokenizer sequences
    probs[0] = 0.0

    # Find OOV index dynamically (handles <OOV> or <oov>)
    oov_idx = None
    for key in tokenizer.word_index.keys():
        if key.lower() == "<oov>":
            oov_idx = tokenizer.word_index[key]
            break
    if oov_idx is not None and oov_idx < probs.size:
        probs[oov_idx] = 0.0

    s = probs.sum()
    return probs / s if s > 0 else probs


# -----------------------
# Core prediction
# -----------------------
def predict_next(
    seed_text: str,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
) -> str:
    """
    If neither temperature nor top_k is provided -> greedy argmax.
    Returns the seed plus the predicted next word, or an explanatory message.
    """
    seed_text = clean_text(seed_text)
    seq = tokenizer.texts_to_sequences([seed_text])
    if not seq or len(seq[0]) == 0:
        return "No prediction (no known tokens after cleaning)."

    token_list = pad_sequences(seq, maxlen=max_sequence_len - 1, padding="pre")
    logits = loaded_model.predict(token_list, verbose=0)[0]  # probabilities

    # Start with raw probs, then mask special tokens
    probs = mask_special_tokens(logits, tokenizer)

    # Apply top-k (optional)
    if top_k is not None and top_k > 0:
        probs = top_k_filter(probs, k=top_k)

    # Temperature or greedy
    if temperature is not None and temperature > 0:
        probs = softmax_temperature(probs, temperature)
        next_index = sample_from_probs(probs)
    else:
        next_index = int(np.argmax(probs))

    # Map back to word (reverse lookup)
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}
    word = index_to_word.get(next_index)
    if word is None:
        return "No prediction found."
    return f"{seed_text} {word}"

def generate_text(
    seed_text: str,
    num_words: int = 5,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
) -> str:
    text = clean_text(seed_text)
    for _ in range(num_words):
        candidate = predict_next(text, temperature=temperature, top_k=top_k)
        if candidate.startswith("No prediction"):
            break
        text = candidate
    return text

# -----------------------
# CLI / Interactive
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Next-word prediction")
    parser.add_argument("--seed", type=str, default=None, help="Seed text")
    parser.add_argument("--num_words", type=int, default=1, help="How many words to generate")
    parser.add_argument("--temp", type=float, default=None, help="Temperature (e.g., 0.7). If omitted -> greedy")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling (e.g., 5)")
    parser.add_argument("--interactive", action="store_true", help="Interactive loop")
    args = parser.parse_args()

    if args.interactive:
        print("🔁 Interactive mode. Press Enter on empty line to quit.")
        while True:
            try:
                seed = input("Enter a seed: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not seed:
                print("Bye!")
                break
            out = generate_text(seed, num_words=args.num_words, temperature=args.temp, top_k=args.top_k)
            print("→", out)
        return

    if args.seed is None:
        # quick demos if no seed provided
        print(predict_next("I love"))
        print(predict_next("deep", temperature=0.7))
        print(generate_text("machine learning", num_words=10, top_k=5))
        return

    result = generate_text(args.seed, num_words=args.num_words, temperature=args.temp, top_k=args.top_k)
    print(result)

if __name__ == "__main__":
    main()
