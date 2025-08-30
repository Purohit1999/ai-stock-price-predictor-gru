
# GENAI Mini-Lab
Next-word prediction + IMDB sentiment analysis built with TensorFlow/Keras.  
Includes ready-to-train scripts, CLI inference, and Jupyter/VS Code support.

---

## Repository Layout
```

GENAI/
├─ model\_creation.py           # train next-word model (LSTM) on toy corpus
├─ predict\_next\_word.py        # CLI/interactive next-word generation
├─ tokenizer.pkl               # saved tokenizer (+ max sequence len)
├─ next\_word\_model.h5          # legacy model format (created by training)
│
├─ LSTM/
│  └─ imdb\_sentiment\_analysis.ipynb  # your notebook (optional)
├─ imbd\_sentiment\_analysis.py  # train/infer IMDB sentiment (BiLSTM)
├─ imdb\_lstm.keras             # saved IMDB model (preferred format)
├─ imdb\_lstm.h5                # legacy IMDB model (optional)
│
├─ requirements.txt            # Python deps
└─ genai\_env/                  # Python virtual env (local)

````

> Files with a ✅ are produced after training:
- ✅ `next_word_model.h5`, `tokenizer.pkl`
- ✅ `imdb_lstm.keras` (and optionally `imdb_lstm.h5`)

---

## Quickstart (Windows / PowerShell)

```powershell
# 1) From GENAI/
python -m venv genai_env
.\genai_env\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# (Optional) Install Jupyter + kernel
pip install jupyter jupyterlab ipykernel
python -m ipykernel install --user --name genai_env
````

---

## Next-Word Prediction

### Train

```powershell
# from GENAI/
python model_creation.py
```

Artifacts saved to:

* `next_word_model.h5`
* `tokenizer.pkl`  (contains tokenizer + `max_sequence_len`)

### Generate (CLI)

```powershell
# Greedy (argmax)
python predict_next_word.py --seed "I love"

# Temperature sampling
python predict_next_word.py --seed "deep" --temp 0.8

# Top-k + multi-token generation
python predict_next_word.py --seed "machine learning" --num_words 10 --top_k 5

# Interactive loop
python predict_next_word.py --interactive --num_words 5 --top_k 5
```

**Notes**

* Input is cleaned to match training (lowercase, basic punctuation strip).
* If you see `<OOV>`, the word wasn’t in the tiny demo corpus (normal for small data).
* Script will prefer `next_word_model.keras` if present; otherwise uses `.h5`.

---

## IMDB Sentiment (BiLSTM)

### Train

```powershell
# from GENAI/
python imbd_sentiment_analysis.py --train
```

Saves:

* `imdb_lstm.keras` (preferred)
* `imdb_lstm.h5` (legacy)

### Predict a custom review

```powershell
python imbd_sentiment_analysis.py --predict "The movie was fantastic! I loved it."
```

### Run a short demo (no flags)

```powershell
python imbd_sentiment_analysis.py
```

**Config (defaults in code)**

* `NUM_WORDS=20000`, `MAXLEN=250`, `EMBED_DIM=128`
* BiLSTM(128) → Dropout → BiLSTM(64) → Dense(64 ReLU) → Dense(1 sigmoid)
* Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`
* Metrics: `accuracy`, `AUC`

---

## Jupyter / VS Code

1. Save your notebook as `LSTM/imdb_sentiment_analysis.ipynb`.
2. Top-right **Select Kernel** → choose **genai\_env**.
3. Run cells. In notebooks, the script ignores unknown Jupyter args automatically.
4. If you run the “demo” block in the notebook, it will quickly train 3 epochs in-memory and print a sample score.

---

## Troubleshooting

* **`Could not open requirements file`**
  Run `pip install -r requirements.txt` **from the folder where the file exists** (`GENAI/`). Use `dir` to confirm.

* **`Model not found` when predicting**
  Run the corresponding training script first (`model_creation.py` or `imbd_sentiment_analysis.py --train`).

* **`<OOV>` token in next-word output**
  Expected on tiny corpora. Add more text to `CORPUS_TEXT` in `model_creation.py` or train on a real corpus.

* **Jupyter adds weird CLI args (`-f ...`)**
  Handled via `parse_known_args()` in the scripts. No action needed.

* **Performance**
  CPU is fine for demos. For faster training, install a GPU-enabled TensorFlow.

---

## Reproducibility

* Fixed seeds (`random`, `numpy`, `tf`) are set for basic reproducibility. GPU/CuDNN and parallelism can still introduce small differences.

---

## Data Sources

* IMDB reviews via `tf.keras.datasets.imdb` (downloaded automatically on first run).

---

## License & Use

Educational project for experimentation and learning. Check individual dataset licenses before redistribution.

---

## Handy Commands

```powershell
# Upgrade pip (optional)
python -m pip install --upgrade pip

# Run Jupyter Lab (optional)
jupyter lab

# Clean and re-train IMDB with fewer epochs
python imbd_sentiment_analysis.py --train --epochs 3 --batch_size 64
```

