

```markdown
# 📈 AI Stock Price Predictor (GRU)

This project is a deep learning system that predicts **next-day stock Opening and Closing prices** using **Gated Recurrent Units (GRUs)**.  
It analyzes **10 years of historical stock data** (Open, High, Low, Close, Volume) and forecasts the next trading day’s Open & Close values.

---

## 🚀 Features
- Downloads stock data automatically via **[yfinance](https://pypi.org/project/yfinance/)**
- Uses multiple features: `Open`, `High`, `Low`, `Close`, `Volume`
- Preprocessing with **MinMaxScaler**
- Sequence modeling using **GRU layers**
- Evaluates with **RMSE (Root Mean Square Error)**
- Shows **last 5 days Actual vs Predicted prices**
- Forecasts **next day’s Opening & Closing prices**
- Model saving in `.keras` format (preferred) and `.h5` (legacy)

---

```markdown
## 📂 Project Structure
```

ai-stock-price-predictor-gru/
├── notebooks/
│   └── predict\_stock.ipynb       # Main Jupyter Notebook
├── models/                       # Saved GRU models (.keras / .h5)
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
└── .gitignore

```

---

## ⚙️ Installation

Clone the repo:
```bash
git clone https://github.com/Purohit1999/ai-stock-price-predictor-gru.git
cd ai-stock-price-predictor-gru
````

Create & activate a virtual environment:

```bash
python -m venv genai_env
genai_env\Scripts\activate   # Windows
# OR
source genai_env/bin/activate  # Mac/Linux
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ Usage

Open the Jupyter Notebook:

```bash
jupyter notebook notebooks/predict_stock.ipynb
```

Or open it in **VS Code** with the Jupyter extension.

Change the stock ticker (e.g., `"TSLA"`, `"GOOG"`, `"MSFT"`) and re-run the notebook:

```python
ticker = "AAPL"
```

---

## 📊 Results

### Actual vs Predicted (Close Price)
![Close Price Prediction](https://github.com/Purohit1999/ai-stock-price-predictor-gru/blob/main/GRU/imgs/close_price_prediction.png?raw=true)


### Example: Last 5 Days (Actual vs Predicted)
| Actual_Open | Pred_Open | Actual_Close | Pred_Close |
|-------------|-----------|--------------|------------|
| 226.17      | 211.87    | 227.76       | 216.38     |
| 226.47      | 211.30    | 227.16       | 215.71     |
| 226.87      | 211.24    | 229.31       | 215.59     |
| 228.61      | 211.28    | 230.49       | 215.34     |
| 230.82      | 212.01    | 232.56       | 215.99     |

---

### Predicted Next Day (example output)

```
Predicted Opening Price for Tomorrow: 229.50
Predicted Closing Price for Tomorrow: 231.12
```

---

## 🧠 Model

* 2 × GRU layers (64 + 32 units)
* Dense layers for regression
* Optimizer: Adam
* Loss: Mean Squared Error (MSE)
* Trained with 10 epochs, batch size 64

---

## 📦 Saving & Loading Model

Save:

```python
model.save("models/gru_stock_model.keras")
```

Load:

```python
from tensorflow import keras
model = keras.models.load_model("models/gru_stock_model.keras")
```

---

## 📌 Notes

* For **inference only**, load with `compile=False` to avoid optimizer warnings:

  ```python
  model = keras.models.load_model("models/gru_stock_model.keras", compile=False)
  ```
* Model files (`.keras`, `.h5`) are ignored in `.gitignore`.
  Use **Git LFS** if you want to version them.

---

## 📜 License

This project is licensed under the MIT License.

---

👨‍💻 **Author**: [Purohit1999](https://github.com/Purohit1999)
📅 Year: 2025



