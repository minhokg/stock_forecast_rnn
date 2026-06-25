# Stock Forecast RNN

Deep learning-based stock price forecasting using Recurrent Neural Networks (RNNs). This project explores multiple sequence modeling architectures, including:

- Vanilla RNN
- LSTM (Long Short-Term Memory)
- Seq2Seq Networks
- Attention-based Models

The models are trained on historical Google stock price data to predict future price movements and evaluate the effectiveness of different recurrent architectures for financial time-series forecasting.

---

## Project Overview

Financial markets generate sequential data where past observations influence future outcomes. Traditional machine learning models often struggle to capture long-term temporal dependencies.

This project investigates how recurrent neural networks can learn patterns from historical stock prices and generate future forecasts.

The repository includes:

- Data preprocessing pipeline
- Feature engineering
- RNN-based forecasting models
- Seq2Seq architecture
- Attention mechanism implementation
- Training and evaluation workflows
- Visualization of predictions

---

## Model Architectures

### 1. Vanilla RNN

A basic recurrent neural network that processes stock prices sequentially while maintaining a hidden state.

**Advantages**
- Simple architecture
- Fast training

**Limitations**
- Vanishing gradient problem
- Difficulty learning long-term dependencies

---

### 2. LSTM

Long Short-Term Memory networks extend RNNs with memory cells and gating mechanisms.

**Advantages**
- Captures long-term dependencies
- More stable training
- Widely used in time-series forecasting

---

### 3. Seq2Seq

Sequence-to-Sequence models use an encoder-decoder architecture.

**Advantages**
- Flexible forecasting horizon
- Better handling of complex temporal patterns

---

### 4. Attention Model

Attention mechanisms allow the model to focus on the most relevant historical observations when generating predictions.

**Advantages**
- Improved interpretability
- Better performance on long sequences
- Enhanced forecasting accuracy

---

## Dataset

The project uses historical Google stock price data.

Typical features include:

- Open
- High
- Low
- Close
- Volume

After preprocessing, the data is transformed into sequences suitable for recurrent neural networks.

---

## Project Structure

```text
stock_forecast_rnn/
│
├── src/
│   └── stock_forecast_rnn/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/minhokg/stock_forecast_rnn.git
cd stock_forecast_rnn
```



### Install Dependencies

```bash
pip install -r requirements.txt
```

