# 📈 Stock Price Prediction using LSTM (Deep Learning)

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)
![YFinance](https://img.shields.io/badge/Data-Yahoo%20Finance-green.svg)

## 📖 Overview

This project utilizes **Long Short-Term Memory (LSTM)** networks, a special kind of Recurrent Neural Network (RNN) capable of learning long-term dependencies, to predict the future stock prices of **Google (GOOGL)**.

By analyzing historical stock data fetched directly from Yahoo Finance, the deep learning model identifies patterns in time-series data to forecast the closing price for the next trading day.

## 📂 Data Sourcing

The data is fetched dynamically using the `yfinance` API:
* **Ticker:** GOOGL (Alphabet Inc.)
* **Start Date:** 2010-01-01
* **End Date:** 2024-12-01
* **Target Feature:** Closing Price (`Close`)

## 🛠️ Technologies & Libraries

* **Python**: Core programming language.
* **TensorFlow / Keras**: Used for building and training the LSTM neural network.
* **Pandas & NumPy**: Used for data manipulation, array processing, and dataset creation.
* **Matplotlib**: Used for visualizing the historical price data and prediction results.
* **Scikit-Learn**: Used for data normalization (`MinMaxScaler`) to scale values between 0 and 1.
* **Yfinance**: Used to download historical market data.

## 🧠 Model Architecture

The model is built using the Keras `Sequential` API and processes sequences of **60 days** of historical data to predict the 61st day.

1.  **LSTM Layer 1**: 50 Units, `return_sequences=True`, Input Shape: (60, 1).
2.  **LSTM Layer 2**: 50 Units, `return_sequences=False` (Does not return sequences to the next layer).
3.  **Dense Layer**: 25 Neurons.
4.  **Output Layer**: 1 Neuron (Predicts the continuous price value).

**Compilation Details:**
* **Optimizer:** Adam
* **Loss Function:** Mean Squared Error (MSE)

## 📊 Training & Results

The model was trained with the following parameters:
* **Batch Size:** 32
* **Epochs:** 10
* **Training/Test Split:** 80% Training / 20% Testing

### Performance
After training, the model visualizes the comparison between the **Actual Stock Prices** and the **Predicted Prices**.

### Future Prediction
The model takes the last 60 days of the dataset to predict the stock price for the next trading day.
* **Predicted Next Day Price for GOOGL:** ~$163.97

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Stock-Price-Prediction-LSTM.git](https://github.com/YOUR_USERNAME/Stock-Price-Prediction-LSTM.git)
    cd Stock-Price-Prediction-LSTM
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy pandas matplotlib yfinance scikit-learn tensorflow
    ```

3.  **Run the Notebook:**
    Open the notebook in Jupyter or Google Colab:
    ```bash
    jupyter notebook "Stock Price Prediction using LSTM (Deep Learning).ipynb"
    ```

## ⚠️ Disclaimer

This project is for **educational purposes only**. Stock market prediction is volatile and complex; this model should not be used for actual financial trading or investment advice.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

