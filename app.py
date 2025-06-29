import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --- Constantes ---
sequence_length = 30
target = 'Close_CL=F'
features = [
    'Close_CL=F', 'High_CL=F', 'Low_CL=F', 'Open_CL=F', 'Volume_CL=F',
    'EMA_7', 'EMA_21', 'RSI', 'MACD', 'Signal_Line',
    'Open Interest (All)', 'Noncommercial Positions-Long (All)',
    'Noncommercial Positions-Short (All)'
]

# --- Chargement du modèle ---
model = load_model("model_lstm.h5")

# --- Données ---
@st.cache_data
def load_data():
    cftc_df = pd.read_csv("annual.csv")
    market_df = yf.download('CL=F', start='2018-01-01')

    cftc_df['As of Date in Form YYYY-MM-DD'] = pd.to_datetime(cftc_df['As of Date in Form YYYY-MM-DD'])
    market_df.reset_index(inplace=True)
    merged_df = pd.merge(
        cftc_df, market_df,
        left_on='As of Date in Form YYYY-MM-DD',
        right_on='Date',
        how='inner'
    )

    def add_indicators(df):
        df['EMA_7'] = df['Close_CL=F'].ewm(span=7).mean()
        df['EMA_21'] = df['Close_CL=F'].ewm(span=21).mean()
        delta = df['Close_CL=F'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['MACD'] = df['Close_CL=F'].ewm(span=12).mean() - df['Close_CL=F'].ewm(span=26).mean()
        df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
        return df.dropna()

    df = add_indicators(merged_df).fillna(method='ffill')
    return df

# --- Prediction ---
def predict_signal(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    X = []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i])
    X = np.array(X)

    last_seq = X[-1]
    pred_scaled = model.predict(np.array([last_seq]))[0][0]
    target_idx = features.index(target)

    full_shape = np.zeros((1, len(features)))
    full_shape[0, target_idx] = pred_scaled
    predicted_price = scaler.inverse_transform(full_shape)[0, target_idx]
    current_price = df[features].iloc[-1][target]

    if predicted_price > current_price * 1.01:
        return "🟢 BUY", current_price, predicted_price
    elif predicted_price < current_price * 0.99:
        return "🔴 SELL", current_price, predicted_price
    else:
        return "🟡 HOLD", current_price, predicted_price

# --- Streamlit UI ---
st.title("🤖 Trading IA – Prédiction LSTM")
df = load_data()
signal, price_now, price_pred = predict_signal(df)

st.metric("Prix actuel", f"{price_now:.2f} $")
st.metric("Prix prédit (demain)", f"{price_pred:.2f} $")
st.subheader(f"📊 Signal IA : {signal}")
st.line_chart(df['Close_CL=F'][-100:])
