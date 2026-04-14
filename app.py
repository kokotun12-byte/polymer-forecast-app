import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Polymer Forecast App", layout="wide")

st.title("Hybrid ARIMAX–LSTM Polymer Import Forecasting App")

# =========================
# Build LSTM manually
# =========================
def build_lstm_model(lookback, n_features):
    model = Sequential([
        LSTM(64, activation="tanh", input_shape=(lookback, n_features), return_sequences=False),
        Dropout(0.1),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    return model

# =========================
# Load Models
# =========================
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    artifacts = joblib.load("hybrid_artifacts.pkl")
    history_y = joblib.load("history_y.pkl")
    history_X = joblib.load("history_X.pkl")
    feature_history = joblib.load("feature_history.pkl")

    lookback = int(artifacts["lookback"])
    n_features = feature_history.shape[1]   # should be 3

    lstm_model = build_lstm_model(lookback, n_features)
    lstm_model.load_weights("lstm_model.weights.h5")

    return scaler, artifacts, history_y, history_X, feature_history, lstm_model

scaler, artifacts, history_y, history_X, feature_history, lstm_model = load_models()

best_order = tuple(artifacts["best_order"])
lookback = int(artifacts["lookback"])
best_weight = float(artifacts["best_weight"])

# =========================
# File Upload
# =========================
uploaded_file = st.file_uploader("Upload Future Exogenous Data (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)

        required_cols = ["Date", "WTI_Price", "Exchange_Rate"]
        if not all(col in df.columns for col in required_cols):
            st.error("Missing required columns: Date, WTI_Price, Exchange_Rate")
        else:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").set_index("Date")

            if (df[["WTI_Price", "Exchange_Rate"]] <= 0).any().any():
                st.error("WTI_Price and Exchange_Rate must be positive.")
            else:
                future_log = np.log(df[["WTI_Price", "Exchange_Rate"]]).copy()
                future_log.columns = ["log_WTI_Price", "log_Exchange_Rate"]

                st.success("Data loaded successfully.")

                if st.button("Run Forecast"):
                    history_y_local = history_y.copy()
                    history_X_local = history_X.copy()
                    feature_history_local = feature_history.copy()

                    arimax_list = []
                    hybrid_list = []
                    weighted_list = []
                    dates = []

                    for i in range(len(future_log)):
                        x_next = future_log.iloc[[i]]
                        date = future_log.index[i]

                        model = SARIMAX(
                            history_y_local,
                            exog=history_X_local,
                            order=best_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        fit = model.fit(disp=False)

                        arimax_pred = fit.get_forecast(steps=1, exog=x_next).predicted_mean.iloc[0]

                        last_window = feature_history_local.iloc[-lookback:].copy()
                        scaled_window = scaler.transform(last_window)
                        X_input = scaled_window.reshape(1, lookback, scaled_window.shape[1])

                        pred_resid_scaled = lstm_model.predict(X_input, verbose=0)[0, 0]

                        dummy = np.zeros((1, feature_history_local.shape[1]))
                        dummy[0, 0] = pred_resid_scaled
                        resid = scaler.inverse_transform(dummy)[0, 0]

                        hybrid_pred = arimax_pred + resid
                        weighted_pred = best_weight * hybrid_pred + (1 - best_weight) * arimax_pred

                        arimax_list.append(arimax_pred)
                        hybrid_list.append(hybrid_pred)
                        weighted_list.append(weighted_pred)
                        dates.append(date)

                        history_y_local = pd.concat([
                            history_y_local,
                            pd.Series([weighted_pred], index=[date])
                        ])
                        history_X_local = pd.concat([history_X_local, x_next])

                        new_row = pd.DataFrame({
                            "residual": [resid],
                            "log_WTI_Price": [x_next.iloc[0, 0]],
                            "log_Exchange_Rate": [x_next.iloc[0, 1]]
                        }, index=[date])

                        feature_history_local = pd.concat([feature_history_local, new_row])

                    result_df = pd.DataFrame({
                        "Date": dates,
                        "ARIMAX": np.exp(arimax_list),
                        "Hybrid": np.exp(hybrid_list),
                        "Weighted_Hybrid": np.exp(weighted_list)
                    }).set_index("Date").round(2)

                    st.subheader("Forecast Results")
                    st.dataframe(result_df)

                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(result_df.index, result_df["ARIMAX"], "--", label="ARIMAX")
                    ax.plot(result_df.index, result_df["Hybrid"], ":", label="Hybrid Residual")
                    ax.plot(result_df.index, result_df["Weighted_Hybrid"], linewidth=2, label="Weighted Hybrid")
                    ax.set_title("Forecast Comparison")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Polymer Import (MT)")
                    ax.legend()
                    ax.grid(True)

                    st.pyplot(fig)

                    csv = result_df.to_csv().encode("utf-8")
                    st.download_button(
                        "Download Results",
                        csv,
                        "forecast.csv",
                        "text/csv"
                    )

    except Exception as e:
        st.error(f"Error: {e}")