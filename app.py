import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import date

from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Polymer Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #0f172a;
}
.card {
    background: white;
    padding: 1rem 1.2rem;
    border-radius: 16px;
    box-shadow: 0 2px 14px rgba(15, 23, 42, 0.06);
    border: 1px solid #e2e8f0;
}
.small-note {
    color: #475569;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Title
# =========================
st.title("📈 Hybrid ARIMAX–LSTM Polymer Import Forecast Dashboard")
st.caption("Forecast polymer import volume using manual input or Excel upload")

# =========================
# Build LSTM manually
# =========================
def build_lstm_model(lookback, n_features):
    model = Sequential([
        LSTM(64, activation="tanh", input_shape=(lookback, n_features), return_sequences=False),
        Dropout(0.2),
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
    n_features = feature_history.shape[1]

    lstm_model = build_lstm_model(lookback, n_features)
    lstm_model.load_weights("lstm_model.weights.h5")

    return scaler, artifacts, history_y, history_X, feature_history, lstm_model

scaler, artifacts, history_y, history_X, feature_history, lstm_model = load_models()

best_order = tuple(artifacts["best_order"])
lookback = int(artifacts["lookback"])
best_weight = float(artifacts["best_weight"])

# =========================
# Helpers
# =========================
MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

def first_day_of_month(year, month):
    return pd.Timestamp(year=int(year), month=int(month), day=1)

def get_allowed_max_date():
    today = pd.Timestamp.today().normalize()
    return pd.Timestamp(today.year, 12, 1)

def get_history_level_series(history_y_log):
    hist = pd.Series(np.exp(history_y_log.values), index=pd.to_datetime(history_y_log.index))
    hist.name = "Historical_Actual"
    return hist

def validate_future_df(df, source_name="input"):
    if df.empty:
        return False, f"No rows found in {source_name}.", None

    required_cols = ["Date", "WTI_Price", "Exchange_Rate"]
    if not all(col in df.columns for col in required_cols):
        return False, "Required columns are: Date, WTI_Price, Exchange_Rate", None

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if df["Date"].duplicated().any():
        return False, "Duplicate Year-Month entries are not allowed.", None

    if (df[["WTI_Price", "Exchange_Rate"]] <= 0).any().any():
        return False, "WTI_Price and Exchange_Rate must be positive values.", None

    # force monthly first day
    if not all(df["Date"].dt.day == 1):
        return False, "Dates must represent month-level input using the first day of month.", None

    # strictly future compared with last historical date
    history_last_date = pd.to_datetime(history_y.index.max())
    if (df["Date"] <= history_last_date).any():
        return False, f"Forecast dates must be after the latest historical date: {history_last_date.strftime('%Y-%m')}.", None

    # consecutive monthly sequence
    expected_dates = pd.date_range(start=df["Date"].iloc[0], periods=len(df), freq="MS")
    if not df["Date"].reset_index(drop=True).equals(pd.Series(expected_dates)):
        return False, "Forecast periods must be continuous monthly dates without gaps.", None

    # forecast horizon restriction
    max_allowed_date = get_allowed_max_date()
    if df["Date"].max() > max_allowed_date:
        return False, (
            f"Forecast is allowed only up to 12 months ahead. "
            f"Maximum allowed month is {max_allowed_date.strftime('%Y-%m')}."
        ), None

    return True, "Validation successful.", df

def parse_uploaded_excel(uploaded_file):
    df = pd.read_excel(uploaded_file)

    cols = [c.strip() for c in df.columns]
    df.columns = cols

    # support either Date column OR Year+Month columns
    if {"Year", "Month", "WTI_Price", "Exchange_Rate"}.issubset(df.columns):
        df["Date"] = pd.to_datetime(
            dict(year=df["Year"].astype(int), month=df["Month"].astype(int), day=1)
        )
        df = df[["Date", "WTI_Price", "Exchange_Rate"]]

    elif {"Date", "WTI_Price", "Exchange_Rate"}.issubset(df.columns):
        df["Date"] = pd.to_datetime(df["Date"]).dt.to_period("M").dt.to_timestamp()
        df = df[["Date", "WTI_Price", "Exchange_Rate"]]

    else:
        raise ValueError(
            "Excel must contain either columns: Year, Month, WTI_Price, Exchange_Rate "
            "or Date, WTI_Price, Exchange_Rate"
        )

    return df

def create_excel_download(df_result):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_result.to_excel(writer, sheet_name="Forecast_Results", index=False)
    output.seek(0)
    return output

def create_template_file():
    today = pd.Timestamp.today()
    start_date = pd.Timestamp(today.year, today.month, 1) + pd.DateOffset(months=1)
    dates = pd.date_range(start=start_date, periods=4, freq="MS")

    template = pd.DataFrame({
        "Year": dates.year,
        "Month": dates.month,
        "WTI_Price": [70.0, 71.5, 72.0, 73.2],
        "Exchange_Rate": [3500, 3520, 3535, 3550]
    })

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        template.to_excel(writer, sheet_name="Template", index=False)
    output.seek(0)
    return output

def run_forecast(future_input_df):
    history_y_local = history_y.copy()
    history_X_local = history_X.copy()
    feature_history_local = feature_history.copy()

    future_input_df = future_input_df.copy()
    future_input_df["Date"] = pd.to_datetime(future_input_df["Date"])
    future_input_df = future_input_df.sort_values("Date").reset_index(drop=True)

    future_log = np.log(future_input_df[["WTI_Price", "Exchange_Rate"]]).copy()
    future_log.columns = ["log_WTI_Price", "log_Exchange_Rate"]
    future_log.index = future_input_df["Date"]

    arimax_list = []
    hybrid_list = []
    weighted_list = []
    dates = []

    for i in range(len(future_log)):
        x_next = future_log.iloc[[i]]
        forecast_date = future_log.index[i]

        model = SARIMAX(
            history_y_local,
            exog=history_X_local,
            order=best_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fit = model.fit(disp=False)

        arimax_pred_log = fit.get_forecast(steps=1, exog=x_next).predicted_mean.iloc[0]

        last_window = feature_history_local.iloc[-lookback:].copy()
        scaled_window = scaler.transform(last_window)
        X_input = scaled_window.reshape(1, lookback, scaled_window.shape[1])

        pred_resid_scaled = lstm_model.predict(X_input, verbose=0)[0, 0]

        dummy = np.zeros((1, feature_history_local.shape[1]))
        dummy[0, 0] = pred_resid_scaled
        resid_log = scaler.inverse_transform(dummy)[0, 0]

        hybrid_pred_log = arimax_pred_log + resid_log
        weighted_pred_log = best_weight * hybrid_pred_log + (1 - best_weight) * arimax_pred_log

        arimax_list.append(np.exp(arimax_pred_log))
        hybrid_list.append(np.exp(hybrid_pred_log))
        weighted_list.append(np.exp(weighted_pred_log))
        dates.append(forecast_date)

        # recursive update in log scale
        history_y_local = pd.concat([
            history_y_local,
            pd.Series([weighted_pred_log], index=[forecast_date])
        ])

        history_X_local = pd.concat([history_X_local, x_next])

        new_row = pd.DataFrame({
            "residual": [resid_log],
            "log_WTI_Price": [x_next.iloc[0, 0]],
            "log_Exchange_Rate": [x_next.iloc[0, 1]]
        }, index=[forecast_date])

        feature_history_local = pd.concat([feature_history_local, new_row])

    result_df = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Year": pd.to_datetime(dates).year,
        "Month": pd.to_datetime(dates).month,
        "Month_Name": [MONTH_NAMES[d.month] for d in pd.to_datetime(dates)],
        "ARIMAX_Level": np.round(arimax_list, 2),
        "Hybrid_Level": np.round(hybrid_list, 2),
        "Weighted_Hybrid_Level": np.round(weighted_list, 2),
        "Input_WTI_Price": future_input_df["WTI_Price"].values,
        "Input_Exchange_Rate": future_input_df["Exchange_Rate"].values
    })

    return result_df

def plot_history_and_forecast(result_df):
    hist_level = get_history_level_series(history_y)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        hist_level.index,
        hist_level.values,
        label="Historical Actual",
        linewidth=2
    )

    ax.plot(
        pd.to_datetime(result_df["Date"]),
        result_df["ARIMAX_Level"],
        linestyle="--",
        label="ARIMAX Forecast",
        linewidth=2
    )

    ax.plot(
        pd.to_datetime(result_df["Date"]),
        result_df["Hybrid_Level"],
        linestyle=":",
        label="Hybrid Residual Forecast",
        linewidth=2
    )

    ax.plot(
        pd.to_datetime(result_df["Date"]),
        result_df["Weighted_Hybrid_Level"],
        label="Weighted Hybrid Forecast",
        linewidth=3
    )

    ax.set_title("Historical Polymer Import and Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Polymer Import Volume (MT)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig

# =========================
# Sidebar Info
# =========================
with st.sidebar:
    st.markdown("### Settings")
    st.info(
        f"""
        **Forecast horizon rule**
        
        - Manual input: maximum **3 months**
        - Upload file: use for **more than 3 months**
        - Maximum forecast month allowed: **{get_allowed_max_date().strftime('%Y-%m')}**
        """
    )

    st.markdown("### Download Template")
    template_file = create_template_file()
    st.download_button(
        label="Download Excel Template",
        data=template_file,
        file_name="forecast_input_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =========================
# Input Section
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Input Future Exogenous Data")
st.markdown(
    '<p class="small-note">Enter Year, Month, WTI price, and exchange rate manually for up to 3 months. '
    'For more than 3 months, please upload an Excel file.</p>',
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["Manual Input (Max 3 Months)", "Excel Upload"])

future_df = None

with tab1:
    st.markdown("#### Manual Input")

    current_year = pd.Timestamp.today().year
    years_range = list(range(current_year, current_year + 2))
    month_options = list(range(1, 13))

    manual_rows = []
    row_cols = st.columns([1.1, 1.1, 1.5, 1.5])

    with row_cols[0]:
        st.markdown("**Year**")
    with row_cols[1]:
        st.markdown("**Month**")
    with row_cols[2]:
        st.markdown("**WTI Price**")
    with row_cols[3]:
        st.markdown("**Exchange Rate**")

    for i in range(3):
        c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.5, 1.5])

        with c1:
            year = st.number_input(
                f"Year {i+1}",
                min_value=current_year,
                max_value=current_year + 1,
                value=current_year,
                step=1,
                key=f"year_{i}"
            )
        with c2:
            month = st.selectbox(
                f"Month {i+1}",
                options=month_options,
                format_func=lambda x: MONTH_NAMES[x],
                key=f"month_{i}"
            )
        with c3:
            wti = st.number_input(
                f"WTI {i+1}",
                min_value=0.0,
                value=0.0,
                step=0.1,
                key=f"wti_{i}"
            )
        with c4:
            exr = st.number_input(
                f"Exchange Rate {i+1}",
                min_value=0.0,
                value=0.0,
                step=1.0,
                key=f"exr_{i}"
            )

        manual_rows.append({
            "Year": int(year),
            "Month": int(month),
            "WTI_Price": float(wti),
            "Exchange_Rate": float(exr)
        })

    manual_submit = st.button("Use Manual Input", type="primary")

    if manual_submit:
        manual_df = pd.DataFrame(manual_rows)

        # keep only rows where both input values are provided and > 0
        manual_df = manual_df[
            (manual_df["WTI_Price"] > 0) &
            (manual_df["Exchange_Rate"] > 0)
        ].copy()

        if manual_df.empty:
            st.warning("Please enter at least one valid month with positive WTI and exchange rate.")
        elif len(manual_df) > 3:
            st.error("Manual input allows maximum 3 months only.")
        else:
            manual_df["Date"] = pd.to_datetime(
                dict(year=manual_df["Year"], month=manual_df["Month"], day=1)
            )
            manual_df = manual_df[["Date", "WTI_Price", "Exchange_Rate"]]

            ok, msg, cleaned_df = validate_future_df(manual_df, "manual input")
            if ok:
                future_df = cleaned_df
                st.success("Manual input is valid and ready for forecasting.")
                st.dataframe(
                    cleaned_df.assign(
                        Year=cleaned_df["Date"].dt.year,
                        Month=cleaned_df["Date"].dt.month
                    )[["Year", "Month", "WTI_Price", "Exchange_Rate"]],
                    use_container_width=True
                )
            else:
                st.error(msg)

with tab2:
    st.markdown("#### Excel Upload")
    uploaded_file = st.file_uploader(
        "Upload .xlsx file",
        type=["xlsx"],
        help="Required columns: either Year, Month, WTI_Price, Exchange_Rate OR Date, WTI_Price, Exchange_Rate"
    )

    if uploaded_file is not None:
        try:
            uploaded_df = parse_uploaded_excel(uploaded_file)

            if len(uploaded_df) <= 3:
                st.warning("Excel upload is mainly for more than 3 months. For up to 3 months, manual input is recommended.")

            ok, msg, cleaned_df = validate_future_df(uploaded_df, "uploaded file")
            if ok:
                future_df = cleaned_df
                st.success("Uploaded file is valid and ready for forecasting.")
                st.dataframe(
                    cleaned_df.assign(
                        Year=cleaned_df["Date"].dt.year,
                        Month=cleaned_df["Date"].dt.month
                    )[["Year", "Month", "WTI_Price", "Exchange_Rate"]],
                    use_container_width=True
                )
            else:
                st.error(msg)

        except Exception as e:
            st.error(f"Upload error: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Forecast Button
# =========================
st.markdown("")
run_btn = st.button("Run Forecast", use_container_width=True)

if run_btn:
    if future_df is None:
        st.error("Please provide valid manual input or upload a valid Excel file first.")
    else:
        with st.spinner("Running forecast..."):
            try:
                result_df = run_forecast(future_df)

                # KPI cards
                st.markdown("## Forecast Summary")
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Forecast Months",
                        len(result_df)
                    )
                with c2:
                    st.metric(
                        "Last Forecast Month",
                        pd.to_datetime(result_df["Date"].max()).strftime("%Y-%m")
                    )
                with c3:
                    st.metric(
                        "Final Weighted Hybrid Forecast (MT)",
                        f"{result_df['Weighted_Hybrid_Level'].iloc[-1]:,.2f}"
                    )

                # Results table
                st.markdown("## Forecast Results")
                display_df = result_df.copy()
                display_df["Date"] = pd.to_datetime(display_df["Date"]).dt.strftime("%Y-%m")
                st.dataframe(display_df, use_container_width=True)

                # Plot
                st.markdown("## Historical and Forecast Plot")
                fig = plot_history_and_forecast(result_df)
                st.pyplot(fig)

                # Download Excel
                excel_file = create_excel_download(result_df)
                st.download_button(
                    label="Download Forecast Results (Excel)",
                    data=excel_file,
                    file_name="polymer_forecast_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"Forecast error: {e}")
