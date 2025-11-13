import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
import joblib
import matplotlib.pyplot as plt


# Constants

SEQ_LEN = 12


# Load models and scaler

@st.cache_resource
def load_models_and_scaler():
    lstm_1m_model = load_model("models/lstm_1month.h5", custom_objects={"mse": MeanSquaredError})
    lstm_6m_model = load_model("models/lstm_6month.h5", custom_objects={"mse": MeanSquaredError})
    feat_scaler = joblib.load("models/scaler.pkl")
    return lstm_1m_model, lstm_6m_model, feat_scaler

lstm_1m_model, lstm_6m_model, feat_scaler = load_models_and_scaler()


# Country & Commodity Encoders

country_list = [
    'Austria', 'Belgium', 'Canada', 'Chile', 'China', 'Colombia', 'Czech Republic', 'Denmark',
    'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland',
    'Israel', 'Italy', 'Japan', 'Korea, Rep.', 'Latvia', 'Lithuania', 'Luxembourg', 'Mexico',
    'Netherlands', 'Norway', 'Poland', 'Portugal', 'Saudi Arabia', 'Slovakia', 'Slovenia',
    'South Africa', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'United Kingdom', 'United States',
    'Algeria', 'Kenya', 'India'
]

commodity_list = ['Meat', 'Dairy', 'Cereals', 'Oils', 'Sugar']

country_decoder = {i: c for i, c in enumerate(country_list)}
commodity_decoder = {i: c for i, c in enumerate(commodity_list)}
country_encoder = {v: k for k, v in country_decoder.items()}
commodity_encoder = {v: k for k, v in commodity_decoder.items()}

# Base Prices (2014–2016 averages)

base_prices_usd_per_ton = {
    "Meat": 3000,
    "Dairy": 3500,
    "Cereals": 250,
    "Oils": 1000,
    "Sugar": 400
}


# Load Default Data

@st.cache_data
def load_default_data():
    df = pd.read_csv("merged.csv", parse_dates=['Date'])
    if np.issubdtype(df['Country'].dtype, np.number):
        df['Country_name'] = df['Country'].map(country_decoder)
    else:
        df['Country_name'] = df['Country']
        df['Country'] = df['Country'].map(country_encoder)

    if np.issubdtype(df['Commodity'].dtype, np.number):
        df['Commodity_name'] = df['Commodity'].map(commodity_decoder)
    else:
        df['Commodity_name'] = df['Commodity']
        df['Commodity'] = df['Commodity'].map(commodity_encoder)
    return df

df = load_default_data()


# Sidebar Controls

st.sidebar.header("Forecasting Controls")

unique_countries = sorted(df['Country_name'].dropna().unique())
unique_commodities = sorted(df['Commodity_name'].dropna().unique())

# Default "Select" options
selected_country = st.sidebar.selectbox("Select Country", ["Select Country"] + list(unique_countries))
selected_commodity = st.sidebar.selectbox("Select Commodity", ["Select Commodity"] + list(unique_commodities))
selected_month = st.sidebar.selectbox("Months ahead to predict", [1, 6])
predict_button = st.sidebar.button("Predict")


# Main Section Layout

st.markdown(
    """
    <h1 style='text-align: center;'>Global Food Price Forecast</h1>
    <p style='text-align: center; font-size:16px;'>
        Predict future Food Price Index (FPI) values for selected countries and commodities based on past trends.
    </p>
    """,
    unsafe_allow_html=True
)


# Validation before prediction

if selected_country == "Select Country" or selected_commodity == "Select Commodity":
    st.info("Please select a Country and a Commodity from the sidebar to begin.")
else:
    if predict_button:
        st.markdown(
            f"<h4 style='text-align: center;'>{selected_month}-Month Ahead Forecast for {selected_country} - {selected_commodity}</h4>",
            unsafe_allow_html=True
        )

        
        # Forecast Logic
   
        encoded_country = country_encoder.get(selected_country)
        encoded_commodity = commodity_encoder.get(selected_commodity)

        df_filtered = df[
            (df['Country'] == encoded_country) &
            (df['Commodity'] == encoded_commodity)
        ].copy()

        if len(df_filtered) < SEQ_LEN:
            st.warning("Not enough data for this selection to predict.")
        else:
            feature_cols = [c for c in df_filtered.columns if c not in ['Date', 'FPI', 'Country_name', 'Commodity_name']]
            last_seq = df_filtered[feature_cols].tail(SEQ_LEN).values

            if last_seq.shape[1] != feat_scaler.n_features_in_:
                st.error(f"Feature mismatch: model expects {feat_scaler.n_features_in_}, got {last_seq.shape[1]}")
            else:
                last_seq_scaled = feat_scaler.transform(last_seq).reshape(1, SEQ_LEN, -1)

                # Predict
                if selected_month == 1:
                    y_pred_scaled = lstm_1m_model.predict(last_seq_scaled)
                    y_pred_real = y_pred_scaled.flatten()[-1:]
                else:
                    preds = []
                    current_seq = last_seq_scaled.copy()
                    for _ in range(6):
                        y_pred_scaled = lstm_6m_model.predict(current_seq)
                        preds.append(y_pred_scaled.flatten()[-1])
                        next_seq = np.roll(current_seq, -1, axis=1)
                        next_seq[0, -1, 0] = preds[-1]
                        current_seq = next_seq
                    y_pred_real = np.array(preds)

                # Reverse approximate scaling
                fpi_min, fpi_max = df_filtered['FPI'].min(), df_filtered['FPI'].max()
                y_pred_real = y_pred_real * (fpi_max - fpi_min) + fpi_min

                # Prepare output
                prediction_df = pd.DataFrame({
                    "Month": [df_filtered['Date'].max() + pd.DateOffset(months=i+1) for i in range(len(y_pred_real))],
                    "Predicted FPI (Index Value)": y_pred_real
                })

                # Convert to estimated price
                base_price = base_prices_usd_per_ton.get(selected_commodity, None)
                if base_price:
                    prediction_df["Estimated Price (USD/ton)"] = base_price * (prediction_df["Predicted FPI (Index Value)"] / 100)
                else:
                    prediction_df["Estimated Price (USD/ton)"] = np.nan

                # Show results
                st.dataframe(prediction_df.style.format({
                    "Predicted FPI (Index Value)": "{:.2f}",
                    "Estimated Price (USD/ton)": "{:,.2f}"
                }))

                # Download option
                csv = prediction_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name=f"FPI_Prediction_{selected_country}_{selected_commodity}.csv",
                    mime="text/csv",
                )

                # Plot
                st.markdown("### FPI Trend Overview")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(df_filtered['Date'], df_filtered['FPI'], color='blue', label='Historical FPI', linewidth=2)
                ax.plot(prediction_df['Month'], prediction_df['Predicted FPI (Index Value)'],
                        color='orange', linestyle='--', marker='o', label='Predicted FPI', linewidth=2)
                ax.set_title(f"FPI Forecast for {selected_country} - {selected_commodity}", fontsize=14)
                ax.set_xlabel("Date")
                ax.set_ylabel("FPI (Index Value)")
                ax.legend()
                st.pyplot(fig)







































































