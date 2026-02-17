import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import xgboost as xgb
import google.generativeai as genai
import os

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="GuardianEye | AI Forensics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_KEY = "AIzaSyB6klLM-_kwRuq2_Iz822ikbw5vnSqT410" 
genai.configure(api_key=API_KEY)

# --- 2. STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    div[data-testid="stMetric"] {
        background-color: #262730; border: 1px solid #464B5C;
        padding: 15px; border-radius: 10px; color: #FAFAFA;
    }
    div[data-testid="stDataFrame"] { border: 1px solid #464B5C; border-radius: 5px; }
    section[data-testid="stSidebar"] { background-color: #262730; }
    h1, h2, h3 { color: #00ADB5 !important; }
    button[data-baseweb="tab"] { font-size: 18px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD MODELS ---
@st.cache_resource
def load_ai_assets():
    try:
        # POINTING TO YOUR SPECIFIC FOLDER
        base_path = ".models/" 
        
        # Load Scaler
        scaler = joblib.load(f'{base_path}safe_scaler.pkl')
        
        # Load LSTM (compile=False fixes the 'mae' error)
        full_lstm = load_model(f'{base_path}advanced_lstm.h5', compile=False)
        encoder_model = Model(inputs=full_lstm.input, outputs=full_lstm.get_layer('bottleneck').output)
        
        # Load XGBoost
        xgb_model = joblib.load(f'{base_path}advanced_xgb.pkl')
        
        return scaler, encoder_model, xgb_model
    except Exception as e:
        st.error(f"❌ Model Loading Failed: {e}\nCheck if files are actually inside the '{base_path}' folder.")
        return None, None, None

scaler, encoder_model, xgb_regressor = load_ai_assets()
# --- 4. UTILS ---
def smart_date_parser(series):
    series_fixed = series.astype(str).str.replace('.', ':', regex=False)
    return pd.to_datetime(series_fixed, errors='coerce', dayfirst=True)

def create_sequences(data, seq_len, feature_col_idx):
    data_val = data[:, feature_col_idx] 
    xs = []
    for i in range(len(data) - seq_len):
        xs.append(data_val[i:(i + seq_len)])
    return np.array(xs)

def get_llm_explanation(row_data, residual, deviation):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except:
        model = genai.GenerativeModel('gemini-2.0-flash')

    # Extract technical parameters with defaults if missing
    volts = row_data.get('VLL', 0)
    amps = row_data.get('ILL', 0)
    
    prompt = f"""
    ACT AS: Senior Revenue Protection Engineer & Forensic Analyst.
    SYSTEM: Hybrid LSTM-XGBoost Anomaly Detection System (GuardianEye).
    
    You are analyzing a flagged 'Non-Technical Loss' (NTL) event.
    
    ---------------------------------------------------------
    [TELEMETRY DATA]
    timestamp: {row_data['FULL_TIME']}
    
    1. ENERGY CONSUMPTION:
       - Actual Recorded:    {row_data['Actual_Energy']:.4f} kW
       - AI Expected Baseline: {row_data['Predicted_Energy']:.4f} kW
       - Net Discrepancy:    {residual:.4f} kW
       - Deviation Severity: {deviation:.0f}%
       
    2. LINE PARAMETERS (Crucial for Diagnosis):
       - Line Voltage (VLL): {volts:.2f} V  (Ref: Normal is ~230V/400V)
       - Line Current (ILL): {amps:.2f} A
    ---------------------------------------------------------

    [FORENSIC ANALYSIS PROTOCOL]
    
    STEP 1: ROOT CAUSE TRIAGE
    - IF (Voltage > 200V) AND (Actual_Load near 0) AND (Predicted > 0.5):
      -> CONCLUSION: "Suspected Meter Bypass / Shunting" (Load exists but not recorded).
    - IF (Voltage < 180V) AND (Actual_Load Low):
      -> CONCLUSION: "Grid Fault / Brownout" (Not Theft).
    - IF (Actual_Load > Predicted):
      -> CONCLUSION: "Unexpected Load Surge" (Equipment malfunction or unauthorized tap).

    STEP 2: GENERATE REPORT
    Provide a professional field report with the following sections:
    
    **1. 🔎 Technical Diagnosis:**
    (State the conclusion clearly using the logic above. Mention if VLL proves power was available.)
    
    **2. 📊 Evidence Interpretation:**
    (Compare the AI's prediction vs reality. E.g., "AI expected 0.8kW based on time-of-day, but meter recorded 0.0kW despite healthy voltage.")
    
    **3. 👷‍♂️ Field Action Plan:**
    (Specific instructions for the crew. E.g., "Inspect seal integrity," "Check CT polarity," "Look for hidden tap before meter.")

    TONE: Strict, Technical, Audit-Ready. No fluff.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Analysis Unavailable. Error: {e}"
def generate_future_features(start_date, periods):
    future_dates = pd.date_range(start=start_date, periods=periods, freq='H')
    df_future = pd.DataFrame({'FULL_TIME': future_dates})
    
    df_future['Hour'] = df_future['FULL_TIME'].dt.hour
    df_future['Hour_Sin'] = np.sin(2 * np.pi * df_future['Hour'] / 24)
    df_future['Hour_Cos'] = np.cos(2 * np.pi * df_future['Hour'] / 24)
    
    explicit_feats = ['VLL', 'ILL', 'PFAVG', 'FREQ', 'Hour_Sin', 'Hour_Cos']
    for feat in explicit_feats:
        if feat not in df_future.columns: df_future[feat] = 0
            
    # Placeholder embeddings for future prediction (assuming consistent behavior)
    for i in range(32): df_future[f'Emb_{i}'] = 0
        
    return df_future, future_dates

# --- 5. MAIN APP ---

# SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2621/2621040.png", width=50)
    st.title("GuardianEye")
    st.caption("v4.0 | Engineering Suite")
    st.divider()
    
    # DATA LOADING
    local_files = ['Test_Data.csv', 'Samples1 - Sheet1.csv']
    local_file_path = next((f for f in local_files if os.path.exists(f)), None)
    
    df = None
    filename = "Unknown"

    if local_file_path:
        st.success(f"Linked: {local_file_path}")
        if st.button("Change Source"):
            local_file_path = None; st.rerun()
        else:
            df = pd.read_csv(local_file_path)
            filename = local_file_path
            
    if df is None:
        uploaded_file = st.file_uploader("Upload Meter Data", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            filename = uploaded_file.name
        else:
            st.info("Waiting for data stream..."); st.stop() 

# PIPELINE
possible_time_cols = ['FULL_TIME', 'Timestamp', 'Date', 'Time']
detected_time_col = next((c for c in df.columns if c in possible_time_cols), df.columns[0])
df = df.rename(columns={detected_time_col: 'FULL_TIME'})
df['FULL_TIME'] = smart_date_parser(df['FULL_TIME'])
df = df.sort_values('FULL_TIME').reset_index(drop=True).dropna(subset=['FULL_TIME'])

# MODEL INFERENCE
if scaler and encoder_model and xgb_regressor:
    cols_to_scale = ['PTOTAL', 'QTOTAL', 'STOTAL', 'ILL', 'VLL', 'PFAVG', 'FREQ', 'Hour_Sin', 'Hour_Cos']
    df['Hour'] = df['FULL_TIME'].dt.hour
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    
    df_proc = df.copy()
    for c in cols_to_scale:
        if c not in df_proc.columns: df_proc[c] = 0
    df_proc[cols_to_scale] = df_proc[cols_to_scale].fillna(0)
    
    scaled_values = scaler.transform(df_proc[cols_to_scale])
    df_proc[cols_to_scale] = scaled_values
    
    SEQ_LEN = 60
    if len(df) < SEQ_LEN: st.error("Data Insufficient."); st.stop()
        
    X_seq = create_sequences(scaled_values, SEQ_LEN, 0)
    X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
    embeddings = encoder_model.predict(X_seq, batch_size=2048, verbose=0)
    df_emb = pd.DataFrame(embeddings, columns=[f'Emb_{i}' for i in range(32)])
    
    df_trim = df_proc.iloc[SEQ_LEN:].reset_index(drop=True)
    explicit_feats = ['VLL', 'ILL', 'PFAVG', 'FREQ', 'Hour_Sin', 'Hour_Cos']
    X_hybrid = pd.concat([df_emb, df_trim[explicit_feats]], axis=1)
    preds = xgb_regressor.predict(X_hybrid)
    
    y_actual = df_trim['PTOTAL'].values
    residuals = np.abs(y_actual - preds)
    
    res_df = pd.DataFrame({
        "FULL_TIME": df['FULL_TIME'].iloc[SEQ_LEN:].reset_index(drop=True),
        "Actual_Energy": y_actual,
        "Predicted_Energy": preds,
        "Residual": residuals
    })
    
    window = 24
    res_df["Threshold"] = res_df["Residual"].rolling(window=window).mean() + (2.0 * res_df["Residual"].rolling(window=window).std())
    res_df["Anomaly"] = res_df["Residual"] > res_df["Threshold"]
    res_df["Deviation_%"] = (res_df["Residual"] / (res_df["Actual_Energy"] + 1e-6)) * 100

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["🔮 Future Prediction", "🚨 Anomaly Detection", "🤖 XAI Forensics"])

    # --- TAB 1: FUTURE CONSUMPTION PREDICTION ---
    with tab1:
        st.header("🔮 Future Consumption Forecasting")
        st.markdown("Predict future energy load (kW) and total consumption (kWh) using XGBoost.")
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("Forecast Parameters")
            forecast_unit = st.radio("Time Horizon Unit:", ["Months", "Years"], horizontal=True)
            forecast_val = st.slider(f"Number of {forecast_unit}", 1, 12, 1)
            
            # Calculate hours
            if forecast_unit == "Months":
                hours_to_pred = forecast_val * 30 * 24
                display_text = f"{forecast_val} Months"
            else:
                hours_to_pred = forecast_val * 365 * 24
                display_text = f"{forecast_val} Years"
                
            if st.button("Generate Forecast 🚀"):
                with st.spinner(f"Projecting load for next {display_text}..."):
                    last_date = df['FULL_TIME'].max()
                    df_future, future_dates = generate_future_features(last_date, hours_to_pred)
                    X_future = df_future[[f'Emb_{i}' for i in range(32)] + explicit_feats]
                    future_preds = xgb_regressor.predict(X_future)
                    
                    # Create Future DataFrame
                    future_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted_Load_kW': future_preds
                    })
                    
                    # Aggregate for cleaner charting (Daily Total kWh)
                    daily_agg = future_df.resample('D', on='Date').sum().reset_index()
                    daily_agg = daily_agg.rename(columns={'Predicted_Load_kW': 'Total_Energy_kWh'})
                    
                    st.session_state['forecast_data'] = future_df
                    st.session_state['daily_agg'] = daily_agg
                    st.success("Prediction Complete")

        with c2:
            if 'forecast_data' in st.session_state:
                f_df = st.session_state['forecast_data']
                d_agg = st.session_state['daily_agg']
                
                # Engineering Metrics
                total_kwh = f_df['Predicted_Load_kW'].sum()
                peak_load = f_df['Predicted_Load_kW'].max()
                avg_load = f_df['Predicted_Load_kW'].mean()
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Forecasted Energy", f"{total_kwh:,.0f} kWh")
                m2.metric("Peak Load Demand", f"{peak_load:.3f} kW")
                m3.metric("Avg Base Load", f"{avg_load:.3f} kW")
                
                # VISUALIZATION SELECTION
                viz_type = st.selectbox("Select Chart View:", ["Daily Consumption (Bar)", "Load Curve Trend (Line)"])
                
                fig_pred = go.Figure()
                
                if viz_type == "Daily Consumption (Bar)":
                    fig_pred.add_trace(go.Bar(
                        x=d_agg['Date'], y=d_agg['Total_Energy_kWh'],
                        name='Daily Energy (kWh)', marker_color='#00ADB5'
                    ))
                    fig_pred.update_layout(yaxis_title="Energy (kWh)")
                else:
                    # Resample to 4H or 1D mean for smoother line chart if too many points
                    line_data = f_df.set_index('Date').resample('D').mean().reset_index()
                    fig_pred.add_trace(go.Scatter(
                        x=line_data['Date'], y=line_data['Predicted_Load_kW'],
                        mode='lines', name='Avg Load (kW)',
                        line=dict(color='#00ADB5', width=2)
                    ))
                    fig_pred.update_layout(yaxis_title="Average Load (kW)")

                fig_pred.update_layout(
                    title=f"Predicted Energy Pattern ({display_text})",
                    paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font=dict(color="white"),
                    xaxis_title="Timeline"
                )
                st.plotly_chart(fig_pred, use_container_width=True)

    # --- TAB 2: ANOMALY DETECTION ---
    with tab2:
        st.header("🚨 Anomaly Detection Dashboard")
        anomalies = res_df[res_df['Anomaly'] == True]
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Anomalies", f"{len(anomalies)}", delta="Alert", delta_color="inverse")
        k2.metric("Max Deviation", f"{res_df['Deviation_%'].max():.0f}%")
        k3.metric("System Status", "CRITICAL" if len(anomalies) > 0 else "NORMAL")
        
        st.divider()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res_df['FULL_TIME'], y=res_df['Predicted_Energy'], mode='lines', name='AI Baseline', line=dict(color='#00ADB5')))
        fig.add_trace(go.Scatter(x=res_df['FULL_TIME'], y=res_df['Actual_Energy'], mode='lines', name='Actual Reading', line=dict(color='#EEEEEE', dash='dot')))
        if not anomalies.empty:
            fig.add_trace(go.Scatter(x=anomalies['FULL_TIME'], y=anomalies['Actual_Energy'], mode='markers', name='Anomaly Detected', marker=dict(color='#FF2E63', size=8, symbol='x')))
        
        fig.update_layout(title="Live Energy Forensics", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font=dict(color="white"), height=400)
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 3: XAI ---
    with tab3:
        st.header("🤖 Forensic Investigator")
        if not anomalies.empty:
            sorted_anom = anomalies.sort_values('Deviation_%', ascending=False)
            sel_idx = st.selectbox("Select Incident:", sorted_anom.index, format_func=lambda x: f"{sorted_anom.loc[x, 'FULL_TIME']} (Dev: {sorted_anom.loc[x, 'Deviation_%']:.0f}%)")
            
            if st.button("Generate Report 🔍"):
                with st.spinner("Analyzing..."):
                    row = sorted_anom.loc[sel_idx]
                    rpt = get_llm_explanation(row, row['Residual'], row['Deviation_%'])
                    st.info(rpt)
        else:
            st.success("No anomalies to investigate.")