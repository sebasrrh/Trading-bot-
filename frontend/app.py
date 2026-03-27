import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time

# Configure page
st.set_page_config(page_title="AI Trading Dashboard", page_icon="📈", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .metric-card { background-color: #1e1e24; border-radius: 10px; padding: 20px; text-align: center; border: 1px solid #333; height: 100%;}
    .metric-value { font-size: 2.5rem; font-weight: bold; color: #fff; }
    .metric-label { font-size: 1rem; color: #aaa; text-transform: uppercase; }
    .explanation-box { background-color: #1e1e24; border-left: 5px solid #2196f3; padding: 20px; margin-top: 20px; font-size: 1.2rem; border-radius: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .signal-BUY { color: #00e676; font-weight: bold; }
    .signal-SELL { color: #ff5252; font-weight: bold; }
    .signal-HOLD { color: #ffea00; font-weight: bold; }
    .stock-title { font-size: 3rem; font-weight: bold; color: #2196f3; margin-bottom: 0; }
</style>
""", unsafe_allow_html=True)

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

@st.cache_data(ttl=20)
def fetch_analysis(ticker=None):
    try:
        if ticker:
            url = f"{BACKEND_URL}/api/analyze?ticker={ticker}"
        else:
            url = f"{BACKEND_URL}/api/state"
            
        response = requests.get(url, timeout=25)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            st.error(f"Stock ticker '{ticker}' not found or no data available.")
        else:
            st.error("Error connecting to backend API. HTTP Status " + str(response.status_code))
    except requests.exceptions.Timeout:
        st.error("Backend took too long to analyze this stock. Try again.")
    except Exception as e:
        st.error(f"Failed to reach backend: {e}")
    return None

def create_advanced_chart(df, ticker):
    if df.empty: return go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    time_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    if time_col not in df.columns: df[time_col] = df.index
        
    fig.add_trace(go.Candlestick(
        x=df[time_col], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
        name='Price', increasing_line_color='#00e676', decreasing_line_color='#ff5252'
    ), row=1, col=1)
    
    if 'SMA_20' in df.columns: fig.add_trace(go.Scatter(x=df[time_col], y=df['SMA_20'], name='SMA 20', line=dict(color='#2196f3', width=1.5)), row=1, col=1)
    if 'SMA_50' in df.columns: fig.add_trace(go.Scatter(x=df[time_col], y=df['SMA_50'], name='SMA 50', line=dict(color='#ff9800', width=1.5)), row=1, col=1)
    
    if 'MACD_12_26_9' in df.columns:
        fig.add_trace(go.Scatter(x=df[time_col], y=df['MACD_12_26_9'], name='MACD', line=dict(color='#2196f3')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df[time_col], y=df['MACDs_12_26_9'], name='Signal', line=dict(color='#ff9800')), row=2, col=1)
        colors = ['#00e676' if val and val >= 0 else '#ff5252' for val in df['MACDh_12_26_9']]
        fig.add_trace(go.Bar(x=df[time_col], y=df['MACDh_12_26_9'], name='Histogram', marker_color=colors), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=600, margin=dict(l=40, r=40, t=40, b=40), xaxis_rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333')
    return fig

def main():
    col_t, col_s = st.columns([2, 1])
    with col_t:
        st.title("📈 AI Quantitative Trading Dashboard")
    with col_s:
        with st.form(key='search_form'):
            st.markdown("<b>Search & Analyze Custom Ticker:</b>", unsafe_allow_html=True)
            query = st.text_input("Ticker Symbol:", placeholder="e.g. MSTR, TSLA, NVDA")
            submit = st.form_submit_button("Run Full ML Pipeline")
            
    if submit and query:
        with st.spinner(f"Fetching live data, building indicators, and retraining XG-Boost model for {query.upper()}..."):
            state = fetch_analysis(query.upper())
            if state:
                st.session_state['state'] = state
    else:
        if 'state' not in st.session_state:
            state = fetch_analysis()
            if state: st.session_state['state'] = state

    state = st.session_state.get('state')
    
    if not state or not state.get("latest_data"):
        st.warning("⏳ Waiting for data... Ensure FastAPI backend is running.")
        time.sleep(5)
        st.rerun()
        return

    # Extract
    df = pd.DataFrame(state["latest_data"])
    signal = state.get("signal", "HOLD")
    explanation = state.get("explanation", "Model analysis complete.")
    ticker = state.get("ticker", "AAPL")
    
    st.markdown(f'<div class="stock-title">{ticker} <span style="font-size:1.5rem; color:#aaa; font-weight:normal;">Live Analysis</span></div>', unsafe_allow_html=True)
    
    # ------------------ EXPLANATION BOX ----------------
    st.markdown(f'''
    <div class="explanation-box">
        <span style="font-size: 1.5rem; display:block; margin-bottom: 5px;">🤖 AI Reasoning & Explanation</span>
        {explanation}
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------ METRICS ----------------
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">NEXT INTERVAL</div><div class="metric-value signal-{signal}">{signal}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">MODEL CONFIDENCE</div><div class="metric-value">{state.get("confidence", 0)*100:.1f}%</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">NEWS SENTIMENT</div><div class="metric-value">{state.get("sentiment", 0):.2f}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-label">WALK-FORWARD ACCURACY</div><div class="metric-value">{state.get("model_accuracy", 0)*100:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------ CHARTS ----------------
    col_chart, col_info = st.columns([3, 1])
    with col_chart:
        fig = create_advanced_chart(df, ticker)
        st.plotly_chart(fig, use_container_width=True)
        
    with col_info:
        st.subheader("📑 Fundamentals")
        f = state.get("fundamentals", {})
        if f:
            st.write(f"**P/E Ratio:** {f.get('pe_ratio', 'N/A')}")
            st.write(f"**P/B Ratio:** {f.get('pb_ratio', 'N/A')}")
            st.write(f"**Yield:** {f.get('dividend_yield', 0)*100:.2f}%" if f.get('dividend_yield') else "**Yield:** N/A")
            st.write(f"**52W High:** ${f.get('52_week_high', 'N/A')}")
            st.write(f"**52W Low:** ${f.get('52_week_low', 'N/A')}")
            
        st.markdown("---")
        st.subheader("⚙️ Current Technicals")
        last = df.iloc[-1]
        st.write(f"**Close:** ${last.get('Close', 0):.2f}")
        if 'SMA_20' in last: st.write(f"**SMA 20:** ${last['SMA_20']:.2f}")
        if 'SMA_50' in last: st.write(f"**SMA 50:** ${last['SMA_50']:.2f}")
        if 'RSI_14' in last: st.write(f"**RSI (14):** {last['RSI_14']:.2f}")

    # No auto rerun so the user can read the explanation.
    # User can refresh via Streamlit UI or the input box.

if __name__ == "__main__":
    main()
