import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def load_data(timeframes):
    dataframes = {}
    for timeframe in timeframes:
        file_path = f"XAUUSD_MT5_{timeframe}.csv"
        df = pd.read_csv(file_path, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        dataframes[timeframe] = df
    return dataframes

def calculate_returns(dataframes):
    for timeframe, df in dataframes.items():
        df['return'] = df['close'].pct_change()
        df.dropna(inplace=True)
    return dataframes

def detect_outliers(dataframes, threshold=3):
    mu_sigma = {}
    for timeframe, df in dataframes.items():
        mu = df['return'].mean()
        sigma = df['return'].std()
        df['z_score'] = (df['return'] - mu) / sigma
        df['outlier'] = np.abs(df['z_score']) > threshold
        mu_sigma[timeframe] = (mu, sigma)
    return dataframes, mu_sigma

def plot_outliers(dataframes, mu_sigma, threshold=3):
    for timeframe, df in dataframes.items():
        mu, sigma = mu_sigma[timeframe]
        fig = px.scatter(df, x=df.index, y='return',
                         color=df['outlier'].map({True: 'red', False: 'blue'}),
                         title=f'XAUUSD Hourly Return with Outliers ({timeframe})',
                         labels={'return': 'Hourly Return'},
                         hover_data=[df.index])
        
        fig.add_hline(y=mu, line_dash="dash", line_color="#00CC96", annotation_text=f"Mean ({timeframe})")
        fig.add_hline(y=mu + threshold * sigma, line_dash="dash", line_color="#FF5733", annotation_text=f"Upper Bound ({timeframe})")
        fig.add_hline(y=mu - threshold * sigma, line_dash="dash", line_color="#FF5733", annotation_text=f"Lower Bound ({timeframe})")
        
        latest_date = df.index[-1]
        latest_return = df['return'].iloc[-1]
        fig.add_trace(go.Scatter(x=[latest_date], y=[latest_return], mode='markers+text', 
                                 marker=dict(color='yellow', size=12, line=dict(color='black', width=2)),
                                 text=[latest_date.strftime('%Y-%m-%d %H:%M')], textposition='top center',
                                 name='Latest Data'))
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(layout="wide")
    st.title("📊 XAUUSD Return Outlier Detection")
    
    timeframes = ["1H", "1H_ALL"]
    dataframes = load_data(timeframes)
    dataframes = calculate_returns(dataframes)
    dataframes, mu_sigma = detect_outliers(dataframes)
    
    st.write("### 📌 Outliers Found:")
    for timeframe, df in dataframes.items():
        st.write(f"#### {timeframe}")
        st.dataframe(df[df['outlier']], height=300)
    
    plot_outliers(dataframes, mu_sigma)
    
    for timeframe, df in dataframes.items():
        latest_date = df.index[-1]
        latest_return = df['return'].iloc[-1]
        st.success(f"### ✅ Latest Data Point ({timeframe}): {latest_date}, Return: {latest_return:.6f}")

if __name__ == "__main__":
    main()
