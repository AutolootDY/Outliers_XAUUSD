import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def load_data(timeframe):
    file_path = f"XAUUSD_MT5_{timeframe}.csv"
    data = pd.read_csv(file_path, parse_dates=["Date"])
    data.set_index("Date", inplace=True)
    return data

def calculate_returns(data):
    data['return'] = data['close'].pct_change()
    data.dropna(inplace=True)
    return data

def detect_outliers(data, threshold=3):
    mu = data['return'].mean()
    sigma = data['return'].std()
    data['z_score'] = (data['return'] - mu) / sigma
    data['outlier'] = np.abs(data['z_score']) > threshold
    return data, mu, sigma

def plot_outliers(data, mu, sigma, threshold=3):
    fig = px.scatter(data, x=data.index, y='return', 
                     color=data['outlier'].map({True: 'red', False: 'blue'}),
                     title='XAUUSD Hourly Return with Outliers', 
                     labels={'return': 'Hourly Return'},
                     hover_data=[data.index])
    
    fig.add_hline(y=mu, line_dash="dash", line_color="#00CC96", annotation_text="Mean")
    fig.add_hline(y=mu + threshold * sigma, line_dash="dash", line_color="#FF5733", annotation_text="Upper Bound")
    fig.add_hline(y=mu - threshold * sigma, line_dash="dash", line_color="#FF5733", annotation_text="Lower Bound")
    
    latest_date = data.index[-1]
    latest_return = data['return'].iloc[-1]
    fig.add_trace(go.Scatter(x=[latest_date], y=[latest_return], mode='markers+text', 
                             marker=dict(color='yellow', size=12, line=dict(color='black', width=2)),
                             text=[latest_date.strftime('%Y-%m-%d %H:%M')], textposition='top center',
                             name='Latest Data'))
    
    st.plotly_chart(fig, use_container_width=True)

def plot_distribution(data, mu, sigma, threshold=3, nbins=100):
    hist_fig = px.histogram(data, x='return', nbins=nbins, marginal="box", 
                            opacity=0.7, color_discrete_sequence=['#0072B2'],
                            title='XAUUSD Hourly Return Normal Distribution', 
                            labels={'return': 'Hourly Return'},
                            histnorm='probability density', hover_data=[data.index])
    
    hist_fig.add_vline(x=mu, line_dash="dash", line_color="#00CC96", annotation_text="Mean")
    hist_fig.add_vline(x=mu + threshold * sigma, line_dash="dash", line_color="#FF5733", annotation_text="Upper Bound")
    hist_fig.add_vline(x=mu - threshold * sigma, line_dash="dash", line_color="#FF5733", annotation_text="Lower Bound")
    
    latest_date = data.index[-1]
    latest_return = data['return'].iloc[-1]
    hist_fig.add_trace(go.Scatter(x=[latest_return], y=[0], mode='markers+text', 
                                  marker=dict(color='black', size=10),
                                  text=[latest_date.strftime('%Y-%m-%d %H:%M')], textposition='top center',
                                  name='Latest Data', hovertext=[latest_date.strftime('%Y-%m-%d %H:%M')]))
    
    st.plotly_chart(hist_fig, use_container_width=True)

def main():
    st.set_page_config(layout="wide")
    st.title("📊 XAUUSD Return Outlier Detection")
    
    timeframe = st.sidebar.radio("Select Timeframe", ["1H", "1H_ALL"], index=0)
    nbins = st.sidebar.slider("Select Number of Bins for Histogram", min_value=10, max_value=1000, value=100, step=10)
    
    data = load_data(timeframe)
    data = calculate_returns(data)
    data, mu, sigma = detect_outliers(data)
    
    st.write("### 📌 Outliers Found:")
    st.dataframe(data[data['outlier']], height=300)
    
    plot_outliers(data, mu, sigma)
    plot_distribution(data, mu, sigma, nbins=nbins)
    
    latest_date = data.index[-1]
    latest_return = data['return'].iloc[-1]
    st.success(f"### ✅ Latest Data Point: {latest_date}, Return: {latest_return:.6f}")

if __name__ == "__main__":
    main()