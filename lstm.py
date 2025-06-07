import streamlit as st
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="LSTM Time Series Forecasting - LA Air Quality",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #000;
    }
    .stMarkdown {
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

DATA_URLS = ["data/LA_pm10_2020.csv", "data/LA_pm10_2021.csv", "data/LA_pm10_2022.csv"]
DATE = "Date"
DATA_COL = "Daily Mean PM10 Concentration"

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def load_data(url):
    df = pd.read_csv(url)
    df[DATE] = pd.to_datetime(df[DATE]).dt.date
    return df

def merge_data():
    merged = pd.DataFrame()
    for url in DATA_URLS:
        dat = load_data(url)
        merged = pd.concat([merged, dat])
    merged[DATA_COL] = (merged[DATA_COL] - merged[DATA_COL].mean()) / merged[DATA_COL].std()
    return merged

def split_data(sequence, nsteps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + nsteps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def input_seq():
    merged = merge_data()
    daily_pm10 = merged[DATA_COL].values.tolist()
    daily_pm10 = daily_pm10[0:len(daily_pm10)-1]
    x_train, y_train = split_data(daily_pm10, 10)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    return torch.FloatTensor(x_train), torch.FloatTensor(y_train)

def create_lstm(hidden_size, num_layers, dropout):
    model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    return model

def build_lstm(hidden_size, num_layers, dropout):
    xtrain, ytrain = input_seq()
    model = create_lstm(hidden_size, num_layers, dropout)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    time1 = time.perf_counter()
    st.text("model training...")
    
    model.train()
    for epoch in range(25):
        optimizer.zero_grad()
        outputs = model(xtrain)
        loss = criterion(outputs.squeeze(), ytrain)
        loss.backward()
        optimizer.step()
    
    time2 = time.perf_counter()
    st.text("model running time: " + str(time2-time1))
    
    torch.save(model.state_dict(), 'models/lstm_model_11.pt')
    return model

def load():
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, dropout=0.6)
    model.load_state_dict(torch.load('models/lstm_model_10.pt', weights_only=False))
    model.eval()
    return model

def make_prediction(start_date, stop_date):
    merged = merge_data()
    merged[DATE] = pd.to_datetime(merged[DATE])
    
    # Convert input dates to datetime for comparison
    start_date = pd.to_datetime(start_date)
    stop_date = pd.to_datetime(stop_date)
    
    # Validate dates
    if start_date < merged[DATE].min() or start_date > merged[DATE].max():
        st.error(f"Start date must be between {merged[DATE].min().date()} and {merged[DATE].max().date()}")
        return None
    
    if stop_date < merged[DATE].min() or stop_date > merged[DATE].max():
        st.error(f"End date must be between {merged[DATE].min().date()} and {merged[DATE].max().date()}")
        return None
    
    if start_date >= stop_date:
        st.error("Start date must be before end date")
        return None
    
    # Convert dates to indices
    start_idx = merged[merged[DATE] == start_date].index[0]
    stop_idx = merged[merged[DATE] == stop_date].index[0]
    
    # Ensure we have enough data points for prediction
    if stop_idx - start_idx < 10:
        st.error("Please select a date range with at least 10 days")
        return None
    
    model = load()
    #predict
    data = merged[DATA_COL][start_idx:stop_idx].values
    # Create sequences of 10 days for prediction
    sequences = []
    for i in range(len(data) - 9):
        sequences.append(data[i:i+10])
    data = np.array(sequences)
    data = torch.FloatTensor(data)
    with torch.no_grad():
        yhat = model(data)
    
    #normalize
    merged['daily_pm10_normalized'] = (merged[DATA_COL] - merged[DATA_COL].mean()) / merged[DATA_COL].std()
    yhat = yhat.numpy().flatten()
    yhat = (yhat - yhat.mean()) / yhat.std()
    actual = (merged['daily_pm10_normalized'][start_idx+9:stop_idx+1]).to_list()
    
    # Create DataFrame with dates
    data = pd.DataFrame({
        'Predicted': yhat,
        'Actual': actual,
        'Difference': np.abs(np.array(yhat) - np.array(actual)),
        'Date': merged[DATE][start_idx+9:stop_idx+1]
    })
    return data

def plot_predictions(data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=data['Date'], y=data['Actual'], name="Actual Values",
                  line=dict(color='#1E88E5')),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=data['Date'], y=data['Predicted'], name="Predicted Values",
                  line=dict(color='#FFA726')),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=data['Date'], y=data['Difference'], name="Difference",
                  line=dict(color='#EF5350')),
        secondary_y=True,
    )
    
    fig.update_layout(
        title="PM10 Concentration Predictions",
        xaxis_title="Date",
        hovermode="x unified",
        template="plotly_white",
        height=600
    )
    
    fig.update_yaxes(title_text="Normalized PM10 Values", secondary_y=False)
    fig.update_yaxes(title_text="Absolute Difference", secondary_y=False)
    
    return fig

def site_points():
    merged = merge_data()
    coords = merged[['Site Name', 'SITE_LATITUDE', 'SITE_LONGITUDE']].rename(columns={'SITE_LATITUDE': 'LAT', 'SITE_LONGITUDE': 'LON'})
    st.dataframe(coords, use_container_width=True)
    return st.map(coords[['LAT', 'LON']])

# Sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
    This application demonstrates an LSTM-based time series forecasting model for predicting PM10 air quality in Los Angeles.
    
    **Features:**
    - Interactive predictions
    - Real-time visualization
    - Model performance metrics
    - Station location mapping
    """)
    
    st.markdown("---")
    st.markdown("### Connect with Me")
    st.markdown("""
    - [Website](https://sameehaafr.github.io/sameehaafr/)
    - [GitHub](https://github.com/sameehaafr)
    - [LinkedIn](https://www.linkedin.com/in/sameeha-afrulbasha/)
    - [Medium](https://sameehaafr.medium.com/)
    """)

# Main content
st.title("LSTM Time Series Forecasting for LA Air Quality")
st.markdown("""
This application uses a Long Short-Term Memory (LSTM) neural network to forecast PM10 air quality levels in Los Angeles, California.
The model is trained on historical data from 2020-2022 and can make predictions for future dates.
""")

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🤖 Model Details", "🔮 Make Predictions", "📈 Performance Metrics"])

with tab1:
    st.header("Data Overview")
    st.markdown("""
    The data used in this project comes from the EPA's air quality monitoring stations in Los Angeles.
    We focus on PM10 (Particulate Matter with a diameter of 10 micrometers or less) as our primary metric.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Full Dataset")
        merged_full = merge_data()
        st.dataframe(merged_full, use_container_width=True)
    
    with col2:
        st.subheader("Normalized PM10 Values")
        merged_sub = merged_full[[DATE, DATA_COL]]
        st.dataframe(merged_sub, use_container_width=True)
    
    st.subheader("Data Distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=merged_full[DATA_COL], name="PM10 Distribution"))
    fig.update_layout(title="Distribution of PM10 Values", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📍 Air Quality Monitoring Stations")
    st.markdown("""
    Below is a map showing the locations of all air quality monitoring stations in Los Angeles
    that contribute data to our model.
    """)
    site_points()

with tab2:
    st.header("LSTM Model Architecture")
    st.markdown("""
    Our LSTM model is designed to capture complex patterns in air quality data:
    - Single LSTM layer with 50 units
    - ELU activation function
    - Dropout rate of 0.6
    - L2 regularization (0.02)
    - 25 training epochs
    """)
    
    model = load()
    st.markdown("### Model Architecture")
    st.markdown("""
    ```python
    LSTMModel(
        (lstm): LSTM(1, 50, batch_first=True, dropout=0.6)
        (fc): Linear(in_features=50, out_features=1, bias=True)
    )
    ```
    """)
    
    st.markdown("### Model Parameters")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    st.markdown(f"""
    - Total Parameters: {total_params:,}
    - Trainable Parameters: {trainable_params:,}
    - Non-trainable Parameters: {total_params - trainable_params:,}
    """)

with tab3:
    st.header("Make Predictions")
    st.markdown("""
    Select a date range to generate predictions. The model will use the previous 10 days of data
    to predict the PM10 values for your selected period.
    """)
    
    merged = merge_data()
    merged[DATE] = pd.to_datetime(merged[DATE])
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            min_value=merged[DATE].min().date(),
            max_value=merged[DATE].max().date(),
            value=merged[DATE].min().date()
        )
    
    with col2:
        stop_date = st.date_input(
            "End Date",
            min_value=start_date,
            max_value=merged[DATE].max().date(),
            value=start_date + pd.Timedelta(days=7)
        )
    
    if st.button("Generate Predictions"):
        with st.spinner("Generating predictions..."):
            predictions = make_prediction(start_date, stop_date)
            
            if predictions is not None:
                st.subheader("Prediction Results")
                st.dataframe(predictions, use_container_width=True)
                
                st.subheader("Visualization")
                fig = plot_predictions(predictions)
                st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Model Performance")
    st.markdown("""
    The model's performance is evaluated using standard regression metrics:
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    """)
    
    if 'predictions' in locals() and predictions is not None:
        mse = mean_squared_error(predictions['Actual'], predictions['Predicted'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(predictions['Actual'], predictions['Predicted'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MSE", f"{mse:.4f}")
        with col2:
            st.metric("RMSE", f"{rmse:.4f}")
        with col3:
            st.metric("MAE", f"{mae:.4f}")
    else:
        st.info("Please generate predictions in the 'Make Predictions' tab to see performance metrics.")