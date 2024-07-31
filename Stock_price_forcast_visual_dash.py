# -- coding: utf-8 --
"""
Created on Fri Nov 24 14:10:57 2023

@author: DELL
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

import yfinance as yf
import plotly.express as px
from datetime import datetime as dt
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Stock Information Dashboard", style={'text-align': 'center', 'background-color': '#007BFF', 'color': 'white', 'padding': '20px', 'border-radius': '10px'}),

    # Main container for content
    html.Div([
        # Container for buttons on the left
        html.Div([
            html.Label("Enter Company code:", style={'color': 'black'}),
            dcc.Input(id="input-symbol", type="text", value="", style={'margin-bottom': '20px', 'padding': '10px', 'border-radius': '5px', 'color': 'black'}),
            
            dcc.DatePickerRange(id='date-picker-range', start_date=dt(2023, 1, 1), end_date=dt(2023, 12, 31), style={'margin-bottom': '10px'}),
            
            dcc.Input(id='forecast-days-input', type='number', placeholder='Enter days for forecast', style={'margin-bottom': '10px', 'padding': '10px', 'border-radius': '5px'}),
            
            html.Button('Update Visualization', id='update-button', style={'background-color': '#4CAF50', 'color': 'white', 'padding': '10px', 'border-radius': '5px'}),
        ], style={'width': '30%', 'float': 'left', 'padding': '20px', 'background-color': '#f0f0f0'}),

        # Container for information and graphs on the right
        html.Div([
            # Callback to update company information and stock visualization
            html.Div(id="company-info", style={'margin-bottom': '20px', 'text-align': 'center'}),
            dcc.Graph(id="stock-price-plot", style={'margin-bottom': '20px'}),
            dcc.Graph(id="stock-prediction-plot", style={'margin-bottom': '20px'}),
        ], style={'width': '65%', 'float': 'left', 'padding': '20px', 'text-align': 'center'}),
    ], style={'display': 'flex'}),
])

# Callback to update company information and stock visualization
@app.callback(
    [Output("company-info", "children"),
     Output("stock-price-plot", "figure")],
    [Input("update-button", "n_clicks")],
    [Input("input-symbol", "value"),
     Input("forecast-days-input", "value"),
     State("date-picker-range", "start_date"),
     State("date-picker-range", "end_date")]
)
def update_visualization(n_clicks, input_symbol, forecast_days, start_date, end_date):
    if not input_symbol or n_clicks is None:
        return [], {}

    try:
        # Get historical stock data
        stock_data = yf.download(input_symbol, start=start_date, end=end_date)

        # Create a figure for the stock price plot
        fig_stock_price = px.line(stock_data, x=stock_data.index, y='Close', title=f"{input_symbol} Stock Price Over Time")

        # Add indicators to the figure
        stock_data['SMA'] = stock_data['Close'].rolling(window=20).mean()
        fig_stock_price.add_scatter(x=stock_data.index, y=stock_data['SMA'], mode='lines', name='SMA')

        # Get company information
        company = yf.Ticker(input_symbol)
        info = company.info

        # Style for the company name
        company_name_style = {'font-weight': 'bold', 'font-size': '24px', 'color': 'black'}  # Bold and black

        # Display the company name in large font, bold, and black
        company_name_display = html.H2(f"{info.get('shortName', '')}", style=company_name_style)

        # Display the rest of the company information in a paragraph, all in bold
        company_info = [
            company_name_display,
            html.P(f"Industry: {info.get('industry', '')}", style={'font-weight': 'bold'}),
            html.P(f"Sector: {info.get('sector', '')}", style={'font-weight': 'bold'}),
            html.P(f"Market Cap: {info.get('marketCap', '')}", style={'font-weight': 'bold'}),
           
            html.P(f"Forward P/E: {info.get('forwardPE', '')}", style={'font-weight': 'bold'}),
            html.P(f"Beta: {info.get('beta', '')}", style={'font-weight': 'bold'}),
            html.P(f"Previous Close: {info.get('regularMarketPreviousClose', '')}", style={'font-weight': 'bold'}),
           
            html.P(f"Recommendation: {info.get('recommendationKey', '')}", style={'font-weight': 'bold'})
        ]

        return company_info, fig_stock_price

    except Exception as e:
        print(f"Error details: {e}")
        return [], {}

# Callback to update stock prediction plot
@app.callback(
    Output("stock-prediction-plot", "figure"),
    [Input("update-button", "n_clicks")],
    [Input("input-symbol", "value"),
     Input("forecast-days-input", "value"),
     State("date-picker-range", "start_date"),
     State("date-picker-range", "end_date")]
)
def update_stock_prediction_plot(n_clicks, input_symbol, forecast_days, start_date, end_date):
    if not input_symbol or n_clicks is None or not forecast_days:
        return {}

    try:
        stock_data = yf.download(input_symbol, start=start_date, end=end_date)

        # Extract closing prices and prepare data
        closing_prices = stock_data['Close']
        X = np.arange(1, len(closing_prices) + 1).reshape(-1, 1)
        y = closing_prices

        # Normalize data
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=2)

        # Define the parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'kernel': ['rbf']
        }

        # Create SVR model
        svr = SVR()

        # Create GridSearchCV object
        grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error')

        # Fit the grid search to the data
        grid_search.fit(X_train, y_train)

        # Use the best model for prediction
        best_svr = grid_search.best_estimator_

        # Concatenate training data and future data for prediction
        X_combined = np.concatenate((X_scaled, scaler_X.transform(np.arange(len(X_scaled) + 1, len(X_scaled) + forecast_days + 1).reshape(-1, 1))))

        # Predict prices for combined data
        y_pred_combined = best_svr.predict(X_combined)

        # Inverse transform the predicted values to the original scale
        y_pred_original_scale = scaler_y.inverse_transform(y_pred_combined.reshape(-1, 1)).flatten()

        # Create a figure for the stock price prediction plot
        fig_stock_prediction = go.Figure()

        # Plot actual prices
        fig_stock_prediction.add_trace(go.Scatter(x=X.flatten(), y=scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten(), mode='lines', name='Actual Prices', line=dict(color='blue')))

        # Plot predicted prices with tuned model
        fig_stock_prediction.add_trace(go.Scatter(x=np.arange(1, len(X_combined) + 1), y=y_pred_original_scale, mode='lines', name='Predicted Prices', line=dict(color='green')))

        fig_stock_prediction.update_layout(
            title=f'Stock Price Prediction for {input_symbol} ({forecast_days} days)',
            xaxis=dict(title='Days'),
            yaxis=dict(title='Closing Prices'),
            legend=dict(x=0, y=1, traceorder='normal'),
            showlegend=True
        )

        return fig_stock_prediction

    except Exception as e:
        print(f"Error details: {e}")
        return {}


if __name__ == '_main_':
    app.run_server(debug=True)