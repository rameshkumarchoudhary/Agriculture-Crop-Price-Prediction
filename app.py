from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
from datetime import datetime
import requests
import plotly.express as px
import os
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# Load the model and encoders
model = joblib.load('random_forest_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Initialize Flask app
server = Flask(__name__)

# Function to get weather data
def get_weather(city):
    api_key = os.getenv('WEATHER_API_KEY', '404d2b9df39c5985f0780d3b80566bf8') 
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city
    try:
        response = requests.get(complete_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(e)
        return None

# Landing page route
@server.route('/', methods=['GET', 'POST'])
def landing():
    city = "Pune"  # Default city
    if request.method == 'POST':
        city = request.form['city']
    
    weather = get_weather(city)
    if weather:
        try:
            weather_description = weather['weather'][0]['description']
            temperature = weather['main']['temp'] - 273.15  # Convert from Kelvin to Celsius
        except KeyError:
            weather_description = "Weather data not available"
            temperature = "N/A"
    else:
        weather_description = "Weather data not available"
        temperature = "N/A"
    
    return render_template('landing.html', 
                           city=city,
                           weather_description=weather_description,
                           temperature=temperature)

# Prediction page route
@server.route('/predict')
def home():
    commodity_options = label_encoders['commodity_name'].classes_
    state_options = label_encoders['state'].classes_
    district_options = label_encoders['district'].classes_
    market_options = label_encoders['market'].classes_

    return render_template('index.html', 
                           commodity_options=commodity_options,
                           state_options=state_options,
                           district_options=district_options,
                           market_options=market_options)

# Prediction processing route
@server.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            commodity_name_input = request.form['commodity_name']
            state_input = request.form['state']
            district_input = request.form['district']
            market_input = request.form['market']
            date_input = request.form['date']

            date_parsed = datetime.strptime(date_input, '%Y-%m-%d')
            month_input = date_parsed.strftime('%B')
            day_input = date_parsed.weekday()

            commodity_name_encoded = label_encoders['commodity_name'].transform([commodity_name_input])[0]
            state_encoded = label_encoders['state'].transform([state_input])[0]
            district_encoded = label_encoders['district'].transform([district_input])[0]
            market_encoded = label_encoders['market'].transform([market_input])[0]
            month_encoded = label_encoders['month_column'].transform([month_input])[0]
            season_input = determine_season(month_input)
            season_encoded = label_encoders['season_names'].transform([season_input])[0]

            user_input = [[commodity_name_encoded, state_encoded, district_encoded, market_encoded, month_encoded, season_encoded, day_input]]
            user_input_df = pd.DataFrame(user_input, columns=['commodity_name', 'state', 'district', 'market', 'month_column', 'season_names', 'day'])

            prediction = model.predict(user_input_df)

            return render_template('result.html', prediction=prediction[0])
        except Exception as e:
            print(e)
            return render_template('error.html', message="An error occurred while processing your request.")

def determine_season(month):
    if month in ["January", "February"]:
        return "winter"
    elif month in ["March", "April"]:
        return "spring"
    elif month in ["May", "June"]:
        return "summer"
    elif month in ["July", "August"]:
        return "monsoon"
    elif month in ["September", "October"]:
        return "autumn"
    else:
        return "pre winter"

# Dash app initialization
app = Dash(__name__, server=server, url_base_pathname='/historical_data/', external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'])

# Load the dataset
historical_df = pd.read_csv('historical_data.csv')
historical_df['date'] = pd.to_datetime(historical_df['date'])  # Ensure 'date' is in datetime format
historical_df = historical_df.sort_values(by='date', ascending=True)  # Sort by date in ascending order

# Dash app layout
app.layout = html.Div([
    html.H1("Historical Crop Prices"),
    html.Div([
        html.Label("Select Market:"),
        dcc.Dropdown(
            id='market-dropdown',
            options=[{'label': market, 'value': market} for market in historical_df['market'].unique()],
            value=historical_df['market'].unique()[0]
        ),
        html.Label("Select Commodity:"),
        dcc.Dropdown(
            id='commodity-dropdown',
            options=[{'label': commodity, 'value': commodity} for commodity in historical_df['commodity_name'].unique()],
            value=historical_df['commodity_name'].unique()[0]
        )
    ], style={'width': '50%', 'display': 'inline-block'}),
    dcc.Graph(id='price-graph')
])

# Callback to update graph based on selected commodity and market
@app.callback(
    Output('price-graph', 'figure'),
    [Input('market-dropdown', 'value'), Input('commodity-dropdown', 'value')]
)
def update_graph(selected_market, selected_commodity):
    filtered_df = historical_df[(historical_df['market'] == selected_market) & (historical_df['commodity_name'] == selected_commodity)]
    fig = px.line(filtered_df, x='date', y='modal_price', title=f'Prices for {selected_commodity} in {selected_market}')
    return fig

if __name__ == '__main__':
    server.run(debug=True)
