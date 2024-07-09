import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv(r'C:\Users\Ramesh Choudhary\OneDrive\Desktop\Website\Website\New Dataset.csv')
data2 = df.copy()
data2 = data2.head(83265)

# Convert prices from per quintal to per kilogram
data2['min_price'] = data2['min_price'] / 100
data2['max_price'] = data2['max_price'] / 100
data2['modal_price'] = data2['modal_price'] / 100

# Extract month and season
Dict = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
month_column = [Dict[int(date.split('-')[1])] for date in data2["date"]]
data2["month_column"] = month_column

# Map months to seasons
season_names = []
for month in data2["month_column"]:
    if month in ["January", "February"]:
        season_names.append("winter")
    elif month in ["March", "April"]:
        season_names.append("spring")
    elif month in ["May", "June"]:
        season_names.append("summer")
    elif month in ["July", "August"]:
        season_names.append("monsoon")
    elif month in ["September", "October"]:
        season_names.append("autumn")
    else:
        season_names.append("pre winter")
data2["season_names"] = season_names

# Add day of the week
day_of_week = [pd.Timestamp(date).dayofweek for date in data2["date"]]
data2['day'] = day_of_week

# Drop original date column
data2 = data2.drop("date", axis=1)

# Remove outliers
Q1 = np.percentile(data2['modal_price'], 25)
Q3 = np.percentile(data2['modal_price'], 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data2 = data2[(data2['modal_price'] > lower_bound) & (data2['modal_price'] < upper_bound)]

# Encode categorical features
label_encoders = {}
for column in ['commodity_name', 'state', 'district', 'market', 'month_column', 'season_names']:
    le = LabelEncoder()
    data2[column] = le.fit_transform(data2[column])
    label_encoders[column] = le

# Features and Labels
features = data2[['commodity_name', 'state', 'district', 'market', 'month_column', 'season_names', 'day']]
labels = data2['modal_price']

# Train-test split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, labels, test_size=0.2, random_state=2)

# Model Training
regr = RandomForestRegressor(max_depth=1000, random_state=0)
regr.fit(Xtrain, Ytrain)

# Save the model and encoders
joblib.dump(regr, 'random_forest_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')