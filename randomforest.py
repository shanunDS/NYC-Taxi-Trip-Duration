# Random Forest Prediction of Duration and Distance of given coordinates

# Imports
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from addons import *

# Accessing the DataFrame
#%%
df = pd.read_csv('../nyc-taxi-trip-duration/train/train.csv')

df.head()
# %%

# Convert pickup and dropoff datetime columns to datetime format
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

df['trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds()

df['trip_duration_minutes'] = df['trip_duration'] / 60
df['trip_duration_hours'] = df['trip_duration'] / 3

sorted_df = df.sort_values(by='trip_duration_hours', ascending=False)
print(df)
# %%
# Detecting Boundary Outliers

nyc_latitude_bounds = (40.4774, 40.9176)  
nyc_longitude_bounds = (-74.2591, -73.7002) 

bounded_df = df[
    (df['pickup_latitude'].between(*nyc_latitude_bounds)) &
    (df['pickup_longitude'].between(*nyc_longitude_bounds)) &
    (df['dropoff_latitude'].between(*nyc_latitude_bounds)) &
    (df['dropoff_longitude'].between(*nyc_longitude_bounds))
]

print(bounded_df)

#%%
# Calculating the Haversine Distance between the coordinates

bounded_df.loc[:, 'distance'] = bounded_df.apply(haversine_distance, axis=1)

# %%
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = bounded_df[['trip_duration']] 
y = bounded_df['distance'] 

model = LinearRegression()
model.fit(X, y)

predicted_distances = model.predict(X)
residuals = y - predicted_distances
mse = mean_squared_error(y, predicted_distances)
threshold = 2*(mse ** 0.5)

outliers = bounded_df[abs(residuals) > threshold]
outliers.head(10)

# %%
print(google_maps_directions_link("id2890542", sorteds))

# %%
clean_df = bounded_df[~bounded_df['id'].isin(outliers['id'])]

print(clean_df)

#%%
# Featuring the Data Frame
clean_df.drop(columns=["id", "store_and_fwd_flag"], inplace=True)
# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


clean_df['pickup_day_of_week'] = clean_df['pickup_datetime'].dt.dayofweek
clean_df['pickup_hour_of_day'] = clean_df['pickup_datetime'].dt.hour

features = ['trip_duration_minutes', 'distance',
            'pickup_day_of_week', 'pickup_hour_of_day']

# Selecting features and target variable
X = clean_df[features]  # Features (in this case, only 'distance')
y_duration = clean_df['trip_duration']  # Target variable ('trip_duration')

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_duration, test_size=0.1, random_state=42)

# Initialize Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Assuming 'clean_df' is your DataFrame with necessary columns: 'vendor_id', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'duration', 'distance'

# Selecting features and target variables
features = ['vendor_id', 'passenger_count', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',
            'distance', 'pickup_day_of_week', 'pickup_hour_of_day']
target_duration = 'trip_duration'
target_distance = 'distance'

X = clean_df[features]
y_duration = clean_df[target_duration]
y_distance = clean_df[target_distance]

# Splitting the data into train and test sets for duration prediction
X_train_duration, X_test_duration, y_train_duration, y_test_duration = train_test_split(X, y_duration, test_size=0.2, random_state=42)

# Splitting the data into train and test sets for distance prediction
X_train_distance, X_test_distance, y_train_distance, y_test_distance = train_test_split(X, y_distance, test_size=0.2, random_state=42)

# Initialize and train Random Forest models for duration and distance prediction
rf_model_duration = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_distance = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the models
rf_model_duration.fit(X_train_duration, y_train_duration)
rf_model_distance.fit(X_train_distance, y_train_distance)

# Make predictions on the test sets
y_pred_duration = rf_model_duration.predict(X_test_duration)
y_pred_distance = rf_model_distance.predict(X_test_distance)

# Evaluate the models
mse_duration = mean_squared_error(y_test_duration, y_pred_duration)
r2_duration = r2_score(y_test_duration, y_pred_duration)

mse_distance = mean_squared_error(y_test_distance, y_pred_distance)
r2_distance = r2_score(y_test_distance, y_pred_distance)

print(f"Random Forest - Duration - Mean Squared Error (MSE): {mse_duration:.2f}, R-squared (R2): {r2_duration:.2f}")
print(f"Random Forest - Distance - Mean Squared Error (MSE): {mse_distance:.2f}, R-squared (R2): {r2_distance:.2f}")

# %%
