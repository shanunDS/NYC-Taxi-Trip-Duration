# import pandas as pd
# import os
# import numpy as np

# def load_data(file_path):
#     current_directory = os.getcwd()
#     df = pd.read_csv(os.path.join(current_directory, file_path))
#     return df

# def clip_outliers(data, variables, fold):
#     limits = dict()
#     for variable in variables:
#         IQR = data[variable].quantile(0.75) - data[variable].quantile(0.25)
#         lower_limit = data[variable].quantile(0.25) - (IQR * fold)
#         upper_limit = data[variable].quantile(0.75) + (IQR * fold)
#         limits[variable] = (lower_limit, upper_limit)
#     clipped_data = data.copy()
#     for variable, (lower_limit, upper_limit) in limits.items():
#         clipped_data[variable] = clipped_data[variable].clip(lower=lower_limit, upper=upper_limit)
#     return clipped_data

# def haversine_distance(df, lat1, lat2, long1, long2):
#     r = 6371  # average radius of Earth in kilometers
#     phi1 = np.radians(df[lat1])
#     phi2 = np.radians(df[lat2])
#     delta_phi = np.radians(df[lat2] - df[lat1])
#     delta_lambda = np.radians(df[long2] - df[long1])
#     a = np.sin(delta_phi/2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
#     d = r * c
#     return d

# def bearing_array(df, lat1, long1, lat2, long2):
#     AVG_EARTH_RADIUS = 6371  # in km
#     long_delta_rad = np.radians(long2 - long1)
#     lat1, long1, lat2, long2 = map(np.radians, (lat1, long1, lat2, long2))
#     y = np.sin(long_delta_rad) * np.cos(lat2)
#     x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(long_delta_rad)
#     return np.degrees(np.arctan2(y, x))

# def preprocess_data(df):
#     # Your preprocessing steps here
#     df.loc[:, 'trip_distance(km)'] = haversine_distance(df, 'pickup_latitude', 'dropoff_latitude', 'pickup_longitude','dropoff_longitude')
#     df.loc[:, 'center_latitude'] = (df['pickup_latitude'].values + df['dropoff_latitude'].values) / 2
#     df.loc[:, 'center_longitude'] = (df['pickup_longitude'].values + df['dropoff_longitude'].values) / 2
#     df.loc[:, 'direction'] = bearing_array(df,df['pickup_latitude'].values, df['pickup_longitude'].values,df['dropoff_latitude'].values, df['dropoff_longitude'].values)
#     return df

#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

def process_data(df):
    # Your data processing code here
    # Example: Calculate haversine distance
    df.loc[:, 'trip_distance(km)'] = haversine_distance(df, 'pickup_latitude', 'dropoff_latitude', 'pickup_longitude', 'dropoff_longitude')
    
    # Example: Calculate center latitude and center longitude
    df.loc[:, 'center_latitude'] = (df['pickup_latitude'].values + df['dropoff_latitude'].values) / 2
    df.loc[:, 'center_longitude'] = (df['pickup_longitude'].values + df['dropoff_longitude'].values) / 2
    
    # Example: Calculate direction
    df.loc[:, 'direction'] = bearing_array(df, df['pickup_latitude'].values, df['pickup_longitude'].values, df['dropoff_latitude'].values, df['dropoff_longitude'].values)
    
    # Example: Convert datetime to month, week, weekday, hour
    df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
    df['dropoff_datetime'] = pd.to_datetime(df.dropoff_datetime)
    df['month'] = df.pickup_datetime.dt.month
    df['week'] = df['pickup_datetime'].dt.isocalendar().week
    df['weekday'] = df.pickup_datetime.dt.weekday
    df['hour'] = df.pickup_datetime.dt.hour
    df['minute'] = df.pickup_datetime.dt.minute
    df['minute_oftheday'] = df['hour'] * 60 + df['minute']
    df.drop(['minute'], axis=1, inplace=True)
    
    # Example: Calculate average speed
    df.loc[:, 'avg_speed_h'] = 1000 * df['trip_distance(km)'] / df['trip_duration(sec)']
    
    # Example: Categorical encoding for store_and_fwd_flag and vendor_id
    df = pd.concat([df, pd.get_dummies(df['store_and_fwd_flag'], dtype=int)], axis=1)
    df.drop(['store_and_fwd_flag'], axis=1, inplace=True)
    df = pd.concat([df, pd.get_dummies(df['vendor_id'], dtype=int)], axis=1)
    df.drop(['vendor_id'], axis=1, inplace=True)
    
    # Example: Remove rows with trip distance less than 1 metre
    df = df[df['trip_distance(km)'] > 0.001]
    
    return df

def split_data(df):
    y = df["trip_duration(sec)"]
    df.drop(["trip_duration(sec)", 'avg_speed_h'], axis=1, inplace=True)
    df.drop(['id', 'pickup_datetime', 'dropoff_datetime', 'pickup_cluster'], axis=1, inplace=True)
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    return X_train, X_test, y_train, y_test
