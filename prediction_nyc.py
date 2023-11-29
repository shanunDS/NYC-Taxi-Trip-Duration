#!/usr/bin/env python
# coding: utf-8

# In[176]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns


# In[177]:


def haversine_distance(df, lat1, lat2, long1, long2):
    
    r = 6371 #average radius of earth in kilometers
    
    phi1 = np.radians(df[lat1])
    
    phi2 = np.radians(df[lat2])
    
    delta_phi = np.radians(df[lat2] - df[lat1])
    
    delta_lambda = np.radians(df[long2] - df[long1])
    
    a = np.sin(delta_phi/2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    d = r * c
    
    return d
    
    
    


# In[178]:


df = pd.read_csv(r'C:\Users\Asus\OneDrive\Desktop\nyc_taxi\Group5-project\train.csv')


# In[179]:


df.head()


# In[180]:


len(df)


# In[181]:


df.duplicated().sum()


# In[182]:


df['dist_km'] = haversine_distance(df, 'pickup_latitude', 'dropoff_latitude', 'pickup_longitude', 'dropoff_longitude')


# In[183]:


df[df['dist_km'] == 0]


# In[184]:


df = df.loc[df['dist_km'] != 0]


# In[185]:


len(df)


# In[186]:


#finding upper and lower bounds using IQR method - any data point which is beyond (lower limit - 1.5IQR) and (upper limit + 1.5IQR) will be termed as an outlier

#IQR = difference between the 75th quantile and the 25th quantile


# In[187]:


def find_limits(data, variable, fold):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_limit = df[variable].quantile(0.25) - (IQR * fold)
    upper_limit = df[variable].quantile(0.75) + (IQR * fold)
    return lower_limit, upper_limit


# In[188]:


#CHECKING IF THERE ANY ROWS WITH 0 TRIP DURATION


# In[189]:


df[df.trip_duration == 0]  # there arent any


# In[190]:


#removing data points where we have 0 passenger count


# In[191]:


len(df[df['passenger_count'] == 0])


# In[192]:


df = df.loc[df['passenger_count'] != 0]


# In[193]:


df[df.passenger_count == 0]


# In[194]:


df.info()


# In[195]:


df.pickup_latitude.min(), df.pickup_latitude.max()


# In[196]:


lower_limit_pickup_latitude, upper_limit_pickup_latitude = find_limits(df, 'pickup_latitude', 1.5)


# In[197]:


lower_limit_pickup_latitude, upper_limit_pickup_latitude


# In[198]:


df.pickup_longitude.min(), df.pickup_longitude.max()


# In[199]:


lower_limit_pickup_longitude, upper_limit_pickup_longitude = find_limits(df, 'pickup_longitude', 1.5)


# In[200]:


lower_limit_pickup_longitude, upper_limit_pickup_longitude


# In[201]:


df.dropoff_latitude.min(), df.dropoff_latitude.max()


# In[202]:


lower_limit_drop_latitude, upper_limit_drop_latitude = find_limits(df, 'dropoff_latitude', 1.5)


# In[203]:


lower_limit_drop_latitude, upper_limit_drop_latitude


# In[204]:


df.dropoff_longitude.min(), df.dropoff_longitude.max()


# In[205]:


lower_limit_drop_longitude, upper_limit_drop_longitude = find_limits(df, 'dropoff_longitude', 1.5)


# In[206]:


lower_limit_drop_longitude, upper_limit_drop_longitude


# In[207]:


df.dist_km.min(), df.dist_km.max()


# In[208]:


lower_limit_dist_km, upper_limit_dist_km = find_limits(df, 'dist_km', 1.5)


# In[209]:


lower_limit_dist_km, upper_limit_dist_km


# In[210]:


df.trip_duration.min(), df.trip_duration.max()


# In[211]:


lower_limit_trip_duration, upper_limit_trip_duration = find_limits(df, 'trip_duration', 1.5)


# In[212]:


lower_limit_trip_duration, upper_limit_trip_duration


# In[213]:


df['trip_duration'].clip(lower=lower_limit_trip_duration, upper=upper_limit_trip_duration, inplace=True)


# In[214]:


df['pickup_latitude'].clip(lower=lower_limit_pickup_latitude, upper=upper_limit_pickup_latitude, inplace=True)


# In[215]:


df['pickup_longitude'].clip(lower=lower_limit_pickup_longitude, upper = upper_limit_pickup_longitude, inplace=True)


# In[216]:


df['dropoff_latitude'].clip(lower = lower_limit_drop_latitude, upper = upper_limit_drop_latitude, inplace=True)


# In[217]:


df['dropoff_longitude'].clip(lower=lower_limit_drop_longitude, upper = upper_limit_drop_longitude, inplace=True)


# In[218]:


df['dist_km'].clip(lower=lower_limit_dist_km, upper = upper_limit_dist_km, inplace=True)


# In[219]:


df = pd.concat([df, pd.get_dummies(df['store_and_fwd_flag'],dtype=int)], axis=1)
df.drop(['store_and_fwd_flag'], axis=1, inplace=True)
df = pd.concat([df, pd.get_dummies(df['vendor_id'],dtype=int)], axis=1)
df.drop(['vendor_id'], axis=1, inplace=True)


# In[220]:


df.head()


# In[221]:


df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
df['dropoff_datetime'] = pd.to_datetime(df.dropoff_datetime)
df['month'] = df.pickup_datetime.dt.month
df['week'] = df['pickup_datetime'].dt.isocalendar().week
df['weekday'] = df.pickup_datetime.dt.weekday
df['hour'] = df.pickup_datetime.dt.hour
df['minute'] = df.pickup_datetime.dt.minute
df['minute_oftheday'] = df['hour'] * 60 + df['minute']
df.drop(['minute'], axis=1, inplace=True)


# In[222]:


df.head()


# In[223]:


y = df["trip_duration"]
df.drop(["trip_duration"], axis=1, inplace=True)
df.drop(['id'], axis=1, inplace=True)
X = df


# In[224]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[171]:


pip install catboost


# In[225]:


from catboost import CatBoostRegressor, CatBoostClassifier


# In[226]:


model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6)

# Fit the model
model.fit(X_train, y_train)

# Generate predictions
predictions = model.predict(X_test)


# In[227]:


predictions


# In[232]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error



# In[229]:


mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')


# In[230]:


rmse = np.sqrt(mean_squared_error(y_test, predictions))


# In[231]:


print(rmse)


# In[233]:


predictions = np.maximum(predictions, 0)
y_test = np.maximum(y_test, 0)

# Calculate RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_test + 1, predictions + 1))


# In[234]:


rmsle


# In[ ]:




