#%%[markdown]

## New York City Taxi Trip Duration

# Team Member (Team 5):
# 1)   Shanun Randev 
# 2)   Anand Raj
# 3)   Mowzli Sre 
# 4)   Bala krishna Reddy

## Dataset Info:

# The independent variables are as follows : 

# 1.	Id - unique identifier for each trip
# 2.	Vendor_id - a code indicating the provider associated with the trip record
# 3.	Pickup_datetime - date and time when the meter was engaged
# 4.	Dropoff_datetime - date and time when the meter was disengaged
# 5.	Passenger_count - number of passengers in the vehicle
# 6.	Pickup_longitude - Longitude where the meter was engaged
# 7.	Pickup_latitude - Latitude where the meter was engaged
# 8.	Dropoff_latitude - latitude where the meter was disengaged
# 9.	Dropoff_longitude - longitude where the meter was disengaged
# 10.	Store_and_fwd_flag- This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y = store and forward; N = not a store and forward trip.

#%%[markdown]

# Importing Libraries

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import os


# Get the current working directory
current_directory = os.getcwd()

# Reading the CSV file
df = pd.read_csv(os.path.join(current_directory, 'train.csv'))

df.head()

#%%[markdown]

## Data Cleaning

## Must do this part here

#%%
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


#%%[markdown]

#Calculating haversine distance between 2 sets of GPS coordinates in the dataframe

#%%
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

#%%

# df.columns

#%%
df['dist_km'] = haversine_distance(df, 'pickup_latitude', 'dropoff_latitude', 'pickup_longitude', 'dropoff_longitude')

# df.head()

# df.info()

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

# df.info()

# df.head(10)

#%%

# df['pickup_datetime'][10]

# df.tail()

# len(df)

# type(df['pickup_datetime'][0])

#%%
df['Hour'] = df['pickup_datetime'].dt.hour

df['AMorPM'] = np.where(df['Hour'] < 12, 'am', 'pm')

df['weekday'] = df['pickup_datetime'].dt.strftime("%a")

#%%
# df.head()

#%%
# sns.histplot(df, x=np.log(df["trip_duration"]), kde=True)

# sns.histplot(data = df, x = df['trip_duration'], kde = True)

#%%
# y=1 + np.log(df["trip_duration"])

# sns.lineplot(x='passenger_count', y=y, data=df)

# plt.show()

# #%%
# g = sns.FacetGrid(df, col='passenger_count', col_wrap=3, height=4)
# g.map(plt.hist, 'trip_duration', bins=30)
# g.set_axis_labels('Trip Duration', 'Frequency')
# plt.show()

# #%%
# sns.scatterplot(x='dist_km', y='trip_duration', data=df)
# plt.title('Scatter Plot of Passenger Count vs Trip Duration')
# plt.show()
