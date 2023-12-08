#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

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

# %%
df = pd.read_csv(r'nyc-taxi-trip-duration/train.csv')

#%%
df.info()


# %%
len(df)

# %%
df.duplicated().sum()

#%%
df['dist_km'] = haversine_distance(df, 'pickup_latitude', 'dropoff_latitude', 'pickup_longitude', 'dropoff_longitude')

# %%
df[df['dist_km'] == 0]

# %%
df = df.loc[df['dist_km'] != 0]
df[df.trip_duration == 0]

# %%
len(df[df['passenger_count'] == 0])
df = df.loc[df['passenger_count'] != 0]

# %%
df[df.passenger_count == 0]

# %%
df.pickup_latitude.min(), df.pickup_latitude.max()


#%%
def find_limits(df, variable, fold):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_limit = df[variable].quantile(0.25) - (IQR * fold)
    upper_limit = df[variable].quantile(0.75) + (IQR * fold)
    return lower_limit, upper_limit

# %%
lower_limit_pickup_latitude, upper_limit_pickup_latitude = find_limits(df, 'pickup_latitude', 1.5)

# %%
lower_limit_pickup_latitude, upper_limit_pickup_latitude

# %%
df.pickup_longitude.min(), df.pickup_longitude.max()


# %%
lower_limit_pickup_longitude, upper_limit_pickup_longitude = find_limits(df, 'pickup_longitude', 1.5)
# %%
lower_limit_pickup_longitude, upper_limit_pickup_longitude

#%%
df.dropoff_latitude.min(), df.dropoff_latitude.max()


# %%
lower_limit_drop_latitude, upper_limit_drop_latitude = find_limits(df, 'dropoff_latitude', 1.5)
# %%
lower_limit_drop_latitude, upper_limit_drop_latitude

# %%
df.dropoff_longitude.min(), df.dropoff_longitude.max()
# %%
lower_limit_drop_longitude, upper_limit_drop_longitude = find_limits(df, 'dropoff_longitude', 1.5)
# %%
lower_limit_drop_longitude, upper_limit_drop_longitude
# %%
df.dist_km.min(), df.dist_km.max()
# %%
lower_limit_dist_km, upper_limit_dist_km = find_limits(df, 'dist_km', 1.5)
# %%
lower_limit_dist_km, upper_limit_dist_km
# %%
df.trip_duration.min(), df.trip_duration.max()
# %%
lower_limit_trip_duration, upper_limit_trip_duration = find_limits(df, 'trip_duration', 1.5)
# %%
lower_limit_trip_duration, upper_limit_trip_duration
# %%
df['trip_duration'].clip(lower=lower_limit_trip_duration, upper=upper_limit_trip_duration, inplace=True)
# %%
df['pickup_latitude'].clip(lower=lower_limit_pickup_latitude, upper=upper_limit_pickup_latitude, inplace=True)
# %%
df['pickup_longitude'].clip(lower=lower_limit_pickup_longitude, upper = upper_limit_pickup_longitude, inplace=True)

# %%
df['dropoff_latitude'].clip(lower = lower_limit_drop_latitude, upper = upper_limit_drop_latitude, inplace=True)
# %%
df['dropoff_longitude'].clip(lower=lower_limit_drop_longitude, upper = upper_limit_drop_longitude, inplace=True)
# %%
df['dist_km'].clip(lower=lower_limit_dist_km, upper = upper_limit_dist_km, inplace=True)
# %%
df = pd.concat([df, pd.get_dummies(df['store_and_fwd_flag'],dtype=int)], axis=1)
df.drop(['store_and_fwd_flag'], axis=1, inplace=True)
df = pd.concat([df, pd.get_dummies(df['vendor_id'],dtype=int)], axis=1)
df.drop(['vendor_id'], axis=1, inplace=True)
# %%
df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
df['dropoff_datetime'] = pd.to_datetime(df.dropoff_datetime)
df['month'] = df.pickup_datetime.dt.month
df['week'] = df['pickup_datetime'].dt.isocalendar().week
df['weekday'] = df.pickup_datetime.dt.weekday
df['hour'] = df.pickup_datetime.dt.hour
df['minute'] = df.pickup_datetime.dt.minute
df['minute_oftheday'] = df['hour'] * 60 + df['minute']
df.drop(['minute'], axis=1, inplace=True)


# %%
df.head()


########################## MOST pickup points #####################
#%%
df = df.loc[(df['pickup_latitude'] >= 40.637044) & (df['pickup_latitude'] <= 40.855256)]
df = df.loc[(df['pickup_longitude'] >= -74.035735) & (df['pickup_longitude'] <= -73.770272)]
df = df.loc[(df['dropoff_latitude'] >= 40.637044) & (df['dropoff_latitude'] <= 40.855256)]
df = df.loc[(df['dropoff_longitude'] >= -74.035735) & (df['dropoff_longitude'] <= -73.770272)]
df.shape



#%%
data=df.iloc[:,:]

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Assuming df is your DataFrame
pickup_locations = data[['pickup_longitude', 'pickup_latitude']]

# Define the number of clusters (adjust as needed)
num_clusters = 4

# Fit K-means clustering model
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['pickup_cluster'] = kmeans.fit_predict(pickup_locations)

#%%
data.head()

#%%
# Count the occurrences of each cluster
cluster_counts = data['pickup_cluster'].value_counts().reset_index(name='pickup_count')

# Display the most common pickup clusters
top_n = 4  # You can adjust this based on your preference
print(f"Top {top_n} Most Common Pickup Clusters:")
print(cluster_counts.head(top_n))

# Optionally, visualize the clusters on a scatter plot
plt.scatter(data['pickup_latitude'], data['pickup_longitude'], c=data['pickup_cluster'], cmap='viridis', alpha=0.5)
plt.title('Pickup Clusters')
     

#%%
#getting common  pickup from each cluster
most_common_pickup_locations_per_cluster = data.groupby('pickup_cluster')[['pickup_latitude', 'pickup_longitude']].agg(lambda x: x.value_counts().index[0]).reset_index()

# Display the most common pickup locations within each cluster
print("Most Common Pickup Locations Within Each Cluster:")
print(most_common_pickup_locations_per_cluster)


#%%
import folium


avg_latitude = df['pickup_latitude'].mean()
avg_longitude = df['pickup_longitude'].mean()
pickup_map = folium.Map(location=[avg_latitude, avg_longitude], zoom_start=12)
for index, row in most_common_pickup_locations_per_cluster.iterrows():
    folium.Marker([row['pickup_latitude'], row['pickup_longitude']],
                  popup=f"Cluster: {row['pickup_cluster']}",
                  icon=folium.Icon(color='blue')).add_to(pickup_map)

# Display the map
pickup_map




