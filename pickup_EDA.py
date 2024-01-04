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
df.info()


#%%
####################### avg trip duration based on hour and weekday###############
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


grouped_data = df.groupby(['weekday', 'hour'])['trip_duration'].mean().reset_index()

# Create a pivot table for the heatmap
pivot_table = grouped_data.pivot(index="weekday", columns="hour", values="trip_duration")

# Plotting the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, cmap="YlGnBu", annot=True, fmt=".0f")
plt.title('Heatmap of Average Trip Duration by Day of Week and Hour')
plt.ylabel('Day of Week (0=Monday, 6=Sunday)')
plt.xlabel('Hour of Day')
plt.show()

#%%
######################### most peaked hours ################
hourly_counts = df['hour'].value_counts().sort_index()

# Plotting
colormap = 'coolwarm'

# Applying the colormap
plt.figure(figsize=(10, 6))
palette = sns.color_palette('husl', n_colors=24)
# sns.barplot(x=hourly_counts.index, y=hourly_counts.values, palette=sns.color_palette(colormap, len(hourly_counts)))
sns.barplot(x=hourly_counts.index, y=hourly_counts.values, palette=palette)

# Enhancing the plot
plt.title('Number of Rides per Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Rides')

# Display the plot
plt.show()




#%%

######################## Busiest days #################
weekdays = df['weekday'].value_counts().sort_index()

d=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
palette = sns.color_palette('husl', n_colors=24)
# Applying the colormap
plt.figure(figsize=(10, 6))
sns.barplot(x=d ,y=weekdays.values, palette=palette)

# Enhancing the plot
plt.title('Daily Taxi Ride Trends')
plt.xlabel('Day')
plt.ylabel('Number of Rides')

# Display the plot
plt.show() 



#%%

#on what day the busiest hours are
grouped_data = df.groupby(['weekday', 'hour']).size().reset_index(name='ride_count')

# Create a grid of bar plots
g = sns.FacetGrid(grouped_data, col='weekday', col_wrap=4, height=4, aspect=1.5)
g.map(sns.barplot, 'hour', 'ride_count', order=range(24))

# Enhancing the plot
g.set_titles('{col_name}')
g.set_axis_labels('Hour of Day', 'Number of Rides')
plt.xticks(range(0, 24))

# Display the plot
plt.show()



#%%

#############line plot ###############
grouped_data = df.groupby(['weekday', 'hour']).size().reset_index(name='ride_count')

# Create a line plot with multiple lines
plt.figure(figsize=(12, 6))
sns.lineplot(data=grouped_data, x='hour', y='ride_count', hue='weekday', palette='tab10')

# Enhancing the plot
plt.title('Number of Taxi Rides per Hour for Each Day of the Week')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Rides')
plt.legend(title='Day of the Week')
plt.xticks(range(0, 24))  # Setting x-ticks for each hour of the day
# plt.grid(True)

# Display the plot
plt.show()

#%%
grouped_data = df.groupby(['weekday', 'hour']).size().reset_index(name='ride_count')

# grouped_data.head()

#%%
# Create a scatter plot
plt.figure(figsize=(12, 6))
sns.scatterplot(data=grouped_data, x='hour', y='weekday', size='ride_count', sizes=(20, 200), hue='ride_count', palette='viridis')

# Enhancing the plot
plt.title('Taxi Rides Scatter Plot by Day of the Week and Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Day of the Week')
plt.legend(title='Number of Rides', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()


########################## MOST pickup points #####################
#%%
df = df.loc[(df['pickup_latitude'] >= 40.637044) & (df['pickup_latitude'] <= 40.855256)]
df = df.loc[(df['pickup_longitude'] >= -74.035735) & (df['pickup_longitude'] <= -73.770272)]
df = df.loc[(df['dropoff_latitude'] >= 40.637044) & (df['dropoff_latitude'] <= 40.855256)]
df = df.loc[(df['dropoff_longitude'] >= -74.035735) & (df['dropoff_longitude'] <= -73.770272)]
df.shape


#%%

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate a simulated dataset of pickup latitude and longitude points
np.random.seed(0)


pickup_locations = np.column_stack((df['pickup_latitude'], df['pickup_longitude']))

# Applying k-means for a range of k values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(pickup_locations)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method For Optimal k (Pickup Locations)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-cluster Sum of Squares (WCSS)')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()





#%%
data=df.iloc[:,:]

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Assuming df is your DataFrame
pickup_locations = data[['pickup_longitude', 'pickup_latitude']]

# Define the number of clusters (adjust as needed)
num_clusters = 10

# Fit K-means clustering model
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['pickup_cluster'] = kmeans.fit_predict(pickup_locations)

#%%
data.head()






#%%
# Count the occurrences of each cluster
cluster_counts = data['pickup_cluster'].value_counts().reset_index(name='pickup_count')

# Display the most common pickup clusters
top_n = 10  # You can adjust this based on your preference
print(f"Top {top_n} Most Common Pickup Clusters:")
print(cluster_counts.head(top_n))

# Optionally, visualize the clusters on a scatter plot
plt.scatter(data['pickup_latitude'], data['pickup_longitude'], c=data['pickup_cluster'], cmap='viridis', alpha=0.5)
plt.xlabel('Latitudes')
plt.ylabel('Longitudes')
plt.title('Pickup Clusters')
     

#%%
def top_5_locations(group):
    # Count the frequency of each (latitude, longitude) pair
    freq = group.groupby(['pickup_latitude', 'pickup_longitude']).size()
    
    # Sort the frequencies in descending order and select the top 5
    return freq.sort_values(ascending=False).head(5)

# Apply the custom function to each group
top_locations_per_cluster = data.groupby('pickup_cluster').apply(top_5_locations).reset_index()

# Renaming the columns for clarity
top_locations_per_cluster.columns = ['pickup_cluster', 'pickup_latitude', 'pickup_longitude', 'frequency']

print(top_locations_per_cluster)


#%%
import folium


avg_latitude = df['pickup_latitude'].mean()
avg_longitude = df['pickup_longitude'].mean()
pickup_map = folium.Map(location=[avg_latitude, avg_longitude], zoom_start=12)
for index, row in top_locations_per_cluster.iterrows():
    folium.Marker([row['pickup_latitude'], row['pickup_longitude']],
                  popup=f"Cluster: {row['pickup_cluster']}",
                  icon=folium.Icon(color='blue')).add_to(pickup_map)

# Display the map
pickup_map


#%%
import io
from PIL import Image

img_data = pickup_map._to_png(5)
img = Image.open(io.BytesIO(img_data))
img.save('image.png')


############################ EDA ##############

#%%
data.head()


#%%



# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#%%
correlation_matrix = df.iloc[:,1:].corr()

# Plotting the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# %%
################################# MODEL BUILDING ##########################################
data.head()

# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'hour', 'weekday', 'dist_km']


y = df["trip_duration"]
X=df[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
model = LinearRegression()


#%%

# Train the model
model.fit(X_train, y_train)


#%%
# Make predictions on the test set
y_pred = model.predict(X_test)


# %%
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# %%
 