
#%%

def find_limits(data, variables, fold):
    limits = dict()
    for variable in variables:
        IQR = data[variable].quantile(0.75) - data[variable].quantile(0.25)
        lower_limit = data[variable].quantile(0.25) - (IQR * fold)
        upper_limit = data[variable].quantile(0.75) + (IQR * fold)
        limits[variable] = (lower_limit, upper_limit)
    return limits

#%%
def clip_variables(data, limits):
    clipped_data = data.copy()
    for variable, (lower_limit, upper_limit) in limits.items():
        clipped_data[variable] = clipped_data[variable].clip(lower=lower_limit, upper=upper_limit)
    return clipped_data
    
#%%[markdown]
# # Outlier Detection of Pickup Latitude and Pickup Longitude & Dropoff Latitude & Dropoff Longitude
#%%
# import folium

# def mapping_outliers(df):
#     # Plotting Pickup cordinates which are outside of New-York (pickup)
#     outlier_locations_pickup = df[((df.pickup_longitude <= -74.15) | (df.pickup_latitude <= 40.5774)| \
#                     (df.pickup_longitude >= -73.7004) | (df.pickup_latitude >= 40.9176))]

#     map_osm_pickup = folium.Map(location=[40.734695, -73.990372], zoom_start=3, tiles="cartodb positron")

#     # Plotting Outliers on Map (pickup)
#     sample_locations = outlier_locations_pickup.head(10000)
#     for i,j in sample_locations.iterrows():
#         if int(j['pickup_latitude']) != 0:
#             folium.Marker(list((j['pickup_latitude'],j['pickup_longitude']))).add_to(map_osm_pickup)
#     map_osm_pickup

#     # Plotting dropoff cordinates which are outside of New-York (dropoff)
#     outlier_locations_dropoff = df[((df.dropoff_longitude <= -74.15) | (df.dropoff_latitude <= 40.5774)| \
#                     (df.dropoff_longitude >= -73.7004) | (df.dropoff_latitude >= 40.9176))]

#     map_osm_dropoff = folium.Map(location=[40.734695, -73.990372], zoom_start=3, tiles="cartodb positron")

#     # Plotting Outliers on Map (dropoff)
#     sample_locations = outlier_locations_dropoff.head(10000)
#     for i,j in sample_locations.iterrows():
#         if int(j['pickup_latitude']) != 0:
#             folium.Marker(list((j['dropoff_latitude'],j['dropoff_longitude']))).add_to(map_osm_dropoff)
#     map_osm_dropoff

#%%