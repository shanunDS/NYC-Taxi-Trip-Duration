import pandas as pd
from geopy.geocoders import Nominatim
import folium

def get_address(latitude, longitude):
    geolocator = Nominatim(user_agent="shanun")
    location = geolocator.reverse((latitude, longitude), language='en')
    return location.address if location else "Address not found"

def visualize_clusters(df, top_clusters):
    mean_latitude = df['pickup_latitude'].mean()
    mean_longitude = df['pickup_longitude'].mean()
    mymap = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=12)
    # Your visualization code here
    return mymap

def encode_categorical(df):
    df = pd.concat([df, pd.get_dummies(df['store_and_fwd_flag'], dtype=int)], axis=1)
    df.drop(['store_and_fwd_flag'], axis=1, inplace=True)
    df = pd.concat([df, pd.get_dummies(df['vendor_id'], dtype=int)], axis=1)
    df.drop(['vendor_id'], axis=1, inplace=True)
    column_mapping = {1: '1', 2: '2'}
    df = df.rename(columns=column_mapping)
    return df
