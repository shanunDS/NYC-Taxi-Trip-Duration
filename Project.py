import pandas as pd
import numpy as np
from geopy.distance import geodesic as GD
from geopy.geocoders import Nominatim
from geopy.geocoders import options
from geopy.exc import GeocoderTimedOut
import ssl



data=pd.read_csv(r'train.csv',parse_dates=['dropoff_datetime','pickup_datetime'], infer_datetime_format=True)



print(data.isna().sum())


data['pick_time']=data.pickup_datetime.dt.time
data['pick_date']=data.pickup_datetime.dt.date

data['drop_time']=data.dropoff_datetime.dt.time
data['drop_date']=data.dropoff_datetime.dt.date



# print(data.head())

def distance(long,lat,long1,lat1):
    pickup=(lat,long)
    drop=(lat1,long1)
    d=GD(pickup,drop).km
    return d





data['distance'] = data.apply(lambda row: distance(row['pickup_longitude'], row['pickup_latitude'], row['dropoff_longitude'],row['dropoff_latitude']), axis=1)

#################  calculate jrny distance ####################

data['journey_time'] = data['dropoff_datetime'] - data['pickup_datetime']
data['speed_kph'] = data['distance'] / (data['journey_time'].dt.total_seconds() / 3600)

print(data.head())