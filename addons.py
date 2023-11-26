
from math import radians, sin, cos, sqrt, atan2

def google_maps_directions_link(trip_id, df):
    row = df[df['id'] == trip_id].iloc[0]
    pickup_coords = (row['pickup_latitude'], row['pickup_longitude'])
    dropoff_coords = (row['dropoff_latitude'], row['dropoff_longitude'])
    
    base_url = "https://www.google.com/maps/dir/?api=1&"
    pickup = f"origin={pickup_coords[0]},{pickup_coords[1]}"
    dropoff = f"destination={dropoff_coords[0]},{dropoff_coords[1]}"
    full_url = f"{base_url}{pickup}&{dropoff}"
    print(f'The calculated duration is {row["trip_duration_hours"]:.2f} hours for {row["distance"]/1.609:.2f} miles')
    return f"For Trip ID {trip_id}, click here for directions: {full_url}"

def haversine_distance(row):
    lat1, lon1 = radians(row['pickup_latitude']), radians(row['pickup_longitude'])
    lat2, lon2 = radians(row['dropoff_latitude']), radians(row['dropoff_longitude'])

    # Haversine formula
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = sin(d_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(d_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    radius_of_earth = 6371  # Radius of Earth in kilometers

    # Calculate the distance
    distance = radius_of_earth * c  # Distance in kilometers
    return distance