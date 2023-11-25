
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import os


from main import df 


print(df.head(10))

def plot_boxplot_and_hist(data, variable):
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios" : (0.50, 0.85)})
    sns.boxplot(x = data[variable], ax = ax_box)
    sns.histplot(data = data, x = variable, ax = ax_hist)
    ax_box.set(xlabel = "")
    plt.title(variable)
    plt.show()
    
#mean - 3 * standard deviation is the lower limit
#mean + 3 * standard deviation is the upper limit   
    
    
def find_limits(data, variable, fold):
    lower_limit = data[variable].mean() - fold * data[variable].std()
    upper_limit = data[variable].mean() + fold * data[variable].std()
    return lower_limit, upper_limit


#Outlier clipping


lower_limit_pick_lat, upper_limit_pick_lat = find_limits(df, 'pickup_latitude', 3)

df['pickup_latitude'].clip(lower = lower_limit_pick_lat, upper=upper_limit_pick_lat, inplace=True)

lower_limit_drop_lat, upper_limit_drop_lat = find_limits(df, 'dropoff_latitude', 3)

df['dropoff_latitude'].clip(lower=lower_limit_drop_lat, upper=upper_limit_drop_lat, inplace=True)

lower_limit_pickup_long, upper_limit_pickup_long = find_limits(df, 'pickup_longitude', 3)

df['pickup_longitude'].clip(lower=lower_limit_pickup_long, upper=upper_limit_pickup_long, inplace=True)

lower_limit_drop_long, upper_limit_drop_long = find_limits(df, 'dropoff_longitude', 3)

df['dropoff_longitude'].clip(lower=lower_limit_drop_long, upper=upper_limit_drop_long, inplace=True)

lower_limit_trip_duration, upper_limit_trip_duration = find_limits(df, 'trip_duration', 3)

df['trip_duration'].clip(lower=lower_limit_trip_duration, upper=upper_limit_trip_duration, inplace=True)



