# New York City Taxi Trip Duration Prediction

**Overview:**  
The project aims to predict the total ride duration of taxi trips in New York City. Given the challenges of high traffic and short distances, the goal is to create a low-latency system for predicting "Trip Duration" based on pickup and drop-off coordinates.

**Team Members (Team 5):**
- Shanun Randev
- Anand Raj
- Mowzli Sre
- Bala Krishna Reddy

**Dataset Details:**
- **Id:** Unique identifier for each trip
- **Vendor_id:** Code indicating the provider associated with the trip record
- **Pickup_datetime:** Date and time when the meter was engaged
- **Dropoff_datetime:** Date and time when the meter was disengaged
- **Passenger_count:** Number of passengers in the vehicle
- **Pickup_longitude, Pickup_latitude, Dropoff_longitude, Dropoff_latitude:** Geographic coordinates
- **Store_and_fwd_flag:** Flag indicating whether the trip record was held in the vehicle memory before sending to the vendor
- **Target Variable:** Trip Duration (seconds)

## Data Analysis

### [Predicting Distance](#predicting-distance)
- Explore the relationship between pickup/dropoff coordinates and the total trip distance.

### [Peak Times and Days](#peak-times-and-days)
- Identify the days and times with maximum trip duration.

### [Top Locations](#top-locations)
- Discover the top 4 locations with high demand for taxis in NYC.

### [Speed Analysis](#speed-analysis)
- Investigate how vehicle speed relates to trip duration.

## Observations on the Dataset

![Observation](https://github.com/shanunDS/Group5-project/assets/66896800/888d0266-a8ed-4757-a454-0ad729d0af82)

## Cleaning and Feature Engineering

![Cleaning](https://github.com/shanunDS/Group5-project/assets/66896800/276c96de-1209-4c9e-87bb-79d889886a9c)

### Data Cleaning and Preprocessing

#### [Outlier Correction](#outlier-correction)
- Employing the 1.5 IQR method to identify and correct outliers.

![Outlier Correction](https://github.com/shanunDS/Group5-project/assets/66896800/8c3634e4-f54d-4c0c-bbe3-cdc6a09380be)

#### [Cleaning Passenger Counts](#cleaning-of-passenger-counts)
- Removing trips with 0 or more than 6 passengers.

#### [Cleaning Trip Durations](#cleaning-of-trip-durations)
- Setting a threshold of 100 hours for practicality.

### [Haversine Formula](#haversine-formula)
- Computing trip distance using the Haversine formula.

### [Cleaning Trip Distance](#cleaning-of-trip-distance)
- Setting a minimum meaningful trip distance and handling outliers.

### [Computing Directional Angle](#computing-directional-angle)
- Calculating the compass bearing between two geographic points.

### [Feature Engineering](#feature-engineering)
- Creating new columns like hour of the day, day of the trip, etc.

### Density Plots after Preprocessing

![Density Plots](https://github.com/shanunDS/Group5-project/assets/66896800/efce6fdf-3329-43e9-a4d7-81545a406624)

### Correlation of Trip Duration with Other Features

![Correlation](https://github.com/shanunDS/Group5-project/assets/66896800/1349599b-892b-4c4e-be91-b073ef918260)

### Relationship of Trip Distance with Trip Duration

![Scatter Plot](https://github.com/shanunDS/Group5-project/assets/66896800/1e8f7bb9-d559-4aaa-b178-08375163c1d6)

## Time Series Analysis

- Peaks and drops related to weather conditions.
- Weekly and daily trends analyzed.

![Time Series Analysis](https://github.com/shanunDS/Group5-project/assets/66896800/1aae8867-232f-4424-acf4-61e84d691a75)

### [Daily Taxi Ride Trends](#daily-taxi-ride-trends)
- Observe the gradual increase in rides from Monday to Saturday.

![Daily Trends](https://github.com/shanunDS/Group5-project/assets/66896800/f85859af-0dbe-4e56-bb16-767b93e0bb3a)

### [Taxi Ride Trends by Hour](#taxi-ride-trends-by-hour)
- Note the peak demand hours in the evening.

![Hourly Trends](https://github.com/shanunDS/Group5-project/assets/66896800/5b0eb8bd-9be2-4bd4-a99b-4ecd7b029745)

### [Weekly Taxi Ride Patterns](#weekly-taxi-ride-patterns)
- Identify weekly patterns with prominent peaks.

![Weekly Patterns](https://github.com/shanunDS/Group5-project/assets/66896800/fc6d633e-cd58-425e-ae85-554893a10b55)

## Demand Prediction using K-means Clustering

### [Demand Prediction](#demand-prediction)

![Demand Prediction](https://github.com/shanunDS/Group5-project/assets/66896800/80083246-25dc-4e9f-9111-695cef96bb73)

### [Determining 'k' for Clusters](#determining-k-for-clusters)
- Using the elbow method to determine the number of clusters.

![Cluster Elbow](https://github.com/shanunDS/Group5-project/assets/66896800/1217656c-e60a-42a7-937a-44755854bd51)

### [Fitting and Identifying Clusters](#fitting-and-identifying-the-clusters)

![Cluster Identification](https://github.com/shanunDS/Group5-project/assets/66896800/fffc41e6-c4e7-4641-949f-00767ccd9f1c)

### [Top Locations (Displayed using Folium)](#top-locations-displayed-using-folium)

![Top Locations](https://github.com/shanunDS/Group5-project/assets/66896800/58ffaf58-24a0-42fa-b14d-0b0c4ea2938f)

## Avg. Speed and Trip Duration Analysis

### [Computing Avg. Traffic Speeds](#computing-avg-traffic-speeds)
- Analyzing traffic speed patterns.

![Avg. Speed Analysis](https://github.com/shanunDS/Group5-project/assets/66896800/088e4bf1-5cab-470d-9cec-6342f9b0c4cc)

### [Linear Regression using CV](#linear-regression-using-cv)
- Evaluating performance metrics for linear regression.

### [XGBoost](#xgboost)
- Utilizing XGBoost for better model performance.

### [Compare LR vs XGBoost](#compare-lr-vs-xgboost)
- Visual comparison between Linear Regression and XGBoost.

![Comparison](https://github.com/shanunDS/Group5-project/assets/66896800/fe12c505-f4da-4b39-986b-638ce45ed8ab)

## Conclusion

- Utilized Haversine Formula for accurate distance computation.
- Identified high-demand clusters around Manhattan.
- Analyzed time series, finding peak hours on Friday and Saturday.
- Max Avg. Speed recorded at 24 Km/hr with higher activity on weekends.
- XGBoost with Hyperparameter Tuning achieved the best model performance with ~3.7 minutes error.
