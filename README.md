# New York City Taxi Trip Duration Prediction

![Taxi](https://github.com/shanunDS/Group5-project/assets/66896800/016534b4-12b4-4670-932f-8716049e68fa)

## Overview
This project aims to predict the total ride duration of taxi trips in New York City. Considering the high traffic and short distances in the city, predicting ride duration is crucial for efficient transportation. The dependent variable is "trip_duration," and the project involves feature engineering, data exploration, time series analysis, and model building.

## Team Members (Team 5):
- Shanun Randev
- Anand Raj
- Mowzli Sre
- Bala Krishna Reddy

## Dataset Details:
The dataset includes various features such as id, vendor_id, pickup/dropoff datetime, passenger count, and geographical coordinates. The target variable is trip duration in seconds.

## Data Analysis Questions:
- [Predicting Distance](#predicting-distance)
- [Peak Times and Days](#peak-times-and-days)
- [Top Locations](#top-locations)
- [Speed and Duration Analysis](#avg-speed-and-trip-duration-analysis)

## Observations on the Dataset:
![Observation](https://github.com/shanunDS/Group5-project/assets/66896800/888d0266-a8ed-4757-a454-0ad729d0af82)

## Cleaning and Feature Engineering:
- [Outlier Correction](#outlier-correction)
- [Cleaning Passenger Counts](#cleaning-of-passenger-counts)
- [Cleaning Trip Durations](#cleaning-of-trip-durations)
- [Haversine Formula](#haversine-formula)
- [Cleaning Trip Distance](#cleaning-of-trip-distance)
- [Computing Directional Angle](#computing-directional-angle)
- [Feature Engineering](#feature-engineering)

![Cleaning](https://github.com/shanunDS/Group5-project/assets/66896800/8c3634e4-f54d-4c0c-bbe3-cdc6a09380be)

## Density Plots after Preprocessing:
![Density](https://github.com/shanunDS/Group5-project/assets/66896800/efce6fdf-3329-43e9-a4d7-81545a406624)

## Correlation of Trip Duration with Features:
Strong positive correlation (0.8) between trip duration and trip distance.

![Correlation](https://github.com/shanunDS/Group5-project/assets/66896800/1349599b-892b-4c4e-be91-b073ef918260)

## Time Series Analysis:
- Peaks and drops in trend graph related to weather conditions.
- Weekly and daily trends analyzed.

![Time Series](https://github.com/shanunDS/Group5-project/assets/66896800/1aae8867-232f-4424-acf4-61e84d691a75)

## Demand Prediction using K-means Clustering:
- [Demand Prediction](#demand-prediction)
- [Determining 'k' for Clusters](#determining-k-for-clusters)
- [Fitting and Identifying Clusters](#fitting-and-identifying-the-clusters)
- [Top Locations (Displayed using Folium)](#top-locations-displayed-using-folium)

![Demand Prediction](https://github.com/shanunDS/Group5-project/assets/66896800/80083246-25dc-4e9f-9111-695cef96bb73)

## Avg. Speed and Trip Duration Analysis:
- [Computing Avg. Traffic Speeds](#computing-avg-traffic-speeds)
- [Linear Regression using CV](#linear-regression-using-cv)
- [XGBoost](#xgboost)
- [Compare LR vs XGBoost](#compare-lr-vs-xgboost)

![Avg. Speed](https://github.com/shanunDS/Group5-project/assets/66896800/088e4bf1-5cab-470d-9cec-6342f9b0c4cc)

## Conclusion
- [Conclusion](#conclusion)

### Comparison: Linear Regression vs XGBoost
![Comparison](https://github.com/shanunDS/Group5-project/assets/66896800/fe12c505-f4da-4b39-986b-638ce45ed8ab)
