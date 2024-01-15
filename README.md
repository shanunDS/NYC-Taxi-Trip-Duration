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
- Predicting distance using pickup/dropoff coordinates and analyzing its relationship with trip duration.
- Identifying peak times and days for maximum trip durations.
- Determining top 4 locations with high taxi demand.
- Analyzing the relationship between speed, trip duration, and other features.

## Observations on the Dataset:
![Observation](https://github.com/shanunDS/Group5-project/assets/66896800/888d0266-a8ed-4757-a454-0ad729d0af82)

## Cleaning and Feature Engineering:
- Outlier correction using the 1.5 IQR method.
- Cleaning passenger counts and trip durations.
- Implementing Haversine formula for accurate distance calculations.
- Computing directional angle and additional features.

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
- Determining optimal number of clusters using the elbow method.
- Identifying top locations using Folium.

![Demand Prediction](https://github.com/shanunDS/Group5-project/assets/66896800/80083246-25dc-4e9f-9111-695cef96bb73)

## Avg. Speed and Trip Duration Analysis:
- Computing average traffic speeds.
- Linear Regression and XGBoost models for prediction.

![Avg. Speed](https://github.com/shanunDS/Group5-project/assets/66896800/088e4bf1-5cab-470d-9cec-6342f9b0c4cc)

## Conclusion:
- Use of Haversine formula for accurate distance computation.
- Identification of high-demand clusters in Manhattan.
- Analysis of time series trends and peak hours.
- Maximum average speed recorded with higher activity on weekends.
- XGBoost with Hyperparameter Tuning as the best-fit model.

### Comparison: Linear Regression vs XGBoost
![Comparison](https://github.com/shanunDS/Group5-project/assets/66896800/fe12c505-f4da-4b39-986b-638ce45ed8ab)
