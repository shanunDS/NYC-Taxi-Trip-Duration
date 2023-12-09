import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



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
