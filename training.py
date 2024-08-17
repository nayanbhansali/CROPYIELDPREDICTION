import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import joblib
import numpy as np

# Load the dataset
file_path = 'crop_yield.csv'
data = pd.read_csv(file_path)

# Identify categorical columns for encoding
categorical_columns = ['Crop']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Features and target, excluding 'Season' and 'State'
features = data[['Crop', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
target = data['Yield']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

# Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_scaled, y_train)
y_pred_dt = dt_model.predict(X_test_scaled)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
r2_dt = r2_score(y_test, y_pred_dt)

# Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

# Gradient Boosting
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
r2_gb = r2_score(y_test, y_pred_gb)

# AdaBoost
ab_model = AdaBoostRegressor(random_state=42)
ab_model.fit(X_train_scaled, y_train)
y_pred_ab = ab_model.predict(X_test_scaled)
rmse_ab = np.sqrt(mean_squared_error(y_test, y_pred_ab))
r2_ab = r2_score(y_test, y_pred_ab)

# K-Nearest Neighbors
knn_model = KNeighborsRegressor()
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
r2_knn = r2_score(y_test, y_pred_knn)

# Support Vector Machine
svm_model = SVR()
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
rmse_svm = np.sqrt(mean_squared_error(y_test, y_pred_svm))
r2_svm = r2_score(y_test, y_pred_svm)

# Collecting results
results = {
    'Linear Regression': (rmse_lr, r2_lr),
    'Decision Tree': (rmse_dt, r2_dt),
    'Random Forest': (rmse_rf, r2_rf),
    'Gradient Boosting': (rmse_gb, r2_gb),
    'AdaBoost': (rmse_ab, r2_ab),
    'K-Nearest Neighbors': (rmse_knn, r2_knn),
    'Support Vector Machine': (rmse_svm, r2_svm)
}

# Finding and saving the best model
best_model_name = min(results, key=lambda x: results[x][0])
best_model = {
    'Linear Regression': lr_model,
    'Decision Tree': dt_model,
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model,
    'AdaBoost': ab_model,
    'K-Nearest Neighbors': knn_model,
    'Support Vector Machine': svm_model
}[best_model_name]

# Save the best model with joblib
joblib.dump(best_model, 'best_model.joblib')

print(f'The best model is {best_model_name} with RMSE: {results[best_model_name][0]} and R²: {results[best_model_name][1]}')

# Save the label encoders with joblib
joblib.dump(label_encoders, 'label_encoders.joblib')
