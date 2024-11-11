import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def regression(data):
    if data is not None:
        features = data.drop(['G3'], axis=1)
        labels = data['G3']

        features = pd.get_dummies(features)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R-squared Score: {r2}")

        return (y_test.values, y_pred)

        #plt.figure(figsize=(10, 5))
        #plt.plot(y_test.values, label="Actual Grades", color="blue")
        #plt.plot(y_pred, label="Predicted Grades", color="red")
        #plt.xlabel("Samples")
        #plt.ylabel("Grades (G3)")
        #plt.title("Actual vs Predicted Grades")
        #plt.legend()
        #plt.show()
