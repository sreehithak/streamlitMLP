import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def logistic_regression(processed_data):
    def categorize_grade(grade):
        if grade <= 9:
            return 'Low'
        elif grade <= 13:
            return 'Medium'
        else:
            return 'High'
    X = processed_data.drop('G3', axis=1)
    y = processed_data['G3']
    
  
    y_cat = y.apply(categorize_grade)

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        C=1.0,
        random_state=42
    )
    model.fit(X_train, y_train)

    #predict
    y_pred = model.predict(X_test)
    
    actual_grades = y_test
    predicted_grades = y_pred

    return actual_grades, predicted_grades