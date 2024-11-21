import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def logistic_regression(data_path):
    # Read the data
    data = pd.read_csv(data_path)

    # Create grade categories
    def categorize_grade(grade):
        if grade <= 9:
            return 'Low'
        elif grade <= 13:
            return 'Medium'
        else:
            return 'High'

    data['grade_category'] = data['G3'].apply(categorize_grade)

    # Prepare features
    features = data.drop(['G3', 'grade_category', 'subject'], axis=1)
    labels = data['grade_category']

    # Encode categorical variables
    categorical_columns = ['sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                           'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 
                           'activities', 'nursery', 'higher', 'internet', 'romantic']

    for col in categorical_columns:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col])

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

    # Train Logistic Regression
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        C=1.0,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix with matplotlib
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Logistic Regression')
    plt.colorbar()
    classes = np.unique(labels)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Annotate the confusion matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], 
                     horizontalalignment="center", 
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.show()

    # Optional: Print model coefficients
    print("\nModel Coefficients:")
    coef_df = pd.DataFrame({
        'feature': features.columns,
        'coefficient': np.abs(model.coef_[0])
    }).sort_values('coefficient', ascending=False)
    print(coef_df.head(10))

    return model

# Example usage
if __name__ == "__main__":
    data_path = './clean/cleaned_student_performance.csv'
    model = logistic_regression(data_path)