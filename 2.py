import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1 Load Dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
data = pd.read_csv(url)
print("First 5 rows:")
print(data.head())

# Features and Target
X = data.drop("species", axis=1)
y = data["species"]

# 2 Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)

# 3 Train Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4 Predictions & Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 5 Save Model
joblib.dump(model, "iris_model.pkl")
print("\nModel saved as iris_model.pkl")
