from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample student data
# [study_hours, attendance_percentage]
X = [
    [1, 60],
    [2, 65],
    [2, 70],
    [3, 72],
    [3, 75],
    [4, 80],
    [4, 82],
    [5, 85],
    [5, 88],
    [6, 90],
    [6, 92],
    [7, 95]
]

# 0 = Low Performance, 1 = High Performance
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Student Performance Prediction")
print("--------------------------------")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Predict performance for a new student
study_hours = float(input("\nEnter study hours per day: "))
attendance = float(input("Enter attendance percentage: "))

prediction = model.predict([[study_hours, attendance]])

if prediction[0] == 1:
    print("Predicted Performance: High")
else:
    print("Predicted Performance: Low")
