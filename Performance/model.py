import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
data = pd.read_csv('student_placement_data.csv')  # Replace 'student_placement_data.csv' with your dataset filename

# Assuming 'Placed' column indicates whether a student is placed or not (1 for placed, 0 for not placed)
# X contains features, y contains target variable
X = data.drop('Placed', axis=1)
y = data['Placed']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print(classification_report(y_test, y_pred))
