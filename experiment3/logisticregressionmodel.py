# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load a sample dataset
iris = datasets.load_iris()

# Define features (X) and target labels (y)
X = iris.data
y = iris.target

# Use only two classes (binary classification) for Logistic Regression
X_binary = X[y != 2]
y_binary = y[y != 2]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.3, random_state=42)

# Create the Logistic Regression model
logistic_model = LogisticRegression(random_state=42, solver='liblinear')

# Train the model on the training data
logistic_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = logistic_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
