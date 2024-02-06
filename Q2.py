from flask import Flask
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
# Import utilities for tree visualization
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Initialize Flask application
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

# Uncomment the following line to run the Flask app if you're running this script outside this environment
# app.run(host="0.0.0.0")

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Define a function for decision tree modeling
def decision_tree(X, y):
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)

    # Fit the model with the training data
    dt_classifier.fit(X_train, y_train)

    # Make predictions on the test set using the .predict() method
    y_pred = dt_classifier.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Visualization of the decision tree
    plt.figure(figsize=(20,10))
    plot_tree(dt_classifier, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
    plt.show()

app.run(host="0.0.0.0")
