from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to create and train a decision tree model
def train_decision_tree(X_train, y_train, random_state=42):
    dt_classifier = DecisionTreeClassifier(random_state=random_state)
    dt_classifier.fit(X_train, y_train)
    return dt_classifier

# Train the decision tree model
dt_model = train_decision_tree(X_train, y_train)

# You can add similar functions for KNN, Linear Regression, etc.
# For example, for KNN:
def train_knn(X_train, y_train, n_neighbors=3):
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(X_train, y_train)
    return knn_classifier

# And for Linear Regression (note: not typically used for classification like Iris dataset):
def train_linear_regression(X_train, y_train):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    return lr_model

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['GET'])
def predict():
    # Extract query parameters for model prediction
    try:
        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid input parameters'}), 400

    # Make prediction using the decision tree model
    features = [sepal_length, sepal_width, petal_length, petal_width]
    predicted_class = iris.target_names[dt_model.predict([features])[0]]

    # Standard API response
    response = {
        'prediction': predicted_class
        # You can add more fields to the response as needed
    }

    return jsonify(response)

# Uncomment the following line to run the Flask app if you're running this script outside this environment
# app.run(host="0.0.0.0", debug=True)
