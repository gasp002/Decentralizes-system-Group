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

# Function to train and evaluate a decision tree model
def train_decision_tree(X_train, y_train, X_test, y_test):
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    y_pred = dt_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return dt_classifier, accuracy

# Function to train and evaluate a KNN model
def train_knn(X_train, y_train, X_test, y_test):
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return knn_classifier, accuracy

# Function to train and evaluate a linear regression model
# Since Linear Regression is not for classification, we use it here for demonstration only
def train_linear_regression(X_train, y_train, X_test, y_test):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    # Placeholder: Linear regression accuracy not applicable, you'd normally use a regression metric
    return lr_model, 'N/A'

# Train and store the models and their accuracies
dt_model, dt_accuracy = train_decision_tree(X_train, y_train, X_test, y_test)
knn_model, knn_accuracy = train_knn(X_train, y_train, X_test, y_test)
lr_model, lr_accuracy = train_linear_regression(X_train, y_train, X_test, y_test)

models_accuracy = {
    'Decision Tree': dt_accuracy,
    'KNN': knn_accuracy,
    'Linear Regression': lr_accuracy  # Placeholder for an appropriate regression metric
}

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
        'prediction': predicted_class,
        'model': 'Decision Tree',
        'accuracy': dt_accuracy
    }

    return jsonify(response)

@app.route('/model_accuracies', methods=['GET'])
def model_accuracies():
    # API endpoint to provide accuracies of all models
    return jsonify(models_accuracy)

# Uncomment the following line to run the Flask app if you're running this script outside this environment
# app.run(host="0.0.0.0", debug=True)
