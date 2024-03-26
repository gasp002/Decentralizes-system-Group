from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

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
def train_knn(iris):
    X = iris.data[:, :1]  # Use only the first feature, sepal length to make the model accuracy less than 1
    y = iris.target
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return knn_classifier, accuracy

# Function to train and evaluate a linear regression model
def train_linear_regression(iris):
    X = iris.data[:, :1]  # Use only the first feature, sepal length
    y = iris.target

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Logistic Regression model
    model = LogisticRegression(max_iter=200)  # Increase max_iter if needed
    model.fit(X_train, y_train)

    # Predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Save the trained model
    joblib.dump(model, 'iris_logistic_regression_model_one_feature.joblib')

    return model, accuracy

# Train and store the models and their accuracies
dt_model, dt_accuracy = train_decision_tree(X_train, y_train, X_test, y_test)
knn_model, knn_accuracy = train_knn(iris)
lr_model, lr_accuracy = train_linear_regression(iris)

# Dictionary to store models, their accuracies, initial deposit, and weights
# instead of complex calculationb for weight we just link it to the model's accuracy
models = {
    'Decision Tree': {'model': dt_model, 'accuracy': dt_accuracy, 'deposit': 1000, 'weight': dt_accuracy},
    'KNN': {'model': knn_model, 'accuracy': knn_accuracy, 'deposit': 1000, 'weight': knn_accuracy},
    'Logistic Regression': {'model': lr_model, 'accuracy': lr_accuracy, 'deposit': 1000, 'weight': lr_accuracy}
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
    
    features = [sepal_length, sepal_width, petal_length, petal_width]
    single_feature = [sepal_length]  # We used the sepal_length for training
    predictions = []
    for model_name, model_info in models.items():
        if model_name in ['Logistic Regression', 'KNN']:
            # For Logistic Regression and KNN, we use only the first feature for less than perfect accuracy
            predicted_class_index = model_info['model'].predict([single_feature])[0]
        else:
            # For the Decision Tree model, we use all features for perfect accuracy
            predicted_class_index = model_info['model'].predict([features])[0]
        predictions.append(iris.target_names[predicted_class_index])
    
    consensus_prediction = max(set(predictions), key=predictions.count)

    # Implement the slashing mechanism
    for model_name, model_info in models.items():
        # Make prediction using the current model
        if model_name in ['Logistic Regression', 'KNN']:
            predicted_class_index = model_info['model'].predict([[sepal_length]])[0]
        else:
            predicted_class_index = model_info['model'].predict([features])[0]
        model_prediction = iris.target_names[predicted_class_index]
        
        # Check if the model's prediction matches the consensus
        if model_prediction != consensus_prediction:
            # Slash the model's balance if its prediction differs from the consensus
            slash_amount = 100  # Define the amount to be slashed
            model_info['deposit'] -= slash_amount
            # Ensure the deposit does not become negative
            model_info['deposit'] = max(model_info['deposit'], 0)

    # Updated part of the response to include the new balances
    response = {
        'consensus_prediction': consensus_prediction,
        'individual_predictions': predictions,
        'models_info': {model: {'accuracy': info['accuracy'], 
                                'deposit': info['deposit'], 
                                'weight': info['weight']} 
                        for model, info in models.items()}
    }

    return jsonify(response)

@app.route('/model_accuracies', methods=['GET'])
def model_accuracies():
    # API endpoint to provide accuracies of all models
    return jsonify({model: info['accuracy'] for model, info in models.items()})

# Run the Flask app
app.run(host="0.0.0.0", debug=True)
