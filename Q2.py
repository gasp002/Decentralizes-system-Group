from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize Flask application
app = Flask(__name__)

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier and fit the model
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Evaluate the model's performance
y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

def make_prediction(model, features):
    """
    This function receives a trained model and a list of features to make a prediction.
    """
    prediction = model.predict([features])[0]
    predicted_class = iris.target_names[prediction]
    return predicted_class

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['GET'])
def predict():
    # Extract query parameters for model prediction
    # For simplicity, we're expecting all four feature values to be provided
    try:
        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid input parameters'}), 400

    # Make prediction using the encapsulated function
    features = [sepal_length, sepal_width, petal_length, petal_width]
    predicted_class = make_prediction(dt_classifier, features)

    # Standard API response
    response = {
        'prediction': predicted_class,
        'confidence': None,  # If needed, calculate the confidence of the prediction
        'accuracy': accuracy
    }

    return jsonify(response)

# Uncomment the following line to run the Flask app if you're running this script outside this environment
# app.run(host="0.0.0.0", debug=True)
