# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:17:51 2024

@author: romai
"""
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display basic information and first few rows of the dataset
iris_info = iris_df.info()
iris_head = iris_df.head()

# Plot pairplot to visualize relationships between features
sns.pairplot(iris_df, hue="species")
plt.show()

(iris_info, iris_head)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Splitting the data into training and testing sets
X = iris_df.iloc[:, :-1]  # Features - sepal length, sepal width, petal length, petal width
y = iris_df['species']    # Labels - species

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 70% training, 30% testing

# Training the Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Predicting on the test set
y_pred = random_forest.predict(X_test)

# Evaluation
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print (classification_rep, confusion_mat)
