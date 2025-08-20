# Iris Dataset Classification using KNN

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)

# Define KNN model
knn = KNeighborsClassifier(n_neighbors=7, weights="distance")

# Train model
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Evaluation
print("Model Used: K-Nearest Neighbors")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
print("\nMetrics:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# User Input
def predict_iris_species():
    print("\nEnter flower measurements to predict Iris species:")
    sepal_length = float(input("Sepal Length (cm): "))
    sepal_width  = float(input("Sepal Width (cm): "))
    petal_length = float(input("Petal Length (cm): "))
    petal_width  = float(input("Petal Width (cm): "))

    sample = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = knn.predict(sample)[0]
    print(f"\nPredicted Species: {iris.target_names[prediction]}")

# Call the function
predict_iris_species()