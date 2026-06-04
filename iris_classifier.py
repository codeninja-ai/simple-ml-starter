"""
Simple ML Example: Iris Flower Classification
This script demonstrates basic machine learning concepts:
1. Loading data
2. Splitting data into training and testing sets
3. Training a machine learning model
4. Making predictions
5. Evaluating model performance
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the Iris dataset (comes built-in with scikit-learn)
print("Loading Iris dataset...")
iris = datasets.load_iris()
X = iris.data  # Features (flower measurements)
y = iris.target  # Labels (flower species)

print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {iris.target_names}")
print()

# Step 2: Split data into training (80%) and testing (20%) sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print()

# Step 3: Create and train the model
print("Training the Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete!")
print()

# Step 4: Make predictions on the test set
print("Making predictions on test data...")
y_pred = model.predict(X_test)
print()

# Step 5: Evaluate the model
print("=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")
print()

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print()

# Bonus: Feature importance
print("=" * 50)
print("FEATURE IMPORTANCE")
print("=" * 50)
feature_importance = model.feature_importances_
for name, importance in zip(iris.feature_names, feature_importance):
    print(f"{name}: {importance:.4f}")
print()

# Bonus: Make a prediction on a sample
print("=" * 50)
print("SAMPLE PREDICTION")
print("=" * 50)
sample_flower = X_test[0].reshape(1, -1)
prediction = model.predict(sample_flower)[0]
prediction_proba = model.predict_proba(sample_flower)[0]

print(f"Sample measurements: {X_test[0]}")
print(f"Predicted species: {iris.target_names[prediction]}")
print(f"Confidence scores: {dict(zip(iris.target_names, prediction_proba))}")
