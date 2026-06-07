import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Save model as pickle file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")

# Load model
with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Make prediction
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = loaded_model.predict(sample)

print("Prediction:", prediction)

# ### Recommended Approach
# import joblib

# # Save
# joblib.dump(model, "model.pkl")

# # Load
# loaded_model = joblib.load("model.pkl")