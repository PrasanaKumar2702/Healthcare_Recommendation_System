
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the test data and trained model
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
model = load_model("healthcare_recommendation_model.h5")

# Reshape test data for LSTM input
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model performance
mse = model.evaluate(X_test, y_test)
print(f"Mean Squared Error: {mse}")
