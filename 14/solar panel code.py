import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("solar-data.csv")

# Show dataset preview
print("Dataset Preview:\n", data.head())

# Features (input) and target (output)
X = data[["Temp.", "hours", "humidity", "w.speed"]]
y = data["p.output"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Results
print("\nActual values:", list(y_test))
print("Predicted values:", list(y_pred))

# Performance
print("\nModel Performance:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Test with new input
sample = pd.DataFrame([[30, 8, 40, 10]], columns=["Temp.", "hours", "humidity", "w.speed"])
prediction = model.predict(sample)

print("\nSample Input: Temperature=30, Sunlight=8, Humidity=40, Wind=10")
print("Predicted Power Output:", prediction[0])
