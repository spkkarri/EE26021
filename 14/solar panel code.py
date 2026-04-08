import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'irradiance': [200, 400, 600, 800, 1000, 300, 500, 700, 900],
    'temperature': [25, 30, 35, 40, 45, 28, 33, 38, 42],
    'power': [50, 120, 200, 280, 350, 80, 160, 240, 310]
}

df = pd.DataFrame(data)

# Input and output
X = df[['irradiance', 'temperature']]
y = df['power']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Print results
print("Actual:", y_test.values)
print("Predicted:", y_pred)

# Test with new data
new_data = pd.DataFrame([[850, 37]], columns=['irradiance', 'temperature'])
predicted_power = model.predict(new_data)

print("Predicted Solar Power Output:", predicted_power[0])

# ================= GRAPH 1: Actual vs Predicted =================
plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred, s=80, alpha=0.7)

# Ideal line (perfect prediction)
plt.plot([min(y_test), max(y_test)],
         [min(y_test), max(y_test)],
         linestyle='--')

plt.xlabel("Actual Power Output (W)", fontsize=12)
plt.ylabel("Predicted Power Output (W)", fontsize=12)
plt.title("Solar Power Prediction: Actual vs Predicted", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()


# ================= GRAPH 2: Error Analysis =================
errors = y_test - y_pred

plt.figure(figsize=(8,5))

plt.bar(range(len(errors)), errors)
plt.axhline(0)

plt.xlabel("Test Sample Index")
plt.ylabel("Error (Actual - Predicted)")
plt.title("Prediction Error Analysis")
plt.grid(True)
plt.tight_layout()
plt.show()
