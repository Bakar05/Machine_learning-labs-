import numpy as np
import matplotlib.pyplot as plt

# If you have the style file, uncomment the next line
# plt.style.use('./deeplearning.mplstyle')

# ==============================
# Training Data
# ==============================

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

print("Training Data")
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# ==============================
# Number of Training Examples
# ==============================

print("\nTraining Examples")
print(f"x_train.shape = {x_train.shape}")

m = x_train.shape[0]
print(f"Number of training examples = {m}")

# Alternative method
m = len(x_train)
print(f"Number of training examples (using len) = {m}")

# ==============================
# Access Individual Training Example
# ==============================

i = 0

x_i = x_train[i]
y_i = y_train[i]

print(f"\nTraining Example {i}")
print(f"(x^{i}, y^{i}) = ({x_i}, {y_i})")

# ==============================
# Plot Training Data
# ==============================

plt.figure(figsize=(6,4))

plt.scatter(x_train, y_train,
            marker='x',
            color='red',
            s=100)

plt.title("Housing Prices")
plt.xlabel("Size (1000 sqft)")
plt.ylabel("Price (1000s of dollars)")

plt.grid(True)
plt.show()

# ==============================
# Model Parameters
# ==============================

w = 500
b = 300

print("\nInitial Parameters")
print(f"w = {w}")
print(f"b = {b}")

# ==============================
# Model Function
# ==============================

def compute_model_output(x, w, b):
    """
    Computes predictions of linear regression model

    Parameters:
        x : ndarray
            Input data
        w : float
            Weight
        b : float
            Bias

    Returns:
        f_wb : ndarray
            Predicted values
    """

    m = x.shape[0]

    f_wb = np.zeros(m)

    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb

# ==============================
# Compute Predictions
# ==============================

tmp_f_wb = compute_model_output(x_train, w, b)

print("\nPredictions")
print(tmp_f_wb)

# ==============================
# Plot Prediction vs Actual Data
# ==============================

plt.figure(figsize=(6,4))

plt.plot(x_train,
         tmp_f_wb,
         color='blue',
         label='Model Prediction')

plt.scatter(x_train,
            y_train,
            marker='x',
            color='red',
            s=100,
            label='Actual Data')

plt.title("Housing Prices")
plt.xlabel("Size (1000 sqft)")
plt.ylabel("Price (1000s of dollars)")

plt.legend()
plt.grid(True)

plt.show()

# ==============================
# Best Parameters
# ==============================

w = 200
b = 100

print("\nBest Parameters")
print(f"w = {w}")
print(f"b = {b}")

# Compute predictions again

tmp_f_wb = compute_model_output(x_train, w, b)

plt.figure(figsize=(6,4))

plt.plot(x_train,
         tmp_f_wb,
         color='green',
         linewidth=2,
         label='Best Fit Line')

plt.scatter(x_train,
            y_train,
            marker='x',
            color='red',
            s=100,
            label='Actual Data')

plt.title("Best Fit Linear Regression")
plt.xlabel("Size (1000 sqft)")
plt.ylabel("Price (1000s of dollars)")

plt.legend()
plt.grid(True)

plt.show()

# ==============================
# Predict Price of 1200 sqft House
# ==============================

x_new = 1.2

predicted_price = w * x_new + b

print("\nPrediction")
print(f"Price of 1200 sqft house = ${predicted_price:.0f} thousand dollars")
