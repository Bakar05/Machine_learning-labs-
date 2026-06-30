import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Training Data
# -----------------------------
x_train = np.array([1.0, 2.0])      # Size (1000 sqft)
y_train = np.array([300.0, 500.0])  # Price (1000s of dollars)


# -----------------------------
# Cost Function
# -----------------------------
def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Input data
        y (ndarray): Target values
        w (float): Weight
        b (float): Bias

    Returns:
        total_cost (float): Cost of the model
    """

    m = x.shape[0]

    cost_sum = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost

    total_cost = cost_sum / (2 * m)

    return total_cost


# -----------------------------
# Test the Cost Function
# -----------------------------
w = 200
b = 100

cost = compute_cost(x_train, y_train, w, b)

print("Cost =", cost)


# -----------------------------
# Larger Dataset
# -----------------------------
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480, 430, 630, 730])

w = 209
b = 2.4

cost = compute_cost(x_train, y_train, w, b)

print("Cost on larger dataset =", cost)

