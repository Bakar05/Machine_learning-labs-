import copy
import math
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Training Data
# ==========================

X_train = np.array([
    [2104, 5, 1, 45],
    [1416, 3, 2, 40],
    [852, 2, 1, 35]
])

y_train = np.array([460, 232, 178])

# ==========================
# Prediction Function
# ==========================

def predict(x, w, b):
    return np.dot(x, w) + b


# ==========================
# Cost Function
# ==========================

def compute_cost(X, y, w, b):

    m = X.shape[0]
    cost = 0

    for i in range(m):

        prediction = np.dot(X[i], w) + b
        cost += (prediction - y[i]) ** 2

    return cost / (2 * m)


# ==========================
# Compute Gradient
# ==========================

def compute_gradient(X, y, w, b):

    m, n = X.shape

    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):

        err = (np.dot(X[i], w) + b) - y[i]

        for j in range(n):
            dj_dw[j] += err * X[i, j]

        dj_db += err

    dj_dw /= m
    dj_db /= m

    return dj_db, dj_dw


# ==========================
# Gradient Descent
# ==========================

def gradient_descent(
    X,
    y,
    w_in,
    b_in,
    alpha,
    iterations
):

    w = copy.deepcopy(w_in)
    b = b_in

    J_history = []

    for i in range(iterations):

        dj_db, dj_dw = compute_gradient(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(X, y, w, b)
        J_history.append(cost)

        if i % math.ceil(iterations / 10) == 0:
            print(f"Iteration {i:4d}: Cost {cost:.2f}")

    return w, b, J_history


# ==========================
# Training
# ==========================

initial_w = np.zeros(4)
initial_b = 0

alpha = 5e-7
iterations = 1000

w_final, b_final, J_history = gradient_descent(
    X_train,
    y_train,
    initial_w,
    initial_b,
    alpha,
    iterations
)

print("\nFinal Parameters")
print("----------------")
print("Weights:", w_final)
print("Bias:", b_final)

# ==========================
# Prediction Example
# ==========================

new_house = np.array([1200, 3, 1, 40])

prediction = predict(new_house, w_final, b_final)

print(f"\nPredicted price: ${prediction*1000:,.0f}")

# ==========================
# Plot Learning Curve
# ==========================

plt.figure(figsize=(8,5))

plt.plot(J_history, linewidth=2)

plt.title("Learning Curve")
plt.xlabel("Iterations")
plt.ylabel("Cost")

plt.grid(True)

plt.tight_layout()

plt.savefig("output.png")

plt.show()
