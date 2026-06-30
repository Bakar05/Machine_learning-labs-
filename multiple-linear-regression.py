import copy
import math
import numpy as np

# =========================
# Data
# =========================

X_train = np.array([
    [2104, 5, 1, 45],
    [1416, 3, 2, 40],
    [852, 2, 1, 35]
])

y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])

print("X shape:", X_train.shape)
print("y shape:", y_train.shape)
print("w shape:", w_init.shape)

# =========================
# Prediction (loop version)
# =========================

def predict_single_loop(x, w, b):
    n = x.shape[0]
    p = 0
    for i in range(n):
        p += x[i] * w[i]
    return p + b

x_vec = X_train[0]
print("Loop prediction:", predict_single_loop(x_vec, w_init, b_init))

# =========================
# Prediction (vectorized)
# =========================

def predict(x, w, b):
    return np.dot(x, w) + b

print("Dot prediction:", predict(x_vec, w_init, b_init))

# =========================
# Cost Function
# =========================

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0

    for i in range(m):
        f_wb = np.dot(X[i], w) + b
        cost += (f_wb - y[i]) ** 2

    return cost / (2 * m)

print("Initial cost:", compute_cost(X_train, y_train, w_init, b_init))

# =========================
# Gradient Computation
# =========================

def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]

        for j in range(n):
            dj_dw[j] += err * X[i, j]

        dj_db += err

    dj_dw /= m
    dj_db /= m

    return dj_db, dj_dw

db, dw = compute_gradient(X_train, y_train, w_init, b_init)
print("Gradient b:", db)
print("Gradient w:", dw)

# =========================
# Gradient Descent
# =========================

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    w = copy.deepcopy(w_in)
    b = b_in

    J_history = []

    for i in range(num_iters):

        dj_db, dj_dw = compute_gradient(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(X, y, w, b)
        J_history.append(cost)

        if i % max(1, num_iters // 10) == 0:
            print(f"Iteration {i}: Cost {cost:.2f}")

    return w, b, J_history

# =========================
# Train Model
# =========================

initial_w = np.zeros_like(w_init)
initial_b = 0.0

iterations = 1000
alpha = 5.0e-7

w_final, b_final, J_hist = gradient_descent(
    X_train, y_train,
    initial_w, initial_b,
    alpha, iterations
)

print("\nFinal parameters:")
print("b:", b_final)
print("w:", w_final)

# =========================
# Predictions after training
# =========================

m = X_train.shape[0]

for i in range(m):
    pred = np.dot(X_train[i], w_final) + b_final
    print(f"Prediction: {pred:.2f}, Target: {y_train[i]}")
