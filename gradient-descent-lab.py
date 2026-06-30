import math
import numpy as np
import matplotlib.pyplot as plt

# =========================
# DATASET
# =========================
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

# =========================
# COST FUNCTION
# =========================
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i])**2

    total_cost = (1 / (2 * m)) * cost
    return total_cost

# =========================
# GRADIENT FUNCTION
# =========================
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        error = f_wb - y[i]

        dj_dw += error * x[i]
        dj_db += error

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

# =========================
# GRADIENT DESCENT
# =========================
def gradient_descent(x, y, w_in, b_in, alpha, num_iters):

    J_history = []
    w = w_in
    b = b_in

    for i in range(num_iters):

        dj_dw, dj_db = compute_gradient(x, y, w, b)

        # simultaneous update
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(x, y, w, b)
        J_history.append(cost)

        if i % (num_iters // 10) == 0:
            print(f"Iteration {i:4}: Cost {cost:.4f}, w {w:.4f}, b {b:.4f}")

    return w, b, J_history

# =========================
# TRAIN MODEL
# =========================
w_init = 0
b_init = 0
iterations = 1000
alpha = 0.01

w_final, b_final, J_history = gradient_descent(
    x_train,
    y_train,
    w_init,
    b_init,
    alpha,
    iterations
)

print("\nFinal parameters:")
print("w =", w_final)
print("b =", b_final)

# =========================
# PREDICTIONS
# =========================
def predict(x):
    return w_final * x + b_final

print("\nPredictions:")
print("1000 sqft:", predict(1.0))
print("1200 sqft:", predict(1.2))
print("2000 sqft:", predict(2.0))

# =========================
# COST PLOT
# =========================
plt.plot(J_history)
plt.title("Cost vs Iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()
