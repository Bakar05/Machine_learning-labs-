import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)


def zscore_normalize_features(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm


def predict(X, w, b):
    return np.dot(X, w) + b


def compute_cost(X, y, w, b):
    m = X.shape[0]
    predictions = predict(X, w, b)
    cost = np.sum((predictions - y) ** 2)
    return cost / (2 * m)


def compute_gradient(X, y, w, b):
    m = X.shape[0]
    predictions = predict(X, w, b)
    error = predictions - y
    dj_dw = (1 / m) * np.dot(X.T, error)
    dj_db = (1 / m) * np.sum(error)
    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, alpha, iterations):
    w = w_in.copy()
    b = b_in
    J_history = []

    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        cost = compute_cost(X, y, w, b)
        J_history.append(cost)

        if i % max(1, iterations // 10) == 0:
            print(f"Iteration {i:6d}: Cost = {cost:.6f}")

    print("\nTraining Complete")
    print("Weights:", w)
    print("Bias:", b)
    return w, b


# =====================================================
# Example 1: Linear Regression without Feature Engineering
# =====================================================

print("=" * 60)
print("Example 1: Linear Regression without Feature Engineering")
print("=" * 60)

x = np.arange(0, 20, 1)
y = 1 + x ** 2
X = x.reshape(-1, 1)

initial_w = np.zeros(X.shape[1])
initial_b = 0

model_w, model_b = gradient_descent(
    X, y,
    initial_w,
    initial_b,
    iterations=1000,
    alpha=1e-2
)

plt.figure(figsize=(6, 4))
plt.scatter(x, y, marker='x', c='r', label='Actual Value')
plt.plot(x, X @ model_w + model_b, label='Predicted Value')
plt.title("No Feature Engineering")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# Example 2: Polynomial Feature x²
# =====================================================

print("\n" + "=" * 60)
print("Example 2: Polynomial Feature x²")
print("=" * 60)

x = np.arange(0, 20, 1)
y = 1 + x ** 2
X = x ** 2
X = X.reshape(-1, 1)

initial_w = np.zeros(X.shape[1])
initial_b = 0

model_w, model_b = gradient_descent(
    X, y,
    initial_w,
    initial_b,
    iterations=10000,
    alpha=1e-5
)

plt.figure(figsize=(6, 4))
plt.scatter(x, y, marker='x', c='r', label='Actual Value')
plt.plot(x, np.dot(X, model_w) + model_b, label='Predicted Value')
plt.title("Feature Engineering (x²)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# Example 3: Multiple Polynomial Features
# =====================================================

print("\n" + "=" * 60)
print("Example 3: Multiple Polynomial Features (x, x², x³)")
print("=" * 60)

x = np.arange(0, 20, 1)
y = x ** 2
X = np.c_[x, x ** 2, x ** 3]

initial_w = np.zeros(X.shape[1])
initial_b = 0

model_w, model_b = gradient_descent(
    X, y,
    initial_w,
    initial_b,
    iterations=10000,
    alpha=1e-7
)

plt.figure(figsize=(6, 4))
plt.scatter(x, y, marker='x', c='r', label='Actual Value')
plt.plot(x, X @ model_w + model_b, label='Predicted Value')
plt.title("Features: x, x², x³")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# Example 4: Visualizing Individual Features
# =====================================================

print("\n" + "=" * 60)
print("Example 4: Visualizing Individual Features")
print("=" * 60)

x = np.arange(0, 20, 1)
y = x ** 2
X = np.c_[x, x ** 2, x ** 3]
feature_names = ["x", "x²", "x³"]

fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

for i in range(3):
    ax[i].scatter(X[:, i], y)
    ax[i].set_xlabel(feature_names[i])

ax[0].set_ylabel("y")
plt.tight_layout()
plt.show()

# =====================================================
# Example 5: Z-Score Normalization
# =====================================================

print("\n" + "=" * 60)
print("Example 5: Z-Score Normalization")
print("=" * 60)

x = np.arange(0, 20, 1)
X = np.c_[x, x ** 2, x ** 3]

print("Peak-to-Peak Range (Original)")
print(np.ptp(X, axis=0))

X_norm = zscore_normalize_features(X)

print("\nPeak-to-Peak Range (Normalized)")
print(np.ptp(X_norm, axis=0))

# =====================================================
# Example 6: Polynomial Regression with Feature Scaling
# =====================================================

print("\n" + "=" * 60)
print("Example 6: Polynomial Regression with Feature Scaling")
print("=" * 60)

x = np.arange(0, 20, 1)
y = x ** 2
X = np.c_[x, x ** 2, x ** 3]
X = zscore_normalize_features(X)

initial_w = np.zeros(X.shape[1])
initial_b = 0

model_w, model_b = gradient_descent(
    X, y,
    initial_w,
    initial_b,
    iterations=100000,
    alpha=1e-1
)

plt.figure(figsize=(6, 4))
plt.scatter(x, y, marker='x', c='r', label='Actual Value')
plt.plot(x, X @ model_w + model_b, label='Predicted Value')
plt.title("Normalized Polynomial Features")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# Example 7: Modeling a Complex Function
# =====================================================

print("\n" + "=" * 60)
print("Example 7: Modeling a Complex Function (Cosine)")
print("=" * 60)

x = np.arange(0, 20, 1)
y = np.cos(x / 2)

X = np.c_[
    x,
    x ** 2,
    x ** 3,
    x ** 4,
    x ** 5,
    x ** 6,
    x ** 7,
    x ** 8,
    x ** 9,
    x ** 10,
    x ** 11,
    x ** 12,
    x ** 13
]

X = zscore_normalize_features(X)

initial_w = np.zeros(X.shape[1])
initial_b = 0

model_w, model_b = gradient_descent(
    X, y,
    initial_w,
    initial_b,
    iterations=1000000,
    alpha=1e-1
)

plt.figure(figsize=(6, 4))
plt.scatter(x, y, marker='x', c='r', label='Actual Value')
plt.plot(x, X @ model_w + model_b, label='Predicted Value')
plt.title("Polynomial Regression on Cosine Function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
