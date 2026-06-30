import numpy as np
import time

# =========================
# Vector Creation
# =========================

a = np.zeros(4)
print(a, a.shape, a.dtype)

a = np.zeros((4,))
print(a, a.shape, a.dtype)

a = np.random.random_sample(4)
print(a, a.shape, a.dtype)

a = np.arange(4.)
print(a, a.shape, a.dtype)

a = np.random.rand(4)
print(a, a.shape, a.dtype)

a = np.array([5, 4, 3, 2])
print(a, a.shape, a.dtype)

a = np.array([5., 4, 3, 2])
print(a, a.shape, a.dtype)

# =========================
# Indexing & Slicing (1D)
# =========================

a = np.arange(10)
print(a)

print(a[2])
print(a[-1])

try:
    print(a[10])
except Exception as e:
    print(e)

a = np.arange(10)
print(a)

print(a[2:7:1])
print(a[2:7:2])
print(a[3:])
print(a[:3])
print(a[:])

# =========================
# Vector Operations
# =========================

a = np.array([1, 2, 3, 4])

print(-a)
print(np.sum(a))
print(np.mean(a))
print(a**2)

b = np.array([-1, -2, 3, 4])
print(a + b)

c = np.array([1, 2])
try:
    print(a + c)
except Exception as e:
    print(e)

print(5 * a)

# =========================
# Dot Product (Manual)
# =========================

def my_dot(a, b):
    x = 0
    for i in range(a.shape[0]):
        x += a[i] * b[i]
    return x

a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])

print(my_dot(a, b))

# =========================
# NumPy Dot Product
# =========================

print(np.dot(a, b))
print(np.dot(b, a))

# =========================
# Speed Comparison
# =========================

np.random.seed(1)

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()

print("Vectorized np.dot:", c)
print("Time:", (toc - tic) * 1000, "ms")

tic = time.time()
c = my_dot(a, b)
toc = time.time()

print("Loop dot:", c)
print("Time:", (toc - tic) * 1000, "ms")

del a, b

# =========================
# Matrices (2D)
# =========================

a = np.zeros((1, 5))
print(a, a.shape)

a = np.zeros((2, 1))
print(a, a.shape)

a = np.random.random_sample((1, 1))
print(a, a.shape)

a = np.array([[5], [4], [3]])
print(a, a.shape)

# =========================
# Matrix Indexing
# =========================

a = np.arange(6).reshape(-1, 2)
print(a)

print(a[2, 0])
print(a[2])

# =========================
# Matrix Slicing
# =========================

a = np.arange(20).reshape(-1, 10)
print(a)

print(a[0, 2:7:1])
print(a[:, 2:7:1])
print(a[:, :])
print(a[1, :])
print(a[1])
