import numpy as np

np.set_printoptions(precision=6, suppress=True)

# ----------------------------
# PARÂMETROS DO EXERCÍCIO
# ----------------------------
x = np.array([[0.5, -0.2]])   # shape (1,2)
y = np.array([[1.0]])           # shape (1,1)

W1 = np.array([[0.3, -0.1],     # shape (2,2)
               [0.2,  0.4]])
b1 = np.array([[0.1, -0.2]])    # shape (1,2)

W2 = np.array([[ 0.5],          # shape (2,1)
               [-0.3]])
b2 = np.array([[0.2]])          # shape (1,1)

eta = 0.3                        # learning rate
# ativação: tanh
# ----------------------------


def tanh(z):
    return np.tanh(z)

def dtanh(z):
    t = np.tanh(z)
    return 1.0 - t*t

def mse(y_true, y_pred):
    return 0.5 * np.mean((y_true - y_pred)**2)

def fmt(a):
    if isinstance(a, np.ndarray):
        return np.array2string(a, formatter={'float_kind':lambda v: f"{v:.6f}"})
    return f"{a:.6f}"

# ---------- FORWARD ----------
z1 = x @ W1 + b1          # (1,2)
h  = tanh(z1)             # (1,2)
z2 = h @ W2 + b2          # (1,1)
y_hat = tanh(z2)          # (1,1)
L = mse(y, y_hat)

print("=== FORWARD ===")
print("x       =", fmt(x))
print("W1      =", fmt(W1))
print("b1      =", fmt(b1))
print("z1=xW1+b1       =", fmt(z1))
print("h=tanh(z1)      =", fmt(h))
print("W2      =", fmt(W2))
print("b2      =", fmt(b2))
print("z2=hW2+b2       =", fmt(z2))
print("y_hat=tanh(z2)  =", fmt(y_hat))
print("Loss L=0.5*(y - y_hat)^2 =", fmt(L))
print()

# ---------- BACKPROP ----------
dL_dyhat = (y_hat - y)                 # (1,1)
dyhat_dz2 = dtanh(z2)                  # (1,1)
delta2 = dL_dyhat * dyhat_dz2          # (1,1)

# Gradientes W2, b2
dW2 = h.T @ delta2                      # (2,1)
db2 = delta2                            # (1,1)

# Propaga para oculta:
delta1 = (delta2 @ W2.T) * dtanh(z1)    # (1,2)

# Gradientes W1, b1
dW1 = x.T @ delta1                      # (2,2)
db1 = delta1                            # (1,2)

print("=== BACKPROP ===")
print("dL/dy_hat          =", fmt(dL_dyhat))
print("d(y_hat)/dz2       =", fmt(dyhat_dz2))
print("delta2=dL/dz2      =", fmt(delta2))
print("dW2 = h^T @ delta2 =", fmt(dW2))
print("db2 = delta2       =", fmt(db2))
print("delta1=dL/dz1      =", fmt(delta1))
print("dW1 = x^T @ delta1 =", fmt(dW1))
print("db1 = delta1       =", fmt(db1))
print()

# ---------- UPDATE (gradient descent) ----------
W2_new = W2 - eta * dW2
b2_new = b2 - eta * db2
W1_new = W1 - eta * dW1
b1_new = b1 - eta * db1

print("=== UPDATE (eta = {:.4f}) ===".format(eta))
print("W2_new =", fmt(W2_new))
print("b2_new =", fmt(b2_new))
print("W1_new =", fmt(W1_new))
print("b1_new =", fmt(b1_new))
print()

# ---------- (opcional) Recalcula forward com pesos atualizados ----------
z1_n = x @ W1_new + b1_new
h_n  = tanh(z1_n)
z2_n = h_n @ W2_new + b2_new
y_hat_n = tanh(z2_n)
L_n = mse(y, y_hat_n)

print("=== FORWARD APÓS UPDATE (opcional) ===")
print("y_hat' =", fmt(y_hat_n))
print("L'     =", fmt(L_n))
