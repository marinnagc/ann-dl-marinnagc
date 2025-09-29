# mlp_scratch.py
import numpy as np

# --------- Ativações ---------
def tanh(x): return np.tanh(x)
def dtanh(x): 
    t = np.tanh(x); 
    return 1.0 - t*t

def sigmoid(x): return 1/(1+np.exp(-x))
def dsigmoid(x): 
    s = sigmoid(x)
    return s*(1-s)

def relu(x): return np.maximum(0, x)
def drelu(x): return (x > 0).astype(x.dtype)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

# --------- Perdas ---------
def mse_loss(y_true, y_pred):  # y_pred já é ativação final
    return 0.5*np.mean((y_true - y_pred)**2)

def bce_loss(y_true, y_hat):   # binário: y in {0,1}, y_hat in (0,1)
    # estabilidade numérica
    eps = 1e-12
    y_hat = np.clip(y_hat, eps, 1-eps)
    return -np.mean(y_true*np.log(y_hat) + (1-y_true)*np.log(1-y_hat))

def ce_loss(y_true_onehot, p): # multiclasse: one-hot e softmax(p)
    eps = 1e-12
    p = np.clip(p, eps, 1-eps)
    return -np.mean(np.sum(y_true_onehot*np.log(p), axis=1))

# --------- Camada densa ---------
class Dense:
    def __init__(self, in_dim, out_dim, act='tanh', seed=42):
        rng = np.random.default_rng(seed)
        # Xavier/He simples conforme ativação
        if act == 'relu':
            scale = np.sqrt(2/in_dim)
        else:
            scale = np.sqrt(1/in_dim)
        self.W = rng.normal(0, scale, size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))
        self.z = None
        self.a = None
        self._act = act
        self._act_f, self._dact_f = {
            'tanh': (tanh, dtanh),
            'sigmoid': (sigmoid, dsigmoid),
            'relu': (relu, drelu),
            'linear': (lambda x:x, lambda x: np.ones_like(x))
        }[act]

    def fwd(self, x):
        self.z = x @ self.W + self.b
        self.a = self._act_f(self.z)
        return self.a

    def back(self, x, grad_out, lr):
        # grad_out é dL/da (do próximo estágio)
        dz = grad_out * self._dact_f(self.z)  # dL/dz
        dW = x.T @ dz / x.shape[0]
        db = dz.mean(axis=0, keepdims=True)
        grad_in = dz @ self.W.T            # dL/dx (para camada anterior)
        # atualização GD
        self.W -= lr * dW
        self.b -= lr * db
        return grad_in

# --------- MLP ---------
class MLP:
    def __init__(self, layers, task='binary', lr=0.1, seed=42):
        """
        layers: lista de tuplas (dim, act), incluindo a saída.
          Ex.: [(2,'tanh'), (8,'tanh'), (1,'sigmoid')] -> binário
               [(4,'relu'), (16,'relu'), (3,'linear+softmax')] -> multiclasse
        task: 'binary' (sigmoid + BCE) ou 'multiclass' (softmax + CE)
        """
        self.lr = lr
        self.task = task
        self.layers = []
        for i in range(len(layers)-1):
            in_dim,_ = layers[i]
            out_dim,act = layers[i+1]
            # truque: saída multiclasse usa 'linear' + softmax fora
            act = 'linear' if (i == len(layers)-2 and task=='multiclass') else act
            self.layers.append(Dense(in_dim, out_dim, act=act, seed=seed+i))

    def _forward(self, X):
        a = X
        caches = [X]
        for L in self.layers:
            a = L.fwd(a)
            caches.append(a)
        if self.task == 'multiclasse' or self.task == 'multiclass':
            a = softmax(a)
        return a, caches

    def fit(self, X, y, epochs=200, verbose=False):
        history = []
        for ep in range(1, epochs+1):
            y_hat, caches = self._forward(X)
            # perdas e gradiente inicial
            if self.task == 'binary':
                loss = bce_loss(y.reshape(-1,1), y_hat)
                # dL/da (saída): (y_hat - y)/(y_hat*(1-y_hat)) * dBCE/dlogit? Não.
                # Para BCE com sigmoid na última camada linear: melhor usar cadeia explícita:
                # Aqui a última camada já aplicou sigmoid (porque definimos 'sigmoid'); então:
                grad = (y_hat - y.reshape(-1,1))  # dL/dz para sigmoid+BCE
            else:
                # y é inteiro [0..K-1] -> vira one-hot
                K = y_hat.shape[1]
                y_one = np.eye(K)[y.astype(int)]
                loss = ce_loss(y_one, y_hat)
                grad = (y_hat - y_one) / X.shape[0] * X.shape[0]  # dL/dz softmax+CE

            history.append(loss)

            # Backprop: varre camadas de trás pra frente
            grad_in = grad
            for i in reversed(range(len(self.layers))):
                x_in = caches[i]  # entrada daquela camada
                grad_in = self.layers[i].back(x_in, grad_in, self.lr)

            if verbose and (ep % max(1, epochs//10) == 0):
                print(f"epoch {ep}/{epochs} - loss {loss:.4f}")
        return np.array(history)

    def predict_proba(self, X):
        y_hat, _ = self._forward(X)
        return y_hat

    def predict(self, X):
        y_hat = self.predict_proba(X)
        if self.task == 'binary':
            return (y_hat.ravel() >= 0.5).astype(int)
        return np.argmax(y_hat, axis=1)
