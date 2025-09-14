Ótimo, agora tenho os prints do enunciado com os parâmetros exatos 👌. Vou te explicar **cada etapa da implementação** que você precisa fazer, relacionando com os exercícios 1 e 2.

---

## 🔹 1. Geração dos dados

### Exercício 1

* Classe 0:

  * Média = \[1.5, 1.5]
  * Covariância = \[\[0.5, 0], \[0, 0.5]] (variância baixa → pontos mais concentrados)
* Classe 1:

  * Média = \[5, 5]
  * Covariância = \[\[0.5, 0], \[0, 0.5]]

👉 Resultado: pontos bem separados (linearmente separáveis, pouca sobreposição).

### Exercício 2

* Classe 0:

  * Média = \[3, 3]
  * Covariância = \[\[1.5, 0], \[0, 1.5]] (variância maior → pontos espalhados)
* Classe 1:

  * Média = \[4, 4]
  * Covariância = \[\[1.5, 0], \[0, 1.5]]

👉 Resultado: pontos se sobrepõem mais → não dá para separar perfeitamente com uma linha reta.

### Código típico (vale para ambos):

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 1000

# Exemplo do Ex1
mean0, cov0 = [1.5, 1.5], [[0.5, 0], [0, 0.5]]
mean1, cov1 = [5, 5], [[0.5, 0], [0, 0.5]]

X0 = np.random.multivariate_normal(mean0, cov0, n)
X1 = np.random.multivariate_normal(mean1, cov1, n)

X = np.vstack([X0, X1])
y = np.hstack([np.zeros(n), np.ones(n)])
```

E plota:

```python
plt.scatter(X0[:,0], X0[:,1], label="Classe 0")
plt.scatter(X1[:,0], X1[:,1], label="Classe 1")
plt.legend(); plt.show()
```

---

## 🔹 2. Implementação do Perceptron

### Estrutura:

* Pesos `w = [w1, w2]` e viés `b`.
* Inicializados aleatoriamente ou zerados.
* Função de decisão:
  $y_{pred} = \text{step}(w \cdot x + b)$

### Regra de atualização:

Para cada ponto `(x, y)`:

* Calcula predição `ŷ`.
* Se errou (`ŷ != y`), atualiza:

  $$
  w = w + \eta \cdot (y - \hat{y}) \cdot x
  $$

  $$
  b = b + \eta \cdot (y - \hat{y})
  $$
* Onde `η` é a taxa de aprendizado (comece com 0.01).

---

## 🔹 3. Treinamento

* Loop por **épocas** (até 100).
* Em cada época, percorre todos os exemplos.
* Conta quantas atualizações foram feitas:

  * Se **nenhuma** → convergiu.
* Guarda a acurácia após cada época para plotar curva.

---

## 🔹 4. Avaliação

* Depois do treino:

  * Acurácia final = nº de acertos ÷ total.
  * Plote:

    * **Limite de decisão**:
      Se w2 ≠ 0:

      $$
      x2 = -\frac{w1}{w2} \cdot x1 - \frac{b}{w2}
      $$
    * **Curva de acurácia** (época × acurácia).
    * **Pontos errados** com “X” ou cor diferente.

---

## 🔹 5. Interpretação dos resultados

* **Exercício 1**
  Dados separáveis → perceptron converge rápido (100% de acurácia).
  Pesos finais definem uma linha clara entre as classes.

* **Exercício 2**
  Dados sobrepostos → perceptron não chega a 100% de acurácia.
  Pode oscilar entre épocas.
  Mesmo assim encontra um limite razoável que separa “o grosso” dos dados.

---

## 🔹 Fluxo completo resumido

1. Gerar dados com `np.random.multivariate_normal`.
2. Plotar nuvem de pontos.
3. Inicializar perceptron (`w, b`).
4. Loop de treino com regra de atualização.
5. Calcular acurácia por época.
6. Plotar:

   * Dados + linha de decisão.
   * Curva de acurácia.
   * Pontos errados.
7. Reportar: `w, b, acurácia final` + discussão.

---

Quer que eu escreva **um código único**, que roda **Ex1 e Ex2** já com esses parâmetros do enunciado, plota tudo e mostra pesos/acurácia final no console?
