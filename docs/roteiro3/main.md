# Relatório - Multi-Layer Perceptrons (MLPs)

Este relatório apresenta os resultados obtidos nos exercícios 2, 3 e 4, que envolvem a implementação e avaliação de redes neurais do tipo Multi-Layer Perceptron (MLP) aplicadas a problemas de classificação binária e multiclasse.

## Exercício 1 - Forward e Backpropagation Manual

**Configuração inicial:**
- Entrada: \(x = [0.5, -0.2]\)
- Saída esperada: \(y = 1.0\)
- Pesos e vieses:
  \[
  W^{(1)} =
  \begin{bmatrix}
  0.3 & -0.1 \\
  0.2 & 0.4
  \end{bmatrix}, \quad
  b^{(1)} = [0.1, -0.2]
  \]
  \[
  W^{(2)} =
  \begin{bmatrix}
  0.5 \\
  -0.3
  \end{bmatrix}, \quad
  b^{(2)} = 0.2
  \]
- Taxa de aprendizado: \(\eta = 0.3\)  
- Função de ativação: \(\tanh\)  
- Função de custo: MSE

---

## Forward Pass
- Pré-ativação camada oculta:
  \[
  z^{(1)} = xW^{(1)} + b^{(1)} = [0.21, -0.33]
  \]
- Ativação camada oculta:
  \[
  h = \tanh(z^{(1)}) = [0.206966, -0.318521]
  \]
- Pré-ativação saída:
  \[
  z^{(2)} = hW^{(2)} + b^{(2)} = 0.399039
  \]
- Saída final:
  \[
  \hat{y} = \tanh(z^{(2)}) = 0.379127
  \]
- Função de perda:
  \[
  L = \tfrac{1}{2}(y - \hat{y})^2 = 0.192742
  \]

---

## Backpropagation
- Gradiente na saída:
  \[
  \delta^{(2)} = ( \hat{y} - y ) \cdot (1 - \tanh^2(z^{(2)})) = -0.531631
  \]
- Gradientes da camada de saída:
  \[
  \frac{\partial L}{\partial W^{(2)}} =
  \begin{bmatrix}
  -0.110030 \\
   0.169335
  \end{bmatrix}, \quad
  \frac{\partial L}{\partial b^{(2)}} = -0.531631
  \]
- Gradiente propagado para camada oculta:
  \[
  \delta^{(1)} = [-0.254429, \; 0.143308]
  \]
- Gradientes da camada oculta:
  \[
  \frac{\partial L}{\partial W^{(1)}} =
  \begin{bmatrix}
  -0.127215 & 0.071654 \\
   0.050886 & -0.028662
  \end{bmatrix}, \quad
  \frac{\partial L}{\partial b^{(1)}} = [-0.254429, 0.143308]
  \]

---

## Atualização dos Parâmetros (\(\eta = 0.3\))
- Novos pesos e vieses após atualização:

\[
W^{(2)}_{\text{new}} =
\begin{bmatrix}
0.533009 \\
-0.350801
\end{bmatrix}, \quad
b^{(2)}_{\text{new}} = 0.359489
\]

\[
W^{(1)}_{\text{new}} =
\begin{bmatrix}
0.338164 & -0.121496 \\
0.184734 & 0.408598
\end{bmatrix}, \quad
b^{(1)}_{\text{new}} = [0.176329, -0.242992]
\]

---

## Forward Após Atualização
- Nova saída:
  \[
  \hat{y}' = 0.570172
  \]
- Novo erro:
  \[
  L' = 0.092376
  \]

---

## Conclusão
Após uma única atualização de pesos e vieses, a perda caiu de **0.1927 para 0.0924**, mostrando que o processo de backpropagation e gradient descent ajustou corretamente os parâmetros, aproximando a saída \(\hat{y}\) do valor esperado \(y\).


---

## Exercício 2 - Classificação Binária

**Arquitetura**: 2 neurônios de entrada, 1 camada oculta e 1 neurônio de saída (ativação `tanh`).

- **Curva de Loss**:  
  ![Loss Ex2](loss_ex2.png)

  Observa-se uma redução constante da função de perda, estabilizando próximo de 0.54 após 300 épocas.

- **Matriz de Confusão**:  
  ![CM Ex2](cm_ex2.png)

  - Classe 0: 90 acertos, 9 erros.  
  - Classe 1: 136 acertos, 57 erros.

- **Métricas principais**:

  | Classe | Precisão | Recall | F1-score |
  |--------|----------|--------|----------|
  | 0      | 0.6122   | 0.9091 | 0.7317   |
  | 1      | 0.9379   | 0.7047 | 0.8047   |

  **Acurácia total: 77.4%**

---

## Exercício 3 - Classificação em 3 Classes

**Arquitetura**: 2 neurônios de entrada, 1 camada oculta com mais unidades, saída com 3 neurônios (ativação `tanh`).

- **Curva de Loss**:  
  ![Loss Ex3](loss_ex3.png)

  A perda cai até aproximadamente 0.56, mostrando boa convergência.

- **Matriz de Confusão**:  
  ![CM Ex3](cm_ex3.png)

  - Classe 0: 85 acertos, 15 erros.  
  - Classe 1: 70 acertos, 30 erros.  
  - Classe 2: 74 acertos, 26 erros.  

- **Métricas principais**:

  | Classe | Precisão | Recall | F1-score |
  |--------|----------|--------|----------|
  | 0      | 0.7944   | 0.8500 | 0.8213   |
  | 1      | 0.8140   | 0.7000 | 0.7527   |
  | 2      | 0.6916   | 0.7400 | 0.7150   |

  **Acurácia total: 76.3%**

---

## Exercício 4 - MLP Mais Profundo (3 Classes)

**Arquitetura**: semelhante ao Ex3, mas com **duas camadas ocultas**, aumentando a profundidade da rede.

- **Curva de Loss**:  
  ![Loss Ex4](loss_ex4.png)

  O modelo converge mais rápido e atinge valores menores de perda (~0.44).

- **Matriz de Confusão**:  
  ![CM Ex4](cm_ex4.png)

  - Classe 0: 91 acertos, 9 erros.  
  - Classe 1: 75 acertos, 25 erros.  
  - Classe 2: 78 acertos, 22 erros.  

- **Métricas principais**:

  | Classe | Precisão | Recall | F1-score |
  |--------|----------|--------|----------|
  | 0      | 0.8505   | 0.9100 | 0.8792   |
  | 1      | 0.8065   | 0.7500 | 0.7772   |
  | 2      | 0.7800   | 0.7800 | 0.7800   |

  **Acurácia total: 81.3%**

---

## Comparativo Geral

- **Curvas de Loss (Ex2, Ex3, Ex4)**:  
  ![Loss Comparativa](loss_curves_all.png)

- O modelo mais simples (Ex2) já obteve desempenho razoável (77%), mas com desequilíbrio entre classes.  
- O Ex3 ampliou a rede para 3 classes, mantendo acurácia próxima de 76%.  
- O Ex4, com arquitetura mais profunda, apresentou o melhor resultado (81%), além de convergência mais rápida.  

**Conclusão**: o aumento de profundidade melhorou a capacidade de generalização da rede, reduzindo a perda final e aumentando a acurácia geral.

---
