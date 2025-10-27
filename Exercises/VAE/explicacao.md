perfeito — vou te entregar um “guia-receita” pronto pra você **entender e implementar** um VAE do zero no MNIST/Fashion-MNIST, com **código organizado em funções, muitos comentários**, e um roteirinho de **como avaliar/visualizar/entregar no GitHub Pages**.
Se quiser, você pode literalmente salvar o arquivo abaixo como `vae_mnist.py` e rodar.

---

# Visão geral do que vamos fazer

1. **Preparar dados** (MNIST/FashionMNIST, normalizar e dividir).
2. **Definir o modelo** (Encoder → μ & logσ²; reparametrização; Decoder).
3. **Treinar** (calcular `recon_loss + KL`, otimizar).
4. **Avaliar** (loss em validação; amostrar novas imagens).
5. **Visualizar** (reconstruções, amostras, e latente 2D/3D ou redução).
6. **Gerar imagens/figuras** para seu **GitHub Pages**.
7. **Checklist do relatório** (o que você precisa escrever/explicar).

---

## Como rodar

```bash
# 1) dentro do env
pip install torch torchvision matplotlib scikit-learn umap-learn

```

Arquivos gerados em `./out_vae/`:

* `epXXX_a_input.png` (originais)
* `epXXX_b_recon.png` (reconstruções)
* `epXXX_c_samples.png` (amostras novas)
* `epXXX_d_latent2d.png` (dispersão do latente em 2D, se latent_dim=2)
* `history.csv` (tabela com losses por época)
* `best.pt` (checkpoint do melhor modelo)

> Quer usar **FashionMNIST**? No topo do arquivo, mude `CFG.dataset = "fashion"`.

---

## Entendendo cada função (passo a passo, em português simples)

* `get_dataloaders()`
  baixa o dataset, normaliza para `[0,1]`, divide treino/val, e monta DataLoaders.

* `Encoder`
  transforma a imagem em dois vetores: **μ** (média) e **logσ²** (log-variância) — eles definem uma normal ( \mathcal N(\mu, \sigma^2) ) no latente.

* `VAE.reparameterize(mu, logvar)`
  implementa ( z = \mu + \sigma \cdot \varepsilon ) com (\varepsilon \sim \mathcal N(0,1)).
  Isso permite **fazer backprop** apesar da amostragem.

* `Decoder`
  pega `z` e reconstrói a imagem (com `sigmoid` se usar BCE, porque queremos saída em `[0,1]`).

* `reconstruction_loss`
  mede “o quanto a reconstrução parece com o original”.
  *BCE* é comum para MNIST; *MSE* é uma alternativa.

* `kl_divergence`
  mede “o quanto a distribuição do encoder (por amostra) está próxima da Normal padrão `N(0, I)`”.
  Isso **organiza** o espaço latente para ser contínuo e navegável.

* `train_one_epoch` / `evaluate`
  laços de treino/validação. Calculam **loss total = rec + β*KL** e atualizam pesos (no treino).

* `save_reconstructions` / `save_random_samples` / `save_latent_scatter_2d`
  salvam figuras para **seu relatório/GitHub Pages**:

  * imagens originais vs reconstruídas;
  * amostras novas (z ~ N(0,I));
  * dispersão do latente (quando 2D).

---

## Visualizações (o que colocar na página)

* **Reconstruções** (antes/depois).
* **Amostras** geradas (grelha 6×6).
* **Latente 2D** (se `latent_dim=2`), colorido por classe.

  * Se `latent_dim > 3`, gere uma **redução com t-SNE/UMAP/PCA** e plote.

---

## Avaliação: o que discutir no relatório

1. **Corretude da implementação**

   * Mostrar fórmula da loss (recon + KL).
   * Mostrar reparametrização (z = \mu + \sigma \cdot \varepsilon).

2. **Treino & validação**

   * Curvas de loss (tabela `history.csv` → gráfico simples).
   * Comentar sobre `rec_loss` vs `KL`.

3. **Amostras**

   * Qualidade visual, diversidade.
   * Alguma observação sobre ruído/artefatos.

4. **Latente**

   * Se `latent_dim=2`, dispersão “faz sentido”? As classes se agrupam?
   * Se maior, usar t-SNE/UMAP/PCA e comentar a separabilidade.

5. **Visualizações**

   * Grids limpos, legíveis e com legendas.

6. **Relato**

   * Dificuldades e soluções (ex.: explodiu gradiente? precisou reduzir `lr`?).
   * Impacto de mudar `latent_dim` (extra).
   * Comparação com **AE** simples (extra): VAE geralmente gera melhor e o latente fica mais organizado.

> **Dica de ouro**: se as amostras estiverem “lavadas”, tente **aumentar `epochs`**, reduzir `lr`, ou usar **annealing do β** (começar em 0 e ir até 1 nas primeiras épocas).

---

## GitHub Pages (entrega)

Crie uma página (MkDocs/Material ou simples HTML) com:

* **Seções**: Introdução, Metodologia, Modelo, Treino, Resultados, Visualizações, Conclusão.
* **Imagens**: use as figuras salvas em `out_vae/`.
* **Tabelas/Gráficos**: plote `history.csv` (loss vs época).
* **Citação de IA**: se usou assistência (como aqui), cite na seção de “Apoio/Referências”.

---

## Extra (opcional)

* **AE baseline**: replique o mesmo encoder/decoder **sem** μ/logσ² e **sem KL**; compare reconstrução vs geração (spoiler: AE reconstrói bem, mas “gerar do nada” é ruim).
* **Latent sweep**: com `latent_dim=2`, varra uma grade em (z₁, z₂) e gere um mosaico (mostra transições suaves).

---

se quiser, eu também te preparo um **template de página (MkDocs)** com as imagens já referenciadas e um `Makefile` para publicar no GitHub Pages. Quer que eu já te entregue esse esqueleto de docs?
