# 🔍 Interpretabilidade em CNNs para Detecção de Imagens Sintéticas (CIFAKE + XAI)

Este repositório contém o código-fonte e os artefatos de pesquisa do projeto de Iniciação Científica focado na detecção forense e interpretação de imagens geradas por Inteligência Artificial (Modelos de Difusão Latente), utilizando técnicas de **Inteligência Artificial Explicável (XAI)**.

## 📖 Sobre o Projeto

A evolução dos Modelos de Difusão Latente, como o *Stable Diffusion*, tem impulsionado a geração de imagens sintéticas ultrarrealistas. Embora promissora, essa tecnologia suscita sérios riscos éticos e de segurança associados à disseminação de *deepfakes*. Diferente de abordagens tradicionais de detecção — que focam no desempenho estatístico global e frequentemente aprendem vieses do conjunto de dados (CORVI et al., 2023) —, este projeto propõe uma análise diagnóstica direcionada às **assinaturas de alta frequência** inerentes ao processo de geração.

O objetivo central é investigar, mapear e interpretar a exata natureza morfológica e espectral que diferencia fotografias reais de imagens artificiais. Para isso, propõe-se o treinamento de uma rede **All Convolutional Net (All-CNN-C)** com o *dataset* **CIFAKE**, escolhido por preservar integralmente os micro-ruídos da difusão.

Para mitigar a natureza de "caixa-preta" do modelo, aplicamos três métodos complementares de explicabilidade (*post-hoc*):
1. **Grad-CAM** (*Gradient-weighted Class Activation Mapping*)
2. **Guided Backpropagation**
3. **Deconvolution**

## 📂 Estrutura do Repositório

O projeto segue uma arquitetura modular inspirada em boas práticas de *Data Science* para garantir a reprodutibilidade metodológica:

```text
cifake-xai-detection/
├── data/                      # ⚠️ DIRETÓRIO IGNORADO PELO GIT
│   ├── raw/                   # Imagens originais (CIFAKE - 120k imagens 32x32)
│   └── processed/             # Imagens normalizadas para treinamento (média 0, variância 1)
├── models/                    # ⚠️ DIRETÓRIO IGNORADO PELO GIT
│   └── saved_weights/         # Pesos salvos da All-CNN-C (.keras / .h5)
├── notebooks/                 # Ambientes Jupyter para experimentação e EDA
├── reports/                   # Documentação, figuras e o Catálogo Analítico de Assinaturas
│   └── figures/               # Matrizes de confusão e Curvas ROC/AUC
├── src/                       # Código-fonte principal em Python
│   ├── data_prep.py           # Scripts de ingestão e normalização
│   ├── train.py               # Treinamento da rede preditiva e Grid Search
│   ├── evaluate.py            # Avaliação de métricas preditivas (Acurácia, F1-Score, Recall)
│   ├── architecture/          # Definição matemática da All-CNN-C (ausência de Max-Pooling)
│   └── xai/                   # Implementações post-hoc (Grad-CAM, etc.) e auditorias quantitativas
├── .gitignore                 # Arquivos e diretórios ocultos (bloqueio de dados locais)
├── requirements.txt           # Dependências e bibliotecas do ecossistema Python
└── README.md                  # Documentação principal do projeto
```

## 📊 Conjunto de Dados (CIFAKE)

O projeto utiliza o conjunto **CIFAKE** (Bird & Lotfi, 2023), composto por 120.000 imagens balanceadas com restrição espacial de `32x32` pixels:
* **60.000 Imagens Reais:** Extraídas diretamente do repositório clássico CIFAR-10.
* **60.000 Imagens Sintéticas:** Geradas via *Stable Diffusion*.

> **Nota metodológica:** A dimensão reduzida de 32x32 obriga o modelo preditivo a descartar grande parte do contexto semântico macroscópico, focando em artefatos espectrais e estatísticos (ruídos residuais de difusão), evitando que a rede aprenda anomalias transitórias (como membros deformados).

## 🚀 Como Executar o Projeto

**1. Clone o repositório para o seu ambiente local:**
```bash
git clone https://github.com/KaykyTajima/cifake-xai-detection.git
cd cifake-xai-detection
```

**2. Instale as dependências (recomenda-se o uso de um ambiente virtual):**
```bash
pip install -r requirements.txt
```

**3. Obtenção dos Dados:**
* Baixe o conjunto de dados CIFAKE via [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images).
* Extraia e posicione as imagens dentro do diretório `data/raw/`.
* *Atenção:* O arquivo `.gitignore` já está configurado para **não** subir as imagens e pesos pesados para a nuvem.

## 🔬 Auditoria Quantitativa (Métricas de XAI)

Este projeto supera a avaliação visual subjetiva aplicando um rigoroso protocolo de auditoria matemática para avaliar a **fidelidade explicativa** dos filtros:

* **Sanity Checks (Testes de Sanidade):** Randomização progressiva dos pesos (*cascading randomization*) para descartar métodos que funcionem apenas como detectores de bordas genéricos.
* **Curvas de Deleção/Inserção (AUC):** Avaliação quantitativa preditiva baseada na oclusão estritamente no nível do pixel, medindo o decréscimo ou recuperação da confiança do modelo.
* **ROAR (Remove and Retrain):** Validação algorítmica global que envolve a re-inicialização e re-treinamento da arquitetura em bases adulteradas, garantindo mitigação do viés *Out-of-Distribution* (OOD).

---
