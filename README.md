# 💳 SmartLoan — Sistema Inteligente de Análise de Crédito

> Projeto completo de Machine Learning e MLOps para análise de risco de crédito, com deploy em API (FastAPI), dashboard interativo (Streamlit) e versionamento automatizado (DVC + MLflow).

---

## 📘 Sumário
1. [Visão Geral](#visão-geral)
2. [Arquitetura do Projeto](#arquitetura-do-projeto)
3. [Funcionalidades](#funcionalidades)
4. [Tecnologias Utilizadas](#tecnologias-utilizadas)
5. [Pipeline de Machine Learning](#pipeline-de-machine-learning)
6. [MLOps e Versionamento](#mlops-e-versionamento)
7. [Como Executar o Projeto](#como-executar-o-projeto)
8. [API (FastAPI)](#api-fastapi)
9. [Dashboard (Streamlit)](#dashboard-streamlit)
10. [Estrutura de Pastas](#estrutura-de-pastas)
11. [Autor](#autor)

---

## 🧠 Visão Geral
O **SmartLoan** é um sistema automatizado de análise de crédito que prevê o **risco de inadimplência** de um cliente com base em dados históricos e características socioeconômicas.  
Utiliza **modelos supervisionados** (como *Logistic Regression* e *XGBoost*) para classificar o cliente como **“bom pagador”** ou **“mau pagador”**, e fornece uma **pontuação de risco** em tempo real via API.

---

## 🏗️ Arquitetura do Projeto

```bash
SmartLoan_Project/
│
├── data/
│   ├── raw/              # Dados brutos
│   ├── processed/        # Dados limpos e prontos para modelagem
│   └── features/         # Features geradas no pipeline
│
├── notebooks/            # Experimentos e análises exploratórias
│   └── EDA.ipynb
│
├── src/
│   ├── data_prep/        # Limpeza, tratamento e feature engineering
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── feature_engineering.py
│   │
│   ├── models/           # Treinamento e avaliação dos modelos
│   │   ├── train_model.py
│   │   └── evaluate.py
│   │
│   ├── mlops/            # MLflow e DVC
│   │   ├── tracking.py
│   │   └── versioning.py
│   │
│   ├── api/              # FastAPI
│   │   └── main.py
│   │
│   └── dashboard/        # Streamlit Dashboard
│       └── app.py
│
├── requirements.txt      # Dependências do projeto
├── dvc.yaml              # Pipeline de versionamento de dados/modelos
├── mlflow.db             # Banco de tracking MLflow
├── Dockerfile            # Imagem do projeto
├── .gitignore
└── README.md
````

---

## ⚙️ Funcionalidades

* 🔍 **Análise de Crédito** — Previsão de inadimplência usando dados históricos.
* 📊 **Comparação de Modelos** — Avaliação de diferentes algoritmos (Logistic Regression, Random Forest, XGBoost).
* 🧩 **Feature Engineering** — Geração de variáveis derivadas (renda, idade, histórico, etc.).
* 🔁 **MLOps Automatizado** — MLflow + DVC para rastrear experimentos e versionar datasets e modelos.
* 🚀 **API FastAPI** — Predições em tempo real via endpoint `/predict`.
* 📈 **Dashboard Streamlit** — Visualização de métricas, curvas ROC e importância das features.
* 🧪 **A/B Testing** — Testes de performance entre modelos em produção.

---

## 🧰 Tecnologias Utilizadas

| Categoria                | Ferramentas              |
| ------------------------ | ------------------------ |
| **Linguagem**            | Python 3.10              |
| **Machine Learning**     | scikit-learn, XGBoost    |
| **Manipulação de Dados** | pandas, numpy            |
| **Visualização**         | matplotlib, seaborn      |
| **MLOps**                | DVC, MLflow              |
| **API**                  | FastAPI                  |
| **Dashboard**            | Streamlit                |
| **Infraestrutura**       | Docker                   |
| **Versionamento**        | Git + GitHub             |
| **Outros**               | joblib, pyyaml, requests |

---

## 🔄 Pipeline de Machine Learning

1. **Coleta e Limpeza de Dados**

   * Tratamento de valores ausentes e outliers.
   * Normalização e codificação de variáveis.

2. **Feature Engineering**

   * Criação de variáveis derivadas de histórico financeiro.
   * Aplicação de *scaling* e *one-hot encoding*.

3. **Treinamento dos Modelos**

   * Modelos testados: `LogisticRegression`, `RandomForest`, `XGBoost`.
   * Avaliação via métricas: AUC, Accuracy, Recall, F1-Score.

4. **Comparação e Seleção de Modelos**

   * Tracking automático com **MLflow**.
   * Melhor modelo salvo em `models/best_model.joblib`.

5. **Deploy e Monitoramento**

   * API via **FastAPI**.
   * Dashboard via **Streamlit**.
   * Versionamento de datasets e modelos com **DVC**.

---

## 🧩 MLOps e Versionamento

### 🔹 DVC (Data Version Control)

Usado para versionar datasets e modelos:

```bash
dvc init
dvc add data/processed/dataset.csv
dvc add models/best_model.joblib
git add .dvc .gitignore
git commit -m "Versionamento de dados e modelo"
```

### 🔹 MLflow (Tracking de Experimentos)

Registra métricas, parâmetros e artefatos:

```bash
mlflow run .
mlflow ui
```

Interface disponível em: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧠 Como Executar o Projeto

### 1️⃣ Clonar o repositório

```bash
git clone https://github.com/cl4y70n/smartloan-project.git
cd smartloan-project
```

### 2️⃣ Criar e ativar o ambiente virtual

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4️⃣ Rodar o treinamento

```bash
python src/models/train_model.py
```

### 5️⃣ Executar a API

```bash
uvicorn src.api.main:app --reload
```

### 6️⃣ Acessar o Dashboard

```bash
streamlit run src/dashboard/app.py
```

---

## ⚡ API (FastAPI)

### Endpoint: `/predict`

**Método:** `POST`
**Exemplo de JSON:**

```json
{
  "idade": 35,
  "renda_mensal": 5000,
  "historico_credito": 3,
  "dividas_atuais": 2000,
  "numero_dependentes": 1
}
```

**Resposta:**

```json
{
  "risco_inadimplencia": "baixo",
  "probabilidade": 0.12
}
```

---

## 📊 Dashboard (Streamlit)

Exibe:

* Curvas ROC/AUC dos modelos.
* Comparação de métricas.
* Importância das variáveis.
* Histórico de experimentos MLflow.

Acesso local:

```bash
streamlit run src/dashboard/app.py
```

---

## 🗂️ Estrutura de Pastas Resumida

```
SmartLoan_Project/
├── data/
├── src/
│   ├── data_prep/
│   ├── models/
│   ├── mlops/
│   ├── api/
│   └── dashboard/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 👨‍💻 Autor

**Clayton Júnior (cl4y70n)**
📍 Desenvolvedor e Engenheiro de IA
📧 Contato: claytonjunioe334@gmail.com
🌐 GitHub: [github.com/cl4y70n](https://github.com/cl4y70n)

---

⭐ *Se gostou do projeto, deixe uma estrela no repositório!* ⭐

```

---

Quer que eu gere esse `README.md` automaticamente e te envie o **arquivo pronto para colar na pasta do projeto** (ou em formato `.zip`)?
```
