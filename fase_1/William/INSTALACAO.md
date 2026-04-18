# ============================================================================
# INSTALACAO DE DEPENDENCIAS
# ============================================================================

## Opcao 1: Usando requirements.txt (Recomendado)

### Para Producao (minimo)
```bash
pip install -r requirements-prod.txt
```
Tamanho: ~500 MB
Tempo: ~3-5 min
Uso: Para rodar modelos treinados em producao

### Para Desenvolvimento (completo)
```bash
pip install -r requirements-dev.txt
```
Tamanho: ~2 GB (incluindo PyTorch)
Tempo: ~10-15 min
Uso: Para desenvolvimento, testes, notebooks

### Sem pin de versões (latest)
```bash
pip install pandas numpy scikit-learn torch mlflow fastapi pytest jupyter
```

---

## Opcao 2: Usando pyproject.toml (Modern Python)

### Para Producao
```bash
pip install .
```

### Para Desenvolvimento
```bash
pip install -e ".[dev,ml]"
```

### Explicacao
- `pip install .` instala em modo producao
- `pip install -e .` instala em modo editable (dev)
- `[dev,ml]` instala grupos opcionais

---

## Opcao 3: Setup Completo com Makefile

```bash
make install          # Instala requirements-dev.txt
make install-prod     # Instala apenas producao
make lint             # Verifica codigo
make test             # Roda testes
```

---

## Dependencias por Categoria

### Core Data Science
- pandas: DataFrames
- numpy: Computacao numerica
- scikit-learn: ML classicos
- scipy: Funcoes cientificas

### Deep Learning
- torch: PyTorch (redes neurais)
- torchvision: Utilidades visao

### MLOps
- mlflow: Rastreamento experimentos

### API & Web
- fastapi: Framework API
- uvicorn: ASGI server
- pydantic: Validacao dados

### Desenvolvimento
- pytest: Testing framework
- black: Code formatter
- ruff: Linter (rapido)
- mypy: Type checking
- jupyter: Notebooks

### Analise Avancada
- fairlearn: Fairness analysis
- optuna: Hyperparameter tuning
- shap: Feature importance

### Tree Models
- xgboost: Gradient boosting
- lightgbm: Light gradient boosting

---

## Verificando Instalacao

### Listar pacotes instalados
```bash
pip list | grep -E "pandas|torch|mlflow|scikit"
```

### Testar imports
```bash
python -c "import pandas; import torch; import mlflow; print('OK')"
```

### Versoes criticas
```bash
python -c "
import pandas as pd
import numpy as np
import sklearn
import torch
import mlflow

print(f'pandas: {pd.__version__}')
print(f'numpy: {np.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
print(f'torch: {torch.__version__}')
print(f'mlflow: {mlflow.__version__}')
"
```

---

## Troubleshooting

### Erro: "No module named torch"
```bash
# PyTorch geralmente precisa ser instalado separadamente
pip install torch --index-url https://download.pytorch.org/whl/cpu
# Ou com CUDA se tiver GPU
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Erro: "Microsoft Visual C++ 14.0 is required" (Windows)
```bash
# Baixe Visual Studio Build Tools em:
# https://visualstudio.microsoft.com/downloads/
# Selete "Desktop development with C++"
```

### Erro: "command not found: make" (Windows)
```bash
# Use Python para rodar scripts diretamente
python 02_train_baselines.py
# ou
jupyter notebook
```

### Descrepancia de versoes
```bash
# Force compatibilidade
pip install --upgrade -r requirements.txt
```

---

## Tamanho Esperado

| Opcao | Tamanho | Tempo |
|-------|---------|-------|
| prod | ~500 MB | 3-5 min |
| dev | ~2 GB | 10-15 min |
| CPU | ~1.5 GB | 5-10 min |
| GPU | ~3 GB | 15-20 min |

---

## Update de Dependencias

### Verificar desatualizado
```bash
pip list --outdated
```

### Update tudo
```bash
pip install -U -r requirements-dev.txt
```

### Gerar requirements atualizado
```bash
pip freeze > requirements-latest.txt
```

---

## Ambientes Virtuais

### Criar novo ambiente
```bash
python -m venv .venv
```

### Ativar ambiente
```bash
# Linux/Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Instalar neste ambiente
```bash
pip install -r requirements-dev.txt
```

---

## Docker (Bonus)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY . .

CMD ["gunicorn", "src.api.main:app"]
```

---

## Proximas Etapas

Depois de instalar, teste:

```bash
# 1. Script baseline
python 02_train_baselines.py

# 2. Notebook
jupyter notebook notebooks/02_baseline_models.ipynb

# 3. MLflow
mlflow ui
```

---

**Dicas**: Para CI/CD, use `requirements-prod.txt`. Para desenvolvimento local, use `requirements-dev.txt`.
