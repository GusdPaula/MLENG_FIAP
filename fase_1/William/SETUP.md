# 🚀 SETUP RAPIDO - TELCO CHURN PREDICTION

## Inicio Rapido (5 minutos)

### Opcao 1: Com Make (Linux/Mac/Windows)
```bash
cd William

# Setup completo + instalar
make install-full

# Rodar tudo
make run-all

# Ver MLflow
make mlflow-ui
```

### Opcao 2: Com pip direto (Windows/Linux/Mac)
```bash
cd William

# 1. Criar ambiente virtual
python -m venv .venv

# 2. Ativar
# Linux/Mac:
source .venv/bin/activate
# Windows:
.\.venv\Scripts\activate

# 3. Instalar (escolha uma)

# Opcao A: Apenas producao
pip install -r requirements-prod.txt

# Opcao B: Desenvolvimento completo
pip install -r requirements-dev.txt

# Opcao C: Usando pyproject.toml
pip install -e ".[dev,ml]"

# 4. Testar
python -c "import pandas, torch, mlflow; print('OK')"

# 5. Rodar baselines
python 02_train_baselines.py

# 6. Ver MLflow
mlflow ui
```

---

## O que Cada requirements.txt Contém?

### `requirements-prod.txt` (minimo - 500 MB)
Para rodar modelos em producao:
- pandas, numpy, scikit-learn
- torch (PyTorch)
- mlflow
- fastapi, uvicorn
- xgboost, lightgbm

**Use se**: Vai rodar SO modelos treinados, nao escrever codigo

### `requirements-dev.txt` (completo - 2 GB)
Para desenvolvimento:
- Tudo de producao +
- pytest, pytest-cov
- black, ruff, mypy
- jupyter, jupyterlab
- fairlearn, optuna, shap

**Use se**: Vai desenvolver, testar, experimentar

### `requirements.txt` (intermediario)
Versao intermedia (sem CI tools avancados)

**Use se**: Quer package basico + dev tools

---

## Passo a Passo Completo

### 1. Clone o Repositorio (ja feito?)
```bash
cd /c/Users/Will/dev/postech/postech-ml-projeto/MLENG_FIAP/fase_1/William
```

### 2. Criar Ambiente Virtual
```bash
python -m venv .venv
```

### 3. Ativar Ambiente
```bash
# Windows
.\.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

Deve aparecer `(.venv)` no inicio do prompt

### 4. Upgrade pip e ferramentas
```bash
python -m pip install --upgrade pip setuptools wheel
```

### 5. Instalar Dependencias
```bash
# Opcao recomendada (completo)
pip install -r requirements-dev.txt

# Ou apenas producao
pip install -r requirements-prod.txt
```

Tempo esperado: 10-15 minutos (primeira vez)

### 6. Verificar Instalacao
```bash
python -c "
import pandas as pd
import numpy as np
import torch
import mlflow
import sklearn
print('Versoes:')
print(f'  pandas: {pd.__version__}')
print(f'  numpy: {np.__version__}')
print(f'  torch: {torch.__version__}')
print(f'  scikit-learn: {sklearn.__version__}')
print(f'  mlflow: {mlflow.__version__}')
print('STATUS: OK!')
"
```

### 7. Rodar Baselines
```bash
python 02_train_baselines.py
```

Saida esperada:
```
======================================================================
ETAPA 3: TREINAMENTO DE BASELINES COM MLFLOW
======================================================================
[OK] Dataset carregado: 7043 linhas x 33 colunas
[OK] DummyClassifier treinado
[OK] LogisticRegression treinado
[OK] Treinamento concluido com sucesso!
======================================================================
```

### 8. Abrir Notebook
```bash
jupyter notebook notebooks/02_baseline_models.ipynb
```

### 9. Visualizar MLflow
```bash
mlflow ui
```
Acesse: http://localhost:5000

---

## Usando Makefile (Dica Pro)

```bash
# Ver todos comandos
make help

# Instalar (escolha 1)
make install                 # Via pyproject.toml
make install-prod            # Apenas producao
make install-full            # Desarrollo completo

# Executar
make run-baselines           # Treinar baselines
make run-eda                 # Rodar EDA
make run-all                 # EDA + Baselines
make run-notebook            # Abrir Jupyter

# Qualidade de codigo
make lint                    # Verificar codigo
make format                  # Formatar codigo
make test                    # Rodar testes
make clean                   # Limpar cache

# MLflow
make mlflow-ui              # Abrir dashboard
```

---

## Resolucao de Problemas

### Erro: "No module named torch"
```bash
# PyTorch nao instalou corretamente
# Reinstale:
pip install torch --force-reinstall
# Ou com GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Erro: "pip: command not found" (Mac/Linux)
```bash
# Use pip3
pip3 install -r requirements-dev.txt
```

### Erro: "Python 3.10+ required"
```bash
# Verifique versao
python --version
python3 --version
# Use a versao correta (3.10, 3.11, ou 3.12)
python3.10 -m venv .venv
```

### Erro: "Microsoft Visual C++ is required" (Windows)
```bash
# Alguns packages precisam compilar
# Baixe Visual Studio Build Tools:
# https://visualstudio.microsoft.com/downloads/
# Selecione "Desktop development with C++"
```

### Erro: "Permission denied" (Linux/Mac)
```bash
# Talvez precise chmod
chmod +x 02_train_baselines.py
# Ou use sudo (nao recomendado)
sudo pip install -r requirements-dev.txt
```

---

## Ambiente Diferente por Maquina

### Windows PowerShell
```powershell
# Ativar venv
.\.venv\Scripts\Activate.ps1

# Se der erro de permissao:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Windows CMD
```cmd
.\.venv\Scripts\activate.bat
```

### Linux/Mac Bash/Zsh
```bash
source .venv/bin/activate
```

### Verificar ativo
```bash
which python   # Deve mostrar .venv/bin/python
# ou
pip list | head -5
```

---

## Cidades Diferentes

### Para CPU Apenas (Recomendado se sem GPU)
```bash
pip install torch torchvision torchaudio
```

### Para GPU NVIDIA (CUDA 12.1)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Para GPU NVIDIA (CUDA 11.8)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Para GPU Apple Silicon (Mac M1/M2)
```bash
# MacOS com conda eh melhor:
conda install pytorch torchvision torchaudio -c pytorch
```

---

## Usar Sem Sistema Virtual (Nao Recomendado)

```bash
# Instalar globalmente
pip install -r requirements-dev.txt

# Pode conflitar com outros projetos!
# Sempre use venv para ML projects
```

---

## Checklist de Setup

- [ ] Python 3.10+ instalado
- [ ] cd para pasta William
- [ ] Criou .venv
- [ ] Ativou .venv
- [ ] Rodou pip install
- [ ] Testou imports (python -c ...)
- [ ] Rodou python 02_train_baselines.py
- [ ] Abriu mlflow ui
- [ ] Abriu notebook

Se tudo [✓], voce esta pronto para:
- [x] Etapa 1 & 2: EDA + ML Canvas (COMPLETO)
- [x] Etapa 3: Baselines (COMPLETO)
- [ ] Etapa 3 cont: RandomForest + XGBoost + MLP
- [ ] Etapa 4: API + Docker
- [ ] Etapa 5: Documentacao + Video

---

## Proximos Comandos

```bash
# Depois de setup, rodar:

# 1. Script Python
python 02_train_baselines.py

# 2. Jupyter Notebook
jupyter notebook notebooks/02_baseline_models.ipynb

# 3. MLflow Dashboard
mlflow ui
# Abrir http://localhost:5000

# 4. Rodar testes (quando tiver testes)
pytest tests/

# 5. Verificar qualidade de codigo
make lint
make format
```

---

## Tamanho Esperado & Tempo

| Step | Size | Time |
|------|------|------|
| Python install | - | 2 min |
| Create venv | 50 MB | <1 min |
| pip install-prod | 500 MB | 5 min |
| pip install-dev | 2 GB | 15 min |
| Train baselines | - | 2 min |

**Total com dev**: ~25 minutos (primeira vez)

---

## Support

Se tiver problema:
1. Verifique Python >= 3.10: `python --version`
2. Verifique venv ativo: `which python` (Linux/Mac)
3. Tente `pip install --upgrade pip`
4. Delete .venv e comece denovo
5. Leia INSTALACAO.md para mais detalhes

---

**Status**: Pronto para Desenvolvimento ML 🚀
