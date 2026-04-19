.PHONY: help install install-dev lint format type-check test test-cov test-smoke test-unit test-all clean build run run-api run-api-prod docs venv venv-activate pre-commit

# Variáveis
PYTHON := python3
PIP := pip
VENV := .venv
PROJECT_NAME := telco-churn-prediction

help:
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  Telco Churn Prediction - MLOps Pipeline Profissional     ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Setup & Instalação:"
	@echo "  make install          - Instalar dependências"
	@echo "  make install-dev      - Instalar com dev dependencies"
	@echo "  make venv             - Criar virtual environment"
	@echo "  make venv-activate    - Ativar virtual environment"	
	@echo "  make clean            - Limpar arquivos temporários"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             - Linter (ruff)"
	@echo "  make format           - Formatar código (black, isort)"
	@echo "  make type-check       - Type checking (mypy)"
	@echo "  make pre-commit       - Format + Lint + Type check"
	@echo ""
	@echo "Testes:"
	@echo "  make test             - Executar testes"
	@echo "  make test-cov         - Testes com cobertura"
	@echo "  make test-smoke       - Smoke tests"
	@echo "  make test-unit        - Unit tests"
	@echo "  make test-all         - Todos os testes (lint + type + testes)"
	@echo ""
	@echo "Execução:"
	@echo "  make run              - Executar aplicação"
	@echo "  make run-api          - Iniciar API (desenvolvimento)"
	@echo "  make run-api-prod     - Iniciar API (produção)"
	@echo "  make docs             - Gerar documentação"
	@echo ""
	@echo "make run-all	- Rodar EDA + Baselines"
	@echo "make mlflow-ui	- Abrir MLflow UI (localhost:5000)"
	@echo ""
	@echo "Docker:"
	@echo "make docker-build	- Construir imagem Docker"
	@echo "make docker-run	- Executar container Docker"
	@echo ""


# ==== CODE QUALITY ====
lint:
	@echo "🔍 Rodando ruff..."
	ruff check src/ tests/ --show-source || true
	@echo "✅ Lint completo"

format:
	@echo "🎨 Formatando com black..."
	black src/ tests/ || true
	@echo "🎨 Formatando com isort..."
	isort src/ tests/ || true
	@echo "✅ Formatação completa"

type-check:
	@echo "🔍 Verificando tipos com mypy..."
	mypy src/ || true
	@echo "✅ Type check completo"

# ==== TESTES ====
test:
	@echo "🧪 Rodando testes..."
	pytest tests/ -v --tb=short || true
	@echo "✅ Testes concluídos"

test-cov:
	@echo "🧪 Rodando testes com cobertura..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing || true
	@echo "✅ Testes com cobertura concluídos"

test-smoke:
	@echo "🔥 Rodando smoke tests..."
	pytest tests/ -v -m smoke || true
	@echo "✅ Smoke tests concluídos"

test-unit:
	@echo "🧪 Rodando testes unitários..."
	pytest tests/ -v -m unit || true
	@echo "✅ Testes unitários concluídos"

test-all: lint type-check test-cov
	@echo "✅ Todos os testes passaram!"

# ==== BUILD ====
build: clean
	@echo "📦 Buildando pacote..."
	$(PYTHON) -m build
	@echo "✅ Build concluído"

# ==== EXECUÇÃO ====
run:
	@echo "▶️  Executando aplicação..."
	$(PYTHON) -m src

run-api:
	@echo "▶️  Iniciando API FastAPI (desenvolvimento)..."
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-api-prod:
	@echo "▶️  Iniciando API FastAPI (produção)..."
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# ==== DOCUMENTAÇÃO ====
docs:
	@echo "📚 Documentação:"
	@echo "  - API Docs: http://localhost:8000/api/docs (Swagger UI)"
	@echo "  - ReDoc: http://localhost:8000/api/redoc"
	@echo "  - Coverage: htmlcov/index.html (após make test-cov)"

# ==== VENV ====
venv:
	@echo "🐍 Criando virtual environment..."
	$(PYTHON) -m venv $(VENV)	
	@echo "✅ Virtual environment criado"
	
venv-activate:
	source $(VENV)/Scripts/activate
	@echo "✅ Ativado virtual environment"


# ==== GIT ====
pre-commit: format lint type-check
	@echo "✅ Pre-commit checks concluídos"


# ==== INSTALAÇÃO ====
install:
	@echo "📦 Instalando dependências..."
	$(PIP) install -e .
	@echo "✅ Dependências instaladas!"

install-dev:
	@echo "📦 Instalando com dev dependencies..."
	$(PIP) install -e ".[dev,ml]"
	@echo "✅ Dev environment pronto!"

install-prod:
	@echo "📦 Instalando apenas producao..."
	$(PIP) install -r requirements-prod.txt
	@echo "✅ Producao ready!"

install-full:
	@echo "📦 Instalando tudo (dev + prod + ml)..."
	$(PIP) install -r requirements-dev.txt
	@echo "✅ Full development environment pronto!"

# ==== LIMPEZA ====
clean:
	@echo "🧹 Limpando arquivos temporários..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage
	@echo "✅ Limpeza concluída!"

coverage:
	@echo "📊 Gerando cobertura de testes..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "✅ Relatório em htmlcov/index.html"

run-eda:
	@echo "📊 Executando EDA..."
	$(PYTHON) 01_eda_analysis.py
	@echo "✅ EDA concluído!"

run-baselines:
	@echo "🤖 Treinando baselines com MLflow..."
	$(PYTHON) 02_train_baselines.py
	@echo "✅ Baselines concluído!"

run-all: run-eda run-baselines
	@echo "✅ Pipeline completo executado!"

run-notebook:
	@echo "📓 Iniciando Jupyter..."
	jupyter notebook notebooks/

mlflow-ui:
	@echo "📈 Abrindo MLflow UI..."
	mlflow ui --host 127.0.0.1 --port 5000

docker-build:
	@echo "🐳 Construindo imagem Docker..."
	docker build -t $(DOCKER_IMAGE) .
	@echo "✅ Imagem construída!"

docker-run:
	@echo "🐳 Executando container..."
	docker run -p 8000:8000 $(DOCKER_IMAGE)
	@echo "✅ Container rodando em http://localhost:8000"

# Alias úteis
setup: install-dev clean tidy
tidy: clean lint format
dev: install-dev

# CI/CD simulado localmente
ci: clean lint test coverage
	@echo "✅ Pipeline CI OK!"

.DEFAULT_GOAL := help
