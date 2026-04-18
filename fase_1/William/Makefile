.PHONY: help install install-dev clean test lint format run-eda run-notebook docker-build docker-run install-prod run-baselines run-all

# Variáveis
PYTHON := python3
PIP := pip
PROJECT_NAME := telco-churn-prediction
DOCKER_IMAGE := $(PROJECT_NAME):latest

help:
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  $(PROJECT_NAME) - Make Commands                 ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Setup & Install:"
	@echo "  make install          - Instalar via pyproject.toml"
	@echo "  make install-dev      - Instalar com dev dependencies"
	@echo "  make install-prod     - Instalar apenas producao (requirements-prod.txt)"
	@echo "  make install-full     - Instalar tudo completo (requirements-dev.txt)"
	@echo "  make clean            - Limpar arquivos temporarios"
	@echo ""
	@echo "Desenvolvimento:"
	@echo "  make lint             - Validar codigo (ruff)"
	@echo "  make format           - Formatar codigo (black, isort)"
	@echo "  make test             - Executar testes com pytest"
	@echo "  make coverage         - Gerar relatorio de cobertura"
	@echo ""
	@echo "Execucao:"
	@echo "  make run-eda          - Executar script de EDA"
	@echo "  make run-baselines    - Treinar baselines com MLflow"
	@echo "  make run-notebook     - Iniciar Jupyter notebook"
	@echo "  make run-all          - Rodar EDA + Baselines"
	@echo "  make mlflow-ui        - Abrir MLflow UI (localhost:5000)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Construir imagem Docker"
	@echo "  make docker-run       - Executar container Docker"
	@echo ""

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

lint:
	@echo "🔍 Validando código com ruff..."
	ruff check src/ tests/ --show-source
	@echo "✅ Linting OK!"

format:
	@echo "✨ Formatando código..."
	isort src/ tests/
	black src/ tests/
	@echo "✅ Formatação concluída!"

test:
	@echo "🧪 Executando testes..."
	pytest tests/ -v --tb=short
	@echo "✅ Testes concluídos!"

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
