# Módulo de API de Predição

Este módulo fornece uma API baseada em classes e seguindo princípios SOLID para gerar predições usando modelos de recomendação treinados. Segue os princípios de Responsabilidade Única, Aberto/Fechado, Substituição de Liskov, Segregação de Interface e Inversão de Dependência.

## Funcionalidades

- **Múltiplas Estratégias de Predição**: Preditores para usuário único, em lote e top-k de recomendações
- **Detecção de Desvio de Modelo e Dados**: Monitoramento estatístico para detectar mudanças de distribuição e degradação de performance
- **Validação Pydantic**: Modelos de solicitação/resposta com validação automática e type-safe
- **Logging Abrangente**: Logging detalhado em todo o pipeline de predição
- **Padrão Factory**: Registro e criação extensíveis de preditores
- **Design Modular**: Separação clara de responsabilidades seguindo princípios SOLID

## Arquitetura

```
api/
├── __init__.py           # Exportações da API pública
├── main.py               # Ponto de entrada da aplicação FastAPI
├── exceptions.py         # Exceções personalizadas
├── controllers/          # Rotas e controle HTTP
│   ├── __init__.py
│   └── routes.py         # Endpoints da API
├── domain/               # Lógica de domínio e predição
│   ├── __init__.py
│   ├── base_predictor.py # Interface abstrata BasePredictor
│   ├── predictor_factory.py  # PredictorFactory (padrão de registro)
│   └── predictors.py     # Implementações concretas de preditores
├── services/             # Serviços de aplicação
│   ├── __init__.py
│   ├── prediction_service.py  # PredictionService (orquestração)
│   └── monitoring_service.py  # Detecção de desvio de modelo e dados
└── models/               # Modelos de dados Pydantic (DTOs)
    ├── __init__.py
    └── schemas.py        # Modelos de solicitação/resposta
```

## Componentes Principais

### BasePredictor

Interface abstrata que todos os preditores devem implementar. Define o contrato para operações de predição.

```python
from api.domain.base_predictor import BasePredictor
from api.models.schemas import PredictionRequest, PredictionResponse

class CustomPredictor(BasePredictor):
    name = "custom"

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        # Implementação
        pass

    def predict_batch(self, requests: list[PredictionRequest]) -> list[PredictionResponse]:
        # Implementação
        pass
```

### Preditores Concretos

- **SingleUserPredictor**: Para predições de usuário único contra itens especificados
- **TopKRecommendationPredictor**: Para recomendações top-k de itens
- **BatchPredictor**: Otimizado para predições em lote com único forward pass

### PredictorFactory

Factory baseada em registro para criar instâncias de preditores.

```python
from api.domain.predictor_factory import PredictorFactory

# Obter preditores disponíveis
predictors = PredictorFactory.available_predictors()
# Retorna: ['batch', 'single_user', 'top_k']

# Criar um preditor
predictor = PredictorFactory.create(
    predictor_type="single_user",
    model=model,
    user2idx=user2idx,
    item2idx=item2idx,
)
```

### PredictionService

Serviço de alto nível para carregamento de modelo e orquestração de predição com monitoramento embutido.

```python
from api.services.prediction_service import PredictionService
from api.models.schemas import PredictionRequest

# Inicializar serviço com monitoramento habilitado
service = PredictionService(
    model_path="models/model.pt",
    predictor_type="single_user",
    device="cpu",
    enable_monitoring=True,
    shift_threshold=0.05,
    drift_threshold=2.0,
    monitoring_window_size=1000,
)

# Fazer predições
request = PredictionRequest(user_id=123, item_ids=[1, 2, 3])
response = service.predict(request)

# Definir baselines de monitoramento após coletar predições iniciais
service.set_monitoring_baselines()

# Verificar desvios
shift_results = service.check_shifts()

# Obter resumo de monitoramento
summary = service.get_monitoring_summary()
```

## Modelos de Dados

### Modelos de Solicitação

```python
from api.models.schemas import PredictionRequest

# Solicitação de predição única
request = PredictionRequest(
    user_id=123,
    item_ids=[1, 2, 3, 4, 5],
    k=None  # Opcional para top-k
)

# Solicitação de recomendação top-k
request = PredictionRequest(
    user_id=123,
    k=10  # Obter top 10 recomendações
)
```

### Modelos de Resposta

```python
from api.models.schemas import PredictionResponse, RecommendationResponse

# Resposta de predição
response = PredictionResponse(
    user_id=123,
    item_scores={1: 0.95, 2: 0.87, 3: 0.72},
    metadata={"predictor": "single_user", "model_type": "ncf"}
)

# Resposta de recomendação
response = RecommendationResponse(
    user_id=123,
    recommendations=[(1, 0.95), (2, 0.87), (3, 0.72)],
    metadata={"predictor": "top_k", "k": 10}
)
```

## Monitoramento e Detecção de Desvio

### MonitoringService

Monitoramento abrangente para detectar desvios de modelo e dados.

```python
from api.services.monitoring_service import MonitoringService

# Inicializar serviço de monitoramento
monitoring = MonitoringService(
    shift_threshold=0.05,      # Limiar de p-value para desvio de dados
    drift_threshold=2.0,       # Limiar de z-score para desvio de performance
    window_size=1000,          # Número de predições a rastrear
)

# Registrar predições
monitoring.record_predictions(
    scores=[0.95, 0.87, 0.72, 0.68],
    user_ids=[123, 123, 123, 123],
    item_ids=[1, 2, 3, 4],
)

# Definir baselines após coletar dados iniciais
monitoring.set_baselines()

# Verificar desvios
results = monitoring.check_shifts()
# Retorna: {
#     'data_shift': ShiftDetectionResult(...),
#     'performance_drift': ShiftDetectionResult(...)
# }

# Obter resumo de monitoramento
summary = monitoring.get_monitoring_summary()
# Retorna: {
#     'performance_stats': {'mean': 0.80, 'std': 0.12, ...},
#     'has_baseline': True,
#     'window_size': 1000,
#     'history_size': 500,
#     ...
# }
```

### Métodos de Detecção de Desvio

- **DataShiftDetector**: Usa teste Kolmogorov-Smirnov para detectar mudanças de distribuição em scores de predição
- **ModelPerformanceMonitor**: Usa análise de z-score para detectar degradação de performance

```python
from api.services.monitoring_service import DataShiftDetector, ModelPerformanceMonitor

# Detecção de desvio de dados
detector = DataShiftDetector(threshold=0.05)
detector.set_baseline(baseline_metrics)
result = detector.detect_shift(current_metrics)

# Detecção de desvio de performance
monitor = ModelPerformanceMonitor(window_size=1000)
monitor.record_predictions(scores=[0.95, 0.87, ...])
monitor.set_baseline()
result = monitor.detect_performance_drift(threshold=2.0)
```

## Exemplos de Uso

### Predição Básica

```python
from api.services.prediction_service import PredictionService
from api.models.schemas import PredictionRequest

# Inicializar serviço
service = PredictionService(
    model_path="models/model.pt",
    predictor_type="single_user",
)

# Criar solicitação
request = PredictionRequest(
    user_id=123,
    item_ids=[1, 2, 3, 4, 5]
)

# Obter predições
response = service.predict(request)
print(f"Scores do usuário {response.user_id}: {response.item_scores}")
```

### Recomendações Top-K

```python
from api.services.prediction_service import PredictionService

# Inicializar com preditor top-k
service = PredictionService(
    model_path="models/model.pt",
    predictor_type="top_k",
)

# Obter top 10 recomendações
response = service.recommend(user_id=123, k=10)
print(f"Top recomendações: {response.recommendations}")
```

### Predições em Lote

```python
from api.services.prediction_service import PredictionService
from api.models.schemas import PredictionRequest

# Inicializar com preditor batch
service = PredictionService(
    model_path="models/model.pt",
    predictor_type="batch",
)

# Criar múltiplas solicitações
requests = [
    PredictionRequest(user_id=123, item_ids=[1, 2, 3]),
    PredictionRequest(user_id=456, item_ids=[4, 5, 6]),
]

# Obter predições em lote
response = service.predict_batch(requests)
print(f"Predições para {len(response.predictions)} usuários")
```

### Com Monitoramento

```python
from api.services.prediction_service import PredictionService
from api.models.schemas import PredictionRequest

# Inicializar com monitoramento habilitado
service = PredictionService(
    model_path="models/model.pt",
    predictor_type="single_user",
    enable_monitoring=True,
    shift_threshold=0.05,
    drift_threshold=2.0,
)

# Fazer predições (automaticamente registradas)
for i in range(100):
    request = PredictionRequest(user_id=123, item_ids=[1, 2, 3])
    service.predict(request)

# Definir baselines
service.set_monitoring_baselines()

# Continuar fazendo predições
for i in range(100):
    request = PredictionRequest(user_id=456, item_ids=[4, 5, 6])
    service.predict(request)

# Verificar desvios
shift_results = service.check_shifts()
for shift_type, result in shift_results.items():
    if result.has_shift:
        print(f"{shift_type} detectado: {result.message}")
```

### Predit Personalizado

```python
from api.domain.base_predictor import BasePredictor
from api.domain.predictor_factory import PredictorFactory
from api.models.schemas import PredictionRequest, PredictionResponse
from api.services.prediction_service import PredictionService

@PredictorFactory.register("custom")
class CustomPredictor(BasePredictor):
    name = "custom"

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        # Lógica de predição personalizada
        user_idx = self._get_user_idx(request.user_id)
        item_indices = self._get_item_indices(request.item_ids)

        # Sua lógica de predição personalizada aqui
        scores = self._custom_scoring(user_idx, item_indices)

        item_scores = dict(zip(request.item_ids, scores))

        return PredictionResponse(
            user_id=request.user_id,
            item_scores=item_scores,
            metadata={"predictor": self.name}
        )

    def predict_batch(self, requests: list[PredictionRequest]) -> list[PredictionResponse]:
        return [self.predict(req) for req in requests]

    def _custom_scoring(self, user_idx: int, item_indices: list[int]) -> list[float]:
        # Implemente sua lógica de scoring personalizada
        pass

# Usar o preditor personalizado
service = PredictionService(
    model_path="models/model.pt",
    predictor_type="custom",  # Usa seu preditor personalizado
)
```

## Configuração

### Parâmetros do PredictionService

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|---------|-------------|
| `model_path` | str/Path | Obrigatório | Caminho para o artefato do modelo salvo |
| `predictor_type` | str | "single_user" | Tipo de preditor a usar |
| `device` | str | "cpu" | Dispositivo para predições ("cpu" ou "cuda") |
| `enable_monitoring` | bool | True | Habilitar monitoramento de desvio de modelo/dados |
| `shift_threshold` | float | 0.05 | Limiar de p-value para desvio de dados |
| `drift_threshold` | float | 2.0 | Limiar de z-score para desvio de performance |
| `monitoring_window_size` | int | 1000 | Número de predições a rastrear |

### Parâmetros de Monitoramento

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|---------|-------------|
| `shift_threshold` | float | 0.05 | Limiar de p-value para teste KS |
| `drift_threshold` | float | 2.0 | Limiar de z-score para detecção de desvio |
| `window_size` | int | 1000 | Número de predições a manter em memória |

## Tratamento de Exceções

A API fornece exceções personalizadas para tratamento de erros:

```python
from api.exceptions import (
    PredictionError,
    ModelLoadError,
    InvalidInputError,
    PredictorNotFoundError,
)

try:
    service = PredictionService(model_path="models/model.pt")
except ModelLoadError as e:
    print(f"Falha ao carregar modelo: {e}")

try:
    response = service.predict(request)
except InvalidInputError as e:
    print(f"Entrada inválida: {e}")
except PredictorNotFoundError as e:
    print(f"Predit não encontrado: {e}")
```

## Logging

A API usa o módulo padrão de logging do Python. Configure logging em sua aplicação:

```python
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Os loggers da API seguem a hierarquia:
# api - Nível de módulo
# api.controllers - Operações de controle HTTP
# api.services - Operações de serviço
# api.domain - Operações de predição
# api.domain.predictor_factory - Operações de factory
# api.services.monitoring_service - Operações de monitoramento
```

## Dependências

- `torch>=2.0` - PyTorch para inferência de modelo
- `pydantic>=2.0` - Validação e serialização de dados
- `scipy` - Testes estatísticos para detecção de desvio
- `numpy` - Operações numéricas
- `pandas` - Manipulação de dados (se necessário)

## Melhores Práticas

1. **Habilitar Monitoramento**: Use monitoramento em produção para detectar degradação de modelo
2. **Definir Baselines**: Chame `set_monitoring_baselines()` após coletar dados de predição iniciais
3. **Verificar Desvios Regularmente**: Chame periodicamente `check_shifts()` para monitorar a saúde do modelo
4. **Usar Predit Batch**: Para múltiplas predições, use o preditor batch para melhor performance
5. **Tratar Exceções**: Trate adequadamente as exceções da API em sua aplicação
6. **Configurar Logging**: Defina níveis de log apropriados para seu ambiente

## Extendendo a API

### Adicionando Preditores Personalizados

1. Crie uma classe herdando de `BasePredictor`
2. Implemente os métodos `predict()` e `predict_batch()`
3. Registre com o decorador `@PredictorFactory.register("name")`
4. Use via `predictor_type="name"` no PredictionService

### Adicionando Detectores de Desvio Personalizados

1. Crie uma classe herdando de `BaseShiftDetector`
2. Implemente o método `detect_shift()`
3. Use com `MonitoringService` ou standalone

## Considerações de Performance

- **BatchPredictor**: Mais eficiente para lotes grandes (único forward pass)
- **Monitoramento**: Adiciona overhead mínimo (~1-2ms por predição)
- **Tamanho da Janela**: Janelas maiores usam mais memória mas fornecem melhores baselines
- **Dispositivo**: Use GPU (`device="cuda"`) para inferência mais rápida com modelos grandes

## Licença

Este módulo é parte do projeto de sistema de recomendação.
