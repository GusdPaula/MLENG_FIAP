"""Schemas Pydantic para validação de requests e responses da API.

Define modelos de dados para:
  - Health check
  - Predição simples (single)
  - Predição em lote (batch)
"""

from pydantic import BaseModel, Field, field_validator


class CustomerFeatures(BaseModel):
    """Schema estruturado seguindo o padrão snake_case do payload de teste."""

    gender: str = Field(..., description="Gênero (Male/Female)")
    senior_citizen: str = Field(
        ..., description="Cidadão idoso (Yes/No ou 0/1 dependendo do teste)"
    )
    partner: str = Field(..., description="Possui parceiro (Yes/No)")
    dependents: str = Field(..., description="Possui dependentes (Yes/No)")
    tenure_months: int = Field(..., ge=0, description="Meses de contrato")
    phone_service: str = Field(..., description="Serviço telefônico")
    multiple_lines: str = Field(..., description="Múltiplas linhas")
    internet_service: str = Field(..., description="Tipo de internet")
    online_security: str = Field(..., description="Segurança online")
    online_backup: str = Field(..., description="Backup online")
    device_protection: str = Field(..., description="Proteção de dispositivo")
    tech_support: str = Field(..., description="Suporte técnico")
    streaming_tv: str = Field(..., description="Streaming de TV")
    streaming_movies: str = Field(..., description="Streaming de filmes")
    contract: str = Field(..., description="Tipo de contrato")
    paperless_billing: str = Field(..., description="Fatura sem papel")
    payment_method: str = Field(..., description="Método de pagamento")
    monthly_charges: float = Field(..., gt=0, description="Cobrança mensal")
    total_charges: float = Field(..., description="Cobrança total")


class HealthCheckResponse(BaseModel):
    """Response do endpoint de health check.

    Indica status da API e se modelo está carregado.
    """

    status: str = Field(..., description="Status da API")
    version: str = Field(..., description="Versão da API")
    model_loaded: bool = Field(..., description="Se modelo está carregado")


class PredictionRequest(BaseModel):
    """Request para predição de churn de um cliente."""

    features: CustomerFeatures = Field(..., description="Objeto com features estruturadas")
    return_probability: bool = Field(default=True, description="Retornar probabilidades?")


class PredictionResponse(BaseModel):
    """Response com resultado de predição de churn.

    Retorna: predição (0/1), probabilidade opcional e tempo de processamento.
    """

    prediction: int = Field(..., description="0=No Churn, 1=Churn")
    probability: float | None = Field(None, description="Probabilidade de churn (0-1)")
    confidence: float | None = Field(None, description="Confiança da predição (0-1)")
    processing_time_ms: float = Field(..., description="Tempo de processamento em ms")


class BatchPredictionRequest(BaseModel):
    """Request para predição em lote."""

    samples: list[CustomerFeatures] = Field(..., description="Lista de objetos de features")
    return_probabilities: bool = Field(default=True, description="Retornar probabilidades?")

    @field_validator("samples")
    @classmethod  # V2 field_validators must be class methods
    def samples_not_empty(cls, v):
        if not v or len(v) == 0:
            raise ValueError("A lista de samples não pode estar vazia")
        return v


class BatchPredictionResponse(BaseModel):
    """Response com resultado de predições em lote.

    Retorna: lista de predições, probabilidades opcionais, tamanho do lote e tempo.
    """

    predictions: list[int] = Field(..., description="Lista de predições")
    probabilities: list[float] | None = Field(None, description="Lista de probabilidades")
    batch_size: int = Field(..., description="Número de amostras processadas")
    processing_time_ms: float = Field(..., description="Tempo total em ms")


class ModelInfoResponse(BaseModel):
    """Informações do modelo."""

    model_type: str = Field(..., description="Tipo de modelo")
    model_version: str = Field(..., description="Versão do modelo")
    n_features: int = Field(..., description="Número de features")
    features_used: list[str] = Field(..., description="Nomes das features")


class ErrorResponse(BaseModel):
    """Response de erro."""

    error: str = Field(..., description="Mensagem de erro")
    status_code: int = Field(..., description="Código HTTP")
    timestamp: str = Field(..., description="Timestamp do erro")


class ChurnReasonResponse(BaseModel):
    """Explicação de predição de churn."""

    churn_probability: float = Field(..., description="Probabilidade de churn")
    top_risk_factors: list[dict] = Field(..., description="Fatores de risco mais importantes")
    recommendation: str = Field(..., description="Recomendação de ação")
