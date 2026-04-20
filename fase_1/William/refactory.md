# Plano de Refatoração: Unificação do Preprocessamento

## Objetivo

Eliminar a duplicação de lógica entre `TelcoDataPreprocessor` (preprocessing.py) e `FeatureTransformer` (feature_transformer.py), criando uma única classe responsável por preprocessamento tanto no treinamento quanto na inferência.

---

## Estado Atual (Antes)

```
Treinamento:
  CSV → TelcoDataPreprocessor.pipeline_completo() → arrays normalizados → modelo.fit()

Inferência:
  JSON dict → FeatureTransformer._encode() → StandardScaler (recriado) → modelo.predict()
                    ↑
                    Lógica de encoding DUPLICADA e hardcoded
                    ↑
                    Usa TelcoDataPreprocessor internamente para recriar o scaler
```

Problemas:
- Duas implementações da mesma lógica de encoding
- `FeatureTransformer._encode()` tem ~60 linhas de if/else hardcoded que replicam `encode_binary_features()` + `encode_categorical_features()`
- Se uma mudar e a outra não, as predições quebram silenciosamente
- O `FeatureTransformer` já depende do `TelcoDataPreprocessor` para criar o scaler

---

## Estado Desejado (Depois)

```
Treinamento:
  CSV → TelcoDataPreprocessor.pipeline_completo() → arrays normalizados → modelo.fit()

Inferência:
  JSON dict → TelcoDataPreprocessor.transform_single() → array normalizado → modelo.predict()
                    ↑
                    Mesma classe, mesma lógica, zero duplicação
```

---

## Arquivos Afetados

| Arquivo | Ação |
|---|---|
| `src/data/preprocessing.py` | **Modificar** — adicionar métodos de inferência |
| `src/api/feature_transformer.py` | **Remover** — substituído pelo TelcoDataPreprocessor |
| `src/api/main.py` | **Modificar** — trocar import e chamadas |
| `src/api/__init__.py` | **Nenhuma** — não exporta feature_transformer |
| `src/data/__init__.py` | **Nenhuma** — já exporta TelcoDataPreprocessor |
| `tests/test_preprocessing.py` | **Modificar** — adicionar testes para novos métodos |
| `tests/test_models.py` | **Nenhuma** — só faz smoke test de import |

---

## Passo a Passo

### Passo 1: Adicionar métodos de inferência ao `TelcoDataPreprocessor`

**Arquivo:** `src/data/preprocessing.py`

Adicionar 3 novos métodos à classe `TelcoDataPreprocessor`:

#### 1.1 — `fit_for_inference(data_path)`

Método que prepara o preprocessor para uso em inferência. Deve:
- Carregar o dataset CSV completo
- Executar `drop_leakage_columns()`
- Executar `extract_target()` (para separar X de y)
- Executar `encode_binary_features()` no X inteiro
- Executar `encode_categorical_features()` no X inteiro
- Armazenar `self.feature_names` com os nomes das 30 colunas resultantes
- Criar e ajustar `self.scaler = StandardScaler()` nos dados codificados (sem normalizar)
- Marcar `self._fitted_for_inference = True` como flag de segurança

Esse método reutiliza os métodos de encoding que já existem na classe. Não duplica lógica.

#### 1.2 — `transform_single(features_dict)`

Método que transforma um dicionário de features em array 30D normalizado. Deve:
- Verificar que `fit_for_inference()` foi chamado antes (raise ValueError se não)
- Criar um `pd.DataFrame` de 1 linha a partir do dicionário recebido
- Aplicar `encode_binary_features()` nesse DataFrame
- Aplicar `encode_categorical_features()` nesse DataFrame
- Reordenar colunas para corresponder a `self.feature_names` (usar `df.reindex(columns=self.feature_names, fill_value=0)`)
- Aplicar `self.scaler.transform()` para normalizar
- Retornar array numpy com shape (1, 30)

#### 1.3 — `transform_batch(features_list)`

Método que transforma uma lista de dicionários em array 2D normalizado. Deve:
- Verificar que `fit_for_inference()` foi chamado antes
- Criar um `pd.DataFrame` a partir da lista de dicionários
- Aplicar `encode_binary_features()` nesse DataFrame
- Aplicar `encode_categorical_features()` nesse DataFrame
- Reordenar colunas para corresponder a `self.feature_names`
- Aplicar `self.scaler.transform()` para normalizar
- Retornar array numpy com shape (n_samples, 30)

#### Importante — Ajuste no `encode_binary_features()`

O método atual detecta colunas binárias verificando `df[col].nunique() == 2`. Quando o DataFrame tem apenas 1 linha (inferência single), uma coluna com valor "Yes" terá `nunique() == 1` e **não seria detectada como binária**.

Solução: o método precisa de uma lista fixa de colunas binárias conhecidas (ou pelo menos um fallback), em vez de depender da detecção automática. Adicionar um atributo `self.binary_columns` populado durante `fit_for_inference()` e reutilizado no transform.

Da mesma forma, `encode_categorical_features()` detecta colunas categóricas com `nunique() > 2`. Com 1 linha, `nunique()` seria 1 para todas. Mesmo problema. Guardar `self.categorical_columns` durante o fit.

---

### Passo 2: Ajustar a detecção de colunas em `encode_binary_features()` e `encode_categorical_features()`

**Arquivo:** `src/data/preprocessing.py`

Atualmente os dois métodos detectam colunas por análise do DataFrame recebido. Isso funciona no treinamento (DataFrame grande), mas falha na inferência (1 linha).

Modificar ambos os métodos para:
1. Se `self.binary_columns` / `self.categorical_columns` já estiverem definidos (populados no fit), usar esses valores
2. Se não, manter a detecção automática atual (retrocompatibilidade com o pipeline de treinamento)

Isso garante que a detecção acontece uma vez (no fit) e é reutilizada nos transforms.

---

### Passo 3: Modificar `src/api/main.py`

**Arquivo:** `src/api/main.py`

#### 3.1 — Trocar import

Remover:
```python
from .feature_transformer import feature_transformer
```

Adicionar:
```python
from src.data.preprocessing import TelcoDataPreprocessor
```

#### 3.2 — Inicializar o preprocessor

Dentro de `create_app()`, antes ou depois de carregar o modelo, criar uma instância do `TelcoDataPreprocessor` e chamar `fit_for_inference()`:

```
preprocessor = TelcoDataPreprocessor()
preprocessor.fit_for_inference('data/processed/telco_churn_processed.csv')
app.state.preprocessor = preprocessor
```

#### 3.3 — Substituir chamadas nos endpoints

3 lugares precisam ser alterados:

1. **Endpoint `/api/predict`** — trocar `feature_transformer.transform(request.features)` por `app.state.preprocessor.transform_single(request.features)`

2. **Endpoint `/api/predict-batch`** — trocar `feature_transformer.transform_batch(request.samples)` por `app.state.preprocessor.transform_batch(request.samples)`

3. **Endpoint `/api/model-info`** — trocar `feature_transformer.feature_order` por `app.state.preprocessor.feature_names`

---

### Passo 4: Remover `src/api/feature_transformer.py`

**Arquivo:** `src/api/feature_transformer.py`

Deletar o arquivo inteiro. Toda a lógica foi absorvida pelo `TelcoDataPreprocessor`.

Verificar que `src/api/__init__.py` não importa nada desse arquivo (confirmado: não importa).

---

### Passo 5: Remover a constante `FEATURE_ORDER`

A lista hardcoded de 30 features que existia no `feature_transformer.py` não é mais necessária. O `TelcoDataPreprocessor` descobre os nomes das features dinamicamente durante `fit_for_inference()` via `X.columns.tolist()` após o encoding.

Isso é melhor porque se as features mudarem (ex: novo campo no dataset), o código se adapta automaticamente.

---

### Passo 6: Adicionar testes unitários para os novos métodos

**Arquivo:** `tests/test_preprocessing.py`

Adicionar uma nova classe de testes `TestPreprocessorInference` com os seguintes testes:

#### 6.1 — `test_fit_for_inference_creates_scaler`
Verificar que após chamar `fit_for_inference()`, `preprocessor.scaler` não é None e `preprocessor.feature_names` é uma lista com 30 elementos.

#### 6.2 — `test_transform_single_returns_correct_shape`
Passar um dicionário com 19 features e verificar que o retorno tem shape (1, 30).

#### 6.3 — `test_transform_single_without_fit_raises_error`
Chamar `transform_single()` sem ter chamado `fit_for_inference()` antes. Deve levantar `ValueError`.

#### 6.4 — `test_transform_batch_returns_correct_shape`
Passar lista com 3 dicionários e verificar que o retorno tem shape (3, 30).

#### 6.5 — `test_transform_single_output_is_normalized`
Verificar que os valores retornados não são os mesmos que os originais (o scaler deve ter alterado os valores). Verificar que não são todos zeros.

#### 6.6 — `test_transform_consistency`
Transformar o mesmo dicionário duas vezes e verificar que o resultado é idêntico (determinismo).

#### 6.7 — `test_transform_single_matches_pipeline`
Teste de integração que garante a equivalência entre treinamento e inferência. Pegar um registro do dataset original, processá-lo pelo `pipeline_completo()` e separadamente pelo `transform_single()`. Comparar os arrays resultantes — devem ser iguais (ou muito próximos, dentro de tolerância numérica). Este é o teste mais importante da refatoração.

#### Fixture necessária

Os testes de inferência precisam do dataset real (`data/processed/telco_churn_processed.csv`). Criar uma fixture `real_data_path` que retorna o caminho do CSV e pula o teste se o arquivo não existir (`pytest.mark.skipif`).

Também criar uma fixture `sample_features_dict` com um dicionário de exemplo com as 19 features nomeadas.

---

### Passo 7: Ajustar teste de smoke em `tests/test_models.py`

**Arquivo:** `tests/test_models.py`

No teste `test_module_imports`, verificar que `TelcoDataPreprocessor` continua importável (já está ok). Não é necessário ajustar nada aqui, pois o smoke test não importa `FeatureTransformer`.

---

### Passo 8: Limpar `src/api/examples.py` (se ainda existir)

Se o arquivo ainda existir, pode ser removido — não é usado pela API e referencia o formato antigo.

---

### Passo 9: Limpar o `loader.py`

**Arquivo:** `src/data/loader.py`

O `TelcoDataLoader` não é usado em nenhum lugar do projeto (confirmado por busca). Considerar:
- Remover o arquivo
- Remover o import em `src/data/__init__.py` (`from .loader import TelcoDataLoader`)
- Remover o import em `src/__init__.py` (`from .data import TelcoDataLoader, TelcoDataPreprocessor` → `from .data import TelcoDataPreprocessor`)
- Remover das listas `__all__` em ambos os `__init__.py`

---

## Ordem de Execução

1. **Primeiro:** Modificar `preprocessing.py` (Passos 1 e 2) — adicionar métodos sem quebrar nada existente
2. **Segundo:** Adicionar testes (Passo 6) — validar que os novos métodos funcionam
3. **Terceiro:** Modificar `main.py` (Passo 3) — trocar a API para usar o preprocessor
4. **Quarto:** Remover `feature_transformer.py` (Passo 4)
5. **Quinto:** Limpar `loader.py` e `__init__.py` (Passos 8 e 9)
6. **Sexto:** Rodar todos os testes e validar a API manualmente

---

## Riscos e Cuidados

### Detecção de colunas binárias com 1 registro
O problema mais sutil. O `encode_binary_features()` atual usa `nunique() == 2` para detectar colunas binárias. Com 1 linha, `nunique()` é sempre 1. Se isso não for tratado (Passo 2), o transform_single vai retornar dados não codificados. Testar explicitamente.

### Ordem das colunas após `get_dummies()`
O `pd.get_dummies()` gera colunas em ordem alfabética das categorias. Se o DataFrame de inferência tiver menos categorias do que o de treino, colunas vão faltar. O `df.reindex(columns=self.feature_names, fill_value=0)` resolve isso — mas precisa ser testado.

### Gender não é Yes/No
O campo Gender tem valores "Male"/"Female", não "Yes"/"No". O `encode_binary_features()` atual trata isso com um fallback: `{unique_vals[0]: 0, unique_vals[1]: 1}`. Verificar que esse fallback funciona com 1 registro (mesmo problema do `nunique`).

### StandardScaler warning
O scaler é treinado com nomes de features (DataFrame), mas na inferência recebe array numpy. Isso gera um warning do sklearn: "X does not have valid feature names". Considerar usar `scaler.transform(df)` em vez de `scaler.transform(df.values)` para evitar o warning.

### Retrocompatibilidade do `pipeline_completo()`
Não alterar a assinatura ou comportamento do `pipeline_completo()`. Ele deve continuar funcionando exatamente como antes para treinamento. Os novos métodos são adições, não substituições.
