# Próximos Passos...

1. ~~Validar o código~~
   - ~~Executar testes unitários e linting~~
   - ~~Confirmar que os scripts do pipeline de dados executam corretamente~~
2. ~~Configurar chaves para GCP e Kaggle~~
   - ~~Configurar `GOOGLE_APPLICATION_CREDENTIALS`~~
   - ~~Definir credenciais do Kaggle para `kagglehub`~~
3. Adicionar DVC ao pipeline
   - Rastrear datasets brutos e processados
   - Definir estágios DVC para preprocessamento, feature_eng, treino e avaliação
4. ~~Realizar EDA~~
   - ~~Explorar `events.csv`, `category_tree.csv` e propriedades de item combinadas~~
   - ~~Identificar oportunidades de engenharia de features~~
5. Construir e treinar o modelo
   - Implementar o treinamento do modelo de recomendação
   - Registrar experimentos no MLflow
6. ~~Revisar pre-commit e pipeline de CI~~
   - ~~Definir hooks de pre-commit para formatação e checagens estáticas~~
   - ~~Adicionar workflow de CI para executar testes e validar mudanças no pipeline~~
7. Refinar Estratégia de Dados (Baseado na EDA)
   - Priorizar o uso do dataset `events` para a modelagem de interações
   - Investigar documentação externa para decifrar as propriedades numéricas de `item_properties`
   - Validar a estrutura de hierarquia do `category_tree` para verificar se pode ser integrada ao modelo
