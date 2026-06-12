# Análise Exploratória de Dados (EDA)

Este diretório contém notebooks para a análise inicial dos dados do sistema de recomendação de e-commerce.

## Insights Iniciais

Com base na análise realizada no notebook `data_analysis.ipynb`, foram observadas as seguintes conclusões preliminares:

### 1. Conjunto de Dados `events`
O dataset de eventos é, no momento, o único conjunto de dados com utilidade imediata e clara. Ele fornece informações sobre as interações dos usuários com os itens, permitindo entender o comportamento de navegação e compra.

### 2. Conjunto de Dados `item_properties`
A análise deste dataset revelou que a coluna `property` contém valores mistos (numéricos e strings), tornando a interpretação difícil. Sem a documentação técnica ou informações externas sobre o que cada propriedade numérica representa, é complexo extrair insights significativos ou utilizá-lo para engenharia de atributos.

### 3. Conjunto de Dados `category_tree`
O dataset de árvore de categorias apresenta uma estrutura que requer maior compreensão do domínio do negócio. Sem a definição exata de como a hierarquia de categorias é montada e como ela se relaciona com os itens, sua utilidade permanece limitada.

## Conclusão
Para avançar na análise e modelagem, é recomendável buscar documentação adicional sobre o significado das propriedades dos itens e a estrutura da árvore de categorias. Caso contrário, o foco principal do modelo de recomendação deverá residir nos dados de interação contidos no dataset de eventos.

## Experimentos de Modelos

O notebook `model_experiments_mlflow.ipynb` executa os modelos de recomendação com diferentes estratégias de processamento de dados, registrando:

- parâmetros
- métricas
- datasets processados
- artefatos dos modelos

Ele também salva cada dataframe processado em arquivos compatíveis com DVC, para que os dados usados em cada experimento possam ser versionados separadamente.
