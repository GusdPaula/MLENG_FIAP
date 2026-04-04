## Configuração do Google BigQuery

### Pré-requisitos
1. Uma conta do Google Cloud Platform (GCP)
2. Um projeto GCP criado (ID do projeto: `ibm-churn`)
3. API do BigQuery habilitada no seu projeto GCP

### Passo 1: Habilitar a API do BigQuery
1. Acesse o [Google Cloud Console](https://console.cloud.google.com/)
2. Vá para "APIs e Serviços" > "Biblioteca"
3. Procure por "BigQuery API"
4. Clique nela e habilite a API se ainda não estiver habilitada

### Passo 2: Instalar o SDK do Google Cloud
O Google Cloud SDK é necessário para autenticação e outras operações.

1. Baixe e instale o Google Cloud SDK do site oficial: [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)
2. Após a instalação, inicialize o SDK:
   ```bash
   gcloud init
   ```
3. Faça login com sua conta Google:
   ```bash
   gcloud auth login
   ```
4. Defina o projeto padrão:
   ```bash
   gcloud config set project ibm-churn
   ```

### Passo 3: Instalar Bibliotecas Python Necessárias
Instale as bibliotecas Python usando pip:

```bash
pip install google-cloud-bigquery pandas
```

### Passo 4: Aceitar Acesso à Tabela Compartilhada
A tabela já foi criada no dataset `ibmchurn` com o nome `ibm_churn`. Para acessar a tabela compartilhada:

1. Acesse o [Console do BigQuery](https://console.cloud.google.com/bigquery)
2. No painel esquerdo, você deve ver o dataset compartilhado `ibm-churn:ibmchurn`
3. Clique no dataset para expandi-lo e visualizar a tabela `ibm_churn`
4. Se aparecer uma solicitação de autorização, clique em "Aceitar" ou "Permitir"
5. Certifique-se de que sua conta tem as permissões necessárias (normalmente "Visualizador de BigQuery" ou superior)

### Usando BigQuery com Python
O arquivo `db_setup/query.py` contém um exemplo de como consultar tabelas do BigQuery usando Python. Ele usa a biblioteca `google-cloud-bigquery` para conectar e executar consultas.

Exemplo de uso:
```python
from db_setup.query import query_bigquery_table

df = query_bigquery_table('ibm-churn', 'ibmchurn', 'ibm_churn')
```

### Solução de Problemas
- **Erros de autenticação**: Certifique-se de ter feito login com `gcloud auth login` e definido o projeto correto
- **Permissão negada**: Verifique se você tem acesso ao dataset compartilhado
- **API não habilitada**: Confirme que a BigQuery API está habilitada no seu projeto GCP
- **Projeto não encontrado**: Verifique o ID do projeto

Para mais informações, consulte a [Documentação do Google Cloud BigQuery](https://cloud.google.com/bigquery/docs).
