from google.cloud import bigquery


def query_bigquery_table(
    project_id="ibm-churn", dataset_id="ibmchurn", table_id="ibm_churn", query=None
):
    # Inicializa o cliente do BigQuery
    # Se a chave JSON não estiver no ambiente, você pode passar explicitamente:
    # client = bigquery.Client.from_service_account_json('caminho/para/chave.json')
    client = bigquery.Client(project=project_id)

    # Define a query SQL (usando f-string para facilitar)
    if query is None:
        query = f"""
            SELECT *
            FROM `{project_id}.{dataset_id}.{table_id}`
        """

    try:
        print("Iniciando a consulta...")
        # Executa a query e converte diretamente para Pandas DataFrame
        # O método to_dataframe() utiliza o pyarrow para alta performance
        df = client.query(query).to_dataframe()

        print(f"Sucesso! {len(df)} linhas carregadas.")
        return df

    except Exception as e:
        print(f"Erro ao consultar o BigQuery: {e}")
        return None


# Execução
if __name__ == "__main__":
    df = query_bigquery_table()

    if df is not None:
        # Exemplo de visualização
        print(df.head())
