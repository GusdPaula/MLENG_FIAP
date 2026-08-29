from pathlib import Path

import kagglehub
import pandas as pd

DATASET_SLUG = "saharalaa/medical-abstracts-tc-corpus"

EXPECTED_FILES = {
    "train":  "medical_tc_train.csv",
    "test":   "medical_tc_test.csv",
    "labels": "medical_tc_labels.csv",
}


def load_data() -> dict[str, pd.DataFrame]:
    """Baixa (ou usa cache) o dataset e carrega os três CSVs.

    Returns:
        Dicionário com chaves "train", "test" e "labels" mapeando
        para os respectivos DataFrames.

    Raises:
        ImportError: Se kagglehub não estiver instalado.
        FileNotFoundError: Se algum arquivo CSV não for encontrado no
            caminho retornado pelo kagglehub.
        Exception: Qualquer exceção lançada por kagglehub é propagada
            sem modificação.
    """
    path = Path(kagglehub.dataset_download(DATASET_SLUG))

    dataframes: dict[str, pd.DataFrame] = {}
    for key, filename in EXPECTED_FILES.items():
        file_path = path / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"Arquivo esperado não encontrado: '{filename}' "
                f"no caminho '{path}'"
            )
        dataframes[key] = pd.read_csv(file_path)

    return dataframes
