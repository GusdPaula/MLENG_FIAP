from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PACKAGE_ROOT / "data-pipeline"
MODULE_PATH = SOURCE_DIR / "bigquery_query.py"

spec = importlib.util.spec_from_file_location("data_pipeline.bigquery_query", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules["data_pipeline.bigquery_query"] = module
spec.loader.exec_module(module)

BigQueryQuery = module.BigQueryQuery
