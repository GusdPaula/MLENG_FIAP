ecommerce_recommender/
│
├── data/
│   ├── dataset.py                ✅ already exists
│   ├── processors.py             ✅ already exists
│   ├── splits.py                 NEW
│   └── cache.py                  NEW
│
├── models/
│   ├── ...
│
├── training/
│   ├── trainer.py                ✅
│   ├── early_stopping.py         ✅
│   ├── metrics.py                ✅
│   ├── evaluator.py              NEW
│   ├── checkpoint.py             NEW
│   └── experiment.py             NEW (most important)
│
├── mlflow/
│   ├── toolkit.py                ✅
│   └── callbacks.py              NEW
│
├── utils/
│   ├── device.py                 NEW
│   └── cleanup.py                NEW
│
└── notebooks/
    experiment.ipynb