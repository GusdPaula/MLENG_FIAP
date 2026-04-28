import random
import numpy as np

# Global seed value for the entire project
GLOBAL_SEED = 42

def set_all_seeds(seed: int = 42):
    """
    Set seed for random, numpy, torch and torch.cuda (if available).
    Ensures full reproducibility across libraries.
    If no seed is provided, uses GLOBAL_SEED.
    """
    if seed is None:
        seed = GLOBAL_SEED
    """
    Set seed for random, numpy, torch and torch.cuda (if available).
    Ensures full reproducibility across libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
