"""Reproducibility utilities for global deterministic seeding."""

import os
import random
import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set global random seed for python random, numpy, and environment.

    Args:
        seed: Integer seed value.
        deterministic: If True, set PYTHONHASHSEED for deterministic hashing.
    """
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        os.environ["PYTHONHASHSEED"] = str(seed)
