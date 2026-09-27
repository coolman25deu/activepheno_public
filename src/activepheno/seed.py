"""CPU-only deterministic seeding; identifiers never depend on random state."""
import random

import numpy as np


def seed_everything(seed: int) -> None:
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
