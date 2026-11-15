"""Load CAMELYON17-WILDS patches by hospital."""
import numpy as np
from wilds import get_dataset

_DS = None


def _dataset(root="data"):
    global _DS
    if _DS is None:
        _DS = get_dataset(dataset="camelyon17", download=False, root_dir=root)
    return _DS


def load_wilds(hospitals, n, seed=0, root="data"):
    """Return (patches uint8 (n,96,96,3), labels (n,), hospital ids (n,)) for the given hospitals."""
    ds = _dataset(root)
    col = ds.metadata_fields.index("hospital")
    hosp = ds.metadata_array[:, col].numpy()
    pool = np.where(np.isin(hosp, list(hospitals)))[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(pool, size=min(n, len(pool)), replace=False)
    x = np.stack([np.asarray(ds.get_input(int(i)).convert("RGB")) for i in idx])
    y = ds.y_array[idx].numpy()
    return x, y, hosp[idx]