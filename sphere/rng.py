import hashlib


def fold_in(seed: int, *args):
    """
    fold extra values into a seed

    args : anything hashable, e.g. step, rank
    out  : int seed
    """
    data = str((seed,) + args)
    h = hashlib.sha256(data.encode("utf-8")).hexdigest()
    folded_seed = int(h, 16) % (2**63)
    return folded_seed
