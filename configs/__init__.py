import importlib

DATASETS = ("heat", "wind")


def dataset_module(name: str):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}', expected one of {DATASETS}")
    return importlib.import_module(f"configs.{name}")
