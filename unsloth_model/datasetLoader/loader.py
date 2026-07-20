from datasets import load_dataset
from typing import Any
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict


def load_data(
    path: str,
    tokenizer: Any,
    load_from_cache: bool = False,
    cache_dir: str = "./dataset",
    mapper_func: Any | None = None,
) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:

    ### Load raw Dataset
    dataset = load_dataset(path=path, cache_dir=cache_dir, split="train")

    ### Remap dataset
    dataset_col_name = dataset.column_names
    dataset = dataset.map(
        mapper_func,
        fn_kwargs={"tokenizer": tokenizer},
        load_from_cache_file=load_from_cache,
        remove_columns=dataset_col_name,
    )

    return dataset
