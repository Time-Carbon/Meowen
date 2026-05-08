from datasets import load_dataset


def load_data(
    path, tokenizer, load_from_cache=False, cache_dir="./dataset", mapper_func=None
):

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
