from datasets import Dataset
from transformers import PreTrainedTokenizer


def split_chunks(arr, step):
    for i in range(0, len(arr), step):
        yield arr[i : i + step]


def tokenize(prompt: str, tokenizer: PreTrainedTokenizer, cutoff_len: int):
    result = tokenizer(
        prompt, truncation=True, max_length=cutoff_len + 1, padding="max_length"
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
        "labels": result["input_ids"][:-1].copy(),
    }


def load_text_file(
    file_path: str,
    tokenizer: PreTrainedTokenizer,
    cutoff_len: int = 2048,
    overlap_len: int = 512,
):
    with open(file_path, "r") as file:
        raw_text = file.read()
    tokens = tokenizer.encode(raw_text)
    del raw_text
    tokens = list(split_chunks(tokens, cutoff_len - overlap_len))
    for i in range(1, len(tokens)):
        tokens[i] = tokens[i - 1][-overlap_len:] + tokens[i]
    text_chunks = [tokenizer.decode(x) for x in tokens]
    print("Got {} text chunks".format(len(text_chunks)))
    del tokens
    dataset = Dataset.from_list(
        [tokenize(x, tokenizer, cutoff_len) for x in text_chunks]
    )

    return dataset
