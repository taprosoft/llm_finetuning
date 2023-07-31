# Prompt templates

This directory contains template styles for the prompts used to finetune LoRA models.

## Format

A template is described via a JSON file with the following keys:

- `prompt_input`: The template to use when input is not None. Uses `{instruction}` and `{input}` placeholders.
- `prompt_no_input`: The template to use when input is None. Uses `{instruction}` placeholders.
- `description`: A short description of the template, with possible use cases.
- `response_split`: The text to use as separator when cutting real response from the model output.

No `{response}` placeholder was used, since the response is always the last element of the template and is just to be concatenated to the rest.

## Example template

The default template, used unless otherwise specified, is `alpaca.json`

```json
{
    "description": "Template used by Alpaca-LoRA.",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:"
}

```

## Current templates

### alpaca

Default template used for generic LoRA fine tunes so far.

### alpaca_legacy

Legacy template used by the original alpaca repo, with no `\n` after the response field. Kept for reference and experiments.

### alpaca_short

A trimmed down alpaca template which seems to perform just as well and spare some tokens. Models created with the default template seem to be queryable by the short tempalte as well. More experiments are welcome.

### sharegpt

This template is used for the ShareGPT conversation dataset such as [ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered). Basic structure is:

```json
{"conversations": [
  {"from": "human", "value": "hi"},
  {"from": "gpt", "value": "hi"},
  {"from": "human", "value": "how are you"},
  {"from": "gpt", "value": "good"}
]}

```

### text
Currently, we also support supplying raw text file as the training data. Simply add the path to the text file as the parameter of `--data_path`. Long text file will be splitted in smaller chunks to be used in the model.
