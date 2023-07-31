![test workflow](https://github.com/taprosoft/llm_finetuning/actions/workflows/tests.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit\&logoColor=white)](https://github.com/pre-commit/pre-commit)

# Memory Efficient Fine-tuning of Large Language Models (LoRA + quantization)

This repository contains a convenient wrapper for fine-tuning and inference of Large Language Models (LLMs) in memory-constrained environment. Two major components that democratize the training of LLMs are: Parameter-Efficient Fine-tuning ([PEFT](https://github.com/huggingface/peft)) (e.g: LoRA, Adapter) and quantization techniques (8-bit, 4-bit). However, there exists many quantization techniques and corresponding implementations which make it hard to compare and test different training configurations effectively. This repo aims to provide a common fine-tuning pipeline for LLMs to help researchers quickly try most common quantization-methods and create compute-optimized training pipeline.

This repo is built upon these materials:

* [alpaca-lora](https://github.com/tloen/alpaca-lora) for the original training script.
* [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) for the efficient GPTQ quantization method.
* [exllama](https://github.com/turboderp/exllama) for the high-performance inference engine.

## Key Features

* Memory-efficient fine-tuning of LLMs on consumer GPUs (<16GiB) by utilizing LoRA (Low-Rank Adapter) and quantization techniques.

* Support most popular quantization techniques: 8-bit, 4-bit quantization from [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) and [GPTQ](https://github.com/qwopqwop200/GPTQ-for-LLaMa).

* Correct PEFT checkpoint saving at regular interval to minimize risk of progress loss during long training.

* Correct checkpoint resume for all quantization methods.

* Support distributed training on multiple GPUs (with examples).

* Support gradient checkpointing for both `GPTQ` and `bitsandbytes`.

* Switchable prompt templates to fit different pretrained LLMs.

* Support evaluation loop to ensure LoRA is correctly loaded after training.

* Inference and deployment examples.

* Fast inference with [exllama](https://github.com/turboderp/exllama) for GPTQ model.

## Usage

### Demo notebook

See [notebook](llm_finetuning.ipynb) or on Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taprosoft/llm_finetuning/blob/main/llm_finetuning.ipynb).

### Setup

1. Install default dependencies

   ```bash
   pip install -r requirements.txt
   ```

2. If `bitsandbytes` doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17)

3. To use 4-bit efficient CUDA kernel from ExLlama and GPTQ for training and inference

   ```bash
   pip install -r cuda_quant_requirements.txt
   ```
Note that the installation of above packages requires the installation of CUDA to compile custom kernels. If you have issue, looks for help in the original repos [GPTQ](https://github.com/qwopqwop200/GPTQ-for-LLaMa), [exllama](https://github.com/turboderp/exllama) for advices.

### Data Preparation

Prepare the instruction data to fine-tune the model in the following JSON format.

```json
[
    {
        "instruction": "do something with the input",
        "input": "input string",
        "output": "output string"
    }
]
```

You can supply a single JSON file as training data and perform auto split for validation. Or, prepare two separate `train.json` and `test.json` in the same directory to supply as train and validation data.

You should also take a look at [templates](templates/README.md) to see different prompt template to combine the instruction, input, output pair into a single text. During the training process, the model is trained using CausalLM objective (text completion) on the combined text. The prompt template must be compatible with the base LLM to maximize performance. Read the detail of the model card on HF ([example](https://huggingface.co/WizardLM/WizardLM-30B-V1.0)) to get this information.

Prompt template can be switched as command line parameters at training and inference step.

We also support for raw text file input and ShareGPT conversation style input. See [templates](templates/README.md).

### Training (`finetune.py`)

This file contains a straightforward application of PEFT to the LLaMA model,
as well as some code related to prompt construction and tokenization.
We use common HF trainer to ensure the compatibility with other library such as [accelerate](https://github.com/huggingface/accelerate).

Simple usage:

```bash
bash scripts/train.sh

# OR

python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-output'
```

where `data_path` is the path to a JSON file or a directory contains `train.json` and `test.json`. `base_model` is the model name on HF model hub or path to a local model on disk.

We can also tweak other hyperparameters (see example in [train.sh](scripts/train.sh)):

```bash
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-output' \
    --mode 4 \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 0.2 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --resume_from_checkpoint checkpoint-29/adapter_model/
```

Some notables parameters:

```
micro_batch_size: size of the batch on each GPU, greatly affect VRAM usage
batch_size: actual batch size after gradient accumulation
cutoff_len: maximum length of the input sequence, greatly affect VRAM usage
gradient_checkpointing: use gradient checkpointing to save memory, however training speed will be lower
mode: quantization mode to use, acceptable values [4, 8, 16 or "gptq"]
resume_from_checkpoint: resume training from existings LoRA checkpoint
```

#### Download model from HF hub (`download.py`)

You can use the helper script `python download_model.py <model_name>` to download a model from HF model hub and store it locally. By default it will save the model to `models` of the current path. Make sure to create this folder or change the output location `--output`.

#### Quantization mode selection

On the quantization mode effects on training time and memory usage, see [note](benchmark/README.md). Generally, `16` and `gptq` mode has the best performance, and should be selected to reduce training time. However, most of the time you will hit the memory limitation of the GPU with larger models, which mode `4` and `gptq` provides the best memory saving effect. Overall, `gptq` mode has the best balance between memory saving and training speed.

**NOTE**: To use `gptq` mode, you must install the required package in `cuda_quant_requirements`. Also, since GPTQ is a post-hoc quantization technique, only GTPQ-quantized model can be used for training. Look for model name which contains `gptq` on HF model hub, such as [TheBloke/orca_mini_v2_7B-GPTQ](https://huggingface.co/TheBloke/orca_mini_v2_7B-GPTQ). To correctly load the checkpoint, GPTQ model requires offline checkpoint download as described in previous section.

If you don't use `wandb` and want to disable the prompt at start of every training. Run `wandb disabled`.

### Training on multiple GPUs

By default, on multi-GPUs environment, the training script will load the model weight and split its layers accross different GPUs. This is done to reduce VRAM usage, which allows loading larger model than a single GPU can handle. However, this essentially wastes the power of mutiple GPUs since the computation only run on 1 GPU at a time, thus training time is mostly similar to single GPU run.

To correctly run the training on multiple GPUs in parallel, you can use `torchrun` or `accelerate` to launch distributed training. Check the example in [train_torchrun.sh](scripts/train_torchrun.sh) and [train_accelerate.sh](scripts/train_accelerate.sh). Training time will be drastically lower. However, you should modify `batch_size` to be divisible by the number of GPUs.

```bash
bash scripts/train_torchrun.sh
```

### Evaluation

Simply add `--eval` and `--resume_from_checkpoint` to perform evaluation on validation data.

```bash
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'yahma/alpaca-cleaned' \
    --resume_from_checkpoint output/checkpoint-29/adapter_model/ \
    --eval
```

### Inference (`inference.py`)

This file loads the fine-tuned LoRA checkpoint with the base model and performs inference on the selected dataset. Output is printed to terminal output and stored in `sample_output.txt`.

Example usage:

```bash
python inference.py  \
   --base models/TheBloke_vicuna-13b-v1.3.0-GPTQ/  \
   --delta lora-output  \
   --mode exllama       \
   --type local         \
   --data data/test.json
```

Important parameters:

```
base: model id or path to base model
delta: path to fine-tuned LoRA checkpoint (optional)
data: path to evaluation dataset
mode: quantization mode to load the model, acceptable values [4, 8, 16, "gptq", "exllama"]
type: inference type to use, acceptable values ["local", "api", "guidance"]
```

Note that `gptq` and `exllama` mode are only compatible with GPTQ models. `exllama` is currently provide the best inference speed thus is recommended.

Inference type `local` is the default option (use local model loading). To use inference type `api`, we need an instance of `text-generation-inferece` server described in [deployment](deployment/README.md).

Inference type `guidance` is an advanced method to ensure the structure of the text output (such as JSON). Check the command line `inference.py --help` and [guidance](https://github.com/microsoft/guidance) for more information

### Checkpoint export (`merge_lora_checkpoint.py`)

This file contain scripts that merge the LoRA weights back into the base model
for export to Hugging Face format.
They should help users
who want to run inference in projects like [llama.cpp](https://github.com/ggerganov/llama.cpp)
or [text-generation-inference](https://github.com/huggingface/text-generation-inference).

Currently, we do not support the merge of LoRA to GPTQ base model due to incompatibility issue of quantized weight.

### Deployment

See [deployment](deployment/README.md).

### Quantization with GPTQ

To convert normal HF checkpoint go GPTQ checkpoint we need a conversion script. See [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) and [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) for more information.

### Benchmarking

This [document](benchmark/README.md) provides a comprehensive summary of different quantization methods and some suggestions for efficient training & inference.

### Recommended models

Recommended models to start:

* 7B: [TheBloke/vicuna-7B-v1.3-GPTQ](https://huggingface.co/TheBloke/vicuna-7B-v1.3-GPTQ), [lmsys/vicuna-7b-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3)
* 13B: [TheBloke/vicuna-13b-v1.3.0-GPTQ](https://huggingface.co/TheBloke/vicuna-13b-v1.3.0-GPTQ), [lmsys/vicuna-13b-v1.3](https://huggingface.co/lmsys/vicuna-13b-v1.3)
* 33B: [TheBloke/airoboros-33B-gpt4-1.4-GPTQ](https://huggingface.co/TheBloke/airoboros-33B-gpt4-1.4-GPTQ)

### Resources

- https://github.com/ggerganov/llama.cpp: highly portable Llama inference based on C++
- https://github.com/huggingface/text-generation-inference: production-level LLM serving
- https://github.com/microsoft/guidance: enforce structure to LLM output
- https://github.com/turboderp/exllama/: high-perfomance GPTQ inference
- https://github.com/qwopqwop200/GPTQ-for-LLaMa: GPTQ quantization
- https://github.com/oobabooga/text-generation-webui: a flexible Web UI with support for multiple LLMs back-end
- https://github.com/vllm-project/vllm/: high throughput LLM serving

### Acknowledgements

- @disarmyouwitha [exllama_fastapi](https://github.com/turboderp/exllama/issues/37#issuecomment-1579593517)
- @turboderp [exllama](https://github.com/turboderp/exllama)
- @johnsmith0031 [alpaca_lora_4bit](https://github.com/johnsmith0031/alpaca_lora_4bit)
- @TimDettmers [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- @tloen [alpaca-lora](https://github.com/tloen/alpaca-lora/)
- @oobabooga [text-generation-webui](https://github.com/oobabooga/text-generation-webui)
