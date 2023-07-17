# Deployment

This directory provides a quick overview for deployment of local LLMs as an API service.

### Text Generation Inference (HF) (`text-generation-inference`)

[text-generation-inference](https://github.com/huggingface/text-generation-inference/) is a Rust, Python and gRPC server for LLMs. It is used in production at HuggingFace to power their widgets in the model hub. This is the recommended option to serve LLMs at production level with large number of requests and high availability.

#### Pros

* Fast inference speed due to optimized CUDA kernel (FlashAttention)
* Support model sharding to parallel model execution on multiple GPUs
* Provide streaming + blocking APIs
* Support Continuous batching for increased throughput
* Production-ready (tracing, monitoring)
* Support 8-bit and GPTQ quantization

#### Cons

* Don't support LoRA loading
* GPTQ speed is not the best
* Need to compile from source (Rust + CUDA kernel) if hosted natively (we can use provided Docker container to streamline the process).

Example script to run the inference server is provided in [run_inference_server.sh](../scripts/run_inference_server.sh). Replace your model name in `model=` and run:

```bash
bash scripts/run_inference_server.sh
```

Default port of the server is `8080`. You can then query the model using either the `/generate` or `/generate_stream` routes:

```shell
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17}}' \
    -H 'Content-Type: application/json'
```

```shell
curl 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17}}' \
    -H 'Content-Type: application/json'
```

or from Python:

```shell
pip install text-generation
```

```python
from text_generation import Client

client = Client("http://127.0.0.1:8080")
print(client.generate("What is Deep Learning?", max_new_tokens=17).generated_text)

text = ""
for response in client.generate_stream("What is Deep Learning?", max_new_tokens=17):
    if not response.token.special:
        text += response.token.text
print(text)
```

Check more examples and documentations in [documents](https://github.com/huggingface/text-generation-inference/).

**NOTE**: Since `text-generation-inferece` does not support LoRA at the moment, we need to export the checkpoint to merge LoRA with the original base model as mentioned in [README](../README.md).

### FastAPI + ExLlama (`fastapi_server.py`)

Derived from [source](https://github.com/turboderp/exllama/issues/37#issuecomment-1579593517).
A simple stateless API server for ExLlama is provided in [fastapi_server.py](fastapi_server.py).

#### Pros

* Very fast inference speed for GPTQ quantized model
* Support LoRA loading
* Provide streaming + blocking APIs
* Has a quick Chat-UI demo

#### Cons

* Don't support Continuous batching
* Not production level

To launcher the server, use this command:

```bash
python fastapi_server.py -d <path_to_model> -l <path_to_optional_lora>
```

Default endpoint is `localhost:8080`. Sample API request is provided in [fastapi_request.py](fastapi_request.py). Also a chat UI is provided when accessing the default URL on a browser (suitable for testing purpose).

### VLLM

[Improve inference throughput with PagedAttention](https://github.com/vllm-project/vllm).

*To be updated.*
