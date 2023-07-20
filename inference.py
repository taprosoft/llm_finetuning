import os
import pickle
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

import fire
import requests
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from text_generation import Client
from tqdm import tqdm
from transformers import GenerationConfig, LlamaTokenizer

from utils.prompter import AlpacaPrompter, count_tokens


def load_model_hf(base, delta, lora_config, mode):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kwargs = {"device_map": "balanced"}
    if mode == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["quantization_config"] = bnb_config
    elif mode == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=0.0,
        )
        kwargs["quantization_config"] = bnb_config
    elif mode == 16:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    tokenizer = LlamaTokenizer.from_pretrained(base, use_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # setup PEFT model
    if os.path.exists(delta):
        model = get_peft_model(model, lora_config)
        adapters_weights = torch.load(delta + "/adapter_model.bin", map_location="cpu")
        set_peft_model_state_dict(model, adapters_weights)

    return model, tokenizer


def generate_local(model, tokenizer, prompts, generation_config, max_new_tokens=512):
    inputs = tokenizer(prompts, padding="longest", return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    with torch.no_grad():
        generation_outputs = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
        )
    generated_texts = tokenizer.batch_decode(
        generation_outputs, skip_special_tokens=True
    )

    return generated_texts


def generate_api(
    client,
    prompt,
    max_new_token=512,
    temperature=0.7,
    repetition_penalty=1.1,
    trial=4,
):
    text = None
    for _ in range(trial):
        try:
            text = ""
            for response in client.generate_stream(
                prompt,
                max_new_tokens=max_new_token,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                stop_sequences=["</s>"],
            ):
                if not response.token.special:
                    text += response.token.text
            break
        except IOError:
            traceback.print_exc()
            time.sleep(4)
            continue
    output = text if text is not None else "(error)"
    return output


def generate_api_exllama(
    client,
    prompt,
    max_new_token=512,
    temperature=0.7,
    repetition_penalty=1.1,
    trial=4,
):
    text = None
    for _ in range(trial):
        try:
            text = ""
            data = {
                "message": prompt,
                "max_new_tokens": max_new_token,
                "temperature": temperature,
                "token_repetition_penalty_max": repetition_penalty,
            }
            r = requests.post(client, json=data, stream=True)
            if r.status_code == 200:
                for chunk in r.iter_content(chunk_size=4):
                    if chunk:
                        try:
                            chunk = chunk.decode("utf-8")
                        except UnicodeDecodeError:
                            chunk = "<?>"
                        text += chunk
            else:
                raise IOError(
                    "Request failed with status code: {}".format(r.status_code)
                )
            break
        except IOError:
            traceback.print_exc()
            time.sleep(4)
            continue
    output = text if text is not None else "(error)"
    return output


# noinspection PyTypeChecker
def run_inference(
    base: str,  # path to base model
    data: str,  # path to evaluation data
    delta: str = "",  # path to delta (finetuned) model
    mode: Union[int, str] = 8,  # model loading mode, 4, 8, 16 or 32
    batch_size: Union[int, str] = "auto",  # batch size, int or 'auto'
    type: str = "local",  # inference type, 'local', 'api' or 'guidance'
    max_new_tokens: int = 512,  # generation max number of new tokens
    temperature: float = 0.2,  # generation temperature
    num_beams: int = 1,  # generation num beams
    repetition_penalty: float = 1.1,  # generation repetition_penalty
    overwrite_instruction: str = None,  # overwrite instruction in the
    # dataset with custom prompt
    prompt_template: str = "vicuna",  # prompt template for generation
    selected_ids: List[int] = (),  # list of ids from the dataset to run
    api_url: str = "http://127.0.0.1:8080",  # URL for remote API call
    guidance_template: str = "",  # guidance output template
    gptq_group_size: int = 128,  # gptq group size
):
    """Main inference entry point"""
    if os.path.exists(data):
        dataset = load_dataset("json", data_files=data)["train"]
    else:
        dataset = load_dataset(data)["train"]
    model = None

    # setup inference instance
    prompter = AlpacaPrompter(prompt_template)
    # support 2 API server: TGI and exllama FastAPI
    client = Client(api_url) if mode != "exllama" else (api_url + "/generate")
    lora_config = LoraConfig.from_pretrained(delta) if delta else None

    if base:
        if type != "api":
            if isinstance(mode, int):
                model, tokenizer = load_model_hf(base, delta, lora_config, mode)
            elif mode == "exllama":
                from utils.loader.exllama_hf_loader import load_model_exllama

                model, tokenizer = load_model_exllama(base, delta)
            elif mode == "gptq":
                from utils.loader.gptq_loader import load_model_gptq

                model, tokenizer = load_model_gptq(
                    base,
                    lora_config,
                    lora_path=delta,
                    load_lora=delta != "",
                    group_size=gptq_group_size,
                )
            elif mode == "autogptq":
                from utils.loader.autogptq_loader import load_model_autogptq

                model, tokenizer = load_model_autogptq(
                    base,
                    lora_path=delta,
                    load_lora=delta != "",
                )
            else:
                raise NotImplementedError(f"Mode '{mode}' is not supported.")

        if model is not None:
            model.eval()

    # Inference
    instructions, labels = [], []
    outputs = []
    inputs = []

    # local inference using model.generate
    for idx, sample in enumerate(tqdm(dataset)):
        if selected_ids and idx not in selected_ids:
            continue

        sample_instruction = (
            sample["instruction"]
            if not overwrite_instruction
            else overwrite_instruction
        )
        instruction = prompter.generate_prompt(sample_instruction, sample["input"])
        instructions.append(instruction)
        labels.append(sample["output"])
        inputs.append(sample["input"])

    if type == "api":
        # remote inference using text-generation-inference server
        generate_func = generate_api if mode != "exllama" else generate_api_exllama
        if batch_size == "auto":
            batch_size = 4 if mode != "exllama" else 1
        for _id in tqdm(range(0, len(instructions), batch_size)):
            futures = []
            pool = ThreadPoolExecutor(max_workers=batch_size)

            for instruction in instructions[_id : _id + batch_size]:
                print(instruction)
                response = pool.submit(
                    generate_func,
                    client,
                    instruction,
                    max_new_tokens,
                    temperature,
                    repetition_penalty,
                )
                futures.append(response)

            for future in futures:
                text = future.result()
                outputs.append(text)
                print(text)

            pool.shutdown(wait=True)

    elif type == "local":
        # normal HF inference
        generation_config = GenerationConfig(
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
        )
        if batch_size == "auto":
            batch_size = 1
        for _id in tqdm(range(0, len(instructions), batch_size)):
            batch_inputs = instructions[_id : _id + batch_size]
            t0 = time.time()
            generated_texts = generate_local(
                model,
                tokenizer,
                batch_inputs,
                generation_config,
                max_new_tokens=max_new_tokens,
            )
            outputs.extend(generated_texts)

            # calc tokens/sec:
            trimmed_texts = [
                pred_text[len(instruction) :]
                for instruction, pred_text in zip(batch_inputs, generated_texts)
            ]
            new_tokens = count_tokens(trimmed_texts, tokenizer)
            t1 = time.time()
            _sec = t1 - t0
            _tokens_sec = new_tokens / _sec

            print(
                f"Output generated in {_sec} ({_tokens_sec} tokens/s, total {new_tokens} tokens)"
            )

            for output, label in zip(generated_texts, labels[_id : _id + batch_size]):
                print("Pred:", output)
                print("Label:", label, "\n")

    elif type == "guidance":
        import guidance

        assert isinstance(mode, int) or mode in [
            "autogptq"
        ], "Guidance mode only support `bitsandbytes` quantization or `autogptq` for now"
        model_guidance = guidance.llms.Transformers(model=model, tokenizer=tokenizer)
        guidance.llms.Transformers.cache.clear()

        # if guidance_template is path to input file
        if os.path.exists(guidance_template):
            with open(guidance_template, "r") as fi:
                guidance_template = fi.read()

        def generate_guidance(model, prompt):
            input_prompt = prompt + guidance_template
            json_maker = guidance(input_prompt)
            output = json_maker(llm=model)
            return str(output)

        for idx, (instruction, label) in tqdm(enumerate(zip(instructions, labels))):
            output = generate_guidance(model_guidance, instruction)
            outputs.append(output)

            print("#{} Input:".format(idx), instruction)
            print("Pred:", output)
            print("Label:", label, "\n")

    else:
        raise NotImplementedError("Inference type {} not supported".format(type))

    assert len(outputs) == len(instructions)

    with open(os.path.join(delta, "eval.pkl"), "wb") as f:
        pickle.dump((instructions, inputs, outputs, labels), f)
    print("Exported results to eval.pkl")

    fo = open(os.path.join(delta, "sample_output.txt"), "w", encoding="utf-8")
    for pred, input, label in zip(outputs, inputs, labels):
        print(pred)
        fo.write("Input: \n" + "-" * 80 + "\n" + input)
        response = prompter.get_response(pred)
        fo.write("\n\nPred:\n" + "-" * 80 + "\n" + response)
        fo.write("\n\nLabel:\n" + "-" * 80 + "\n" + label)
        fo.write("\n\n\n" + "=" * 80 + "\n")

    fo.close()

    print("Exported results to sample_output.txt")


if __name__ == "__main__":
    fire.Fire(run_inference)
