import sys

import torch
from peft import PeftConfig, PeftModel
from transformers import LlamaForCausalLM  # noqa: F402
from transformers import AutoModelForCausalLM, LlamaTokenizer

model_name = sys.argv[1]
peft_config = PeftConfig.from_pretrained(model_name)
base_model = peft_config.base_model_name_or_path
print("Loading model", base_model)

tokenizer = LlamaTokenizer.from_pretrained(base_model)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    model_name,
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

lora_weight = lora_model.base_model.model.model.layers[0].self_attn.q_proj.weight

assert torch.allclose(first_weight_old, first_weight)

# merge weights - new merging method from peft
lora_model = lora_model.merge_and_unload()

lora_model.train(False)

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

output_name = model_name + "-merged"
LlamaForCausalLM.save_pretrained(
    base_model, output_name, state_dict=deloreanized_sd, max_shard_size="400MB"
)
tokenizer.save_pretrained(output_name)
