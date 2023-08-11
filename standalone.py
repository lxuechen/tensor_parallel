# torchrun --nproc-per-node 2 --master_port=1234 standalone.py
import os
from typing import Sequence

import torch.distributed
import transformers
from peer_lm import constants
from peer_lm.models.flash_llama_v2 import LlamaForCausalLM
from torch import distributed as dist, nn
from torch.nn.modules import conv

import tensor_parallel as tp
from tensor_parallel.config import Config
from tensor_parallel.state_actions import Scale, Split, SplitInGroupedChunks


def get_local_rank():
    return int(os.getenv("LOCAL_RANK", -1))


def get_world_size():
    return int(os.getenv("WORLD_SIZE", 1))


def get_default_config(module: nn.Module, device_ids: Sequence[torch.device]) -> tuple[dict, dict, dict]:
    """Make a generic config that wraps individual linear, embedding and convolutional layers"""
    emb_weights = {m.weight for m in module.modules() if isinstance(m, (nn.Embedding, nn.EmbeddingBag))}

    state_rules = {}
    input_rules = {}
    output_rules = {}
    for name, module in module.named_modules():
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
            assert module.max_norm is None or module.norm_type < 2
            assert getattr(module, "bias", None) is None or module.bias.shape == module.embedding_dim
            state_rules[f"^{name}.weight$"] = Split(world_size=len(device_ids), dim=1)
            if hasattr(module, "bias"):
                state_rules[f"^{name}.bias$"] = Split(world_size=len(device_ids), dim=0)
            output_rules[f"^{name}$"] = {0: "gather -1"}
        elif isinstance(module, nn.Linear) and "lora_A" not in name and "lora_B" not in name:
            assert module.weight.shape == (module.out_features, module.in_features)
            assert module.bias is None or module.bias.shape == (module.out_features,)
            if module.weight not in emb_weights:  # regular linear layer
                state_rules[f"^{name}.(weight|bias)$"] = Split(world_size=len(device_ids), dim=0)
                output_rules[f"^{name}$"] = {0: "gather -1"}
            else:
                # linear weight tied with embeddings; this is a popular special case for language models;
                # since embedding weight will be sliced over dim 1, we should adapt to the input-sliced weight
                input_rules[f"^{name}$"] = {0: Split(world_size=len(device_ids), dim=-1)}
                output_rules[f"^{name}$"] = {0: "sum"}
                if module.bias is not None:
                    state_rules[f"^{name}.bias$"] = Scale(world_size=len(device_ids))
        elif isinstance(module, conv._ConvNd) and module.groups == 1:
            shape = [module.out_channels, module.in_channels] + list(module.kernel_size)
            shape[:2] = shape[:2][::-1] if module.transposed else shape[:2]
            shape = tuple(shape)
            assert module.weight.shape == shape, f"{module.weight.shape} != {shape}"
            assert module.bias is None or module.bias.shape == (module.out_channels,), module.bias.shape
            state_rules[f"^{name}.weight$"] = (
                Split(world_size=len(device_ids), dim=1)
                if module.transposed
                else Split(world_size=len(device_ids), dim=0)
            )
            if module.bias is not None:
                state_rules[f"^{name}.bias$"] = Split(world_size=len(device_ids), dim=0)
            output_rules[f"^{name}$"] = {0: "gather 1"}
        elif isinstance(module, conv._ConvNd) and module.groups != 1:
            # group conv: split each group individually over input channels to avoid changing module.groups
            groups = module.groups
            shape = [module.out_channels // groups, module.in_channels // groups] + list(module.kernel_size)
            shape[:2] = shape[:2][::-1] if module.transposed else shape[:2]
            shape[0] *= module.groups
            shape = tuple(shape)
            assert module.weight.shape == shape, f"{module.weight.shape} != {shape}"
            assert module.bias is None or module.bias.shape == (module.out_channels,), module.bias.shape
            if not module.transposed:
                state_rules[f"^{name}.weight$"] = Split(world_size=len(device_ids), dim=1)
            else:
                state_rules[f"^{name}.weight$"] = SplitInGroupedChunks(
                    world_size=len(device_ids), dim=0, num_groups=groups, chunk_size=1
                )
            if module.bias is not None:
                state_rules[f"^{name}.bias$"] = Scale(world_size=len(device_ids))
            input_rules[f"^{name}$"] = {
                0: SplitInGroupedChunks(world_size=len(device_ids), dim=1, num_groups=groups, chunk_size=1)
            }
            output_rules[f"^{name}$"] = {0: "sum"}
    return state_rules, input_rules, output_rules


dist.init_process_group(backend="nccl", rank=get_local_rank(), world_size=get_world_size())
pg = dist.distributed_c10d._get_default_group()
torch.cuda.set_device(get_local_rank())

all_devices_ids = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
current_device = torch.device(torch.cuda.current_device())
device_ids = [current_device]
print(device_ids)

model = LlamaForCausalLM.from_pretrained(
    constants.SHARED_MODEL_DIR / "llama-teeny", torch_dtype=torch.bfloat16
)
tokenizer = transformers.AutoTokenizer.from_pretrained(constants.SHARED_MODEL_DIR / "llama-teeny", use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

state_rules, input_rules, output_rules = get_default_config(model, device_ids=all_devices_ids)
for rules in [state_rules, input_rules, output_rules]:
    keys = list(rules.keys())
    for key in keys:
        if 'lm_head' in key:
            rules.pop(key)
config = Config(state_rules, input_rules, output_rules, {})
model, _ = tp.tensor_parallel(model, distributed=True, device_ids=device_ids, config=config)
tensors = tokenizer(["how are you?", "I love you and you too."], return_tensors="pt", padding=True)
tensors = {
    "input_ids": tensors["input_ids"],
    "attention_mask": tensors["attention_mask"],
    "labels": tensors["input_ids"],
}

with torch.enable_grad():
    model.train()
    tensors = {k: v.to(current_device) for k, v in tensors.items()}
    outputs = model(**tensors, return_dict=True, output_hidden_states=True)
    print(outputs.logits)
    print(outputs.logits.requires_grad)
    print(outputs.keys())

    hs = outputs.hidden_states
    s0 = hs[0]
    print(s0)
    print(s0.requires_grad)

    print(outputs.loss, outputs.loss.requires_grad)
