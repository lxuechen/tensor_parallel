from typing import Sequence

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, BertModel

from tensor_parallel import TensorParallel, TensorParallelPreTrainedModel, tensor_parallel
from tensor_parallel.pretrained_model import find_predefined_tensor_parallel_config


@pytest.mark.parametrize("use_config", [False, True])
@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3])
@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
def test_forward_gpt2_like(use_config, devices, model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).float().to(devices[0])

    inp1 = torch.randint(1, 1000, size=(2, 3), device=devices[0])
    inp2 = torch.randint(1, 1000, size=(2, 1), device=devices[0])
    inp3 = torch.randint(1, 1000, size=(2, 2), device=devices[0])

    out1_ref = model(inp1, use_cache=True, output_hidden_states=True)
    out2_ref = model(inp2, use_cache=True, past_key_values=out1_ref.past_key_values)
    out3_ref = model(inp3, use_cache=True, past_key_values=out2_ref.past_key_values)

    tp_config = None
    if use_config:
        tp_config = find_predefined_tensor_parallel_config(model.config, devices)
    model_tp = TensorParallel(model, devices, config=tp_config)
    del model

    out1 = model_tp(inp1, use_cache=True, output_hidden_states=True)
    out2 = model_tp(inp2, use_cache=True, past_key_values=out1.past_key_values)
    out3 = model_tp(inp3, use_cache=True, past_key_values=out2.past_key_values)

    torch.testing.assert_close(out1_ref.hidden_states[-1], out1.hidden_states[-1], atol=3e-3, rtol=1e-05)
    torch.testing.assert_close(out1_ref.logits, out1.logits, atol=3e-3, rtol=1e-05)
    torch.testing.assert_close(out2_ref.logits, out2.logits, atol=3e-3, rtol=1e-05)
    torch.testing.assert_close(out3_ref.logits, out3.logits, atol=3e-3, rtol=1e-05)


@pytest.mark.parametrize("use_config", [False, True])
@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3])
@pytest.mark.parametrize("model_name", ["t5-small"])
def test_forward_t5_like(use_config, devices, model_name):
    model = T5ForConditionalGeneration.from_pretrained(model_name, low_cpu_mem_usage=True).float().to(devices[0])

    enc = torch.randint(1, 1000, size=(2, 3), device=devices[0])
    dec1 = torch.randint(1, 1000, size=(2, 3), device=devices[0])
    dec2 = torch.randint(1, 1000, size=(2, 1), device=devices[0])
    dec3 = torch.randint(1, 1000, size=(2, 2), device=devices[0])

    out1_ref = model(enc, decoder_input_ids=dec1, use_cache=True, output_hidden_states=True)
    out2_ref = model(enc, decoder_input_ids=dec2, use_cache=True, past_key_values=out1_ref.past_key_values)
    out3_ref = model(enc, decoder_input_ids=dec3, use_cache=True, past_key_values=out2_ref.past_key_values)

    tp_config = None
    if use_config:
        tp_config = find_predefined_tensor_parallel_config(model.config, devices)
    model_tp = TensorParallel(model, devices, config=tp_config)
    del model

    out1 = model_tp(enc, decoder_input_ids=dec1, use_cache=True, output_hidden_states=True)
    out2 = model_tp(enc, decoder_input_ids=dec2, use_cache=True, past_key_values=out1_ref.past_key_values)
    out3 = model_tp(enc, decoder_input_ids=dec3, use_cache=True, past_key_values=out2_ref.past_key_values)

    torch.testing.assert_close(
        out1_ref.decoder_hidden_states[-1], out1.decoder_hidden_states[-1], atol=3e-3, rtol=1e-05
    )
    torch.testing.assert_close(out1_ref.logits, out1.logits, atol=3e-3, rtol=1e-05)
    torch.testing.assert_close(out2_ref.logits, out2.logits, atol=3e-3, rtol=1e-05)
    torch.testing.assert_close(out3_ref.logits, out3.logits, atol=3e-3, rtol=1e-05)


@pytest.mark.parametrize("use_config", [False, True])
@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3])
@pytest.mark.parametrize("model_name", ["bert-base-uncased"])
def test_forward_bert_like(use_config, devices, model_name):
    model = BertModel.from_pretrained(model_name).to(devices[0])

    inp1 = torch.randint(1, 1000, size=(2, 3), device=devices[0])
    inp2 = torch.randint(1, 1000, size=(2, 1), device=devices[0])
    inp3 = torch.randint(1, 1000, size=(2, 2), device=devices[0])

    out1_ref = model(inp1, output_hidden_states=True)
    out2_ref = model(inp2, output_hidden_states=True)
    out3_ref = model(inp3, output_hidden_states=True)

    tp_config = None
    if use_config:
        tp_config = find_predefined_tensor_parallel_config(model.config, devices)
    model_tp = TensorParallel(model, devices, config=tp_config)
    del model

    out1 = model_tp(inp1, output_hidden_states=True)
    out2 = model_tp(inp2, output_hidden_states=True)
    out3 = model_tp(inp3, output_hidden_states=True)

    torch.testing.assert_close(out1_ref.hidden_states[-1], out1.hidden_states[-1], atol=3e-3, rtol=1e-05)
    torch.testing.assert_close(out2_ref.hidden_states[-1], out2.hidden_states[-1], atol=3e-3, rtol=1e-05)
    torch.testing.assert_close(out3_ref.hidden_states[-1], out3.hidden_states[-1], atol=3e-3, rtol=1e-05)


@pytest.mark.parametrize("generate_kwargs", [{"num_beams": 3}, {}, {"top_p": 0.5}])
@pytest.mark.parametrize("model_name", ["t5-small", "bigscience/bloom-560m"])
@pytest.mark.parametrize("devices", [("cpu",), ("cpu",) * 2, ("cpu",) * 3])
def test_generate(generate_kwargs, model_name, devices):
    def _generate_scores(model, tokenizer, prompt, generate_kwargs):
        scores_tuple = model.generate(
            tokenizer([prompt], return_tensors="pt")["input_ids"].to(devices[0]),
            min_length=5,
            return_dict_in_generate=True,
            output_scores=True,
            **generate_kwargs,
        ).scores
        return torch.stack([scores[0] for scores in scores_tuple], dim=0)

    def _assert_scores_allclose_long_enough(
        first_scores: Sequence[torch.Tensor], second_scores: Sequence[torch.Tensor]
    ) -> int:
        for i in range(3):
            torch.testing.assert_close(
                first_scores[i],
                second_scores[i],
                atol=3e-3,
                rtol=1e-05,
                msg=lambda msg: f"Diverged at {'%d%s' % (i + 1,'tsnrhtdd'[((i + 1)//10%10!=1)*((i + 1)%10<4)*(i + 1)%10::4])} token: {msg}",
            )

    if model_name == "t5-small":
        model = T5ForConditionalGeneration.from_pretrained(model_name, low_cpu_mem_usage=True).float().to(devices[0])
    else:
        model = (
            transformers.AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).float().to(devices[0])
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "Translate from German to English: How are you?"

    scores_ref = _generate_scores(model, tokenizer, prompt, generate_kwargs)

    model_tp = tensor_parallel(model, devices)
    del model

    scores = _generate_scores(model_tp, tokenizer, prompt, generate_kwargs)

    _assert_scores_allclose_long_enough(scores_ref, scores)


@pytest.mark.parametrize("use_predefined_config", [False, True])
@pytest.mark.parametrize("model_name", ["t5-small"])
@pytest.mark.parametrize("sharded", [False, True])
def test_encoder(use_predefined_config, model_name, sharded):
    devices = ["cpu"] * 2
    model = T5ForConditionalGeneration.from_pretrained(model_name, low_cpu_mem_usage=True).float().to(devices[0])

    input = torch.randint(1, 1000, size=(2, 3), device=devices[0])
    out_ref = model.get_encoder()(input)

    if not use_predefined_config:
        model.config.architectures = ["Pretend we don't know this architecture"]
    model_tp = tensor_parallel(model, devices, sharded=sharded)
    assert isinstance(model_tp, TensorParallelPreTrainedModel)
    del model

    out = model_tp.get_encoder()(input)
    torch.testing.assert_close(out_ref, out, atol=3e-3, rtol=1e-05)
