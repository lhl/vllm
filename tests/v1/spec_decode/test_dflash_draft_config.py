# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest

from vllm.model_executor.models.dflash_draft import _get_dflash_rope_parameters
from vllm.model_executor.models.registry import ModelRegistry
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.eagle import EAGLEConfig


@pytest.mark.parametrize(
    ("model_type", "expected_cls"),
    [
        ("gemma4", "DFlashGemma4ForCausalLM"),
        ("llama", "DFlashLlamaForCausalLM"),
        ("phi3", "DFlashPhi4ForCausalLM"),
        ("phi4mm", "DFlashPhi4MMForCausalLM"),
    ],
)
def test_shisa_dflash_draft_config_routes_by_model_type(
    tmp_path,
    model_type,
    expected_cls,
):
    config = {
        "architectures": ["DFlashDraftModel"],
        "model_type": model_type,
        "hidden_size": 2560,
        "num_hidden_layers": 5,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 8192,
        "vocab_size": 262144,
        "num_target_layers": 42,
        "block_size": 16,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 131072,
        "tie_word_embeddings": False,
        "dflash_config": {
            "mask_token_id": 4,
            "target_layer_ids": [1, 11, 20, 30, 40],
        },
        "dtype": "bfloat16",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    hf_config = get_config(str(tmp_path), trust_remote_code=True)
    assert hf_config.model_type == model_type
    assert hf_config.block_size == 16

    wrapped_config = EAGLEConfig(hf_config, method="dflash", model_type="eagle")
    model_config = SimpleNamespace(
        hf_config=wrapped_config,
        model_impl="auto",
        convert_type="none",
    )

    cls, _ = ModelRegistry.resolve_model_cls(["DFlashDraftModel"], model_config)
    assert cls.__name__ == expected_cls


def test_shisa_dflash_uses_llama_rope_parameters():
    config = SimpleNamespace(
        rope_parameters={
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
            "rope_theta": 500000.0,
        }
    )

    assert _get_dflash_rope_parameters(config) == config.rope_parameters


def test_speculators_dflash_config_routes_by_model_type(tmp_path):
    config = {
        "architectures": ["DFlashDraftModel"],
        "speculators_model_type": "dflash",
        "draft_vocab_size": 262144,
        "mask_token_id": 4,
        "aux_hidden_state_layer_ids": [1, 11, 20, 30, 40],
        "block_size": 16,
        "speculators_config": {
            "algorithm": "dflash",
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {
                    "proposal_type": "greedy",
                    "speculative_tokens": 16,
                    "verifier_accept_k": 1,
                    "accept_tolerance": 0.0,
                }
            ],
            "verifier": {
                "architectures": [],
                "name_or_path": "google/gemma-4-31B-it",
            },
        },
        "transformer_layer_config": {
            "model_type": "gemma4",
            "hidden_size": 2560,
            "num_hidden_layers": 5,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 8192,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 1000000.0,
            },
            "max_position_embeddings": 131072,
            "tie_word_embeddings": False,
            "vocab_size": 262144,
        },
        "dtype": "bfloat16",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    hf_config = get_config(str(tmp_path), trust_remote_code=True)
    assert hf_config.model_type == "gemma4"
    assert hf_config.architectures == ["DFlashDraftModel"]
    assert hf_config.dflash_config == {
        "mask_token_id": 4,
        "target_layer_ids": [1, 11, 20, 30, 40],
    }
    assert hf_config.eagle_aux_hidden_state_layer_ids == [1, 11, 20, 30, 40]

    wrapped_config = EAGLEConfig(hf_config, method="dflash", model_type="eagle")
    model_config = SimpleNamespace(
        hf_config=wrapped_config,
        model_impl="auto",
        convert_type="none",
    )

    cls, _ = ModelRegistry.resolve_model_cls(["DFlashDraftModel"], model_config)
    assert cls.__name__ == "DFlashGemma4ForCausalLM"
