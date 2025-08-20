from copy import deepcopy
from typing import Callable, Generator, Optional

import torch

from ...core import VLMBridge
from ...core.util import unwrap_model
from .transformer_config import GLM4VLTransformerConfig


class Glm4VLBridgeBase(VLMBridge):
    """
    Bridge implementation for Glm4VL models.

    This class extends LLMBridge to provide specific configurations and
    optimizations for Glm4VL models, handling the conversion between
    Hugging Face Glm4VL format and Megatron-Core.
    """

    TransformerConfigClass = GLM4VLTransformerConfig

    _DIRECT_MAPPING = {
        "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
        "language_model.output_layer.weight": "lm_head.weight",
        "visual_model.downsample.weight": "model.visual.downsample.weight",
        "visual_model.downsample.bias": "model.visual.downsample.bias",
        "visual_model.merger.gate_proj.weight": "model.visual.merger.gate_proj.weight",
        "visual_model.merger.up_proj.weight": "model.visual.merger.up_proj.weight",
        "visual_model.merger.down_proj.weight": "model.visual.merger.down_proj.weight",
        "visual_model.merger.proj.weight": "model.visual.merger.proj.weight",
        "visual_model.merger.post_projection_norm.weight": "model.visual.merger.post_projection_norm.weight",
        "visual_model.merger.post_projection_norm.bias": "model.visual.merger.post_projection_norm.bias",
        "visual_model.embeddings.position_embedding.weight": "model.visual.embeddings.position_embedding.weight",
        "visual_model.patch_embed.proj.weight": "model.visual.patch_embed.proj.weight",
        "visual_model.patch_embed.proj.bias": "model.visual.patch_embed.proj.bias",
        "visual_model.post_conv_layernorm.weight": "model.visual.post_conv_layernorm.weight",
        "visual_model.post_layernorm.weight": "model.visual.post_layernorm.weight",
    }
    _ATTENTION_MAPPING = {
        # language
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.bias": [
            "model.language_model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.language_model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.language_model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.weight": [
            "model.language_model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_proj.weight": [
            "model.language_model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.input_layernorm.weight"
        ],
    }

    _MLP_MAPPING = {
        "mlp.linear_fc1.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "mlp.linear_fc2.weight": [
            "model.language_model.language_model.layers.{layer_number}.mlp.down_proj.weight"
        ],
        "mlp.shared_experts.linear_fc2.weight": [
            "model.language_model.layers.{layer_number}.mlp.shared_experts.down_proj.weight"
        ],
        "mlp.linear_fc1.weight": [
            "model.language_model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.language_model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.shared_experts.linear_fc1.weight": [
            "model.language_model.layers.{layer_number}.mlp.shared_experts.gate_proj.weight",
            "model.language_model.layers.{layer_number}.mlp.shared_experts.up_proj.weight",
        ],
        "pre_mlp_layernorm.weight": [
            "model.language_model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "mlp.router.weight": [
            "model.language_model.layers.{layer_number}.mlp.gate.weight"
        ],
        "mlp.router.expert_bias": [
            "model.language_model.layers.{layer_number}.mlp.gate.e_score_correction_bias"
        ],
        "mlp.experts.linear_fc1.weight": [
            "model.language_model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
            "model.language_model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
        ],
        "mlp.experts.linear_fc2.weight": [
            "model.language_model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight"
        ],
    }

    _VISUAL_MAPPING = {
        # vision
        "vision_model.blocks.{layer_number}.attn.proj.weight": [
            "model.visual.blocks.{layer_number}.attn.proj.weight"
        ],
        "vision_model.blocks.{layer_number}.attn.qkv.weight": [
            "model.visual.blocks.{layer_number}.attn.qkv.weight"
        ],
        "vision_model.blocks.{layer_number}.norm1.weight": [
            "model.visual.blocks.{layer_number}.norm1.weight"
        ],
        "visual_model.blocks.{layer_number}.mlp.gate_proj.weight": [
            "model.visual.blocks.23.mlp.gate_proj.weight",
        ],
        "visual_model.blocks.{layer_number}.mlp.up_proj.weight": [
            "model.visual.blocks.{layer_number}.mlp.up_proj.weight",
        ],
        "visual_model.blocks.{layer_number}.mlp.down_proj.weight": [
            "model.visual.blocks.{layer_number}.mlp.down_proj.weight",
        ],
        "visual_model.blocks.{layer_number}.norm2.weight": [
            "model.visual.blocks.{layer_number}.norm2.weight"
        ],
    }

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        """
        Map MCore weight names to Hugging Face weight names.

        Args:
            mcore_weights_name: MCore weight name

        Returns:
            list: Corresponding Hugging Face weight names
        """
        assert (
            "_extra_state" not in mcore_weights_name
        ), "extra_state should not be loaded"

        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]

        if "visual_model" in mcore_weights_name:
            return self._weight_name_mapping_visual(mcore_weights_name)
        if "self_attention" in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        else:
            raise NotImplementedError(
                f"Unsupported parameter name: {mcore_weights_name}"
            )

    def _weight_name_mapping_visual(self, name: str):
        split_name = name.split(".")
        layer_number = split_name[2]
        split_name[2] = "{layer_number}"
        key = ".".join(split_name)
        convert_names = []
        mapping_names = self._VISUAL_MAPPING[key]
        convert_names.extend(
            [x.format(layer_number=layer_number) for x in mapping_names]
        )
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    # adapted from qwen vl
    def _weight_name_mapping_attention(self, name: str) -> list[str]:
        split_name = name.split(".")
        layer_number = split_name[3]
        split_name[3] = "{layer_number}"
        key = ".".join(split_name)
        convert_names = []
        mapping_names = self._ATTENTION_MAPPING[key]
        convert_names.extend(
            [x.format(layer_number=layer_number) for x in mapping_names]
        )
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    # adapted from deepseek v3
    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        layer_number = name.split(".")[3]
        convert_names = []
        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                if "{expert_id}" in mapping_names[0]:
                    expert_id = name.split("weight")[-1]
                    convert_names.extend(
                        [
                            x.format(layer_number=layer_number, expert_id=expert_id)
                            for x in mapping_names
                        ]
                    )
                else:
                    convert_names.extend(
                        [x.format(layer_number=layer_number) for x in mapping_names]
                    )
                break
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        """
        Export MCore weights to Hugging Face format.

        Takes MCore weight names and tensor, outputs Hugging Face weight names and tensors.
        Due to MCore's runtime optimizations involving weight merging, output can be a list.

        Args:
            mcore_weights_name: MCore weight name
            mcore_weights: MCore weight tensor

        Returns:
            tuple: (hf_names, hf_weights) - lists of Hugging Face weight names and tensors

        Raises:
            NotImplementedError: If the parameter name is unsupported
        """
        hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
        if len(hf_names) == 1:
            return [hf_names[0]], [mcore_weights]
        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            # split qkv
            assert len(hf_names) == 3
            # split qkv
            num_key_value_heads = self.hf_config.num_key_value_heads
            hidden_dim = self.hf_config.hidden_size
            num_attention_heads = self.hf_config.num_attention_heads

            if "vision_model" in mcore_weights_name:
                num_attention_heads = self.hf_config.vision_config.num_heads
                num_key_value_heads = self.hf_config.vision_config.num_heads
            head_dim = getattr(
                self.hf_config, "head_dim", hidden_dim // num_attention_heads
            )
            out_shape = (
                [num_key_value_heads, -1, hidden_dim]
                if ".bias" not in mcore_weights_name
                else [num_key_value_heads, -1]
            )
            qkv = mcore_weights.view(*out_shape)
            q_len = head_dim * num_attention_heads // num_key_value_heads
            k_len = head_dim
            v_len = head_dim
            single_out_shape = (
                [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]
            )
            q = qkv[:, :q_len].reshape(*single_out_shape)
            k = qkv[:, q_len : q_len + k_len].reshape(*single_out_shape)
            v = qkv[:, q_len + k_len :].reshape(*single_out_shape)
            return hf_names, [q, k, v]

        elif (
            "linear_fc1.weight" in mcore_weights_name
            or "linear_fc1.bias" in mcore_weights_name
        ):
            # split gate_proj and up_proj
            assert len(hf_names) == 2
            gate, up = mcore_weights.chunk(2)
            return hf_names, [gate, up]
        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Import Hugging Face weights to MCore format.

        Takes Hugging Face weight names and tensors, outputs MCore weight tensor.
        Due to MCore's runtime optimizations involving weight merging, input is a list.

        Args:
            mcore_weights_name: MCore weight name
            hf_weights: List of Hugging Face weight tensors

        Returns:
            torch.Tensor: MCore weight tensor

        Raises:
            NotImplementedError: If the parameter name is unsupported
        """
        if len(hf_weights) == 1:
            return hf_weights[0]
        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            # merge qkv
            assert len(hf_weights) == 3
            num_key_value_heads = self.hf_config.num_key_value_heads
            hidden_dim = self.hf_config.hidden_size
            num_attention_heads = self.hf_config.num_attention_heads
            if "vision_model" in mcore_weights_name:
                num_attention_heads = self.hf_config.vision_config.num_heads
                num_key_value_heads = self.hf_config.vision_config.num_heads
            head_dim = getattr(
                self.hf_config, "head_dim", hidden_dim // num_attention_heads
            )
            group_dim = head_dim * num_attention_heads // num_key_value_heads
            q, k, v = hf_weights
            # q k v might be tp split
            real_num_key_value_heads = q.shape[0] // group_dim
            q = q.view(
                [
                    real_num_key_value_heads,
                    group_dim,
                    -1,
                ]
            )
            k = k.view([real_num_key_value_heads, head_dim, -1])
            v = v.view([real_num_key_value_heads, head_dim, -1])
            out_shape = [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]

            qkv = torch.cat([q, k, v], dim=1).view(*out_shape).contiguous()
            return qkv
        elif (
            "linear_fc1.weight" in mcore_weights_name
            or "linear_fc1.bias" in mcore_weights_name
        ):
            # merge gate_proj and up_proj
            assert len(hf_weights) == 2
            gate, up = hf_weights
            return torch.cat([gate, up], dim=0)
        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _get_layer_number(self, vpp_rank: int, local_layer_number: int, models) -> int:
        # map vpp layer number to global layer number
        unwrapped_model = unwrap_model(models[vpp_rank])
        global_layer_number = (
            unwrapped_model.language_model.decoder.layers[
                local_layer_number
            ].layer_number
            - 1
        )
        return global_layer_number

    def _weight_name_mapping_mcore_local_to_global(
        self, model: torch.nn.Module, consider_ep: bool = True
    ) -> dict[str, str]:
        """
        Map local weight names to global weight names, supporting VPP and EP.

        Args:
            model: The model instance

        Returns:
            dict: Mapping from local weight names to global weight names
        """

        # vpp
        local_layer_to_global_layer = {}
        model = unwrap_model(model)
        if hasattr(model, "language_model") and hasattr(
            model.language_model, "decoder"
        ):
            for idx, layer in enumerate(model.language_model.decoder.layers):
                local_layer_to_global_layer[idx] = layer.layer_number - 1
        all_param_names = [
            k for k in model.state_dict().keys() if "_extra_state" not in k
        ]
        ret = {}
        for param_name in all_param_names:
            keyword = "language_model.decoder.layers."
            if keyword in param_name:
                layer_idx = int(param_name.split(keyword)[1].split(".")[0])
                global_layer_idx = local_layer_to_global_layer[layer_idx]
                ret[param_name] = param_name.replace(
                    f"layers.{layer_idx}.", f"layers.{global_layer_idx}."
                )
            else:
                ret[param_name] = param_name

        return ret
