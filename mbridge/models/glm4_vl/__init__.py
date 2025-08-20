from typing import Callable, Generator, Optional

import torch

from ...core import register_model
from .base_bridge import Glm4VLBridgeBase
from megatron.core.transformer.enums


@register_model("glm4v")
class Glm4VLBridgeDense(Glm4VLBridgeBase):
    def _model_provider(
        self, post_model_creation_callbacks: list[Callable[[torch.nn.Module], None]]
    ):
        pass

    def _build_config(self):
        return self._build_base_config(

            moe_router_dtype="fp32",
            disable_bf16_reduced_precision_matmul=True,
            # other
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
        )


@register_model("glm4v_moe")
class Glm4VLBridgeMoe(Glm4VLBridgeBase):
    def _model_provider(
        self, post_model_creation_callbacks: list[Callable[[torch.nn.Module], None]]
    ):
        pass

    def _build_config(self):
        hf_config = self.hf_config
        return self._build_base_config(
            attention_backend=AttnBackend.fused,
            layernorm_epsilon=hf_config.rms_norm_eps,
            ffn_hidden_size=hf_config.intermediate_size,
            moe_router_dtype="fp32",
            disable_bf16_reduced_precision_matmul=True,
            # other
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
        )
