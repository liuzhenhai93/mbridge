from typing import Callable, Generator, Optional

import torch

from ...core import register_model
from .base_bridge import Glm4VLBridgeBase


@register_model("glm4v")
class Glm4VLBridgeDense(Glm4VLBridgeBase):
    def _model_provider(
        self, post_model_creation_callbacks: list[Callable[[torch.nn.Module], None]]
    ):
        pass

    def _build_config(self):
        pass


@register_model("glm4v_moe")
class Glm4VLBridgeMoe(Glm4VLBridgeBase):
    def _model_provider(
        self, post_model_creation_callbacks: list[Callable[[torch.nn.Module], None]]
    ):
        pass

    def _build_config(self):
        pass
