"""
app/model.py
Engine interface + SegFormer loader.
Matches the notebook implementation exactly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import SegformerForSemanticSegmentation

from app.config import CropConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SegformerAdapter — identical to notebook
# ---------------------------------------------------------------------------

class SegformerAdapter(nn.Module):
    def __init__(self, net: SegformerForSemanticSegmentation) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(pixel_values=x).logits          # 1/4 resolution
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return logits


# ---------------------------------------------------------------------------
# BaseEngine protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class BaseEngine(Protocol):
    """Minimal interface every inference engine must satisfy."""

    @property
    def model_id(self) -> str: ...

    @property
    def device_name(self) -> str: ...

    def predict(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Run inference on a single pre-processed CHW float32 tensor.

        Args:
            tensor: shape (1, C, H, W) or (C, H, W) float32.

        Returns:
            uint8 numpy array of shape (H, W) containing argmax class ids.
        """
        ...


# ---------------------------------------------------------------------------
# SegformerEngine
# ---------------------------------------------------------------------------

def _adapt_state_dict_keys(state_dict: dict, target_keys: set) -> dict:
    """
    Auto-detect and remap SegFormer state_dict keys between:
      - New transformers (>=4.49): 'stages.X', 'linear_projections.X', 'q_proj', etc.
      - Older transformers (<=4.48): 'encoder.patch_embeddings.X', 'linear_c.X', 'query', etc.
    Works 100% offline with zero data download.
    """
    import re
    # If all target keys or most target keys are already present, no remapping needed
    if any(k in target_keys for k in state_dict):
        overlap = sum(1 for k in state_dict if k in target_keys)
        if overlap == len(target_keys):
            return state_dict

    # Check if state_dict has 'stages' while target expects 'encoder'
    has_stages = any("segformer.stages." in k for k in state_dict)
    target_expects_encoder = any("segformer.encoder." in k for k in target_keys)

    if has_stages and target_expects_encoder:
        logger.info("Auto-adapting state_dict keys: new transformers (stages) -> older transformers (encoder)")
        new_sd = {}
        for k, v in state_dict.items():
            new_k = k
            new_k = re.sub(r"segformer\.stages\.(\d+)\.patch_embeddings\.", r"segformer.encoder.patch_embeddings.\1.", new_k)
            new_k = re.sub(r"segformer\.stages\.(\d+)\.blocks\.(\d+)\.", r"segformer.encoder.block.\1.\2.", new_k)
            new_k = new_k.replace(".layernorm_before.", ".layer_norm_1.")
            new_k = new_k.replace(".layernorm_after.", ".layer_norm_2.")
            new_k = new_k.replace(".attention.q_proj.", ".attention.self.query.")
            new_k = new_k.replace(".attention.k_proj.", ".attention.self.key.")
            new_k = new_k.replace(".attention.v_proj.", ".attention.self.value.")
            new_k = new_k.replace(".attention.o_proj.", ".attention.output.dense.")
            new_k = new_k.replace(".attention.sequence_reduction.sequence_reduction.", ".attention.self.sr.")
            new_k = new_k.replace(".attention.sequence_reduction.layer_norm.", ".attention.self.layer_norm.")
            new_k = new_k.replace(".mlp.fc1.", ".mlp.dense1.")
            new_k = new_k.replace(".mlp.fc2.", ".mlp.dense2.")
            new_k = re.sub(r"segformer\.stages\.(\d+)\.layer_norm\.", r"segformer.encoder.layer_norm.\1.", new_k)
            new_k = re.sub(r"decode_head\.linear_projections\.(\d+)\.", r"decode_head.linear_c.\1.", new_k)
            new_sd[new_k] = v
        return new_sd

    # Check opposite: state_dict has 'encoder' while target expects 'stages'
    has_encoder = any("segformer.encoder." in k for k in state_dict)
    target_expects_stages = any("segformer.stages." in k for k in target_keys)

    if has_encoder and target_expects_stages:
        logger.info("Auto-adapting state_dict keys: older transformers (encoder) -> new transformers (stages)")
        new_sd = {}
        for k, v in state_dict.items():
            new_k = k
            new_k = re.sub(r"segformer\.encoder\.patch_embeddings\.(\d+)\.", r"segformer.stages.\1.patch_embeddings.", new_k)
            new_k = re.sub(r"segformer\.encoder\.block\.(\d+)\.(\d+)\.", r"segformer.stages.\1.blocks.\2.", new_k)
            new_k = new_k.replace(".layer_norm_1.", ".layernorm_before.")
            new_k = new_k.replace(".layer_norm_2.", ".layernorm_after.")
            new_k = new_k.replace(".attention.self.query.", ".attention.q_proj.")
            new_k = new_k.replace(".attention.self.key.", ".attention.k_proj.")
            new_k = new_k.replace(".attention.self.value.", ".attention.v_proj.")
            new_k = new_k.replace(".attention.output.dense.", ".attention.o_proj.")
            new_k = new_k.replace(".attention.self.sr.", ".attention.sequence_reduction.sequence_reduction.")
            new_k = new_k.replace(".attention.self.layer_norm.", ".attention.sequence_reduction.layer_norm.")
            new_k = new_k.replace(".mlp.dense1.", ".mlp.fc1.")
            new_k = new_k.replace(".mlp.dense2.", ".mlp.fc2.")
            new_k = re.sub(r"segformer\.encoder\.layer_norm\.(\d+)\.", r"segformer.stages.\1.layer_norm.", new_k)
            new_k = re.sub(r"decode_head\.linear_c\.(\d+)\.", r"decode_head.linear_projections.\1.", new_k)
            new_sd[new_k] = v
        return new_sd

    return state_dict


class SegformerEngine:
    """
    Production SegFormer inference engine.

    * fp32 always (no autocast)
    * eval mode
    * loads fine-tuned state_dict on top of pretrained mit-b0 backbone
    """

    _PRETRAINED_ID = "nvidia/mit-b0"

    def __init__(self, config: CropConfig, device: str = "cpu") -> None:
        self._config = config
        self._device = torch.device(device)
        self._model_id: str = config.model.id

        num_classes = config.num_classes()
        weights_path = config.weights_path()

        logger.info(
            "Loading SegformerEngine | model_id=%s arch=%s classes=%d device=%s",
            self._model_id,
            config.model.architecture,
            num_classes,
            device,
        )

        # Build backbone (lazy import so non-HF engines / mocked tests don't require transformers at import time)
        from transformers import SegformerForSemanticSegmentation

        hf_net = SegformerForSemanticSegmentation.from_pretrained(
            self._PRETRAINED_ID,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        model = SegformerAdapter(hf_net)

        # Load fine-tuned weights
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"Weights file not found: {weights_path}\n"
                f"Set model.weights in configs/crops/{config.crop}.yaml correctly."
            )

        logger.info("Loading weights from %s", weights_path)
        state_dict = torch.load(weights_path, map_location="cpu")

        # Unwrap common checkpointing conventions
        if isinstance(state_dict, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in state_dict:
                    logger.debug("Unwrapping checkpoint key '%s'", key)
                    state_dict = state_dict[key]
                    break

        # Auto-adapt keys between transformers versions if needed (100% offline)
        target_keys = set(model.state_dict().keys())
        state_dict = _adapt_state_dict_keys(state_dict, target_keys)

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(self._device)

        self._model: SegformerAdapter = model
        logger.info("SegformerEngine ready on %s", self._device)

    # ---- properties --------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device_name(self) -> str:
        """Return human-readable device name (GPU model name on CUDA, 'cpu' on CPU)."""
        if self._device.type == "cuda":
            try:
                return torch.cuda.get_device_name(self._device)
            except Exception:
                return str(self._device)
        return "cpu"

    # ---- inference ---------------------------------------------------------

    def predict(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Args:
            tensor: (C, H, W) or (1, C, H, W) float32.

        Returns:
            (H, W) uint8 numpy array of argmax class ids.
        """
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)   # → (1, C, H, W)

        tensor = tensor.to(self._device, dtype=torch.float32)

        with torch.no_grad():
            logits = self._model(tensor)   # (1, num_classes, H, W)

        mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return mask
