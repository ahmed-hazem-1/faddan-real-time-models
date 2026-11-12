"""Reference inference matching the executed notebook exactly."""
import io
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import SegformerForSemanticSegmentation
from pathlib import Path

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SegformerAdapter(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        logits = self.net(pixel_values=x).logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return logits


def _adapt_state_dict_keys(state_dict: dict, target_keys: set) -> dict:
    import re
    if any(k in target_keys for k in state_dict):
        overlap = sum(1 for k in state_dict if k in target_keys)
        if overlap == len(target_keys):
            return state_dict

    has_stages = any("segformer.stages." in k for k in state_dict)
    target_expects_encoder = any("segformer.encoder." in k for k in target_keys)

    if has_stages and target_expects_encoder:
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

    has_encoder = any("segformer.encoder." in k for k in state_dict)
    target_expects_stages = any("segformer.stages." in k for k in target_keys)

    if has_encoder and target_expects_stages:
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


def load_reference_model(weights_path: str, num_classes: int, device="cpu"):
    net = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    model = SegformerAdapter(net)
    raw_sd = torch.load(weights_path, map_location="cpu")
    target_keys = set(model.state_dict().keys())
    adapted_sd = _adapt_state_dict_keys(raw_sd, target_keys)
    model.load_state_dict(adapted_sd, strict=True)
    model.eval()
    return model.to(device)


def reference_preprocess(img_bytes: bytes, cfg):
    """Exact notebook preprocessing chain."""
    pil = Image.open(io.BytesIO(img_bytes))
    pil = ImageOps.exif_transpose(pil)
    bgr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    # Gray-world WB
    means = [bgr[:, :, c].mean() for c in range(3)]
    target = np.mean(means)
    gains = np.clip(
        target / np.array(means),
        cfg["wb_gain_clip"][0],
        cfg["wb_gain_clip"][1],
    )
    bgr = np.clip(bgr * gains, 0, 255).astype(np.uint8)

    # CLAHE on L channel
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=cfg["clahe"]["clip"],
        tileGridSize=tuple(cfg["clahe"]["tiles"]),
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Glare removal
    lab2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab2[:, :, 0]
    thr = max(np.percentile(L, cfg["glare"]["pctl"]), cfg["glare"]["min_L"])
    mask = (L >= thr).astype(np.uint8)
    frac = mask.mean()
    if 0 < frac <= cfg["glare"]["max_frac"]:
        kernel = np.ones(
            (cfg["glare"]["dilate"], cfg["glare"]["dilate"]), np.uint8
        )
        mask = cv2.dilate(mask, kernel)
        bgr = cv2.inpaint(bgr, mask, cfg["glare"]["radius"], cv2.INPAINT_TELEA)

    # Bilateral
    bgr = cv2.bilateralFilter(
        bgr,
        cfg["bilateral"]["d"],
        cfg["bilateral"]["sigma"],
        cfg["bilateral"]["sigma"],
    )

    # Unsharp mask
    amount = cfg["unsharp"]["amount"]
    radius = cfg["unsharp"]["radius"]
    blurred = cv2.GaussianBlur(bgr.astype(np.float32), (0, 0), radius)
    bgr = np.clip(
        bgr.astype(np.float32) + amount * (bgr.astype(np.float32) - blurred),
        0,
        255,
    ).astype(np.uint8)

    # Brightness
    lab3 = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    meanL = lab3[:, :, 0].mean()
    lo, hi = cfg["brightness"]["lo"], cfg["brightness"]["hi"]
    if not (lo <= meanL <= hi):
        target_L = (lo + hi) / 2
        g = np.clip(
            target_L / meanL,
            cfg["brightness"]["gain_clip"][0],
            cfg["brightness"]["gain_clip"][1],
        )
        bgr = np.clip(bgr.astype(np.float32) * g, 0, 255).astype(np.uint8)

    # Resize + normalize
    h, w = bgr.shape[:2]
    rs = cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
    x = (rs[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0)
    return t, h, w


def reference_predict(model, img_bytes: bytes, cfg, device="cpu"):
    t, h, w = reference_preprocess(img_bytes, cfg)
    t = t.to(device)
    with torch.no_grad():
        pred = model(t).argmax(1)[0].cpu().numpy().astype(np.uint8)
    if (h, w) != (512, 512):
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    return pred
