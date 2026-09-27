"""Frozen Hugging Face DINO adapters with explicit token pooling."""
import re

import numpy as np


def pool_tokens(hidden, *, register_tokens: int, method: str):
    if method == "cls":
        return hidden[:, 0]
    if method != "patch_mean":
        raise ValueError("token pooling must be cls or patch_mean")
    if register_tokens < 0 or hidden.shape[1] <= 1 + register_tokens:
        raise ValueError("no image patch tokens to pool")
    # Neither CLS nor DINOv3 register tokens represent image patches.
    return hidden[:, 1 + register_tokens:].mean(dim=1)


class FrozenDinoEncoder:
    def __init__(self, model_id: str, revision: str, *, device: str = "cpu",
                 precision: str = "float32", token_pooling: str = "patch_mean",
                 local_files_only: bool = False):
        if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            raise ValueError("revision must be an immutable 40-character Hugging Face commit hash")
        if token_pooling not in {"cls", "patch_mean"}:
            raise ValueError("unsupported token pooling")
        if precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")
        import torch
        import transformers
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("supported devices are cpu and cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if precision == "bfloat16" and (self.device.type != "cuda" or not torch.cuda.is_bf16_supported()):
            raise ValueError("bfloat16 extraction requires a BF16-capable CUDA device")
        self.dtype = torch.float32 if precision == "float32" else torch.bfloat16
        self.model = AutoModel.from_pretrained(model_id, revision=revision,
            local_files_only=local_files_only, trust_remote_code=False, torch_dtype=self.dtype)
        if self.model.config.model_type not in {"dinov2", "dinov3_vit"}:
            raise ValueError("checkpoint must be DINOv2 or DINOv3 ViT")
        self.processor = AutoImageProcessor.from_pretrained(model_id, revision=revision,
            local_files_only=local_files_only, trust_remote_code=False,
            use_fast=self.model.config.model_type == "dinov3_vit")
        self.model.to(self.device).eval().requires_grad_(False)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.token_pooling = token_pooling
        self.output_dim = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.register_tokens = getattr(self.model.config, "num_register_tokens", 0)
        self.descriptor = {"model_id": model_id, "revision": revision,
            "model_type": self.model.config.model_type, "output_dim": self.output_dim,
            "token_pooling": token_pooling, "register_tokens_excluded": self.register_tokens,
            "precision": precision, "device_type": self.device.type,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "processor": self.processor.to_dict(), "processor_overrides": {
                "do_resize": False, "do_center_crop": False, "do_rescale": False},
            "version": "frozen-hf-dino-v1"}

    def runtime(self) -> dict:
        return {"peak_gpu_memory_mb": self.torch.cuda.max_memory_allocated(self.device) / 1024**2
                if self.device.type == "cuda" else 0.0}

    def encode(self, tiles: list[np.ndarray]) -> np.ndarray:
        if not tiles:
            raise ValueError("empty tile batch")
        for tile in tiles:
            if tile.ndim != 3 or tile.shape[2] != 3 or any(n % self.patch_size for n in tile.shape[:2]):
                raise ValueError(f"tile dimensions must be multiples of patch size {self.patch_size}")
        pixels = self.processor(images=tiles, return_tensors="pt", do_resize=False,
                                do_center_crop=False, do_rescale=False)["pixel_values"]
        if tuple(pixels.shape[-2:]) != tuple(tiles[0].shape[:2]):
            raise ValueError("processor unexpectedly changed tile dimensions")
        with self.torch.inference_mode():
            hidden = self.model(pixel_values=pixels.to(device=self.device, dtype=self.dtype)).last_hidden_state
            features = pool_tokens(hidden, register_tokens=self.register_tokens, method=self.token_pooling)
        return features.float().cpu().numpy()
